# Entry point: loads phi-2, defines a greedy generation loop, and runs three
# experiments to verify H2O works correctly (baseline vs big-budget vs small-budget).

from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import torch
from patch import patch_model

model_name = "microsoft/phi-2"

# Tokenizer turns text into integer IDs and back.
# It's loaded separately from the model because they're two different things:
# the tokenizer is a vocabulary + rules; the model is the trained neural network.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load phi-2's trained weights from disk.
# dtype=float32 keeps weights in full 32-bit precision (slower but accurate).
# attn_implementation="eager" forces phi-2 to use plain Python attention.
# sdpa would be faster but it hides the attention weights — H2O needs to read them
# to decide which tokens are heavy hitters.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
    attn_implementation="eager",
)
# eval() disables dropout and other training-only behavior — we're doing inference.
model.eval()

print("done loading")


def generate(prompt, max_new_tokens=30, h2o_config=None, verbose=False):
    """Greedy generation. If h2o_config is None, run a normal cached baseline.

    h2o_config: dict with keys 'max_cache_size' and 'local_window_size'.
    """
    # Decide which cache to use based on whether H2O is requested.
    if h2o_config is not None:
        # patch_model wires our H2OCache into every attention layer and returns:
        #   cache   — the underlying H2OCache object (for inspection)
        #   past    — the adapter we pass to the model as past_key_values
        #   unpatch — a function that removes the hooks when we're done
        cache, past, unpatch = patch_model(
            model,
            max_cache_size=h2o_config["max_cache_size"],
            local_window_size=h2o_config["local_window_size"],
        )
    else:
        # Baseline mode: use HuggingFace's normal DynamicCache (no eviction).
        # IMPORTANT: this cache must persist across steps. If we pass None each step,
        # phi-2 creates a fresh empty cache every step and the model loses all context
        # after the first token.
        cache, past, unpatch = None, DynamicCache(), None

    # Tokenize the prompt. input_ids shape: [1, prompt_len].
    # The [1, ...] front dimension is the batch — we only have one sequence.
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # `generated` is the running sequence of token IDs. We'll keep appending to it.
    generated = input_ids

    # try/finally guarantees unpatch() runs even if something errors mid-loop.
    # Without this, a crash during H2O generation would leave hooks attached to the
    # model and break the next run.
    try:
        # torch.no_grad() turns off gradient tracking — we're not training,
        # so we don't need PyTorch to record operations for backprop.
        # Saves memory and is slightly faster.
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Decide what to feed the model this step.
                # Step 0: feed the full prompt so the cache gets fully populated.
                # Later steps: only feed the newest token. The cache holds the rest,
                # so attention can still look at the whole history.
                model_inputs = generated if step == 0 else generated[:, -1:]

                # Run the forward pass.
                # past_key_values=past tells the model to use our cache.
                # use_cache=True tells the model "yes, write new K/V into the cache."
                outputs = model(model_inputs, past_key_values=past, use_cache=True)

                # outputs.logits shape: [batch, seq_len, vocab_size].
                # vocab_size is 51200 for phi-2 — one score per possible next token.
                # We want the LAST position's logits — that's the prediction for what
                # comes after the input we just fed in.
                # argmax picks the single highest-scoring token (greedy decoding).
                # keepdim=True makes the result shape [1, 1] instead of [1] so it
                # can be concatenated onto `generated` along the sequence axis.
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

                # Append the newly predicted token to the running sequence.
                # dim=1 is the sequence axis (dim=0 is batch).
                generated = torch.cat([generated, next_token], dim=1)

                # Sanity checks for the H2O run only.
                if verbose and cache is not None:
                    # Cache size = how many tokens are currently remembered (layer 0).
                    # We check layer 0 because all 32 layers grow and shrink together.
                    size = cache.key_cache[0].shape[2]
                    # If this assert fires, _evict has a bug — the cache should
                    # never exceed the configured budget.
                    assert size <= h2o_config["max_cache_size"], "cache exceeded budget"
                    # Print every 10 steps to avoid spam, plus the final step.
                    if step % 10 == 0 or step == max_new_tokens - 1:
                        print(f"  step {step:>3}: cache size = {size}")
    finally:
        # Critical cleanup: remove the forward hooks we attached in patch_model.
        # Without this, hooks persist and the next call to generate() (especially
        # a baseline call) would still fire H2O score updates on a stale cache.
        if unpatch is not None:
            unpatch()

    # Convert the final list of token IDs back into a human-readable string.
    # skip_special_tokens=True drops things like end-of-sequence markers.
    return tokenizer.decode(generated[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Phi-2 was trained heavily on this Q&A-style format.
    # It will respond much better than to a plain English instruction.
    prompt = "Instruct: Write a haiku about a cat.\nOutput:"
    n = 30  # generate 30 new tokens per experiment

    # Experiment 1: baseline — no H2O, full cache, no eviction.
    # This is the reference output we'll compare H2O against.
    print("\n=== BASELINE (full cache) ===")
    baseline = generate(prompt, max_new_tokens=n)
    print(baseline)

    # Experiment 2: H2O with a budget bigger than the cache ever grows.
    # Prompt is ~13 tokens + 30 generated = ~43 tokens, budget is 40.
    # Eviction barely fires (maybe last few steps), so output should still
    # be nearly identical to baseline. Mismatch would indicate a bug in our
    # update() method — when eviction doesn't fire, H2O must be a no-op.
    print("\n=== H2O budget=40 window=8 (should match baseline — minimal eviction) ===")
    big = generate(prompt, max_new_tokens=n, h2o_config={"max_cache_size": 40, "local_window_size": 8}, verbose=True)
    print(big)

    # Experiment 3: aggressive eviction with budget=12.
    # Prompt alone is already 13 tokens, so eviction starts immediately.
    # Output WILL diverge from baseline — that proves eviction is firing and
    # actually affecting the model's predictions.
    print("\n=== H2O budget=12 window=4 (heavy eviction — should diverge) ===")
    small = generate(prompt, max_new_tokens=n, h2o_config={"max_cache_size": 12, "local_window_size": 4}, verbose=True)
    print(small)

    # The verdict: two boolean checks that bracket correct H2O behavior.
    # If both print True, the implementation is provably working:
    #   - big budget matching baseline → H2O is faithful when not evicting
    #   - small budget differing       → H2O actually does something when evicting
    print("\n=== verdict ===")
    print(f"big-budget H2O matches baseline: {big == baseline}")
    print(f"small-budget H2O differs from baseline: {small != baseline}")
