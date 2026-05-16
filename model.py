from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import torch
from patch import patch_model

model_name = "microsoft/phi-2"

# Tokenizer converts raw text into token IDs that the model understands.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# attn_implementation="eager" uses the explicit Python attention loop instead of PyTorch's
# optimized sdpa kernel. Slower, but sdpa hides the attention weights — we need to see them.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
    attn_implementation="eager",
)
model.eval()  # disables dropout etc — we're doing inference, not training

print("done loading")


def generate(prompt, max_new_tokens=30, use_h2o=True, max_cache_size=20, local_window_size=4):
    # if H2O is on, patch the model so the cache uses our eviction logic
    if use_h2o:
        cache, adapter = patch_model(
            model,
            max_cache_size=max_cache_size,
            local_window_size=local_window_size,
        )
    else:
        # baseline still needs a persistent cache across generation steps,
        # otherwise the model only sees one token per step and produces garbage
        cache, adapter = None, DynamicCache()

    # turn prompt into token IDs — shape [1, prompt_len]
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated = input_ids

    with torch.no_grad():
        for step in range(max_new_tokens):
            if step == 0:
                # first pass: feed the whole prompt so the cache gets populated
                model_inputs = generated
            else:
                # subsequent passes: only the newest token — cache holds the rest
                model_inputs = generated[:, -1:]

            outputs = model(model_inputs, past_key_values=adapter, use_cache=True)

            # logits shape: [batch, seq_len, vocab_size]
            # we want the LAST position — that's the prediction for the next token
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # append the predicted token to our running sequence
            generated = torch.cat([generated, next_token], dim=1)

            if use_h2o:
                # peek at layer 0's cache to see eviction working
                print(f"step {step}: cache size = {cache.key_cache[0].shape[2]}")

    # turn token IDs back into a string
    return tokenizer.decode(generated[0], skip_special_tokens=True)


if __name__ == "__main__":
    # phi-2 was trained on this Instruct/Output format — it responds well to it
    prompt = "Instruct: Write a haiku about a cat.\nOutput:"

    print("\n--- baseline (no H2O) ---")
    baseline = generate(prompt, max_new_tokens=60, use_h2o=False)
    print(baseline)

    print("\n--- with H2O (budget=15, window=4) ---")
    h2o_output = generate(prompt, max_new_tokens=60, use_h2o=True, max_cache_size=15, local_window_size=4)
    print(h2o_output)
