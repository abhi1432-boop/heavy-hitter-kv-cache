import torch
from transformers import DynamicCache
from h2o_cache import H2OCache


class H2OCacheAdapter(DynamicCache):
    """Makes H2OCache look like HuggingFace's Cache to phi-2.

    Inherits from DynamicCache so the model's helper methods (get_seq_length,
    __len__, etc.) all work out of the box. We only override update() to route
    storage through our H2OCache, which handles eviction.
    """

    def __init__(self, h2o_cache):
        super().__init__()
        self.h2o_cache = h2o_cache

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        del cache_kwargs  # parent class passes this but H2O doesn't need it
        return self.h2o_cache.update(key_states, value_states, layer_idx)

    def get_seq_length(self, layer_idx=0):
        # the model calls this to figure out position IDs for new tokens
        # we report the current size of layer 0's cache
        cache = self.h2o_cache.key_cache[layer_idx]
        if cache is None:
            return 0
        return cache.shape[2]  # seq_len dimension


def patch_model(model, max_cache_size=20, local_window_size=4):
    num_layers = len(model.model.layers)
    cache = H2OCache(max_cache_size, local_window_size, num_layers)

    for layer_idx, layer in enumerate(model.model.layers):
        def make_hook(idx):
            def hook(*args):
                output = args[2]  # PyTorch calls hook(module, inputs, output)
                _, attn_weights = output
                if attn_weights is not None:
                    cache.update_scores(attn_weights, idx)
                return output
            return hook

        layer.self_attn.register_forward_hook(make_hook(layer_idx))

    adapter = H2OCacheAdapter(cache)
    return cache, adapter
