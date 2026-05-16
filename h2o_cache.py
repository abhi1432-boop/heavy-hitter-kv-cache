import torch


class H2OCache:
    def __init__(self, max_cache_size, local_window_size, num_layers):
        # total number of token slots we're allowed to keep in the cache
        self.max_cache_size = max_cache_size

        # how many of those slots are reserved for the most recent tokens (always kept)
        self.local_window_size = local_window_size

        # one K tensor and one V tensor per layer, starts empty
        self.key_cache = [None] * num_layers
        self.value_cache = [None] * num_layers

        # one score per cached position per layer — tracks who's a heavy hitter
        self.accumulated_scores = [None] * num_layers

    def update(self, key_states, value_states, layer_idx):
        # key_states shape: [batch, heads, 1, head_dim] — just the new token's K
        # value_states shape: [batch, heads, 1, head_dim] — just the new token's V

        if self.key_cache[layer_idx] is None:
            # first token — nothing to concatenate, just store directly
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            # append the new token's K and V onto the end of the existing cache
            # dim=2 is the sequence length dimension: [batch, heads, seq_len, head_dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        # return the full cache so attention can run over all positions
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_scores(self, attn_weights, layer_idx):
        # attn_weights shape: [batch, heads, query_len, key_len]
        # we want one score per key position — sum over batch and heads
        # [-1] on the query dimension takes only the last query (the new token we just generated)
        new_scores = attn_weights[0, :, -1, :].sum(dim=0)  # shape: [cache_len]

        if self.accumulated_scores[layer_idx] is None:
            self.accumulated_scores[layer_idx] = new_scores
        else:
            # the cache just grew by 1 (the new token we appended in update())
            # so we extend the scores tensor by 1 slot (new token starts with score 0)
            self.accumulated_scores[layer_idx] = torch.cat([
                self.accumulated_scores[layer_idx],
                torch.zeros(1, device=new_scores.device)
            ])
            # add this step's attention to the running totals
            self.accumulated_scores[layer_idx] += new_scores

        # evict if the cache has grown past our budget
        cache_size = self.key_cache[layer_idx].shape[2]
        if cache_size > self.max_cache_size:
            self._evict(layer_idx)

    def _evict(self, layer_idx):
        cache_size = self.key_cache[layer_idx].shape[2]
        scores = self.accumulated_scores[layer_idx]

        # the local window is the last `local_window_size` positions — always kept
        local_start = cache_size - self.local_window_size

        # look at only the non-window tokens and find the top heavy hitters among them
        num_heavy_hitters = self.max_cache_size - self.local_window_size
        non_window_scores = scores[:local_start]
        _, top_indices = non_window_scores.topk(min(num_heavy_hitters, non_window_scores.shape[0]))

        # sort so we preserve the original left-to-right order in the cache
        top_indices = top_indices.sort().values

        # local window indices are just the last N positions
        local_indices = torch.arange(local_start, cache_size, device=scores.device)

        # final set of positions to keep: heavy hitters + local window
        keep = torch.cat([top_indices, local_indices])

        # slice the cache and scores tensors down to only the kept positions
        self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, keep, :]
        self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, keep, :]
        self.accumulated_scores[layer_idx] = scores[keep]
