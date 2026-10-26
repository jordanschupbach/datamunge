// Inference-only explicit K/V cache with validated append and exact overflow
// rebuild policy: port of tflow's common/ml/kv_cache.h on top of
// lm::CachedDecoder (examples/nn/common/decoder_transformer.hpp).
//
// The cache is an explicit pair of tensors [N, heads, past, head_width] that
// each Append concatenates with the new keys/values (as tflow feeds and
// fetches them through the session). nn::KvCache offers the same growth with
// an amortised buffer inside nn::multi_head_attention; this model's fused
// "qkv" projection is not that layer's layout, so the state stays explicit
// here, which also keeps tflow's validation guards (shape, length, NaN).
#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/decoder_transformer.hpp"

namespace lm {

struct KvState {
  Tensor keys, values;
};

class CachedTransformerLM {
 public:
  DecoderConfig      config;
  nn::ParameterStore store{9001};

  explicit CachedTransformerLM(DecoderConfig c = {}) : config(c) {
    nn::NoGradGuard guard;
    DecoderTransformer(store, nn::zeros({1, c.context, c.input_classes}), c);
  }

  KvState Empty(std::int64_t batch) const {
    if (batch < 1) throw std::invalid_argument("positive cache batch required");
    const nn::Shape shape{batch, config.heads, 0, config.dimensions / config.heads};
    return {nn::zeros(shape), nn::zeros(shape)};
  }

  void Validate(const Tensor& ids, const KvState& state) const {
    if (!ids.defined() || ids.dtype() != nn::DType::Int32 || ids.dim() != 2 || ids.size(0) < 1 || ids.size(1) < 1)
      throw std::invalid_argument("nonempty rank-two int32 cache tokens required");
    for (const Tensor* t : {&state.keys, &state.values}) {
      if (!t->defined() || t->dtype() != nn::DType::Float32 || t->dim() != 4 || t->size(0) != ids.size(0) ||
          t->size(1) != config.heads || t->size(3) != config.dimensions / config.heads)
        throw std::invalid_argument("cache shape must be batch,heads,past,head_width");
      for (float x : t->to_vector<float>())
        if (!std::isfinite(x)) throw std::invalid_argument("finite cache values required");
    }
    if (state.keys.shape() != state.values.shape() || state.keys.size(2) + ids.size(1) > config.context)
      throw std::invalid_argument("matched K/V length and append within context required");
    for (auto v : ids.to_vector<std::int32_t>())
      if (v < 0 || v >= config.input_classes) throw std::invalid_argument("valid cache token IDs required");
  }

  /// Projects only the new tokens, attends to the cached prefix plus the
  /// chunk (causally), and replaces `state` with the grown keys/values.
  std::map<std::string, Tensor> Append(const Tensor& ids, KvState& state) {
    Validate(ids, state);
    nn::NoGradGuard guard;
    const auto      p = CachedDecoder(store, ids, state.keys, state.values, config);
    state             = {p.keys, p.values};
    return {{"logits", p.logits}, {"keys", p.keys}, {"values", p.values}, {"attention", p.attention}, {"hidden", p.hidden}};
  }

  /// Uncached full-context model (the comparison reference).
  Tensor Predict(const Tensor& ids) {
    check_full(ids);
    nn::NoGradGuard guard;
    return DecoderTransformer(store, nn::one_hot(ids, config.input_classes), config).logits;
  }

  /// Absolute positions reset when the last-context window slides: old K/V at
  /// shifted positions are invalid, so rebuild rather than evict one entry.
  std::map<std::string, Tensor> Advance(const std::vector<int>& history, KvState& state, bool* rebuilt = nullptr) {
    if (history.empty()) throw std::invalid_argument("nonempty generation history required");
    if (!state.keys.defined() || state.keys.dim() != 4) throw std::invalid_argument("rank-four cache state required");
    const bool rebuild = history.size() > static_cast<std::size_t>(config.context) ||
                         state.keys.size(2) != static_cast<std::int64_t>(history.size()) - 1;
    if (rebuilt) *rebuilt = rebuild;
    const int count = rebuild ? std::min(static_cast<int>(history.size()), static_cast<int>(config.context)) : 1;
    std::vector<std::int32_t> ids(history.end() - count, history.end());
    const Tensor              t = nn::from_vector(ids, {1, count});
    if (rebuild) {
      auto fresh = Empty(1);
      auto out   = Append(t, fresh);
      state      = fresh;
      return out;
    }
    return Append(t, state);
  }

 private:
  void check_full(const Tensor& ids) const {
    if (ids.dim() != 2 || ids.size(1) != config.context) throw std::invalid_argument("full context matrix");
  }
};

}  // namespace lm
