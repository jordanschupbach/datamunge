#pragma once
/// @file decoder_cache.hpp
/// @brief Owning source-named decoder parameters and explicit caller K/V cache endpoints.
#include <datamunge/nn/autoregressive_components.hpp>
#include <limits>
#include <map>
namespace datamunge::nn::models {
namespace nn = datamunge::nn; ///< Native operations spelling used by source cache methods.
/// @brief Caller-owned live key/value tensor endpoints for a source-named decoder cache.
///
/// Tensor copies share storage; Append replaces endpoints rather than mutating old cache tensors.
/// No optimizer or generation history is owned.
///
/// @par Example
/// @code
/// dm::nn::models::CachedTransformerLM model;auto state=model.Empty(1);
/// auto output=model.Append(dm::nn::from_vector(std::vector<int>{0},{1,1}),state);
/// @endcode
/// @see CachedTransformerLM
struct KvState {
  Tensor keys;   ///< Float32[B,H,past,D/H] cached projected keys.
  Tensor values; ///< Same shape cached projected values.
};

/// @brief Own initialized source decoder parameters and validate explicit caller K/V appends.
///
/// The store uses source seed9001 and creates all named leaves at construction. Copies are disabled
/// by ParameterStore ownership. Calls require synchronization against store/config mutation. Config
/// and store are exposed for original checkpoint workflows; keep geometry fixed after construction.
/// Cache state and token-history loops remain caller-owned. Returned inference endpoints have no
/// training graph.
///
/// @par Example
/// @code
/// dm::nn::models::CachedTransformerLM model;auto state=model.Empty(1);
/// auto output=model.Append(dm::nn::from_vector(std::vector<int>{0},{1,1}),state);
/// @endcode
/// @see CachedDecoder(), DecoderTransformer()
class CachedTransformerLM {
 public:
  DecoderConfig      config;      ///< Owned geometry; preserve it after initialized leaves exist.
  nn::ParameterStore store{9001}; ///< Owned source-named parameters and initializer seed9001.

  /// @brief Initialize all source-named decoder leaves without an optimizer.
  ///
  /// @param c Validated owned decoder configuration; default character65/context64/D16.
  /// @throws std::invalid_argument Invalid geometry or primitive initialization configuration.
  /// @par Example
  /// @code
  /// dm::nn::models::CachedTransformerLM model;auto state=model.Empty(1);
  /// auto output=model.Append(dm::nn::from_vector(std::vector<int>{0},{1,1}),state);
  /// @endcode
  /// @see DecoderConfig
  explicit CachedTransformerLM(DecoderConfig c = {}) : config(c) {
    nn::NoGradGuard guard;
    DecoderTransformer(store, nn::zeros({1, c.context, c.input_classes}), c);
  }

  /// @brief Allocate independent zero-length cache tensors for a positive batch.
  ///
  /// @param batch Positive number of independent sequences.
  /// @return Owning empty Float32[B,H,0,D/H] key/value endpoints.
  /// @throws std::invalid_argument Nonpositive batch or invalid configuration.
  /// @par Example
  /// @code
  /// dm::nn::models::CachedTransformerLM model;auto state=model.Empty(1);
  /// auto output=model.Append(dm::nn::from_vector(std::vector<int>{0},{1,1}),state);
  /// @endcode
  /// @see Append()
  KvState Empty(std::int64_t batch) const {
    config.Validate();
    if (batch < 1)
      throw std::invalid_argument("positive cache batch required");
    const nn::Shape shape{batch, config.heads, 0, config.dimensions / config.heads};
    return {nn::zeros(shape), nn::zeros(shape)};
  }

  /// @brief Check one append request without changing state or parameters.
  ///
  /// The combined past+new length must fit context. Scans all cache values and token IDs; no engine
  /// or parameter state changes.
  ///
  /// @param ids Nonempty Int32[B,new] token IDs within input vocabulary.
  /// @param state Matching finite Float32[B,H,past,D/H] keys/values.
  /// @throws std::invalid_argument Invalid dtype, shape, token range, cache finiteness or context
  /// length.
  /// @par Example
  /// @code
  /// dm::nn::models::CachedTransformerLM model;auto state=model.Empty(1);
  /// auto output=model.Append(dm::nn::from_vector(std::vector<int>{0},{1,1}),state);
  /// model.Validate(dm::nn::from_vector(std::vector<int>{1},{1,1}),state);
  /// @endcode
  /// @see Append()
  void Validate(const Tensor& ids, const KvState& state) const {
    if (!ids.defined() || ids.dtype() != nn::DType::Int32 || ids.dim() != 2 || ids.size(0) < 1 ||
        ids.size(1) < 1)
      throw std::invalid_argument("nonempty rank-two int32 cache tokens required");
    for (const Tensor* t : {&state.keys, &state.values}) {
      if (!t->defined() || t->dtype() != nn::DType::Float32 || t->dim() != 4 ||
          t->size(0) != ids.size(0) || t->size(1) != config.heads ||
          t->size(3) != config.dimensions / config.heads)
        throw std::invalid_argument("cache shape must be batch,heads,past,head_width");
      for (float x : t->to_vector<float>())
        if (!std::isfinite(x))
          throw std::invalid_argument("finite cache values required");
    }
    if (state.keys.shape() != state.values.shape() ||
        state.keys.size(2) + ids.size(1) > config.context)
      throw std::invalid_argument("matched K/V length and append within context required");
    for (auto v : ids.to_vector<std::int32_t>())
      if (v < 0 || v >= config.input_classes)
        throw std::invalid_argument("valid cache token IDs required");
  }

  /// Projects only the new tokens, attends to the cached prefix plus the
  /// chunk (causally), and replaces `state` with the grown keys/values.
  /// @brief Project new tokens, causally attend to retained keys and replace cache endpoints.
  ///
  /// Checks before inference; no parameter creation occurs after construction with unchanged
  /// geometry. Old tensor aliases retain old storage. No gradient recording is enabled.
  ///
  /// @param ids Valid nonempty Int32[B,new] token matrix.
  /// @param state Caller-owned endpoints, replaced after successful inference.
  /// @return Owning map of logits, grown keys/values, attention and hidden tensor endpoints.
  /// @throws std::invalid_argument Invalid append request or native primitive input.
  /// @par Example
  /// @code
  /// dm::nn::models::CachedTransformerLM model;auto state=model.Empty(1);
  /// auto output=model.Append(dm::nn::from_vector(std::vector<int>{0},{1,1}),state);
  /// @endcode
  /// @see Validate(), Advance()
  std::map<std::string, Tensor> Append(const Tensor& ids, KvState& state) {
    Validate(ids, state);
    nn::NoGradGuard guard;
    const auto      p = CachedDecoder(store, ids, state.keys, state.values, config);
    state             = {p.keys, p.values};
    return {
        {"logits", p.logits},
        {"keys", p.keys},
        {"values", p.values},
        {"attention", p.attention},
        {"hidden", p.hidden}};
  }

  /// Uncached full-context model (the comparison reference).
  /// @brief Evaluate one uncached source context as a cache parity reference.
  ///
  /// @param ids Int32[B,context] source tokens with valid vocabulary IDs.
  /// @return No-grad full-context logits[B,context,output_classes].
  /// @throws std::invalid_argument Invalid full-context geometry or primitive input.
  /// @par Example
  /// @code
  /// dm::nn::models::CachedTransformerLM model;auto
  /// logits=model.Predict(dm::nn::zeros({1,64},dm::nn::DType::Int32));
  /// @endcode
  /// @see Append()
  Tensor Predict(const Tensor& ids) {
    check_full(ids);
    nn::NoGradGuard guard;
    return DecoderTransformer(store, nn::one_hot(ids, config.input_classes), config).logits;
  }

  /// Absolute positions reset when the last-context window slides: old K/V at
  /// shifted positions are invalid, so rebuild rather than evict one entry.
  /// @brief Append the final history token or rebuild the last absolute-position context.
  ///
  /// Absolute positions reset when the window slides; existing keys at shifted positions are
  /// discarded. This is one history transition, not a generation loop. No random draws occur.
  ///
  /// @param history Nonempty caller-owned chronological token history, length fitting int.
  /// @param state Caller cache endpoints, replaced after successful inference.
  /// @param rebuilt Optional borrowed flag, default nullptr; written before inference.
  /// @return Inference fields from appending one token or rebuilding the retained context window.
  /// @throws std::invalid_argument Empty history, incompatible state or append geometry.
  /// @par Example
  /// @code
  /// dm::nn::models::CachedTransformerLM model;auto state=model.Empty(1);auto
  /// output=model.Advance({0,1},state);
  /// @endcode
  /// @see Append()
  std::map<std::string, Tensor>
  Advance(const std::vector<int>& history, KvState& state, bool* rebuilt = nullptr) {
    if (history.size() > size_t(std::numeric_limits<int>::max()))
      throw std::invalid_argument("bounded source token history required");
    if (history.empty())
      throw std::invalid_argument("nonempty generation history required");
    if (!state.keys.defined() || state.keys.dim() != 4)
      throw std::invalid_argument("rank-four cache state required");
    const bool rebuild = history.size() > static_cast<std::size_t>(config.context) ||
                         state.keys.size(2) != static_cast<std::int64_t>(history.size()) - 1;
    if (rebuilt)
      *rebuilt = rebuild;
    const int count =
        rebuild ? std::min(static_cast<int>(history.size()), static_cast<int>(config.context)) : 1;
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
  /// @brief Require a full source context matrix before no-grad inference.
  ///
  /// @param ids Candidate full token matrix.
  /// @throws std::invalid_argument Invalid full-context geometry.
  /// @see Predict()
  void check_full(const Tensor& ids) const {
    if (ids.dim() != 2 || ids.size(1) != config.context)
      throw std::invalid_argument("full context matrix");
  }
};

} // namespace datamunge::nn::models
