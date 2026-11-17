#pragma once
/// @file autoregressive_components.hpp
/// @brief Optimizer-free character recurrence and full/cached causal decoder graphs.
#include <cmath>
#include <datamunge/nn/attention.hpp>
#include <datamunge/nn/memory_networks.hpp>
namespace datamunge::nn::models {
/// @brief Own the DecoderConfig compact causal-decoder configuration or live endpoints.
///
/// No optimizer or inference cache is owned; caller controls training and key/value retention.
/// @par Example
/// @code
/// dm::nn::models::DecoderConfig value;
/// @endcode
/// @par References
/// - Vaswani et al. (2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// - Chen et al. (2020), Generative Pretraining From Pixels,
/// https://proceedings.mlr.press/v119/chen20s.html .
/// @see DecoderTransformer(), CachedDecoder()
struct DecoderConfig {
  std::int64_t input_classes  = 65;    ///< Input vocabulary size.
  std::int64_t output_classes = 65;    ///< Output vocabulary size.
  std::int64_t context        = 64;    ///< Maximum absolute positions.
  std::int64_t dimensions     = 16;    ///< Model width divisible by heads.
  std::int64_t heads          = 2;     ///< Attention head count.
  std::int64_t feed_forward   = 32;    ///< Feed-forward hidden width.
  bool         rms_norm       = false; ///< True selects gamma-only RMSNorm instead of LayerNorm.
  bool         quick_gelu     = false; ///< True selects x*sigmoid(1.702*x) instead of tanh GELU.

  /// @brief Check positive geometry and divisibility of model width by heads.
  /// @throws std::invalid_argument Invalid decoder geometry.
  /// @par Example
  /// @code
  /// dm::nn::models::DecoderConfig cfg;cfg.Validate();
  /// @endcode
  void Validate() const {
    if (context < 1 || dimensions < 1 || heads < 1 || dimensions % heads || input_classes < 2 ||
        output_classes < 2 || feed_forward < 1 || context > 65536 || dimensions > 4096 ||
        feed_forward > 16384 || input_classes > 100000 || output_classes > 100000)
      throw std::invalid_argument("valid decoder Transformer configuration");
  }
};

/// @brief Own the DecoderPath compact causal-decoder configuration or live endpoints.
///
/// No optimizer or inference cache is owned; caller controls training and key/value retention.
/// @par Example
/// @code
/// dm::nn::models::DecoderPath value;
/// @endcode
/// @par References
/// - Vaswani et al. (2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// - Chen et al. (2020), Generative Pretraining From Pixels,
/// https://proceedings.mlr.press/v119/chen20s.html .
/// @see DecoderTransformer(), CachedDecoder()
struct DecoderPath {
  Tensor logits;    ///< Live CPU[N,T,output_classes] logits.
  Tensor hidden;    ///< Live CPU[N,T,D] final normalized states.
  Tensor attention; ///< Live CPU[N,heads,T,T+past] causal attention.
  Tensor embedded;  ///< Live CPU[N,T,D] token-plus-position embeddings.
  Tensor keys;      ///< Live CPU[N,heads,T+past,D/heads] keys.
  Tensor values;    ///< Live CPU[N,heads,T+past,D/heads] values.
};

/// @brief Internal source affine, normalization and head-layout graph primitives.
namespace decoder_component_detail {

/// @brief Apply the source normal(.02) affine preset.
/// @param store Caller-owned store.
/// @param name Parameter scope.
/// @param x Live floating input.
/// @param out Output width.
/// @return Live affine output.
/// @throws std::invalid_argument Invalid explicit configuration or tensor geometry.
inline Tensor
Affine(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t out) {
  DenseOptions o;
  o.init = init::normal(.02);
  return dense(store, name, x, out, o);
}

/// @brief Apply source epsilon1e-5 LayerNorm or RMSNorm.
/// @param store Caller-owned store.
/// @param c Normalization configuration.
/// @param x Live floating input.
/// @param name Parameter scope.
/// @return Live normalized output.
/// @throws std::invalid_argument Invalid explicit configuration or tensor geometry.
inline Tensor
Normalize(ParameterStore& store, const DecoderConfig& c, const Tensor& x, const std::string& name) {
  return c.rms_norm ? rms_norm(store, name, x, 1e-5) : layer_norm(store, name, x, 1e-5);
}

/// @brief Transpose [N,T,D] into [N,H,T,D/H].
/// @param x Live rank-three tensor.
/// @param heads Positive divisor of D.
/// @return Live head-separated view.
/// @throws std::invalid_argument Invalid explicit configuration or tensor geometry.
inline Tensor SplitHeads(const Tensor& x, std::int64_t heads) {
  if (x.dim() != 3 || heads < 1 || x.size(0) < 1 || x.size(2) % heads)
    throw std::invalid_argument("positive aligned rank-three head geometry");
  const std::int64_t n = x.size(0), t = x.size(1), d = x.size(2);
  return transpose(reshape(x, {n, t, heads, d / heads}), {0, 2, 1, 3});
}

/// @brief Transpose and combine [N,H,T,d] into [N,T,H*d].
/// @param x Live rank-four tensor.
/// @return Live merged tensor.
/// @throws std::invalid_argument Invalid explicit configuration or tensor geometry.
inline Tensor MergeHeads(const Tensor& x) {
  if (x.dim() != 4 || x.size(0) < 1 || x.size(1) < 1 || x.size(3) < 1)
    throw std::invalid_argument("positive rank-four head geometry");
  const std::int64_t n = x.size(0), h = x.size(1), t = x.size(2), dh = x.size(3);
  return reshape(transpose(x, {0, 2, 1, 3}), {n, t, h * dh});
}

/// @brief Apply the residual attention/feed-forward decoder block.
/// @param store Caller-owned store.
/// @param c Validated geometry.
/// @param x Live CPU[N,T,D] embedded states.
/// @param past_keys Undefined or prior CPU[N,H,P,D/H] keys.
/// @param past_values Aligned prior values.
/// @return Live decoder endpoints with grown key/value tensors.
/// @throws std::invalid_argument Invalid explicit configuration or tensor geometry.
inline DecoderPath Block(
    ParameterStore&      store,
    const DecoderConfig& c,
    const Tensor&        x,
    const Tensor&        past_keys,
    const Tensor&        past_values) {
  const std::int64_t D   = c.dimensions;
  const Tensor       qkv = Affine(store, "qkv", Normalize(store, c, x, "ln1"), 3 * D);
  const Tensor       q   = SplitHeads(slice_axis(qkv, 2, 0, D), c.heads);
  Tensor             k   = SplitHeads(slice_axis(qkv, 2, D, 2 * D), c.heads);
  Tensor             v   = SplitHeads(slice_axis(qkv, 2, 2 * D, 3 * D), c.heads);
  if (past_keys.defined() && past_keys.size(2) > 0) {
    k = concat({past_keys, k}, 2);
    v = concat({past_values, v}, 2);
  }
  AttentionOptions o;
  o.causal       = true;
  o.scale        = 1.0 / std::sqrt(static_cast<double>(D / c.heads));
  o.need_weights = true;
  o.impl         = AttentionImpl::Plain;
  const auto a   = scaled_dot_product_attention(q, k, v, o);
  Tensor     h   = add(x, Affine(store, "attention_out", MergeHeads(a.output), D));
  Tensor     ff  = Affine(store, "ff1", Normalize(store, c, h, "ln2"), c.feed_forward);
  ff             = c.quick_gelu ? mul(ff, sigmoid(mul(ff, 1.702))) : gelu(ff, true);
  h              = add(h, Affine(store, "ff2", ff, D));
  h              = Normalize(store, c, h, "ln_final");
  DecoderPath p;
  p.logits    = Affine(store, "head", h, c.output_classes);
  p.hidden    = h;
  p.attention = a.weights;
  p.embedded  = x;
  p.keys      = k;
  p.values    = v;
  return p;
}

} // namespace decoder_component_detail

/// @brief Apply a full one-block causal decoder to encoded tokens.
/// @param store Caller-owned Float32 CPU store, lazy source normal(.02) parameters.
/// @param encoded Float32 CPU[N,context,input_classes] encoded tokens; soft inputs permitted.
/// @param c Validated geometry; default65classes/context64/D16/H2/FF32.
/// @return Live logits, normalized hidden states, attention, embeddings and keys/values.
/// @throws std::invalid_argument Invalid explicit configuration or tensor geometry.
///
/// Handles retain storage and live paths after local graph collection. The function mutates
/// only lazy parameter creation; it owns no optimizer or fitting loop.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);auto
/// y=m::DecoderTransformer(p,n::zeros({1,64,65}));
/// @endcode
/// @par References
/// - Vaswani et al. (2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// - Chen et al. (2020), Generative Pretraining From Pixels,
/// https://proceedings.mlr.press/v119/chen20s.html .
/// @see DecoderTransformer(), CachedDecoder()
inline DecoderPath
DecoderTransformer(ParameterStore& store, const Tensor& encoded, const DecoderConfig& c = {}) {
  c.Validate();
  if (encoded.dim() != 3 || encoded.size(1) != c.context || encoded.size(2) != c.input_classes)
    throw std::invalid_argument("decoder input must be [N, context, input_classes]");
  const Tensor token    = store.get("token/W", {c.input_classes, c.dimensions}, init::normal(.02));
  const Tensor position = store.get("position/W", {c.context, c.dimensions}, init::normal(.02));
  const Tensor x        = add(matmul(encoded, token), position);
  return decoder_component_detail::Block(store, c, x, Tensor(), Tensor());
}

/// @brief Decode new token IDs at positions following an explicit key/value cache.
/// @param store Caller-owned Float32 CPU store shared with the full decoder.
/// @param ids Int32 CPU[N,T] token IDs within input_classes.
/// @param past_keys Float32 CPU[N,heads,P,D/heads], including zero-length P=0.
/// @param past_values Same geometry as past_keys.
/// @param c Validated full-decoder configuration; default source preset.
/// @return Live new-position logits and grown keys/values; caller owns cache updates.
/// @throws std::invalid_argument Invalid explicit configuration or tensor geometry.
///
/// Handles retain storage and live paths after local graph collection. The function mutates
/// only lazy parameter creation; it owns no optimizer or fitting loop.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);auto
/// y=m::CachedDecoder(p,n::from_vector(std::vector<int>{0},{1,1}),n::zeros({1,2,0,8}),n::zeros({1,2,0,8}));
/// @endcode
/// @par References
/// - Vaswani et al. (2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// - Chen et al. (2020), Generative Pretraining From Pixels,
/// https://proceedings.mlr.press/v119/chen20s.html .
/// @see DecoderTransformer(), CachedDecoder()
inline DecoderPath CachedDecoder(
    ParameterStore&      store,
    const Tensor&        ids,
    const Tensor&        past_keys,
    const Tensor&        past_values,
    const DecoderConfig& c = {}) {
  c.Validate();
  if (ids.dim() != 2 || ids.dtype() != DType::Int32 || !past_keys.defined() ||
      !past_values.defined() || past_keys.dim() != 4 || past_keys.shape() != past_values.shape() ||
      past_keys.size(0) != ids.size(0) || past_keys.size(1) != c.heads ||
      past_keys.size(3) != c.dimensions / c.heads || ids.size(1) < 1 ||
      past_keys.size(2) + ids.size(1) > c.context)
    throw std::invalid_argument("cached decoder shape/context");
  for (int id : ids.to_vector<int>())
    if (id < 0 || id >= c.input_classes)
      throw std::invalid_argument("decoder token range");
  const std::int64_t offset = past_keys.size(2), count = ids.size(1);
  const Tensor token    = store.get("token/W", {c.input_classes, c.dimensions}, init::normal(.02));
  const Tensor position = store.get("position/W", {c.context, c.dimensions}, init::normal(.02));
  const Tensor x = add(gather(token, ids, 0), slice_axis(position, 0, offset, offset + count));
  return decoder_component_detail::Block(store, c, x, past_keys, past_values);
}

/// @brief Initialize the six tensors of the 2025-parameter character Elman preset.
/// @param p Caller-owned Float32 CPU store; creation consumes its local RNG.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(8903);dm::nn::models::CharacterRnnParameters(p);
/// @endcode
/// @par References
/// - Elman (1990), Finding Structure in Time, https://doi.org/10.1207/s15516709cog1402_1 .
/// @see CharacterRnnStep(), CharacterRnnSequence()
inline void CharacterRnnParameters(ParameterStore& p) {
  p.get("embedding/W", {65, 8}, init::normal(.1));
  p.get("recurrent/Wx", {8, 16}, init::normal(.1));
  p.get("recurrent/Wh", {16, 16}, init::normal(.15));
  p.get("recurrent/b", {16}, init::zeros());
  p.get("output/W", {16, 65}, init::normal(.1));
  p.get("output/b", {65}, init::zeros());
}
/// @brief Advance one character Elman step with shared caller-owned weights.
/// @param p Initialized CharacterRnnParameters store.
/// @param encoded Float32 CPU[N,65] encoded character, allowing differentiable soft inputs.
/// @param state Float32 CPU[N,16] previous hidden state.
/// @return Pair of live CPU[N,16] next state and CPU[N,65] logits.
/// @throws std::invalid_argument Incompatible tensor or parameter geometry.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(8903);m::CharacterRnnParameters(p);
/// auto y=m::CharacterRnnStep(p,n::zeros({1,65}),n::zeros({1,16}));
/// @endcode
/// @par References
/// - Elman (1990), Finding Structure in Time, https://doi.org/10.1207/s15516709cog1402_1 .
/// @see CharacterRnnSequence()
inline std::pair<Tensor, Tensor>
CharacterRnnStep(const ParameterStore& p, const Tensor& encoded, const Tensor& state) {
  if (encoded.dim() != 2 || encoded.size(1) != 65 || state.shape() != Shape{encoded.size(0), 16})
    throw std::invalid_argument("character step geometry");
  const auto e = matmul(encoded, p.at("embedding/W"));
  const auto h = tanh(bias_add(
      add(matmul(e, p.at("recurrent/Wx")), matmul(state, p.at("recurrent/Wh"))),
      p.at("recurrent/b")));
  return {h, bias_add(matmul(h, p.at("output/W")), p.at("output/b"))};
}
/// @brief Unroll the character Elman preset from a zero hidden state.
/// @param p Initialized CharacterRnnParameters store, shared at every time step.
/// @param encoded Float32 CPU[N,T,65] encoded characters with T>0.
/// @return Pair of live CPU[N,T,65] logits and CPU[N,T,16] hidden states.
/// @throws std::invalid_argument Invalid encoded rank, width or empty time dimension.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(8903);m::CharacterRnnParameters(p);
/// auto y=m::CharacterRnnSequence(p,n::zeros({1,64,65}));
/// @endcode
/// @par References
/// - Elman (1990), Finding Structure in Time, https://doi.org/10.1207/s15516709cog1402_1 .
/// @see CharacterRnnStep()
inline std::pair<Tensor, Tensor> CharacterRnnSequence(
    const ParameterStore& p,
    const Tensor&         encoded) {
  if (encoded.dim() != 3 || encoded.size(2) != 65 || encoded.size(1) < 1)
    throw std::invalid_argument("character sequence geometry");
  auto h = mul(reshape(slice_axis(slice_axis(encoded, 1, 0, 1), 2, 0, 16), {-1, 16}), 0.);
  std::vector<Tensor> states, scores;
  for (int t = 0; t < encoded.size(1); ++t) {
    auto next = CharacterRnnStep(p, reshape(slice_axis(encoded, 1, t, t + 1), {-1, 65}), h);
    h         = next.first;
    states.push_back(expand_dims(h, 1));
    scores.push_back(expand_dims(next.second, 1));
  }
  return {concat(scores, 1), concat(states, 1)};
}

} // namespace datamunge::nn::models
