// Compact pre-norm decoder Transformer (GPT-style) and its K/V-cached form:
// port of tflow's common/nn/decoder_transformer.h and common/nn/cached_decoder.h.
//
// One block: x = token one-hot @ token/W + position/W; h = x + attention_out(
// causal MHA(ln1(x))) with a fused biased "qkv" projection; h += ff2(gelu_tanh(
// ff1(ln2(h)))); logits = head(ln_final(h)). Parameter names, shapes and
// initialisers (N(0, .02) weights, zero biases, unit/zero LN) are tflow's, so
// TF's saved arrays (.npz) load by name.
//
// The attention is nn::scaled_dot_product_attention with causal = true (the
// library aligns the causal mask bottom-right, which is exactly the
// "key j <= offset + query q" rule of tflow's cached decoder when Tq new
// queries follow Tk - Tq cached keys).
#pragma once

#include <datamunge/nn/nn.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

namespace lm {

namespace nn = datamunge::nn;
using nn::Tensor;

struct DecoderConfig {
  std::int64_t input_classes = 65, output_classes = 65, context = 64, dimensions = 16, heads = 2, feed_forward = 32;
  bool         rms_norm = false, quick_gelu = false;
  void         Validate() const {
    if (context < 1 || dimensions < 1 || heads < 1 || dimensions % heads || input_classes < 2 || output_classes < 2 ||
        feed_forward < 1)
      throw std::invalid_argument("valid decoder Transformer configuration");
  }
};

/// logits [N, T, V], hidden [N, T, D] (after ln_final), attention [N, H, T, T],
/// embedded [N, T, D]; for the cached form also the full keys/values.
struct DecoderPath {
  Tensor logits, hidden, attention, embedded, keys, values;
};

namespace detail {

inline Tensor Affine(nn::ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t out) {
  nn::DenseOptions o;
  o.init = nn::init::normal(.02);
  return nn::dense(store, name, x, out, o);
}

inline Tensor Normalize(nn::ParameterStore& store, const DecoderConfig& c, const Tensor& x, const std::string& name) {
  return c.rms_norm ? nn::rms_norm(store, name, x, 1e-5) : nn::layer_norm(store, name, x, 1e-5);
}

inline Tensor SplitHeads(const Tensor& x, std::int64_t heads) {
  const std::int64_t n = x.size(0), t = x.size(1), d = x.size(2);
  return nn::transpose(nn::reshape(x, {n, t, heads, d / heads}), {0, 2, 1, 3});
}

inline Tensor MergeHeads(const Tensor& x) {
  const std::int64_t n = x.size(0), h = x.size(1), t = x.size(2), dh = x.size(3);
  return nn::reshape(nn::transpose(x, {0, 2, 1, 3}), {n, t, h * dh});
}

/// The block after the embedding: x [N, Tq, D]; past keys/values [N, H, P, dh]
/// (undefined for none). Fills logits/hidden/attention/keys/values.
inline DecoderPath Block(nn::ParameterStore& store, const DecoderConfig& c, const Tensor& x, const Tensor& past_keys,
                         const Tensor& past_values) {
  const std::int64_t D   = c.dimensions;
  const Tensor       qkv = Affine(store, "qkv", Normalize(store, c, x, "ln1"), 3 * D);
  const Tensor       q   = SplitHeads(nn::slice_axis(qkv, 2, 0, D), c.heads);
  Tensor             k   = SplitHeads(nn::slice_axis(qkv, 2, D, 2 * D), c.heads);
  Tensor             v   = SplitHeads(nn::slice_axis(qkv, 2, 2 * D, 3 * D), c.heads);
  if (past_keys.defined() && past_keys.size(2) > 0) {
    k = nn::concat({past_keys, k}, 2);
    v = nn::concat({past_values, v}, 2);
  }
  nn::AttentionOptions o;
  o.causal       = true;
  o.scale        = 1.0 / std::sqrt(static_cast<double>(D / c.heads));
  o.need_weights = true;
  o.impl         = nn::AttentionImpl::Plain;
  const auto a   = nn::scaled_dot_product_attention(q, k, v, o);
  Tensor     h   = nn::add(x, Affine(store, "attention_out", MergeHeads(a.output), D));
  Tensor     ff  = Affine(store, "ff1", Normalize(store, c, h, "ln2"), c.feed_forward);
  ff = c.quick_gelu ? nn::mul(ff, nn::sigmoid(nn::mul(ff, 1.702))) : nn::gelu(ff, true);
  h  = nn::add(h, Affine(store, "ff2", ff, D));
  h  = Normalize(store, c, h, "ln_final");
  DecoderPath p;
  p.logits    = Affine(store, "head", h, c.output_classes);
  p.hidden    = h;
  p.attention = a.weights;
  p.embedded  = x;
  p.keys      = k;
  p.values    = v;
  return p;
}

}  // namespace detail

/// tflow's DecoderTransformer: `encoded` is the float one-hot input [N, context,
/// input_classes] (one-hot so the input Jacobian can be taken).
inline DecoderPath DecoderTransformer(nn::ParameterStore& store, const Tensor& encoded, const DecoderConfig& c = {}) {
  c.Validate();
  if (encoded.dim() != 3 || encoded.size(1) != c.context || encoded.size(2) != c.input_classes)
    throw std::invalid_argument("decoder input must be [N, context, input_classes]");
  const Tensor token    = store.get("token/W", {c.input_classes, c.dimensions}, nn::init::normal(.02));
  const Tensor position = store.get("position/W", {c.context, c.dimensions}, nn::init::normal(.02));
  const Tensor x        = nn::add(nn::matmul(encoded, token), position);
  return detail::Block(store, c, x, Tensor(), Tensor());
}

/// tflow's CachedDecoder: new token ids [N, t] (int32) at absolute positions
/// P..P+t-1 after P cached positions (past keys/values [N, H, P, dh]). Returns
/// logits/hidden/attention for the new positions and the grown keys/values.
inline DecoderPath CachedDecoder(nn::ParameterStore& store, const Tensor& ids, const Tensor& past_keys,
                                 const Tensor& past_values, const DecoderConfig& c = {}) {
  c.Validate();
  const std::int64_t offset   = past_keys.size(2), count = ids.size(1);
  const Tensor       token    = store.get("token/W", {c.input_classes, c.dimensions}, nn::init::normal(.02));
  const Tensor       position = store.get("position/W", {c.context, c.dimensions}, nn::init::normal(.02));
  const Tensor x = nn::add(nn::gather(token, ids, 0), nn::slice_axis(position, 0, offset, offset + count));
  return detail::Block(store, c, x, past_keys, past_values);
}

}  // namespace lm
