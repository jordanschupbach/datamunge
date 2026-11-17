#pragma once
/// @file completion_graphs.hpp
/// @brief Pure sequence, language and vision teaching graphs; all optimization remains
/// caller-owned.
#include <algorithm>
#include <cmath>
#include <datamunge/nn/attention_components.hpp>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/checkpoint.hpp>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
#include <datamunge/nn/sequence.hpp>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <set>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief component detail source-equation components; no optimizer or training driver.
namespace component_detail {
/// @brief Apply Require with the original source equations and parameter names.
///
///
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param valid Boolean CPU[B,T] route eligibility, padding excluded from capacity and balance.
/// @param message Explicit source geometry/control value; defaults match the declaration.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// dm::nn::models::component_detail::Require(true,"valid");
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline void Require(bool valid, const std::string& message) {
  if (!valid)
    throw std::invalid_argument(message);
}
} // namespace component_detail
/// @brief attn source-equation components; no optimizer or training driver.
namespace attn {
/// @brief Apply AttentionAllowed with the original source equations and parameter names.
///
///
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param key_lengths Live CPU source endpoint with the family geometry above.
/// @param query_lengths Live CPU source endpoint with the family geometry above.
/// @param queries Explicit source geometry/control value; defaults match the declaration.
/// @param keys Explicit source geometry/control value; defaults match the declaration.
/// @param causal Explicit source geometry/control value; defaults match the declaration.
/// @param query_offset Explicit source geometry/control value; defaults match the declaration.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto x=n::zeros({1,2,8});auto
/// y=m::attn::MultiHeadAttention(p,"a",x,x,8,2);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
using datamunge::nn::models::AttentionAllowed;
/// @brief Apply ScaledDotProductAttention with the original source equations and parameter names.
///
///
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param q Live CPU source endpoint with the family geometry above.
/// @param k Live CPU source endpoint with the family geometry above.
/// @param v Live CPU source endpoint with the family geometry above.
/// @param d Explicit source geometry/control value; defaults match the declaration.
/// @param mask Source boolean visibility or numeric mask with broadcast-compatible shape.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto x=n::zeros({1,2,8});auto
/// y=m::attn::MultiHeadAttention(p,"a",x,x,8,2);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline nn::Attended ScaledDotProductAttention(
    const nn::Tensor& q,
    const nn::Tensor& k,
    const nn::Tensor& v,
    std::int64_t      d,
    const nn::Tensor& mask = {}) {
  nn::AttentionOptions o;
  o.mask         = mask;
  o.scale        = 1. / std::sqrt(double(d));
  o.need_weights = true;
  o.impl         = nn::AttentionImpl::Plain;
  return nn::scaled_dot_product_attention(q, k, v, o);
}
/// @brief Apply MultiHeadAttention with the original source equations and parameter names.
///
///
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param name Explicit source geometry/control value; defaults match the declaration.
/// @param q Live CPU source endpoint with the family geometry above.
/// @param kv Live CPU source endpoint with the family geometry above.
/// @param d Explicit source geometry/control value; defaults match the declaration.
/// @param heads Explicit source geometry/control value; defaults match the declaration.
/// @param mask Source boolean visibility or numeric mask with broadcast-compatible shape.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto x=n::zeros({1,2,8});auto
/// y=m::attn::MultiHeadAttention(p,"a",x,x,8,2);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline nn::Attended MultiHeadAttention(
    nn::ParameterStore& p,
    const std::string&  name,
    const nn::Tensor&   q,
    const nn::Tensor&   kv,
    std::int64_t        d,
    std::int64_t        heads,
    const nn::Tensor&   mask = {}) {
  component_detail::Require(q.size(2) == d && kv.size(2) == d, "matching attention width");
  nn::MultiHeadOptions o;
  o.mask         = mask;
  o.need_weights = true;
  o.impl         = nn::AttentionImpl::Plain;
  return nn::multi_head_attention(p, name, q, kv, heads, o);
}
} // namespace attn
/// @brief positional source-equation components; no optimizer or training driver.
namespace positional {
using datamunge::nn::models::RelativePositionBias;
}
/// @brief forecast source-equation components; no optimizer or training driver.
namespace forecast {

namespace nn = datamunge::nn;
using nn::Tensor;

/// x [B, 12, 1] (already normalized) -> [B, 3].
/// @brief Apply LstmForecast with the original source equations and parameter names.
///
/// Float32 CPU[B,12,1] observations; output[B,3].
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param x Live CPU source endpoint with the family geometry above.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::forecast::LstmForecast(p,n::zeros({1,12,1}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Hochreiter and Schmidhuber(1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor LstmForecast(nn::ParameterStore& p, const Tensor& x) {
  nn::LstmState s = nn::lstm_zero_state(x.shape()[0], 8, x.dtype());
  for (const auto& step : nn::time_steps(x))
    s = nn::lstm_step(p, "lstm", step, s);
  return nn::dense(p, "head", s.h, 3);
}

/// Prediction in observation units: the net works on (x - mean) / sd.
/// @brief Apply Predict with the original source equations and parameter names.
///
/// Float32 CPU[B,12,1] observations; output[B,3].
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param x Live CPU source endpoint with the family geometry above.
/// @param mean Explicit source geometry/control value; defaults match the declaration.
/// @param sd Explicit source geometry/control value; defaults match the declaration.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::forecast::LstmForecast(p,n::zeros({1,12,1}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Hochreiter and Schmidhuber(1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor Predict(nn::ParameterStore& p, const Tensor& x, double mean, double sd) {
  return nn::add(nn::mul(LstmForecast(p, nn::div(nn::sub(x, mean), sd)), sd), mean);
}

/// Mean squared error in normalized units.

} // namespace forecast

/// @brief midi source-equation components; no optimizer or training driver.
namespace midi {

namespace nn = datamunge::nn;
using nn::Tensor;

/// @brief kInputs, kNotes, kStart, kUnits, kSteps: live source graph endpoint/storage; tensor
/// copies share native storage.
constexpr int kInputs = 13; ///< Exact original geometry/special-token value.
constexpr int kNotes  = 12; ///< Exact original geometry/special-token value.
constexpr int kStart  = 12; ///< Exact original geometry/special-token value.
constexpr int kUnits  = 8;  ///< Exact original geometry/special-token value.
constexpr int kSteps  = 16; ///< Exact original geometry/special-token value.

/// @brief Own Step source graph endpoints or configuration.
///
/// Float32 CPU[B,16,13] encoded notes,8-unit state,12-class logits; no sampling loop.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::midi::MidiLstm(p,n::zeros({1,16,13}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Hochreiter and Schmidhuber(1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Step {
  /// @brief state: live source graph endpoint/storage; tensor copies share native storage.
  nn::LstmState state;
  /// @brief logits: live source graph endpoint/storage; tensor copies share native storage.
  Tensor logits;
};

/// One step from a one-hot note [B, 13] and the previous state.
/// @brief Apply MidiLstmStep with the original source equations and parameter names.
///
/// Float32 CPU[B,16,13] encoded notes,8-unit state,12-class logits; no sampling loop.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param onehot Live CPU source endpoint with the family geometry above.
/// @param previous Explicit source geometry/control value; defaults match the declaration.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::midi::MidiLstm(p,n::zeros({1,16,13}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Hochreiter and Schmidhuber(1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Step
MidiLstmStep(nn::ParameterStore& p, const Tensor& onehot, const nn::LstmState& previous) {
  const Tensor w    = p.get("embedding/W", {kInputs, kUnits}, nn::init::normal(.1));
  const auto   next = nn::lstm_step(p, "lstm", nn::matmul(onehot, w), previous);
  return {next, nn::dense(p, "head", next.h, kNotes)};
}

/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,16,13] encoded notes,8-unit state,12-class logits; no sampling loop.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::midi::MidiLstm(p,n::zeros({1,16,13}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Hochreiter and Schmidhuber(1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief logits, h, c: live source graph endpoint/storage; tensor copies share native storage.
  Tensor logits; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor h;      ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor c; ///< Live source endpoint; owning handle shares storage and derivatives. // [B, 16, 12],

            ///< [B, 17, 8] (incl. the zero initial state)
};

/// Teacher-forced pass over one-hot notes [B, 16, 13] from a zero state.
/// @brief Apply MidiLstm with the original source equations and parameter names.
///
/// Float32 CPU[B,16,13] encoded notes,8-unit state,12-class logits; no sampling loop.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param x Live CPU source endpoint with the family geometry above.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::midi::MidiLstm(p,n::zeros({1,16,13}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Hochreiter and Schmidhuber(1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path MidiLstm(nn::ParameterStore& p, const Tensor& x) {
  const auto          n = x.shape()[0];
  nn::LstmState       s{nn::zeros({n, kUnits}, x.dtype()), nn::zeros({n, kUnits}, x.dtype())};
  std::vector<Tensor> hs{s.h}, cs{s.c}, logits;
  for (const auto& step : nn::time_steps(x)) {
    const auto next = MidiLstmStep(p, step, s);
    s               = next.state;
    hs.push_back(s.h);
    cs.push_back(s.c);
    logits.push_back(next.logits);
  }
  return {nn::stack_time(logits), nn::stack_time(hs), nn::stack_time(cs)};
}

} // namespace midi

/// @brief handwriting source-equation components; no optimizer or training driver.
namespace handwriting {

namespace nn = datamunge::nn;
using nn::Tensor;

/// @brief kSteps, kSymbols, kRaw: live source graph endpoint/storage; tensor copies share native
/// storage.
constexpr int kSteps   = 33; ///< Exact original geometry/special-token value.
constexpr int kSymbols = 21; ///< Exact original geometry/special-token value.
constexpr int kRaw     = 19; ///< Exact original geometry/special-token value.

/// @brief Own Step source graph endpoints or configuration.
///
/// Float32 CPU[B,33,3] offsets and[B,3,21] text;8-unit state,19 raw mixture values per step.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::handwriting::Handwriting(p,n::zeros({1,33,3}),n::zeros({1,3,21}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Hochreiter and Schmidhuber(1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Step {
  /// @brief state: live source graph endpoint/storage; tensor copies share native storage.
  nn::LstmState state;
  /// @brief kappa, window, phi, raw: live source graph endpoint/storage; tensor copies share native
  /// storage.
  Tensor kappa;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor window; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor phi;    ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor raw;    ///< Live source endpoint; owning handle shares storage and derivatives.
};

/// stroke [B, 3], text [B, 3, 21], kappa [B, 2], window [B, 21].
/// @brief Apply HandwritingCell with the original source equations and parameter names.
///
/// Float32 CPU[B,33,3] offsets and[B,3,21] text;8-unit state,19 raw mixture values per step.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param stroke Live CPU source endpoint with the family geometry above.
/// @param text ASCII text; non-ASCII bytes rejected; words longer than100characters become UNK100.
/// @param previous Explicit source geometry/control value; defaults match the declaration.
/// @param kappa Live CPU source endpoint with the family geometry above.
/// @param window Live CPU source endpoint with the family geometry above.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::handwriting::Handwriting(p,n::zeros({1,33,3}),n::zeros({1,3,21}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Hochreiter and Schmidhuber(1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Step HandwritingCell(
    nn::ParameterStore&  p,
    const Tensor&        stroke,
    const Tensor&        text,
    const nn::LstmState& previous,
    const Tensor&        kappa,
    const Tensor&        window) {
  Step out;
  out.state             = nn::lstm_step(p, "lstm", nn::concat({stroke, window}, 1), previous);
  const Tensor params   = nn::dense(p, "window", out.state.h, 6);
  const Tensor alpha    = nn::exp(nn::slice_axis(params, 1, 0, 2)),
               beta     = nn::exp(nn::slice_axis(params, 1, 2, 4));
  out.kappa             = nn::add(kappa, nn::mul(nn::exp(nn::slice_axis(params, 1, 4, 6)), .05));
  const Tensor position = nn::reshape(nn::arange(0, 3, 1, kappa.dtype()), {1, 1, 3});
  const Tensor delta    = nn::sub(nn::expand_dims(out.kappa, 2), position);
  out.phi               = nn::sum(
      nn::mul(
          nn::expand_dims(alpha, 2),
          nn::exp(nn::neg(nn::mul(nn::expand_dims(beta, 2), nn::square(delta))))),
      {1});
  out.window = nn::sum(nn::mul(text, nn::expand_dims(out.phi, 2)), {1});
  out.raw    = nn::dense(p, "mdn", nn::concat({out.state.h, out.window}, 1), kRaw);
  return out;
}

/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,33,3] offsets and[B,3,21] text;8-unit state,19 raw mixture values per step.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::handwriting::Handwriting(p,n::zeros({1,33,3}),n::zeros({1,3,21}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Hochreiter and Schmidhuber(1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief raw, h, c, kappa, window, phi: live source graph endpoint/storage; tensor copies share
  /// native storage.
  Tensor raw;    ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor h;      ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor c;      ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor kappa;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor window; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor phi;    ///< Live source endpoint; owning handle shares storage and derivatives.
};

/// Teacher-forced unroll over stroke [B, 33, 3] from zero states.
/// @brief Apply Handwriting with the original source equations and parameter names.
///
/// Float32 CPU[B,33,3] offsets and[B,3,21] text;8-unit state,19 raw mixture values per step.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param stroke Live CPU source endpoint with the family geometry above.
/// @param text ASCII text; non-ASCII bytes rejected; words longer than100characters become UNK100.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::handwriting::Handwriting(p,n::zeros({1,33,3}),n::zeros({1,3,21}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Hochreiter and Schmidhuber(1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path Handwriting(nn::ParameterStore& p, const Tensor& stroke, const Tensor& text) {
  const auto    B     = stroke.shape()[0];
  nn::LstmState s     = nn::lstm_zero_state(B, 8, stroke.dtype());
  Tensor        kappa = nn::zeros({B, 2}, stroke.dtype()),
         window       = nn::zeros({B, kSymbols}, stroke.dtype());
  std::vector<Tensor> hs{s.h}, cs{s.c}, ks{kappa}, ws{window}, raw, phi;
  for (const auto& input : nn::time_steps(stroke)) {
    const Step next = HandwritingCell(p, input, text, s, kappa, window);
    s               = next.state;
    kappa           = next.kappa;
    window          = next.window;
    hs.push_back(s.h);
    cs.push_back(s.c);
    ks.push_back(kappa);
    ws.push_back(window);
    raw.push_back(next.raw);
    phi.push_back(next.phi);
  }
  return {
      nn::stack_time(raw),
      nn::stack_time(hs),
      nn::stack_time(cs),
      nn::stack_time(ks),
      nn::stack_time(ws),
      nn::stack_time(phi)};
}

/// Mean per-step negative log-likelihood of target strokes (dx, dy, pen) under
/// the 3-component bivariate Gaussian mixture plus a Bernoulli pen.

} // namespace handwriting

/// @brief ctc handwriting source-equation components; no optimizer or training driver.
namespace ctc_handwriting {

namespace nn = datamunge::nn;
using nn::Tensor;

/// @brief kFrames, kPixels, kClasses, kBlank: live source graph endpoint/storage; tensor copies
/// share native storage.
constexpr int kFrames  = 8;  ///< Exact original geometry/special-token value.
constexpr int kPixels  = 49; ///< Exact original geometry/special-token value.
constexpr int kClasses = 11; ///< Exact original geometry/special-token value.
constexpr int kBlank   = 10; ///< Exact original geometry/special-token value.

/// input [B, 8, 49], lengths int32 [B] -> logits [8, B, 11].
/// @brief Apply CtcHandwriting with the original source equations and parameter names.
///
/// Float32 CPU[B,8,49] digit frames,Int32[B] valid lengths,8-unit state,11 classes including
/// blank10; output time-major[8,B,11]. Caller owns fitting, objectives and generation. Graph
/// tensors retain storage and live derivatives; lazy parameter creation mutates only borrowed
/// stores. Synchronize calls against concurrent store mutation. Copies of path fields share
/// storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param input Live CPU source endpoint with the family geometry above.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::ctc_handwriting::CtcHandwriting(p,n::zeros({1,8,49}),n::full({1},8,n::DType::Int32));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Hochreiter and Schmidhuber(1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor CtcHandwriting(nn::ParameterStore& p, const Tensor& input, const Tensor& lengths) {
  const auto   B        = input.shape()[0];
  const Tensor features = nn::reshape(
      nn::tanh(nn::dense(p, "input", nn::reshape(input, {-1, kPixels}), 8)), {B, kFrames, 8});
  const auto          steps = nn::time_steps(features);
  nn::LstmState       s     = nn::lstm_zero_state(B, 8, input.dtype());
  std::vector<Tensor> logits;
  for (int t = 0; t < kFrames; ++t) {
    const auto   next  = nn::lstm_step(p, "lstm", steps[t], s);
    const Tensor valid = nn::expand_dims(nn::greater(lengths, double(t)), 1);
    s                  = {nn::where(valid, next.h, s.h), nn::where(valid, next.c, s.c)};
    logits.push_back(nn::dense(p, "head", s.h, kClasses));
  }
  return nn::transpose(nn::stack_time(logits), {1, 0, 2});
}

} // namespace ctc_handwriting

/// @brief seq2seq source-equation components; no optimizer or training driver.
namespace seq2seq {

namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Map: owning source-compatible native type spelling.
using Map = std::map<std::string, Tensor>;

/// @brief kSteps, kUnits, kBos, kEos: live source graph endpoint/storage; tensor copies share
/// native storage.
constexpr int kSteps = 6; ///< Exact original geometry/special-token value.
constexpr int kUnits = 8; ///< Exact original geometry/special-token value.
constexpr int kBos   = 1; ///< Exact original geometry/special-token value.
constexpr int kEos   = 2; ///< Exact original geometry/special-token value.

/// The fit / validation / test pools of a seq2seq data.npz.

/// one-hot [..., V] x "embedding/W" [V, 8] ~ N(0, .1^2) -> [..., 8].
/// @brief Apply Embedding with the original source equations and parameter names.
///
/// Float32 CPU[B,6,V] encoded tokens,Int32[B] lengths in[0,6],8-unit states;BOS1/EOS2; padding
/// freezes recurrence. Caller owns fitting, objectives and generation. Graph tensors retain storage
/// and live derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param encoded Live Float32 CPU encoded tokens with the family context/vocabulary geometry.
/// @param vocab Explicit source geometry/control value; defaults match the declaration.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::seq2seq::Plain(p,x,x,l,l,z,z,14);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor Embedding(nn::ParameterStore& p, const Tensor& encoded, std::int64_t vocab) {
  const Tensor W     = p.get("embedding/W", {vocab, kUnits}, nn::init::normal(.1));
  nn::Shape    shape = encoded.shape();
  shape.back()       = kUnits;
  return nn::reshape(nn::matmul(nn::reshape(encoded, {-1, vocab}), W), shape);
}

/// "head/W" [in, V] ~ N(0, .1^2), "head/b" = 0 (tflow pre-adds them before Dense).
/// @brief Apply Head with the original source equations and parameter names.
///
/// Float32 CPU[B,6,V] encoded tokens,Int32[B] lengths in[0,6],8-unit states;BOS1/EOS2; padding
/// freezes recurrence. Caller owns fitting, objectives and generation. Graph tensors retain storage
/// and live derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param h Live CPU source endpoint with the family geometry above.
/// @param vocab Explicit source geometry/control value; defaults match the declaration.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::seq2seq::Plain(p,x,x,l,l,z,z,14);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor Head(nn::ParameterStore& p, const Tensor& h, std::int64_t vocab) {
  const Tensor W = p.get("head/W", {h.shape().back(), vocab}, nn::init::normal(.1));
  const Tensor b = p.get("head/b", {vocab}, nn::init::zeros());
  return nn::bias_add(nn::matmul(h, W), b);
}

/// [B, 1] bool: lengths > t.
/// @brief Apply ValidAt with the original source equations and parameter names.
///
/// Float32 CPU[B,6,V] encoded tokens,Int32[B] lengths in[0,6],8-unit states;BOS1/EOS2; padding
/// freezes recurrence. Caller owns fitting, objectives and generation. Graph tensors retain storage
/// and live derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @param t Explicit source geometry/control value; defaults match the declaration.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::seq2seq::Plain(p,x,x,l,l,z,z,14);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor ValidAt(const Tensor& lengths, int t) {
  return nn::expand_dims(nn::greater(lengths, double(t)), 1);
}
/// [B, 6] bool: position < lengths.
/// @brief Apply ValidPositions with the original source equations and parameter names.
///
/// Float32 CPU[B,6,V] encoded tokens,Int32[B] lengths in[0,6],8-unit states;BOS1/EOS2; padding
/// freezes recurrence. Caller owns fitting, objectives and generation. Graph tensors retain storage
/// and live derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::seq2seq::Plain(p,x,x,l,l,z,z,14);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor ValidPositions(const Tensor& lengths) {
  return nn::less(
      nn::reshape(nn::arange(0, kSteps, 1, nn::DType::Int32), {1, -1}),
      nn::reshape(lengths, {-1, 1}));
}

/// @brief Own Encoded source graph endpoints or configuration.
///
/// Float32 CPU[B,6,V] encoded tokens,Int32[B] lengths in[0,6],8-unit states;BOS1/EOS2; padding
/// freezes recurrence. Caller owns fitting, objectives and generation. Graph tensors retain storage
/// and live derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::seq2seq::Plain(p,x,x,l,l,z,z,14);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Encoded {
  /// @brief h, c: live source graph endpoint/storage; tensor copies share native storage.
  std::vector<Tensor> h; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor> c; ///< Live source endpoint; owning handle shares storage and derivatives. //

                         ///< 7 states each: the initial one, then one per source position

  /// @brief memory: live source graph endpoint/storage; tensor copies share native storage.
  Tensor memory; // [B, 6, 8]: the encoder state at valid positions, zero elsewhere
};

/// Length-masked LSTM "encoder" over the embedded source xe [B, 6, 8].
/// @brief Apply Encode with the original source equations and parameter names.
///
/// Float32 CPU[B,6,V] encoded tokens,Int32[B] lengths in[0,6],8-unit states;BOS1/EOS2; padding
/// freezes recurrence. Caller owns fitting, objectives and generation. Graph tensors retain storage
/// and live derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param xe Live CPU source endpoint with the family geometry above.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @param ih Live CPU source endpoint with the family geometry above.
/// @param ic Live CPU source endpoint with the family geometry above.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::seq2seq::Plain(p,x,x,l,l,z,z,14);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Encoded Encode(
    nn::ParameterStore& p,
    const Tensor&       xe,
    const Tensor&       lengths,
    const Tensor&       ih,
    const Tensor&       ic) {
  Encoded       out;
  nn::LstmState s{ih, ic};
  out.h.push_back(s.h);
  out.c.push_back(s.c);
  std::vector<Tensor> memory;
  const auto          xs = nn::time_steps(xe);
  for (int t = 0; t < kSteps; ++t) {
    const auto   next = nn::lstm_step(p, "encoder", xs[t], s);
    const Tensor mask = ValidAt(lengths, t);
    s                 = {nn::where(mask, next.h, s.h), nn::where(mask, next.c, s.c)};
    out.h.push_back(s.h);
    out.c.push_back(s.c);
    memory.push_back(nn::where(mask, s.h, 0.0));
  }
  out.memory = nn::stack_time(memory);
  return out;
}

/// Masked mean token cross-entropy of logits [B, 6, V] (tflow's seq2seq loss).

/// One decoder step from owned state: logits [B, V], next h, c [B, 8] and
/// attention weights [B, 6] (undefined for attention-free decoders).
/// @brief Own DecoderStep source graph endpoints or configuration.
///
/// Float32 CPU[B,6,V] encoded tokens,Int32[B] lengths in[0,6],8-unit states;BOS1/EOS2; padding
/// freezes recurrence. Caller owns fitting, objectives and generation. Graph tensors retain storage
/// and live derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::seq2seq::Plain(p,x,x,l,l,z,z,14);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct DecoderStep {
  /// @brief logits, h, c, attention: live source graph endpoint/storage; tensor copies share native
  /// storage.
  Tensor logits;    ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor h;         ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor c;         ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor attention; ///< Live source endpoint; owning handle shares storage and derivatives.
};
/// @brief StepFn: owning source-compatible native type spelling.
using StepFn = std::function<DecoderStep(const Tensor& token, const Tensor& h, const Tensor& c)>;

/// Greedy decoding from the final encoder state (h0, c0): BOS first, then each
/// row's own previous prediction, frozen after EOS. Returns int32 [B, 6]
/// (0 after EOS); `trace` (when given) receives the attention [B, 6, 6].

/// Masked attention normalisation shared by tflow's attentive decoders:
/// softmax over valid source positions (invalid scores -1e9, then zeroed),
/// divided by max(sum, 1) so that an empty source gives all-zero weights.
/// @brief Apply MaskedAttention with the original source equations and parameter names.
///
/// Float32 CPU[B,6,V] encoded tokens,Int32[B] lengths in[0,6],8-unit states;BOS1/EOS2; padding
/// freezes recurrence. Caller owns fitting, objectives and generation. Graph tensors retain storage
/// and live derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param scores Live CPU source endpoint with the family geometry above.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::seq2seq::Plain(p,x,x,l,l,z,z,14);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor MaskedAttention(const Tensor& scores, const Tensor& lengths) {
  const Tensor valid = ValidPositions(lengths);
  const Tensor raw =
      nn::mul(nn::softmax(nn::where(valid, scores, -1e9), -1), nn::cast(valid, scores.dtype()));
  return nn::div(raw, nn::maximum(nn::sum(raw, {1}, true), nn::scalar(1.0, scores.dtype())));
}

/// sum_t weights[b, t] memory[b, t, :] -> [B, 8].
/// @brief Apply Context with the original source equations and parameter names.
///
/// Float32 CPU[B,6,V] encoded tokens,Int32[B] lengths in[0,6],8-unit states;BOS1/EOS2; padding
/// freezes recurrence. Caller owns fitting, objectives and generation. Graph tensors retain storage
/// and live derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param memory Live source memory endpoints; segment-recurring forwards detach explicit past
/// states.
/// @param weights Live CPU source endpoint with the family geometry above.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::seq2seq::Plain(p,x,x,l,l,z,z,14);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor Context(const Tensor& memory, const Tensor& weights) {
  return nn::sum(nn::mul(memory, nn::expand_dims(weights, 2)), {1});
}

} // namespace seq2seq

// ---- added by job 4.2b: the plain (attention-free) model of tflow's
// common/nn/seq2seq.h and greedy decoding with configurable BOS/EOS/fill. ----
namespace seq2seq {

/// Greedy decoding like Greedy() with the start token `bos`, the stop token
/// `eos` and `fill` written after the stop (tflow's pointer network: -1, 5, -1).

/// tflow's plain encoder-decoder (nn::Seq2seq): the decoder LSTM starts from
/// the final encoder state and sees only the embedded teacher tokens; no
/// attention. 14-token vocabulary: 1326 parameters in 9 tensors.
/// @brief Own PlainPath source graph endpoints or configuration.
///
/// Float32 CPU[B,6,V] encoded tokens,Int32[B] lengths in[0,6],8-unit states;BOS1/EOS2; padding
/// freezes recurrence. Caller owns fitting, objectives and generation. Graph tensors retain storage
/// and live derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::seq2seq::Plain(p,x,x,l,l,z,z,14);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct PlainPath {
  /// @brief encoder_h, encoder_c, decoder_h, decoder_c, logits, bridge_h, bridge_c: live source
  /// graph endpoint/storage; tensor copies share native storage.
  Tensor encoder_h; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor encoder_c; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor decoder_h; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor decoder_c; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor logits;    ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor bridge_h;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor bridge_c;  ///< Live source endpoint; owning handle shares storage and derivatives.
};

/// One plain decoder step: embedded token [B, 8] -> state and logits [B, V].
/// @brief Apply PlainStep with the original source equations and parameter names.
///
/// Float32 CPU[B,6,V] encoded tokens,Int32[B] lengths in[0,6],8-unit states;BOS1/EOS2; padding
/// freezes recurrence. Caller owns fitting, objectives and generation. Graph tensors retain storage
/// and live derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param embedded Live CPU source endpoint with the family geometry above.
/// @param previous Explicit source geometry/control value; defaults match the declaration.
/// @param vocab Explicit source geometry/control value; defaults match the declaration.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::seq2seq::Plain(p,x,x,l,l,z,z,14);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline DecoderStep PlainStep(
    nn::ParameterStore&  p,
    const Tensor&        embedded,
    const nn::LstmState& previous,
    std::int64_t         vocab) {
  const auto s = nn::lstm_step(p, "decoder", embedded, previous);
  return {Head(p, s.h, vocab), s.h, s.c, Tensor()};
}

/// Teacher-forced pass; x, d are one-hot [B, 6, V].
/// @brief Apply Plain with the original source equations and parameter names.
///
/// Float32 CPU[B,6,V] encoded tokens,Int32[B] lengths in[0,6],8-unit states;BOS1/EOS2; padding
/// freezes recurrence. Caller owns fitting, objectives and generation. Graph tensors retain storage
/// and live derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param x Live CPU source endpoint with the family geometry above.
/// @param d Live CPU source endpoint with the family geometry above.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @param out_lengths Int32 CPU[B] valid teacher/output counts in[0,6].
/// @param ih Live CPU source endpoint with the family geometry above.
/// @param ic Live CPU source endpoint with the family geometry above.
/// @param vocab Explicit source geometry/control value; defaults match the declaration.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::seq2seq::Plain(p,x,x,l,l,z,z,14);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline PlainPath Plain(
    nn::ParameterStore& p,
    const Tensor&       x,
    const Tensor&       d,
    const Tensor&       lengths,
    const Tensor&       out_lengths,
    const Tensor&       ih,
    const Tensor&       ic,
    std::int64_t        vocab) {
  const Tensor xe  = Embedding(p, x, vocab);
  const Tensor de  = Embedding(p, d, vocab);
  const auto   enc = Encode(p, xe, lengths, ih, ic);
  PlainPath    out;
  out.bridge_h = enc.h.back();
  out.bridge_c = enc.c.back();
  nn::LstmState       s{out.bridge_h, out.bridge_c};
  std::vector<Tensor> dh{s.h}, dc{s.c}, logits;
  const auto          ds = nn::time_steps(de);
  for (int t = 0; t < kSteps; ++t) {
    const auto   next = PlainStep(p, ds[t], s, vocab);
    const Tensor mask = ValidAt(out_lengths, t);
    s                 = {nn::where(mask, next.h, s.h), nn::where(mask, next.c, s.c)};
    logits.push_back(nn::where(mask, next.logits, 0.0));
    dh.push_back(s.h);
    dc.push_back(s.c);
  }
  out.encoder_h = nn::stack_time(enc.h);
  out.encoder_c = nn::stack_time(enc.c);
  out.decoder_h = nn::stack_time(dh);
  out.decoder_c = nn::stack_time(dc);
  out.logits    = nn::stack_time(logits);
  return out;
}

} // namespace seq2seq

/// @brief bahdanau source-equation components; no optimizer or training driver.
namespace bahdanau {

namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief kSteps: live source graph endpoint/storage; tensor copies share native storage.
using seq2seq::kSteps;
/// @brief kUnits: live source graph endpoint/storage; tensor copies share native storage.
using seq2seq::kUnits;

/// @brief kVocab: live source graph endpoint/storage; tensor copies share native storage.
constexpr int kVocab = 14;

/// @brief Own Attended source graph endpoints or configuration.
///
/// Float32 CPU[B,6,14] source/teacher,Int32[B] lengths in[0,6],8-unit states.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::bahdanau::Bahdanau(p,x,x,l,l,z,z);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Attended {
  /// @brief weights, context: live source graph endpoint/storage; tensor copies share native
  /// storage.
  Tensor weights; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor context; ///< Live source endpoint; owning handle shares storage and derivatives.
};

/// @brief Apply BahdanauContext with the original source equations and parameter names.
///
/// Float32 CPU[B,6,14] source/teacher,Int32[B] lengths in[0,6],8-unit states.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param memory Live source memory endpoints; segment-recurring forwards detach explicit past
/// states.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @param query Live query endpoint or true to use query-feature stabilization, according to the
/// declared type.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::bahdanau::Bahdanau(p,x,x,l,l,z,z);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Attended BahdanauContext(
    nn::ParameterStore& p,
    const Tensor&       memory,
    const Tensor&       lengths,
    const Tensor&       query) {
  const Tensor we = p.get("attention/Wenc", {kUnits, kUnits}, nn::init::normal(.1));
  const Tensor wq = p.get("attention/Wquery", {kUnits, kUnits}, nn::init::normal(.1));
  const Tensor b  = p.get("attention/b", {kUnits}, nn::init::zeros());
  const Tensor v  = p.get("attention/v", {kUnits, 1}, nn::init::normal(.1));
  const Tensor keys =
      nn::reshape(nn::matmul(nn::reshape(memory, {-1, kUnits}), we), {-1, kSteps, kUnits});
  const Tensor q      = nn::expand_dims(nn::matmul(query, wq), 1);
  const Tensor hidden = nn::tanh(nn::add(nn::add(keys, q), b));
  const Tensor scores = nn::reshape(nn::matmul(nn::reshape(hidden, {-1, kUnits}), v), {-1, kSteps});
  Attended     out;
  out.weights = seq2seq::MaskedAttention(scores, lengths);
  out.context = seq2seq::Context(memory, out.weights);
  return out;
}

/// @brief Own Step source graph endpoints or configuration.
///
/// Float32 CPU[B,6,14] source/teacher,Int32[B] lengths in[0,6],8-unit states.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::bahdanau::Bahdanau(p,x,x,l,l,z,z);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Step {
  /// @brief state: live source graph endpoint/storage; tensor copies share native storage.
  nn::LstmState state;
  /// @brief logits, weights, context: live source graph endpoint/storage; tensor copies share
  /// native storage.
  Tensor logits;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor weights; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor context; ///< Live source endpoint; owning handle shares storage and derivatives.
};

/// @brief Apply BahdanauStep with the original source equations and parameter names.
///
/// Float32 CPU[B,6,14] source/teacher,Int32[B] lengths in[0,6],8-unit states.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param embedded Live CPU source endpoint with the family geometry above.
/// @param previous Explicit source geometry/control value; defaults match the declaration.
/// @param memory Live source memory endpoints; segment-recurring forwards detach explicit past
/// states.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::bahdanau::Bahdanau(p,x,x,l,l,z,z);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Step BahdanauStep(
    nn::ParameterStore&  p,
    const Tensor&        embedded,
    const nn::LstmState& previous,
    const Tensor&        memory,
    const Tensor&        lengths) {
  const Attended att = BahdanauContext(p, memory, lengths, previous.h);
  const auto state = nn::lstm_step(p, "decoder", nn::concat({embedded, att.context}, 1), previous);
  return {state, seq2seq::Head(p, state.h, kVocab), att.weights, att.context};
}

/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,6,14] source/teacher,Int32[B] lengths in[0,6],8-unit states.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::bahdanau::Bahdanau(p,x,x,l,l,z,z);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief encoder_h, encoder_c, decoder_h, decoder_c, logits, bridge_h, bridge_c, attention,
  /// contexts, memory: live source graph endpoint/storage; tensor copies share native storage.
  Tensor encoder_h; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor encoder_c; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor decoder_h; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor decoder_c; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor logits;    ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor bridge_h;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor bridge_c;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor contexts;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor memory;    ///< Live source endpoint; owning handle shares storage and derivatives.
};

/// Teacher-forced pass; x, d are one-hot [B, 6, 14].
/// @brief Apply Bahdanau with the original source equations and parameter names.
///
/// Float32 CPU[B,6,14] source/teacher,Int32[B] lengths in[0,6],8-unit states.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param x Live CPU source endpoint with the family geometry above.
/// @param d Live CPU source endpoint with the family geometry above.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @param out_lengths Int32 CPU[B] valid teacher/output counts in[0,6].
/// @param ih Live CPU source endpoint with the family geometry above.
/// @param ic Live CPU source endpoint with the family geometry above.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::bahdanau::Bahdanau(p,x,x,l,l,z,z);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path Bahdanau(
    nn::ParameterStore& p,
    const Tensor&       x,
    const Tensor&       d,
    const Tensor&       lengths,
    const Tensor&       out_lengths,
    const Tensor&       ih,
    const Tensor&       ic) {
  const Tensor xe  = seq2seq::Embedding(p, x, kVocab);
  const Tensor de  = seq2seq::Embedding(p, d, kVocab);
  const auto   enc = seq2seq::Encode(p, xe, lengths, ih, ic);
  Path         out;
  out.memory   = enc.memory;
  out.bridge_h = enc.h.back();
  out.bridge_c = enc.c.back();
  nn::LstmState       s{out.bridge_h, out.bridge_c};
  std::vector<Tensor> dh{s.h}, dc{s.c}, logits, attention, contexts;
  const auto          ds = nn::time_steps(de);
  for (int t = 0; t < kSteps; ++t) {
    const Step   next = BahdanauStep(p, ds[t], s, enc.memory, lengths);
    const Tensor mask = seq2seq::ValidAt(out_lengths, t);
    s                 = {nn::where(mask, next.state.h, s.h), nn::where(mask, next.state.c, s.c)};
    logits.push_back(nn::where(mask, next.logits, 0.0));
    attention.push_back(nn::where(mask, next.weights, 0.0));
    contexts.push_back(nn::where(mask, next.context, 0.0));
    dh.push_back(s.h);
    dc.push_back(s.c);
  }
  out.encoder_h = nn::stack_time(enc.h);
  out.encoder_c = nn::stack_time(enc.c);
  out.decoder_h = nn::stack_time(dh);
  out.decoder_c = nn::stack_time(dc);
  out.logits    = nn::stack_time(logits);
  out.attention = nn::stack_time(attention);
  out.contexts  = nn::stack_time(contexts);
  return out;
}

} // namespace bahdanau

/// @brief translation source-equation components; no optimizer or training driver.
namespace translation {

namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief kSteps: live source graph endpoint/storage; tensor copies share native storage.
using seq2seq::kSteps;
/// @brief kUnits: live source graph endpoint/storage; tensor copies share native storage.
using seq2seq::kUnits;

/// @brief kVocab: live source graph endpoint/storage; tensor copies share native storage.
constexpr int kVocab = 256;

/// @brief Own Attended source graph endpoints or configuration.
///
/// Float32 CPU[B,6,256] source/teacher,Int32[B] lengths in[0,6],8-unit states.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);m::translation::Build(p);auto
/// x=n::zeros({1,6,256}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::translation::Translation(p,x,x,l,l,z,z);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Attended {
  /// @brief weights, context: live source graph endpoint/storage; tensor copies share native
  /// storage.
  Tensor weights; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor context; ///< Live source endpoint; owning handle shares storage and derivatives.
};

/// @brief Apply TranslationContext with the original source equations and parameter names.
///
/// Float32 CPU[B,6,256] source/teacher,Int32[B] lengths in[0,6],8-unit states.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param memory Live source memory endpoints; segment-recurring forwards detach explicit past
/// states.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @param query Live query endpoint or true to use query-feature stabilization, according to the
/// declared type.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);m::translation::Build(p);auto
/// x=n::zeros({1,6,256}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::translation::Translation(p,x,x,l,l,z,z);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Attended TranslationContext(
    nn::ParameterStore& p,
    const Tensor&       memory,
    const Tensor&       lengths,
    const Tensor&       query) {
  const Tensor w         = p.get("attention/W", {kUnits, kUnits}, nn::init::normal(.1));
  const Tensor projected = nn::expand_dims(nn::matmul(query, w), 1);
  const Tensor scores    = nn::sum(nn::mul(projected, memory), {2});
  Attended     out;
  out.weights = seq2seq::MaskedAttention(scores, lengths);
  out.context = seq2seq::Context(memory, out.weights);
  return out;
}

/// @brief Own Step source graph endpoints or configuration.
///
/// Float32 CPU[B,6,256] source/teacher,Int32[B] lengths in[0,6],8-unit states.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);m::translation::Build(p);auto
/// x=n::zeros({1,6,256}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::translation::Translation(p,x,x,l,l,z,z);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Step {
  /// @brief state: live source graph endpoint/storage; tensor copies share native storage.
  nn::LstmState state;
  /// @brief logits, weights, context: live source graph endpoint/storage; tensor copies share
  /// native storage.
  Tensor logits;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor weights; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor context; ///< Live source endpoint; owning handle shares storage and derivatives.
};

/// @brief Apply TranslationStep with the original source equations and parameter names.
///
/// Float32 CPU[B,6,256] source/teacher,Int32[B] lengths in[0,6],8-unit states.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param embedded Live CPU source endpoint with the family geometry above.
/// @param previous Explicit source geometry/control value; defaults match the declaration.
/// @param memory Live source memory endpoints; segment-recurring forwards detach explicit past
/// states.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);m::translation::Build(p);auto
/// x=n::zeros({1,6,256}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::translation::Translation(p,x,x,l,l,z,z);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Step TranslationStep(
    nn::ParameterStore&  p,
    const Tensor&        embedded,
    const nn::LstmState& previous,
    const Tensor&        memory,
    const Tensor&        lengths) {
  const auto     state = nn::lstm_step(p, "decoder", embedded, previous);
  const Attended att   = TranslationContext(p, memory, lengths, state.h);
  const Tensor   wc    = p.get("combine/W", {2 * kUnits, kUnits}, nn::init::normal(.1));
  const Tensor   bc    = p.get("combine/b", {kUnits}, nn::init::zeros());
  const Tensor   attentional =
      nn::tanh(nn::bias_add(nn::matmul(nn::concat({att.context, state.h}, 1), wc), bc));
  return {state, seq2seq::Head(p, attentional, kVocab), att.weights, att.context};
}

/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,6,256] source/teacher,Int32[B] lengths in[0,6],8-unit states.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);m::translation::Build(p);auto
/// x=n::zeros({1,6,256}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::translation::Translation(p,x,x,l,l,z,z);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief encoder_h, encoder_c, decoder_h, decoder_c, logits, bridge_h, bridge_c, attention,
  /// contexts, memory: live source graph endpoint/storage; tensor copies share native storage.
  Tensor encoder_h; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor encoder_c; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor decoder_h; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor decoder_c; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor logits;    ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor bridge_h;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor bridge_c;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor contexts;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor memory;    ///< Live source endpoint; owning handle shares storage and derivatives.
};

/// Teacher-forced pass; x, d are one-hot [B, 6, 256].
/// @brief Apply Translation with the original source equations and parameter names.
///
/// Float32 CPU[B,6,256] source/teacher,Int32[B] lengths in[0,6],8-unit states.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param x Live CPU source endpoint with the family geometry above.
/// @param d Live CPU source endpoint with the family geometry above.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @param out_lengths Int32 CPU[B] valid teacher/output counts in[0,6].
/// @param ih Live CPU source endpoint with the family geometry above.
/// @param ic Live CPU source endpoint with the family geometry above.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);m::translation::Build(p);auto
/// x=n::zeros({1,6,256}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::translation::Translation(p,x,x,l,l,z,z);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path Translation(
    nn::ParameterStore& p,
    const Tensor&       x,
    const Tensor&       d,
    const Tensor&       lengths,
    const Tensor&       out_lengths,
    const Tensor&       ih,
    const Tensor&       ic) {
  const Tensor xe  = seq2seq::Embedding(p, x, kVocab);
  const Tensor de  = seq2seq::Embedding(p, d, kVocab);
  const auto   enc = seq2seq::Encode(p, xe, lengths, ih, ic);
  Path         out;
  out.memory   = enc.memory;
  out.bridge_h = enc.h.back();
  out.bridge_c = enc.c.back();
  nn::LstmState       s{out.bridge_h, out.bridge_c};
  std::vector<Tensor> dh{s.h}, dc{s.c}, logits, attention, contexts;
  const auto          ds = nn::time_steps(de);
  for (int t = 0; t < kSteps; ++t) {
    const Step   next = TranslationStep(p, ds[t], s, enc.memory, lengths);
    const Tensor mask = seq2seq::ValidAt(out_lengths, t);
    s                 = {nn::where(mask, next.state.h, s.h), nn::where(mask, next.state.c, s.c)};
    logits.push_back(nn::where(mask, next.logits, 0.0));
    attention.push_back(nn::where(mask, next.weights, 0.0));
    contexts.push_back(nn::where(mask, next.context, 0.0));
    dh.push_back(s.h);
    dc.push_back(s.c);
  }
  out.encoder_h = nn::stack_time(enc.h);
  out.encoder_c = nn::stack_time(enc.c);
  out.decoder_h = nn::stack_time(dh);
  out.decoder_c = nn::stack_time(dc);
  out.logits    = nn::stack_time(logits);
  out.attention = nn::stack_time(attention);
  out.contexts  = nn::stack_time(contexts);
  return out;
}

/// Creates every parameter in tflow's order.
/// @brief Apply Build with the original source equations and parameter names.
///
/// Float32 CPU[B,6,256] source/teacher,Int32[B] lengths in[0,6],8-unit states.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);m::translation::Build(p);auto
/// x=n::zeros({1,6,256}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::translation::Translation(p,x,x,l,l,z,z);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline void Build(nn::ParameterStore& p) {
  nn::NoGradGuard guard;
  const Tensor    one = nn::one_hot(nn::zeros({1, kSteps}, nn::DType::Int32), kVocab),
               len    = nn::ones({1}, nn::DType::Int32);
  const Tensor z      = nn::zeros({1, kUnits});
  Translation(p, one, one, len, len, z, z);
}

} // namespace translation

/// @brief luong source-equation components; no optimizer or training driver.
namespace luong {

namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief kSteps: live source graph endpoint/storage; tensor copies share native storage.
using seq2seq::kSteps;
/// @brief kUnits: live source graph endpoint/storage; tensor copies share native storage.
using seq2seq::kUnits;

/// @brief kVocab: live source graph endpoint/storage; tensor copies share native storage.
constexpr int kVocab = 14;

/// @brief Own Step source graph endpoints or configuration.
///
/// Float32 CPU[B,6,14] source/teacher,Int32[B] lengths in[0,6],8-unit states.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::luong::Luong(p,x,x,l,l,z,z);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Step {
  /// @brief state: live source graph endpoint/storage; tensor copies share native storage.
  nn::LstmState state;
  /// @brief logits, weights, context: live source graph endpoint/storage; tensor copies share
  /// native storage.
  Tensor logits;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor weights; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor context; ///< Live source endpoint; owning handle shares storage and derivatives.
};

/// @brief Apply LuongStep with the original source equations and parameter names.
///
/// Float32 CPU[B,6,14] source/teacher,Int32[B] lengths in[0,6],8-unit states.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param embedded Live CPU source endpoint with the family geometry above.
/// @param previous Explicit source geometry/control value; defaults match the declaration.
/// @param memory Live source memory endpoints; segment-recurring forwards detach explicit past
/// states.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::luong::Luong(p,x,x,l,l,z,z);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Step LuongStep(
    nn::ParameterStore&  p,
    const Tensor&        embedded,
    const nn::LstmState& previous,
    const Tensor&        memory,
    const Tensor&        lengths) {
  const auto   state = nn::lstm_step(p, "decoder", embedded, previous);
  const auto   att   = translation::TranslationContext(p, memory, lengths, state.h);
  const Tensor wc    = p.get("combine/W", {2 * kUnits, kUnits}, nn::init::normal(.1));
  const Tensor bc    = p.get("combine/b", {kUnits}, nn::init::zeros());
  const Tensor attentional =
      nn::tanh(nn::bias_add(nn::matmul(nn::concat({att.context, state.h}, 1), wc), bc));
  return {state, seq2seq::Head(p, attentional, kVocab), att.weights, att.context};
}

/// @brief Path: owning source-compatible native type spelling.
using Path = translation::Path;

/// Teacher-forced pass; x, d are one-hot [B, 6, 14].
/// @brief Apply Luong with the original source equations and parameter names.
///
/// Float32 CPU[B,6,14] source/teacher,Int32[B] lengths in[0,6],8-unit states.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param x Live CPU source endpoint with the family geometry above.
/// @param d Live CPU source endpoint with the family geometry above.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @param out_lengths Int32 CPU[B] valid teacher/output counts in[0,6].
/// @param ih Live CPU source endpoint with the family geometry above.
/// @param ic Live CPU source endpoint with the family geometry above.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::luong::Luong(p,x,x,l,l,z,z);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path Luong(
    nn::ParameterStore& p,
    const Tensor&       x,
    const Tensor&       d,
    const Tensor&       lengths,
    const Tensor&       out_lengths,
    const Tensor&       ih,
    const Tensor&       ic) {
  const Tensor xe  = seq2seq::Embedding(p, x, kVocab);
  const Tensor de  = seq2seq::Embedding(p, d, kVocab);
  const auto   enc = seq2seq::Encode(p, xe, lengths, ih, ic);
  Path         out;
  out.memory   = enc.memory;
  out.bridge_h = enc.h.back();
  out.bridge_c = enc.c.back();
  nn::LstmState       s{out.bridge_h, out.bridge_c};
  std::vector<Tensor> dh{s.h}, dc{s.c}, logits, attention, contexts;
  const auto          ds = nn::time_steps(de);
  for (int t = 0; t < kSteps; ++t) {
    const Step   next = LuongStep(p, ds[t], s, enc.memory, lengths);
    const Tensor mask = seq2seq::ValidAt(out_lengths, t);
    s                 = {nn::where(mask, next.state.h, s.h), nn::where(mask, next.state.c, s.c)};
    logits.push_back(nn::where(mask, next.logits, 0.0));
    attention.push_back(nn::where(mask, next.weights, 0.0));
    contexts.push_back(nn::where(mask, next.context, 0.0));
    dh.push_back(s.h);
    dc.push_back(s.c);
  }
  out.encoder_h = nn::stack_time(enc.h);
  out.encoder_c = nn::stack_time(enc.c);
  out.decoder_h = nn::stack_time(dh);
  out.decoder_c = nn::stack_time(dc);
  out.logits    = nn::stack_time(logits);
  out.attention = nn::stack_time(attention);
  out.contexts  = nn::stack_time(contexts);
  return out;
}

} // namespace luong

/// @brief pointer network source-equation components; no optimizer or training driver.
namespace pointer_network {

namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief kSteps: live source graph endpoint/storage; tensor copies share native storage.
using seq2seq::kSteps;
/// @brief kUnits: live source graph endpoint/storage; tensor copies share native storage.
using seq2seq::kUnits;

/// @brief kPoints, kStop, kChoices, kBos: live source graph endpoint/storage; tensor copies share
/// native storage.
constexpr int kPoints  = 5;  ///< Exact original geometry/special-token value.
constexpr int kStop    = 5;  ///< Exact original geometry/special-token value.
constexpr int kChoices = 6;  ///< Exact original geometry/special-token value.
constexpr int kBos     = -1; ///< Exact original geometry/special-token value.

/// Additive pointer scores [B, 6] over the memory [B, 6, 8]; invalid input
/// positions (>= length, except STOP) are -1e9.
/// @brief Apply PointerScores with the original source equations and parameter names.
///
/// Float32 CPU[B,5,2] points,encoded[B,6,6] teacher choices,Int32[B,6] pointers;STOP5,BOS-1;8-unit
/// states. Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param memory Live source memory endpoints; segment-recurring forwards detach explicit past
/// states.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @param query Live query endpoint or true to use query-feature stabilization, according to the
/// declared type.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// z=n::zeros({1,8}),l=n::full({1},5,n::DType::Int32);auto
/// y=m::pointer_network::PointerNetwork(p,n::zeros({1,5,2}),n::zeros({1,6,6}),n::full({1,6},-1,n::DType::Int32),l,l,z,z);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor PointerScores(
    nn::ParameterStore& p,
    const Tensor&       memory,
    const Tensor&       lengths,
    const Tensor&       query) {
  const Tensor we   = p.get("pointer/Wmemory", {kUnits, kUnits}, nn::init::normal(.1));
  const Tensor wq   = p.get("pointer/Wquery", {kUnits, kUnits}, nn::init::normal(.1));
  const Tensor bias = p.get("pointer/b", {kUnits}, nn::init::zeros());
  const Tensor v    = p.get("pointer/v", {kUnits, 1}, nn::init::normal(.1));
  const Tensor keys =
      nn::reshape(nn::matmul(nn::reshape(memory, {-1, kUnits}), we), {-1, kChoices, kUnits});
  const Tensor q      = nn::expand_dims(nn::matmul(query, wq), 1);
  const Tensor hidden = nn::tanh(nn::add(nn::add(keys, q), bias));
  const Tensor scores =
      nn::reshape(nn::matmul(nn::reshape(hidden, {-1, kUnits}), v), {-1, kChoices});
  const Tensor positions = nn::reshape(nn::arange(0, kChoices, 1, nn::DType::Int32), {1, -1});
  const Tensor valid     = nn::logical_or(
      nn::less(positions, nn::reshape(lengths, {-1, 1})), nn::equal(positions, double(kStop)));
  return nn::where(valid, scores, -1e9);
}

/// @brief Own Step source graph endpoints or configuration.
///
/// Float32 CPU[B,5,2] points,encoded[B,6,6] teacher choices,Int32[B,6] pointers;STOP5,BOS-1;8-unit
/// states. Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// z=n::zeros({1,8}),l=n::full({1},5,n::DType::Int32);auto
/// y=m::pointer_network::PointerNetwork(p,n::zeros({1,5,2}),n::zeros({1,6,6}),n::full({1,6},-1,n::DType::Int32),l,l,z,z);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Step {
  /// @brief state: live source graph endpoint/storage; tensor copies share native storage.
  nn::LstmState state;
  /// @brief logits, selected: live source graph endpoint/storage; tensor copies share native
  /// storage.
  Tensor logits;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor selected; ///< Live source endpoint; owning handle shares storage and derivatives.
};

/// pointer: one-hot previous pointer [B, 6] (all-zero for BOS); bos: bool [B].
/// @brief Apply PointerStep with the original source equations and parameter names.
///
/// Float32 CPU[B,5,2] points,encoded[B,6,6] teacher choices,Int32[B,6] pointers;STOP5,BOS-1;8-unit
/// states. Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param pointer Live CPU source endpoint with the family geometry above.
/// @param bos Live CPU source endpoint with the family geometry above.
/// @param previous Explicit source geometry/control value; defaults match the declaration.
/// @param memory Live source memory endpoints; segment-recurring forwards detach explicit past
/// states.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// z=n::zeros({1,8}),l=n::full({1},5,n::DType::Int32);auto
/// y=m::pointer_network::PointerNetwork(p,n::zeros({1,5,2}),n::zeros({1,6,6}),n::full({1,6},-1,n::DType::Int32),l,l,z,z);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Step PointerStep(
    nn::ParameterStore&  p,
    const Tensor&        pointer,
    const Tensor&        bos,
    const nn::LstmState& previous,
    const Tensor&        memory,
    const Tensor&        lengths) {
  Tensor       selected = nn::sum(nn::mul(memory, nn::expand_dims(pointer, 2)), {1});
  const Tensor b        = p.get("decoder/BOS", {kUnits}, nn::init::normal(.1));
  selected = nn::add(selected, nn::mul(nn::expand_dims(nn::cast(bos, selected.dtype()), 1), b));
  const auto next = nn::lstm_step(p, "decoder", selected, previous);
  return {next, PointerScores(p, memory, lengths, next.h), selected};
}

/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,5,2] points,encoded[B,6,6] teacher choices,Int32[B,6] pointers;STOP5,BOS-1;8-unit
/// states. Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// z=n::zeros({1,8}),l=n::full({1},5,n::DType::Int32);auto
/// y=m::pointer_network::PointerNetwork(p,n::zeros({1,5,2}),n::zeros({1,6,6}),n::full({1,6},-1,n::DType::Int32),l,l,z,z);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief encoder_h, encoder_c, decoder_h, decoder_c, logits, bridge_h, bridge_c, memory,
  /// selected: live source graph endpoint/storage; tensor copies share native storage.
  Tensor encoder_h; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor encoder_c; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor decoder_h; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor decoder_c; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor logits;    ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor bridge_h;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor bridge_c;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor memory;    ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor selected;  ///< Live source endpoint; owning handle shares storage and derivatives.
};

/// Teacher-forced pass: points [B, 5, 2], teacher_weights one-hot [B, 6, 6], teacher_ids int32 [B,
/// 6].
/// @brief Apply PointerNetwork with the original source equations and parameter names.
///
/// Float32 CPU[B,5,2] points,encoded[B,6,6] teacher choices,Int32[B,6] pointers;STOP5,BOS-1;8-unit
/// states. Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param points Live CPU source endpoint with the family geometry above.
/// @param teacher_weights Live CPU source endpoint with the family geometry above.
/// @param teacher_ids Live CPU source endpoint with the family geometry above.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @param out_lengths Int32 CPU[B] valid teacher/output counts in[0,6].
/// @param ih Live CPU source endpoint with the family geometry above.
/// @param ic Live CPU source endpoint with the family geometry above.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// z=n::zeros({1,8}),l=n::full({1},5,n::DType::Int32);auto
/// y=m::pointer_network::PointerNetwork(p,n::zeros({1,5,2}),n::zeros({1,6,6}),n::full({1,6},-1,n::DType::Int32),l,l,z,z);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path PointerNetwork(
    nn::ParameterStore& p,
    const Tensor&       points,
    const Tensor&       teacher_weights,
    const Tensor&       teacher_ids,
    const Tensor&       lengths,
    const Tensor&       out_lengths,
    const Tensor&       ih,
    const Tensor&       ic) {
  p.get("input/W", {2, kUnits}, nn::init::normal(.1));
  p.get("input/b", {kUnits}, nn::init::zeros());
  const Tensor embedding = nn::reshape(
      nn::dense(p, "input", nn::reshape(points, {-1, 2}), kUnits, nn::Activation::Tanh),
      {-1, kPoints, kUnits});
  const auto xs = nn::time_steps(embedding), teachers = nn::time_steps(teacher_weights),
             ids = nn::time_steps(teacher_ids);
  nn::LstmState       s{ih, ic};
  std::vector<Tensor> eh{s.h}, ec{s.c}, dh, dc, logits, memory, selected;
  for (int t = 0; t < kPoints; ++t) {
    const auto   next  = nn::lstm_step(p, "encoder", xs[t], s);
    const Tensor valid = seq2seq::ValidAt(lengths, t);
    s                  = {nn::where(valid, next.h, s.h), nn::where(valid, next.c, s.c)};
    eh.push_back(s.h);
    ec.push_back(s.c);
    memory.push_back(nn::where(valid, s.h, 0.0));
  }
  const Tensor stop = p.get("pointer/STOP", {kUnits}, nn::init::normal(.1));
  memory.push_back(nn::add(nn::zeros_like(s.h), stop));
  Path out;
  out.memory   = nn::stack_time(memory);
  out.bridge_h = s.h;
  out.bridge_c = s.c;
  dh.push_back(s.h);
  dc.push_back(s.c);
  for (int t = 0; t < kSteps; ++t) {
    const Step   next  = PointerStep(p, teachers[t], nn::less(ids[t], 0.0), s, out.memory, lengths);
    const Tensor valid = seq2seq::ValidAt(out_lengths, t);
    s                  = {nn::where(valid, next.state.h, s.h), nn::where(valid, next.state.c, s.c)};
    logits.push_back(nn::where(valid, next.logits, 0.0));
    selected.push_back(nn::where(valid, next.selected, 0.0));
    dh.push_back(s.h);
    dc.push_back(s.c);
  }
  out.encoder_h = nn::stack_time(eh);
  out.encoder_c = nn::stack_time(ec);
  out.decoder_h = nn::stack_time(dh);
  out.decoder_c = nn::stack_time(dc);
  out.logits    = nn::stack_time(logits);
  out.selected  = nn::stack_time(selected);
  return out;
}

} // namespace pointer_network

/// @brief pointer generator source-equation components; no optimizer or training driver.
namespace pointer_generator {

namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief kSteps: live source graph endpoint/storage; tensor copies share native storage.
using seq2seq::kSteps;
/// @brief kUnits: live source graph endpoint/storage; tensor copies share native storage.
using seq2seq::kUnits;

/// @brief kBase, kExtended: live source graph endpoint/storage; tensor copies share native storage.
constexpr int kBase     = 14; ///< Exact original geometry/special-token value.
constexpr int kExtended = 20; ///< Exact original geometry/special-token value.

/// @brief Own Step source graph endpoints or configuration.
///
/// Float32 CPU[B,6,14] base tokens and[B,6,20] copy IDs,Int32[B] lengths in[0,6].
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::pointer_generator::PointerGenerator(p,x,x,l,l,z,z,n::zeros({1,6,20}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Step {
  /// @brief state: live source graph endpoint/storage; tensor copies share native storage.
  nn::LstmState state;
  /// @brief logits, weights, context: live source graph endpoint/storage; tensor copies share
  /// native storage.
  Tensor logits;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor weights; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor context; ///< Live source endpoint; owning handle shares storage and derivatives.
};

/// One step; `source` is the one-hot extended source [B, 6, 20].
/// @brief Apply PointerGeneratorStep with the original source equations and parameter names.
///
/// Float32 CPU[B,6,14] base tokens and[B,6,20] copy IDs,Int32[B] lengths in[0,6].
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param embedded Live CPU source endpoint with the family geometry above.
/// @param previous Explicit source geometry/control value; defaults match the declaration.
/// @param memory Live source memory endpoints; segment-recurring forwards detach explicit past
/// states.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @param source Live CPU source endpoint with the family geometry above.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::pointer_generator::PointerGenerator(p,x,x,l,l,z,z,n::zeros({1,6,20}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Step PointerGeneratorStep(
    nn::ParameterStore&  p,
    const Tensor&        embedded,
    const nn::LstmState& previous,
    const Tensor&        memory,
    const Tensor&        lengths,
    const Tensor&        source) {
  const auto att   = bahdanau::BahdanauContext(p, memory, lengths, previous.h);
  const auto state = nn::lstm_step(p, "decoder", nn::concat({embedded, att.context}, 1), previous);
  const Tensor features = nn::concat({att.context, state.h, embedded}, 1);
  const Tensor gw       = p.get("gate/W", {3 * kUnits, 1}, nn::init::normal(.1));
  const Tensor gb       = p.get("gate/b", {1}, nn::init::zeros());
  const Tensor learned  = nn::sigmoid(nn::add(nn::matmul(features, gw), gb));
  // With no copyable source, use the normalized generator alone.
  const Tensor gate = nn::where(nn::expand_dims(nn::greater(lengths, 0.0), 1), learned, 1.0);
  const Tensor vocab =
      nn::pad(nn::softmax(seq2seq::Head(p, state.h, kBase), -1), {{0, 0}, {0, kExtended - kBase}});
  const Tensor copy = nn::sum(nn::mul(source, nn::expand_dims(att.weights, 2)), {1});
  const Tensor mixed =
      nn::add(nn::mul(gate, vocab), nn::mul(nn::sub(nn::scalar(1.0, gate.dtype()), gate), copy));
  // Tiny explicit smoothing keeps the log finite at absent extended IDs.
  return {
      state, nn::log(nn::add(mixed, nn::scalar(1e-9, mixed.dtype()))), att.weights, att.context};
}

/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,6,14] base tokens and[B,6,20] copy IDs,Int32[B] lengths in[0,6].
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::pointer_generator::PointerGenerator(p,x,x,l,l,z,z,n::zeros({1,6,20}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief encoder_h, encoder_c, decoder_h, decoder_c, logits, bridge_h, bridge_c, attention,
  /// contexts, memory: live source graph endpoint/storage; tensor copies share native storage.
  Tensor encoder_h; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor encoder_c; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor decoder_h; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor decoder_c; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor logits;    ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor bridge_h;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor bridge_c;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor contexts;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor memory;    ///< Live source endpoint; owning handle shares storage and derivatives.
};

/// Teacher-forced pass; x, d are base one-hot [B, 6, 14], source the extended one-hot [B, 6, 20].
/// @brief Apply PointerGenerator with the original source equations and parameter names.
///
/// Float32 CPU[B,6,14] base tokens and[B,6,20] copy IDs,Int32[B] lengths in[0,6].
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param x Live CPU source endpoint with the family geometry above.
/// @param d Live CPU source endpoint with the family geometry above.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @param out_lengths Int32 CPU[B] valid teacher/output counts in[0,6].
/// @param ih Live CPU source endpoint with the family geometry above.
/// @param ic Live CPU source endpoint with the family geometry above.
/// @param source Live CPU source endpoint with the family geometry above.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::pointer_generator::PointerGenerator(p,x,x,l,l,z,z,n::zeros({1,6,20}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path PointerGenerator(
    nn::ParameterStore& p,
    const Tensor&       x,
    const Tensor&       d,
    const Tensor&       lengths,
    const Tensor&       out_lengths,
    const Tensor&       ih,
    const Tensor&       ic,
    const Tensor&       source) {
  const Tensor xe  = seq2seq::Embedding(p, x, kBase);
  const Tensor de  = seq2seq::Embedding(p, d, kBase);
  const auto   enc = seq2seq::Encode(p, xe, lengths, ih, ic);
  Path         out;
  out.memory   = enc.memory;
  out.bridge_h = enc.h.back();
  out.bridge_c = enc.c.back();
  nn::LstmState       s{out.bridge_h, out.bridge_c};
  std::vector<Tensor> dh{s.h}, dc{s.c}, logits, attention, contexts;
  const auto          ds = nn::time_steps(de);
  for (int t = 0; t < kSteps; ++t) {
    const Step   next = PointerGeneratorStep(p, ds[t], s, enc.memory, lengths, source);
    const Tensor mask = seq2seq::ValidAt(out_lengths, t);
    s                 = {nn::where(mask, next.state.h, s.h), nn::where(mask, next.state.c, s.c)};
    logits.push_back(nn::where(mask, next.logits, 0.0));
    attention.push_back(nn::where(mask, next.weights, 0.0));
    contexts.push_back(nn::where(mask, next.context, 0.0));
    dh.push_back(s.h);
    dc.push_back(s.c);
  }
  out.encoder_h = nn::stack_time(enc.h);
  out.encoder_c = nn::stack_time(enc.c);
  out.decoder_h = nn::stack_time(dh);
  out.decoder_c = nn::stack_time(dc);
  out.logits    = nn::stack_time(logits);
  out.attention = nn::stack_time(attention);
  out.contexts  = nn::stack_time(contexts);
  return out;
}

/// Base-vocabulary ids: min(ids, 13).
/// @brief Apply Base with the original source equations and parameter names.
///
/// Float32 CPU[B,6,14] base tokens and[B,6,20] copy IDs,Int32[B] lengths in[0,6].
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param ids Live CPU source endpoint with the family geometry above.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,14}),z=n::zeros({1,8}),l=n::full({1},6,n::DType::Int32);auto
/// y=m::pointer_generator::PointerGenerator(p,x,x,l,l,z,z,n::zeros({1,6,20}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor Base(const Tensor& ids) {
  auto v = ids.to_vector<std::int32_t>();
  for (auto& x : v)
    x = std::min(x, kBase - 1);
  return nn::from_vector(v, ids.shape());
}

} // namespace pointer_generator

/// @brief nmt source-equation components; no optimizer or training driver.
namespace nmt {

namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Outputs: owning source-compatible native type spelling.
using Outputs = std::map<std::string, Tensor>;

/// tflow's ml::SeqBatch (int32 [N, 6] tokens, [N] lengths and ids).
/// @brief Own SeqBatch source graph endpoints or configuration.
///
/// Float32 CPU[B,6,256] source/teacher,Int32[B] lengths in[0,6];D8,H2,FF16.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,256}),l=n::full({1},6,n::DType::Int32);auto y=m::nmt::TransformerNmt(p,x,x,l,l);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct SeqBatch {
  /// @brief input, target, teacher, input_lengths, output_lengths, source_ids: live source graph
  /// endpoint/storage; tensor copies share native storage.
  Tensor input;          ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor target;         ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor teacher;        ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor input_lengths;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor output_lengths; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor source_ids;     ///< Live source endpoint; owning handle shares storage and derivatives.
};

/// @brief Own TransformerNmtPath source graph endpoints or configuration.
///
/// Float32 CPU[B,6,256] source/teacher,Int32[B] lengths in[0,6];D8,H2,FF16.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,256}),l=n::full({1},6,n::DType::Int32);auto y=m::nmt::TransformerNmt(p,x,x,l,l);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct TransformerNmtPath {
  /// @brief logits, memory, hidden, encoder_attention, decoder_attention, cross_attention,
  /// source_embedded, target_embedded: live source graph endpoint/storage; tensor copies share
  /// native storage.
  Tensor logits;            ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor memory;            ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor hidden;            ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor encoder_attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor decoder_attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor cross_attention;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor source_embedded;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor target_embedded;   ///< Live source endpoint; owning handle shares storage and derivatives.
};

/// Float mask [N, T, 1]: 1 where t < lengths[n].
/// @brief Apply LengthMask with the original source equations and parameter names.
///
/// Float32 CPU[B,6,256] source/teacher,Int32[B] lengths in[0,6];D8,H2,FF16.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @param T Explicit source geometry/control value; defaults match the declaration.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,256}),l=n::full({1},6,n::DType::Int32);auto y=m::nmt::TransformerNmt(p,x,x,l,l);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor LengthMask(const Tensor& lengths, std::int64_t T) {
  const auto         l = nn::cast(lengths, nn::DType::Int64).to_vector<std::int64_t>();
  std::vector<float> m;
  for (auto n : l)
    for (std::int64_t t = 0; t < T; ++t)
      m.push_back(t < n ? 1.f : 0.f);
  return nn::from_vector(m, {static_cast<std::int64_t>(l.size()), T, 1});
}

/// `source` / `teacher` are float one-hot [N, 6, 256].
/// @brief Apply TransformerNmt with the original source equations and parameter names.
///
/// Float32 CPU[B,6,256] source/teacher,Int32[B] lengths in[0,6];D8,H2,FF16.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param store Borrowed Float32 CPU parameter store; lazy source-named initialization only.
/// @param source Live CPU source endpoint with the family geometry above.
/// @param teacher Live CPU source endpoint with the family geometry above.
/// @param source_lengths Int32 CPU[B] valid source counts in[0,6].
/// @param target_lengths Int32 CPU[B] valid target counts in[0,6].
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,6,256}),l=n::full({1},6,n::DType::Int32);auto y=m::nmt::TransformerNmt(p,x,x,l,l);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline TransformerNmtPath TransformerNmt(
    nn::ParameterStore& store,
    const Tensor&       source,
    const Tensor&       teacher,
    const Tensor&       source_lengths,
    const Tensor&       target_lengths) {
  constexpr std::int64_t T = 6, D = 8, H = 2, V = 256, F = 16;
  const Tensor           sm = LengthMask(source_lengths, T), tm = LengthMask(target_lengths, T);
  const Tensor           embedding  = store.get("embedding/W", {V, D}, nn::init::normal(.1));
  const Tensor           sinusoidal = nn::sinusoidal_encoding(T, D);
  auto                   embed      = [&](const Tensor& input, const Tensor& mask) {
    return nn::mul(
        nn::add(nn::mul(nn::matmul(input, embedding), std::sqrt(float(D))), sinusoidal), mask);
  };
  const Tensor sx = embed(source, sm), tx = embed(teacher, tm);
  auto         norm = [&](const Tensor& x, const std::string& name, const Tensor& mask) {
    return nn::mul(nn::layer_norm(store, name, x, 1e-5), mask);
  };
  auto ff = [&](const Tensor& x, const std::string& name) {
    const double limit = std::sqrt(6.f / (D + F));
    const Tensor w1    = store.get(name + "/W1", {D, F}, nn::init::uniform(limit)),
                 b1    = store.get(name + "/b1", {F}, nn::init::zeros());
    const Tensor w2    = store.get(name + "/W2", {F, D}, nn::init::uniform(limit)),
                 b2    = store.get(name + "/b2", {D}, nn::init::zeros());
    return nn::bias_add(nn::matmul(nn::relu(nn::bias_add(nn::matmul(x, w1), b1)), w2), b2);
  };
  auto mask = [&](const Tensor& kl, const Tensor& ql, bool causal) {
    return nn::expand_dims(attn::AttentionAllowed(kl, ql, T, T, causal), 1);
  };
  const auto enc = attn::MultiHeadAttention(
      store, "encoder/self", sx, sx, D, H, mask(source_lengths, source_lengths, false));
  Tensor memory  = norm(nn::add(sx, enc.output), "encoder/ln1", sm);
  memory         = norm(nn::add(memory, ff(memory, "encoder/ff")), "encoder/ln2", sm);
  const auto dec = attn::MultiHeadAttention(
      store, "decoder/self", tx, tx, D, H, mask(target_lengths, target_lengths, true));
  Tensor     hidden = norm(nn::add(tx, dec.output), "decoder/ln1", tm);
  const auto cross  = attn::MultiHeadAttention(
      store, "decoder/cross", hidden, memory, D, H, mask(source_lengths, target_lengths, false));
  hidden              = norm(nn::add(hidden, cross.output), "decoder/ln2", tm);
  hidden              = norm(nn::add(hidden, ff(hidden, "decoder/ff")), "decoder/ln3", tm);
  const Tensor hw     = store.get("head/W", {D, V}, nn::init::uniform(std::sqrt(6.f / (D + V))));
  const Tensor hb     = store.get("head/b", {V}, nn::init::zeros());
  const Tensor logits = nn::mul(nn::bias_add(nn::matmul(hidden, hw), hb), tm);
  return {logits, memory, hidden, enc.weights, dec.weights, cross.weights, sx, tx};
}

/// tflow's ml::TransformerNmtModel: token CE averaged over valid target
/// positions, Adam .003, partials of the loss w.r.t. the source / teacher
/// one-hots and the encoder memory, and the causal prefix gradient
/// d sum(logits[:, :2, :7]) / d teacher one-hot.

} // namespace nmt

/// @brief bert source-equation components; no optimizer or training driver.
namespace bert {

namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Batch: owning source-compatible native type spelling.
using Batch = std::map<std::string, Tensor>;

/// @brief Own EncoderPath source graph endpoints or configuration.
///
/// Float32 CPU[B,T,70] encoded tokens;Int32[B,T] segments0/1 and[B] lengths;D8,H2,FF16;MLM
/// context32. Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,32,70}),segments=n::zeros({1,32},n::DType::Int32),lengths=n::full({1},32,n::DType::Int32);auto
/// y=m::bert::BertMlm(p,x,segments,lengths);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct EncoderPath {
  /// @brief hidden, attention, embedded: live source graph endpoint/storage; tensor copies share
  /// native storage.
  Tensor hidden;    ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor embedded;  ///< Live source endpoint; owning handle shares storage and derivatives.
};
/// @brief Own MlmPath source graph endpoints or configuration.
///
/// Float32 CPU[B,T,70] encoded tokens;Int32[B,T] segments0/1 and[B] lengths;D8,H2,FF16;MLM
/// context32. Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,32,70}),segments=n::zeros({1,32},n::DType::Int32),lengths=n::full({1},32,n::DType::Int32);auto
/// y=m::bert::BertMlm(p,x,segments,lengths);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct MlmPath {
  /// @brief logits, hidden, attention, embedded, transformed: live source graph endpoint/storage;
  /// tensor copies share native storage.
  Tensor logits;      ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor hidden;      ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor attention;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor embedded;    ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor transformed; ///< Live source endpoint; owning handle shares storage and derivatives.
};

/// Float mask [N, T, 1]: 1 where t < lengths[n].
/// @brief Apply LengthMask with the original source equations and parameter names.
///
/// Float32 CPU[B,T,70] encoded tokens;Int32[B,T] segments0/1 and[B] lengths;D8,H2,FF16;MLM
/// context32. Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @param T Explicit source geometry/control value; defaults match the declaration.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,32,70}),segments=n::zeros({1,32},n::DType::Int32),lengths=n::full({1},32,n::DType::Int32);auto
/// y=m::bert::BertMlm(p,x,segments,lengths);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor LengthMask(const Tensor& lengths, std::int64_t T) {
  const auto         l = nn::cast(lengths, nn::DType::Int64).to_vector<std::int64_t>();
  std::vector<float> m;
  for (auto n : l)
    for (std::int64_t t = 0; t < T; ++t)
      m.push_back(t < n ? 1.f : 0.f);
  return nn::from_vector(m, {static_cast<std::int64_t>(l.size()), T, 1});
}

/// @brief Apply Linear with the original source equations and parameter names.
///
/// Float32 CPU[B,T,70] encoded tokens;Int32[B,T] segments0/1 and[B] lengths;D8,H2,FF16;MLM
/// context32. Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param store Borrowed Float32 CPU parameter store; lazy source-named initialization only.
/// @param name Explicit source geometry/control value; defaults match the declaration.
/// @param x Live CPU source endpoint with the family geometry above.
/// @param out Explicit source geometry/control value; defaults match the declaration.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,32,70}),segments=n::zeros({1,32},n::DType::Int32),lengths=n::full({1},32,n::DType::Int32);auto
/// y=m::bert::BertMlm(p,x,segments,lengths);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor
Linear(nn::ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t out) {
  nn::DenseOptions o;
  o.init = nn::init::normal(.02);
  return nn::dense(store, name, x, out, o);
}

/// @brief Apply Heads with the original source equations and parameter names.
///
/// Float32 CPU[B,T,70] encoded tokens;Int32[B,T] segments0/1 and[B] lengths;D8,H2,FF16;MLM
/// context32. Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param x Live CPU source endpoint with the family geometry above.
/// @param heads Explicit source geometry/control value; defaults match the declaration.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,32,70}),segments=n::zeros({1,32},n::DType::Int32),lengths=n::full({1},32,n::DType::Int32);auto
/// y=m::bert::BertMlm(p,x,segments,lengths);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor Heads(const Tensor& x, std::int64_t heads) {
  return nn::transpose(
      nn::reshape(x, {x.size(0), x.size(1), heads, x.size(2) / heads}), {0, 2, 1, 3});
}

/// @brief Apply BertEncoder with the original source equations and parameter names.
///
/// Float32 CPU[B,T,70] encoded tokens;Int32[B,T] segments0/1 and[B] lengths;D8,H2,FF16;MLM
/// context32. Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param store Borrowed Float32 CPU parameter store; lazy source-named initialization only.
/// @param encoded Live Float32 CPU encoded tokens with the family context/vocabulary geometry.
/// @param segments Live CPU source endpoint with the family geometry above.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @param T Explicit source geometry/control value; defaults match the declaration.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,32,70}),segments=n::zeros({1,32},n::DType::Int32),lengths=n::full({1},32,n::DType::Int32);auto
/// y=m::bert::BertMlm(p,x,segments,lengths);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline EncoderPath BertEncoder(
    nn::ParameterStore& store,
    const Tensor&       encoded,
    const Tensor&       segments,
    const Tensor&       lengths,
    std::int64_t        T = 32) {
  constexpr std::int64_t V = 70, D = 8, F = 16;
  if (T < 1)
    throw std::invalid_argument("positive BERT context required");
  const Tensor table     = store.get("token/W", {V, D}, nn::init::normal(.02));
  const Tensor positions = store.get("position/W", {T, D}, nn::init::normal(.02));
  const Tensor types     = store.get("segment/W", {2, D}, nn::init::normal(.02));
  const Tensor mask      = LengthMask(lengths, T);
  auto         norm      = [&](const Tensor& x, const std::string& name) {
    return nn::mul(nn::layer_norm(store, name, x, 1e-5), mask);
  };
  const Tensor tokens = nn::matmul(encoded, table);
  const Tensor embedded =
      norm(nn::add(nn::add(tokens, positions), nn::gather(types, segments, 0)), "embedding_ln");
  const Tensor allowed = nn::expand_dims(attn::AttentionAllowed(lengths, lengths, T, T), 1);
  const Tensor q       = Heads(Linear(store, "attention/q", embedded, D), 2),
               k       = Heads(Linear(store, "attention/k", embedded, D), 2),
               v       = Heads(Linear(store, "attention/v", embedded, D), 2);
  const auto   a       = attn::ScaledDotProductAttention(q, k, v, 4, allowed);
  const Tensor mixed = nn::reshape(nn::transpose(a.output, {0, 2, 1, 3}), {encoded.size(0), T, D});
  Tensor hidden = norm(nn::add(embedded, Linear(store, "attention/out", mixed, D)), "attention_ln");
  const Tensor ff = nn::gelu(Linear(store, "ff1", hidden, F), true);
  hidden          = norm(nn::add(hidden, Linear(store, "ff2", ff, D)), "ff_ln");
  return {hidden, a.weights, embedded};
}

/// @brief Apply BertMlm with the original source equations and parameter names.
///
/// Float32 CPU[B,T,70] encoded tokens;Int32[B,T] segments0/1 and[B] lengths;D8,H2,FF16;MLM
/// context32. Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param store Borrowed Float32 CPU parameter store; lazy source-named initialization only.
/// @param encoded Live Float32 CPU encoded tokens with the family context/vocabulary geometry.
/// @param segments Live CPU source endpoint with the family geometry above.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,32,70}),segments=n::zeros({1,32},n::DType::Int32),lengths=n::full({1},32,n::DType::Int32);auto
/// y=m::bert::BertMlm(p,x,segments,lengths);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline MlmPath BertMlm(
    nn::ParameterStore& store,
    const Tensor&       encoded,
    const Tensor&       segments,
    const Tensor&       lengths) {
  constexpr std::int64_t T = 32, D = 8, V = 70;
  const auto             core        = BertEncoder(store, encoded, segments, lengths, T);
  const Tensor           table       = store.at("token/W");
  const Tensor           mask        = LengthMask(lengths, T);
  const Tensor           transformed = nn::mul(
      nn::layer_norm(store, "mlm_ln", nn::gelu(Linear(store, "mlm", core.hidden, D), true), 1e-5),
      mask);
  const Tensor bias = store.get("output/b", {V}, nn::init::zeros());
  const Tensor logits =
      nn::mul(nn::bias_add(nn::matmul(transformed, table, false, true), bias), mask);
  return {logits, core.hidden, core.attention, core.embedded, transformed};
}

/// tflow's ml::BertMlmModel: CE averaged over the selected positions only
/// (max(count, 1) in the denominator), Adam .003.

} // namespace bert

/// @brief compact bert source-equation components; no optimizer or training driver.
namespace compact_bert {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief A: owning source-compatible native type spelling.
using A = bert::Batch;
/// @brief Own Config source graph endpoints or configuration.
///
/// Float32 CPU[B,32,70],Int32 segments[B,32],Int32[B] lengths in[0,32];config controls embedding
/// width,depth,sharing. Caller owns fitting, objectives and generation. Graph tensors retain
/// storage and live derivatives; lazy parameter creation mutates only borrowed stores. Synchronize
/// calls against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::compact_bert::Forward(p,n::zeros({1,32,70}),n::zeros({1,32},n::DType::Int32),n::full({1},32,n::DType::Int32));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Config {
  /// @brief embedding, layers: live source graph endpoint/storage; tensor copies share native
  /// storage.
  int embedding = 8; ///< Owned source configuration/control value.
  int layers    = 2; ///< Owned source configuration/control value.

  /// @brief shared, segments: live source graph endpoint/storage; tensor copies share native
  /// storage.
  bool shared   = false; ///< Owned source configuration/control value.
  bool segments = true;  ///< Owned source configuration/control value.
};
/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,32,70],Int32 segments[B,32],Int32[B] lengths in[0,32];config controls embedding
/// width,depth,sharing. Caller owns fitting, objectives and generation. Graph tensors retain
/// storage and live derivatives; lazy parameter creation mutates only borrowed stores. Synchronize
/// calls against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::compact_bert::Forward(p,n::zeros({1,32,70}),n::zeros({1,32},n::DType::Int32),n::full({1},32,n::DType::Int32));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief logits, hidden: live source graph endpoint/storage; tensor copies share native storage.
  Tensor logits; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor hidden; ///< Live source endpoint; owning handle shares storage and derivatives.

  /// @brief states, attention: live source graph endpoint/storage; tensor copies share native
  /// storage.
  std::vector<Tensor>
      states; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      attention; ///< Live source endpoint; owning handle shares storage and derivatives.
};
/// @brief Apply Forward with the original source equations and parameter names.
///
/// Float32 CPU[B,32,70],Int32 segments[B,32],Int32[B] lengths in[0,32];config controls embedding
/// width,depth,sharing. Caller owns fitting, objectives and generation. Graph tensors retain
/// storage and live derivatives; lazy parameter creation mutates only borrowed stores. Synchronize
/// calls against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param encoded Live Float32 CPU encoded tokens with the family context/vocabulary geometry.
/// @param segments Live CPU source endpoint with the family geometry above.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @param c Explicit source geometry/control value; defaults match the declaration.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::compact_bert::Forward(p,n::zeros({1,32,70}),n::zeros({1,32},n::DType::Int32),n::full({1},32,n::DType::Int32));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path Forward(
    nn::ParameterStore& p,
    const Tensor&       encoded,
    const Tensor&       segments,
    const Tensor&       lengths,
    Config              c = {}) {
  constexpr int T = 32, D = 8, V = 70;
  int           E = c.embedding;
  component_detail::Require(E > 0 && c.layers > 0, "positive compact embedding/depth");
  auto mask   = bert::LengthMask(lengths, T);
  auto linear = [&](const Tensor& x, const std::string& name, int out) {
    return bert::Linear(p, name, x, out);
  };
  auto norm = [&](const Tensor& x, const std::string& name) {
    return nn::layer_norm(p, name, x, 1e-5) * mask;
  };
  auto table    = p.get("token/W", {V, E}, nn::init::normal(.02)),
       position = p.get("position/W", {T, E}, nn::init::normal(.02));
  auto x        = nn::matmul(encoded, table) + position;
  if (c.segments)
    x = x + nn::gather(p.get("segment/W", {2, E}, nn::init::normal(.02)), segments, 0);
  x = norm(x, "embedding_ln");
  if (E != D)
    x = linear(x, "embedding_projection", D) * mask;
  Path path;
  path.states.push_back(x);
  auto allowed = nn::expand_dims(attn::AttentionAllowed(lengths, lengths, T, T), 1);
  for (int l = 0; l < c.layers; ++l) {
    auto name  = c.shared ? std::string("shared") : "block" + std::to_string(l);
    auto q     = bert::Heads(linear(x, name + "/q", D), 2),
         k     = bert::Heads(linear(x, name + "/k", D), 2),
         v     = bert::Heads(linear(x, name + "/v", D), 2);
    auto a     = attn::ScaledDotProductAttention(q, k, v, 4, allowed);
    auto mixed = nn::reshape(nn::transpose(a.output, {0, 2, 1, 3}), {encoded.size(0), T, D});
    x          = norm(x + linear(mixed, name + "/out", D), name + "/attention_ln");
    x          = norm(
        x + linear(nn::gelu(linear(x, name + "/ff1", 16), true), name + "/ff2", D),
        name + "/ff_ln");
    path.states.push_back(x);
    path.attention.push_back(a.weights);
  }
  path.hidden      = x;
  auto transformed = norm(nn::gelu(linear(x, "mlm", E), true), "mlm_ln");
  path.logits =
      (nn::matmul(transformed, table, false, true) + p.get("output/b", {V}, nn::init::zeros())) *
      mask;
  return path;
}

} // namespace compact_bert

/// @brief peft source-equation components; no optimizer or training driver.
namespace peft {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief A: owning source-compatible native type spelling.
using A = bert::Batch;
/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,128,70],Int32[B] lengths in[0,128];base store is separately borrowed from task
/// store;rank2 alpha4 LoRA or width2 adapters. Caller owns fitting, objectives and generation.
/// Graph tensors retain storage and live derivatives; lazy parameter creation mutates only borrowed
/// stores. Synchronize calls against concurrent store mutation. Copies of path fields share
/// storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);n::ParameterStore task(8);auto
/// y=m::peft::Forward(p,task,n::zeros({1,128,70}),n::full({1},128,n::DType::Int32),"lora");
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief logits, hidden, attention, embedded: live source graph endpoint/storage; tensor copies
  /// share native storage.
  Tensor logits;    ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor hidden;    ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor embedded;  ///< Live source endpoint; owning handle shares storage and derivatives.

  /// @brief updates: live source graph endpoint/storage; tensor copies share native storage.
  std::vector<Tensor> updates;
};
/// @brief Apply Forward with the original source equations and parameter names.
///
/// Float32 CPU[B,128,70],Int32[B] lengths in[0,128];base store is separately borrowed from task
/// store;rank2 alpha4 LoRA or width2 adapters. Caller owns fitting, objectives and generation.
/// Graph tensors retain storage and live derivatives; lazy parameter creation mutates only borrowed
/// stores. Synchronize calls against concurrent store mutation. Copies of path fields share
/// storage.
/// @param base Borrowed pretrained base store, independent from task storage; caller freezes it
/// during fitting.
/// @param task Borrowed trainable adaptation store, separate from base.
/// @param encoded Live Float32 CPU encoded tokens with the family context/vocabulary geometry.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @param mode One of lora,adapter,head; separate task-store adaptation path.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);n::ParameterStore task(8);auto
/// y=m::peft::Forward(p,task,n::zeros({1,128,70}),n::full({1},128,n::DType::Int32),"lora");
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path Forward(
    nn::ParameterStore& base,
    nn::ParameterStore& task,
    const Tensor&       encoded,
    const Tensor&       lengths,
    const std::string&  mode) {
  component_detail::Require(
      mode == "lora" || mode == "adapter" || mode == "head", "lora,adapter,head modes");
  constexpr int T = 128, D = 8;
  auto          mask = bert::LengthMask(lengths, T);
  Path          result;
  auto          linear = [&](const Tensor& x, const std::string& name, int out) {
    auto value = bert::Linear(base, name, x, out);
    if (mode == "lora" && (name == "attention/q" || name == "attention/v")) {
      auto prefix = "lora/" + name;
      auto a      = task.get(prefix + "/A", {x.size(2), 2}, nn::init::normal(.02)),
           b      = task.get(prefix + "/B", {2, out}, nn::init::zeros());
      auto update = nn::matmul(nn::matmul(x, a), b) * 2.;
      result.updates.push_back(update);
      value = value + update;
    }
    return value;
  };
  auto norm = [&](const Tensor& x, const std::string& name) {
    return nn::layer_norm(base, name, x, 1e-5) * mask;
  };
  auto adapter = [&](const Tensor& x, const std::string& name) {
    if (mode != "adapter")
      return x;
    auto prefix = "adapter/" + name;
    auto down   = task.get(prefix + "/down/W", {D, 2}, nn::init::normal(.02)),
         db     = task.get(prefix + "/down/b", {2}, nn::init::zeros()),
         up     = task.get(prefix + "/up/W", {2, D}, nn::init::zeros()),
         ub     = task.get(prefix + "/up/b", {D}, nn::init::zeros());
    auto update = nn::matmul(nn::gelu(nn::matmul(x, down) + db, true), up) + ub;
    result.updates.push_back(update);
    return x + update;
  };
  auto token    = base.get("token/W", {70, D}, nn::init::normal(.02)),
       position = base.get("position/W", {T, D}, nn::init::normal(.02)),
       segment  = base.get("segment/W", {2, D}, nn::init::normal(.02));
  auto embedded = norm(
      nn::matmul(encoded, token) + position + nn::squeeze(nn::slice(segment, {0, 0}, {1, D}), {0}),
      "embedding_ln");
  auto q        = bert::Heads(linear(embedded, "attention/q", D), 2),
       k        = bert::Heads(linear(embedded, "attention/k", D), 2),
       v        = bert::Heads(linear(embedded, "attention/v", D), 2);
  auto attended = attn::ScaledDotProductAttention(
      q, k, v, 4, nn::expand_dims(attn::AttentionAllowed(lengths, lengths, T, T), 1));
  auto mixed = nn::reshape(nn::transpose(attended.output, {0, 2, 1, 3}), {encoded.size(0), T, D});
  auto h = norm(embedded + adapter(linear(mixed, "attention/out", D), "attention"), "attention_ln");
  auto ff          = linear(nn::gelu(linear(h, "ff1", 16), true), "ff2", D);
  h                = norm(h + adapter(ff, "ff"), "ff_ln");
  auto cls         = nn::squeeze(nn::slice(h, {0, 0, 0}, {encoded.size(0), 1, D}), {1});
  result.logits    = bert::Linear(task, "head", cls, 2);
  result.hidden    = h;
  result.attention = attended.weights;
  result.embedded  = embedded;
  return result;
}

} // namespace peft

/// @brief t5 source-equation components; no optimizer or training driver.
namespace t5 {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief A: owning source-compatible native type spelling.
using A = bert::Batch;
/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,48,72] source/teacher;Int32[B] source/target lengths in[0,48];D8,H2,shared token
/// table,relative buckets32. Caller owns fitting, objectives and generation. Graph tensors retain
/// storage and live derivatives; lazy parameter creation mutates only borrowed stores. Synchronize
/// calls against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,48,72}),l=n::full({1},48,n::DType::Int32);auto y=m::t5::Forward(p,x,x,l,l);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief logits, memory, hidden, encoder_attention, decoder_attention, cross_attention: live
  /// source graph endpoint/storage; tensor copies share native storage.
  Tensor logits;            ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor memory;            ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor hidden;            ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor encoder_attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor decoder_attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor cross_attention;   ///< Live source endpoint; owning handle shares storage and derivatives.
};
/// @brief Apply Forward with the original source equations and parameter names.
///
/// Float32 CPU[B,48,72] source/teacher;Int32[B] source/target lengths in[0,48];D8,H2,shared token
/// table,relative buckets32. Caller owns fitting, objectives and generation. Graph tensors retain
/// storage and live derivatives; lazy parameter creation mutates only borrowed stores. Synchronize
/// calls against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param source Live CPU source endpoint with the family geometry above.
/// @param teacher Live CPU source endpoint with the family geometry above.
/// @param sl Int32 CPU[B] valid source counts in[0,48].
/// @param tl Int32 CPU[B] valid target counts in[0,48].
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// x=n::zeros({1,48,72}),l=n::full({1},48,n::DType::Int32);auto y=m::t5::Forward(p,x,x,l,l);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path Forward(
    nn::ParameterStore& p,
    const Tensor&       source,
    const Tensor&       teacher,
    const Tensor&       sl,
    const Tensor&       tl) {
  constexpr int T = 48, D = 8;
  auto table = p.get("token/W", {72, D}, nn::init::normal(.2)), sm = bert::LengthMask(sl, T),
       tm     = bert::LengthMask(tl, T);
  auto linear = [&](const Tensor& x, const std::string& name, int out) {
    return nn::matmul(x, p.get(name, {x.size(2), out}, nn::init::normal(.1)));
  };
  auto norm = [&](const Tensor& x, const std::string& name, const Tensor& mask) {
    return nn::rms_norm(p, name, x, 1e-6) * mask;
  };
  auto relative = [&](const std::string& name, bool bidir) {
    return positional::RelativePositionBias(
        p.get(name, {32, 2}, nn::init::normal(.02)), T, T, bidir, 32, 128);
  };
  auto attention = [&](const Tensor&      x,
                       const Tensor&      kv,
                       const std::string& name,
                       const Tensor&      kl,
                       const Tensor&      ql,
                       bool               causal,
                       const Tensor&      bias) {
    auto q      = bert::Heads(linear(x, name + "/Wq", D), 2),
         k      = bert::Heads(linear(kv, name + "/Wk", D), 2),
         v      = bert::Heads(linear(kv, name + "/Wv", D), 2);
    auto logits = nn::matmul(q, k, false, true);
    if (bias.defined())
      logits = logits + bias;
    auto allowed     = nn::expand_dims(attn::AttentionAllowed(kl, ql, T, T, causal), 1);
    auto probability = nn::softmax(nn::where(allowed, logits, nn::full(logits.shape(), -1e9)), -1) *
                       nn::cast(allowed, nn::DType::Float32);
    auto merged = nn::reshape(
        nn::transpose(nn::matmul(probability, v), {0, 2, 1, 3}), {source.size(0), T, D});
    return std::pair<Tensor, Tensor>{linear(merged, name + "/Wo", D), probability};
  };
  auto ff = [&](const Tensor& x, const std::string& name) {
    return linear(nn::relu(linear(x, name + "/Wi", 16)), name + "/Wo", D);
  };
  auto memory     = nn::matmul(source, table) * sm;
  auto normalized = norm(memory, "encoder/self_norm", sm);
  auto enc        = attention(
      normalized, normalized, "encoder/self", sl, sl, false, relative("encoder/relative", true));
  memory      = (memory + enc.first) * sm;
  memory      = (memory + ff(norm(memory, "encoder/ff_norm", sm), "encoder/ff")) * sm;
  memory      = norm(memory, "encoder/final_norm", sm);
  auto hidden = nn::matmul(teacher, table) * tm;
  auto dn     = norm(hidden, "decoder/self_norm", tm);
  auto dec = attention(dn, dn, "decoder/self", tl, tl, true, relative("decoder/relative", false));
  hidden   = (hidden + dec.first) * tm;
  auto cross =
      attention(norm(hidden, "decoder/cross_norm", tm), memory, "decoder/cross", sl, tl, false, {});
  hidden      = (hidden + cross.first) * tm;
  hidden      = (hidden + ff(norm(hidden, "decoder/ff_norm", tm), "decoder/ff")) * tm;
  hidden      = norm(hidden, "decoder/final_norm", tm);
  auto logits = nn::matmul(hidden * float(1 / std::sqrt(float(D))), table, false, true) * tm;
  return {logits, memory, hidden, enc.second, dec.second, cross.second};
}

} // namespace t5

/// @brief xl source-equation components; no optimizer or training driver.
namespace xl {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief A: owning source-compatible native type spelling.
using A = bert::Batch;
/// @brief Memory: owning source-compatible native type spelling.
using Memory = std::vector<Tensor>;
/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,T,65] and two[B,M,8] memory endpoints;D8,H2,two blocks;memory is explicitly
/// detached. Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);m::xl::Memory
/// memory(2,n::zeros({1,0,8}));auto y=m::xl::Forward(p,n::zeros({1,16,65}),memory);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief logits: live source graph endpoint/storage; tensor copies share native storage.
  Tensor logits;
  /// @brief hids, attention: live source graph endpoint/storage; tensor copies share native
  /// storage.
  std::vector<Tensor> hids; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      attention; ///< Live source endpoint; owning handle shares storage and derivatives.
};
/// @brief Apply Forward with the original source equations and parameter names.
///
/// Float32 CPU[B,T,65] and two[B,M,8] memory endpoints;D8,H2,two blocks;memory is explicitly
/// detached. Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param encoded Live Float32 CPU encoded tokens with the family context/vocabulary geometry.
/// @param memory Live source memory endpoints; segment-recurring forwards detach explicit past
/// states.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);m::xl::Memory
/// memory(2,n::zeros({1,0,8}));auto y=m::xl::Forward(p,n::zeros({1,16,65}),memory);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path Forward(nn::ParameterStore& p, const Tensor& encoded, const Memory& memory) {
  int  b = encoded.size(0), qt = encoded.size(1), mt = memory[0].size(1), kt = mt + qt;
  auto linear = [&](const Tensor& x, const std::string& name, int out) {
    return nn::matmul(x, p.get(name, {x.size(2), out}, nn::init::normal(.15)));
  };
  std::vector<float> rel(qt * kt * 8);
  std::vector<bool>  permitted(qt * kt);
  for (int i = 0; i < qt; ++i)
    for (int j = 0; j < kt; ++j) {
      permitted[i * kt + j] = j <= mt + i;
      for (int d = 0; d < 4; ++d) {
        float angle                   = float(mt + i - j) * std::pow(10000.f, -2.f * d / 8);
        rel[(i * kt + j) * 8 + d]     = std::sin(angle);
        rel[(i * kt + j) * 8 + d + 4] = std::cos(angle);
      }
    }
  auto relative = nn::from_vector(rel, {qt, kt, 8}), allowed = nn::from_vector(permitted, {qt, kt});
  auto u = p.get("relative/u", {2, 1, 4}, nn::init::normal(.02)),
       v = p.get("relative/v", {2, 1, 4}, nn::init::normal(.02));
  auto x = linear(encoded, "token/W", 8) * std::sqrt(8.f);
  Path result;
  for (int l = 0; l < 2; ++l) {
    auto name = "layer" + std::to_string(l);
    result.hids.push_back(x);
    auto cat   = nn::concat({memory[l].detach(), x}, 1);
    auto heads = [&](const Tensor& z, int t) {
      return nn::transpose(nn::reshape(z, {b, t, 2, 4}), {0, 2, 1, 3});
    };
    auto q = heads(linear(x, name + "/Wq", 8), qt), k = heads(linear(cat, name + "/Wk", 8), kt),
         value = heads(linear(cat, name + "/Wv", 8), kt);
    auto r =
        nn::transpose(nn::reshape(linear(relative, name + "/Wr", 8), {qt, kt, 2, 4}), {2, 0, 1, 3});
    auto content  = nn::matmul(q + u, k, false, true),
         position = nn::sum(nn::expand_dims(q + v, 3) * r, {-1});
    auto scores   = (content + position) * .5;
    auto weights  = nn::softmax(nn::where(allowed, scores, nn::full(scores.shape(), -1e9)), -1);
    result.attention.push_back(weights);
    auto mixed = nn::reshape(nn::transpose(nn::matmul(weights, value), {0, 2, 1, 3}), {b, qt, 8});
    x          = nn::layer_norm(p, name + "/ln1", x + linear(mixed, name + "/Wo", 8), 1e-5);
    auto w1 = linear(x, name + "/W1", 16), b1 = p.get(name + "/b1", {16}, nn::init::zeros());
    auto ff = linear(nn::relu(w1 + b1), name + "/W2", 8),
         b2 = p.get(name + "/b2", {8}, nn::init::zeros());
    x       = nn::layer_norm(p, name + "/ln2", x + ff + b2, 1e-5);
  }
  auto bias     = p.get("head/b", {65}, nn::init::zeros());
  result.logits = linear(x, "head/W", 65) + bias;
  return result;
}

} // namespace xl

/// @brief xlnet source-equation components; no optimizer or training driver.
namespace xlnet {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief A: owning source-compatible native type spelling.
using A = bert::Batch;
/// @brief Memory: owning source-compatible native type spelling.
using Memory = std::vector<Tensor>;
/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,16,65],Int32[B,16] permutation ranks, two[B,M,8] memory endpoints;content/query
/// streams share parameters. Caller owns fitting, objectives and generation. Graph tensors retain
/// storage and live derivatives; lazy parameter creation mutates only borrowed stores. Synchronize
/// calls against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);m::xlnet::Memory
/// memory(2,n::zeros({1,0,8}));auto rank=n::reshape(n::arange(0,16,1,n::DType::Int32),{1,16});auto
/// y=m::xlnet::Forward(p,n::zeros({1,16,65}),rank,memory);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief logits: live source graph endpoint/storage; tensor copies share native storage.
  Tensor logits;
  /// @brief content, query: live source graph endpoint/storage; tensor copies share native storage.
  Tensor content; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor query;   ///< Live source endpoint; owning handle shares storage and derivatives.

  /// @brief hids, content_attention, query_attention: live source graph endpoint/storage; tensor
  /// copies share native storage.
  std::vector<Tensor> hids; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      content_attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      query_attention; ///< Live source endpoint; owning handle shares storage and derivatives.
};
inline Path
/// @brief Apply Forward with the original source equations and parameter names.
///
/// Float32 CPU[B,16,65],Int32[B,16] permutation ranks, two[B,M,8] memory endpoints;content/query
/// streams share parameters. Caller owns fitting, objectives and generation. Graph tensors retain
/// storage and live derivatives; lazy parameter creation mutates only borrowed stores. Synchronize
/// calls against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param encoded Live Float32 CPU encoded tokens with the family context/vocabulary geometry.
/// @param rank Int32 CPU[B,16] factorization ranks0..15, physically ordered by token position.
/// @param memory Live source memory endpoints; segment-recurring forwards detach explicit past
/// states.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);m::xlnet::Memory
/// memory(2,n::zeros({1,0,8}));auto rank=n::reshape(n::arange(0,16,1,n::DType::Int32),{1,16});auto
/// y=m::xlnet::Forward(p,n::zeros({1,16,65}),rank,memory);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
Forward(nn::ParameterStore& p, const Tensor& encoded, const Tensor& rank, const Memory& memory) {
  int  b = encoded.size(0), qt = encoded.size(1), mt = memory[0].size(1), kt = mt + qt;
  auto linear = [&](const Tensor& x, const std::string& name, int out) {
    return nn::matmul(x, p.get(name, {x.size(2), out}, nn::init::normal(.15)));
  };
  std::vector<float> rel(qt * kt * 8);
  for (int i = 0; i < qt; ++i)
    for (int j = 0; j < kt; ++j) {
      for (int d = 0; d < 4; ++d) {
        float angle                   = float(mt + i - j) * std::pow(10000.f, -2.f * d / 8);
        rel[(i * kt + j) * 8 + d]     = std::sin(angle);
        rel[(i * kt + j) * 8 + d + 4] = std::cos(angle);
      }
    }
  auto relative = nn::from_vector(rel, {qt, kt, 8});
  auto u        = p.get("relative/u", {2, 1, 4}, nn::init::normal(.02)),
       v        = p.get("relative/v", {2, 1, 4}, nn::init::normal(.02));
  auto table    = p.get("token/W", {65, 8}, nn::init::normal(.2));
  auto h        = nn::matmul(encoded, table);
  auto g        = nn::zeros_like(h) + p.get("query/seed", {8}, nn::init::normal(.2));
  // Both streams share the same token table, relative biases and layer weights.
  Path              result;
  std::vector<bool> cm(b * 16 * kt), qm(cm.size());
  auto              ranks = rank.to_vector<int>();
  for (int row = 0; row < b; ++row)
    for (int i = 0; i < 16; ++i)
      for (int j = 0; j < kt; ++j) {
        int index = (row * 16 + i) * kt + j;
        cm[index] = j < mt || ranks[row * 16 + j - mt] <= ranks[row * 16 + i];
        qm[index] = j < mt || ranks[row * 16 + j - mt] < ranks[row * 16 + i];
      }
  auto content_allowed = nn::from_vector(cm, {b, 1, 16, kt}),
       query_allowed   = nn::from_vector(qm, {b, 1, 16, kt});
  for (int l = 0; l < 2; ++l) {
    auto name = "layer" + std::to_string(l);
    result.hids.push_back(h);
    auto cat   = nn::concat({memory[l].detach(), h}, 1);
    auto heads = [&](const Tensor& z, int t) {
      return nn::transpose(nn::reshape(z, {b, t, 2, 4}), {0, 2, 1, 3});
    };
    auto k     = heads(linear(cat, name + "/Wk", 8), kt),
         value = heads(linear(cat, name + "/Wv", 8), kt);
    auto r =
        nn::transpose(nn::reshape(linear(relative, name + "/Wr", 8), {16, kt, 2, 4}), {2, 0, 1, 3});
    auto stream = [&](const Tensor& x, const Tensor& mask, std::vector<Tensor>& trace) {
      auto q        = heads(linear(x, name + "/Wq", 8), 16);
      auto content  = nn::matmul(q + u, k, false, true),
           position = nn::sum(nn::expand_dims(q + v, 3) * r, {-1});
      auto scores   = (content + position) * .5;
      auto raw      = nn::softmax(nn::where(mask, scores, nn::full(scores.shape(), -1e9)), -1) *
                 nn::cast(mask, nn::DType::Float32);
      auto a = raw / nn::maximum(nn::sum(raw, {-1}, true), nn::scalar(1));
      trace.push_back(a);
      auto mixed = nn::reshape(nn::transpose(nn::matmul(a, value), {0, 2, 1, 3}), {b, 16, 8});
      auto z     = nn::layer_norm(p, name + "/ln1", x + linear(mixed, name + "/Wo", 8), 1e-5);
      auto b1    = p.get(name + "/b1", {16}, nn::init::zeros());
      auto ff    = linear(nn::relu(linear(z, name + "/W1", 16) + b1), name + "/W2", 8);
      auto b2    = p.get(name + "/b2", {8}, nn::init::zeros());
      return nn::layer_norm(p, name + "/ln2", z + ff + b2, 1e-5);
    };
    auto next_h = stream(h, content_allowed, result.content_attention);
    auto next_g = stream(g, query_allowed, result.query_attention);
    h           = next_h;
    g           = next_g;
  }
  result.logits  = nn::matmul(g, table, false, true) + p.get("head/b", {65}, nn::init::zeros());
  result.content = h;
  result.query   = g;
  return result;
}

} // namespace xlnet

/// @brief switch lm source-equation components; no optimizer or training driver.
namespace switch_lm {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief A: owning source-compatible native type spelling.
using A = bert::Batch;
/// @brief Own Moe source graph endpoints or configuration.
///
/// Float32 CPU[B,16,65],Int32[B] lengths in[0,16];four experts,D8;capacity is per sequence.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::switch_lm::Forward(p,n::zeros({1,16,65}),n::full({1},16,n::DType::Int32));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Moe {
  /// @brief fields: live source graph endpoint/storage; tensor copies share native storage.
  A fields;
  /// @brief indices, dispatched, expert_outputs: live source graph endpoint/storage; tensor copies
  /// share native storage.
  std::vector<Tensor>
      indices; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      dispatched; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      expert_outputs; ///< Live source endpoint; owning handle shares storage and derivatives.
};
inline Moe
/// @brief Apply Switch with the original source equations and parameter names.
///
/// Float32 CPU[B,16,65],Int32[B] lengths in[0,16];four experts,D8;capacity is per sequence.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param input Live CPU source endpoint with the family geometry above.
/// @param valid Boolean CPU[B,T] route eligibility, padding excluded from capacity and balance.
/// @param capacity Explicit source geometry/control value; defaults match the declaration.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::switch_lm::Forward(p,n::zeros({1,16,65}),n::full({1},16,n::DType::Int32));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
Switch(nn::ParameterStore& p, const Tensor& input, const Tensor& valid, int capacity = -1) {
  int  n = input.size(0), t = input.size(1), total = n * t;
  auto flat          = nn::reshape(input, {total, 8}),
       rw            = p.get("switch/router/W", {8, 4}, nn::init::normal(.15));
  auto mask          = nn::cast(valid, nn::DType::Float32);
  auto probabilities = nn::softmax(nn::matmul(input, rw), -1) * nn::expand_dims(mask, 2);
  auto chosen        = nn::cast(nn::argmax(probabilities, 2), nn::DType::Int32);
  auto choice        = chosen.to_vector<int>();
  auto mv            = mask.to_vector<float>();
  std::vector<float> assigned(total * 4), accepted(total * 4), slots(total * 4), counts(n * 4),
      used(n * 4), length(n), active(n);
  std::vector<int> caps(n);
  for (int b = 0; b < n; ++b) {
    for (int j = 0; j < t; ++j)
      length[b] += mv[b * t + j];
    active[b] = length[b] > 0;
    caps[b]   = capacity < 0 ? int(std::ceil(length[b] * 1.25f / 4)) : capacity;
    for (int j = 0; j < t; ++j) {
      int i = b * t + j;
      for (int e = 0; e < 4; ++e)
        slots[i * 4 + e] = counts[b * 4 + e];
      if (mv[i]) {
        int e               = choice[i];
        assigned[i * 4 + e] = 1;
        accepted[i * 4 + e] = counts[b * 4 + e] < caps[b];
        counts[b * 4 + e] += 1;
        used[b * 4 + e] += accepted[i * 4 + e];
      }
    }
  }
  auto accept = nn::from_vector(accepted, {n, t, 4}), gate = nn::sum(probabilities * accept, {2}),
       count = nn::from_vector(counts, {n, 4});
  auto denom = nn::expand_dims(nn::maximum(nn::from_vector(length, {n}), nn::scalar(1)), 1);
  auto f = count / denom, prob = nn::sum(probabilities, {1}) / denom;
  auto activity  = nn::from_vector(active, {n});
  auto auxiliary = nn::sum(nn::sum(f * prob, {1}) * activity) /
                   nn::maximum(nn::sum(activity), nn::scalar(1)) * 4.;
  Moe    result;
  Tensor combined = nn::zeros({total, 8});
  for (int e = 0; e < 4; ++e) {
    std::vector<std::int64_t> ids;
    std::vector<int>          inverse(total);
    for (int i = 0; i < total; ++i)
      if (accepted[i * 4 + e]) {
        ids.push_back(i);
        inverse[i] = ids.size();
      }
    auto list  = nn::from_vector(ids, {static_cast<std::int64_t>(ids.size())});
    auto batch = nn::gather(flat, list, 0);
    auto name  = "switch/expert" + std::to_string(e);
    auto w1    = p.get(name + "/W1", {8, 16}, nn::init::normal(.15)),
         b1    = p.get(name + "/b1", {16}, nn::init::zeros()),
         w2    = p.get(name + "/W2", {16, 8}, nn::init::normal(.15)),
         b2    = p.get(name + "/b2", {8}, nn::init::zeros());
    auto out   = nn::matmul(nn::relu(nn::matmul(batch, w1) + b1), w2) + b2;
    result.indices.push_back(list);
    result.dispatched.push_back(batch);
    result.expert_outputs.push_back(out);
    combined =
        combined +
        nn::gather(nn::concat({nn::zeros({1, 8}), out}, 0), nn::from_vector(inverse, {total}), 0);
  }
  result.fields = {
      {"output", nn::reshape(combined, {n, t, 8}) * nn::expand_dims(gate, 2)},
      {"probabilities", probabilities},
      {"chosen", chosen},
      {"assigned", nn::from_vector(assigned, {n, t, 4})},
      {"accepted", accept},
      {"slot", nn::from_vector(slots, {n, t, 4})},
      {"gate", gate},
      {"capacity", nn::from_vector(caps, {n})},
      {"counts", count},
      {"used", nn::from_vector(used, {n, 4})},
      {"auxiliary", auxiliary}};
  return result;
}
/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,16,65],Int32[B] lengths in[0,16];four experts,D8;capacity is per sequence.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::switch_lm::Forward(p,n::zeros({1,16,65}),n::full({1},16,n::DType::Int32));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief logits, hidden, attention, embedded, moe_input: live source graph endpoint/storage;
  /// tensor copies share native storage.
  Tensor logits;    ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor hidden;    ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor embedded;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor moe_input; ///< Live source endpoint; owning handle shares storage and derivatives.

  /// @brief moe: live source graph endpoint/storage; tensor copies share native storage.
  Moe moe;
};
/// @brief Apply Forward with the original source equations and parameter names.
///
/// Float32 CPU[B,16,65],Int32[B] lengths in[0,16];four experts,D8;capacity is per sequence.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param encoded Live Float32 CPU encoded tokens with the family context/vocabulary geometry.
/// @param lengths Int32 CPU[B] valid sequence counts within the source context.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::switch_lm::Forward(p,n::zeros({1,16,65}),n::full({1},16,n::DType::Int32));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path Forward(nn::ParameterStore& p, const Tensor& encoded, const Tensor& lengths) {
  auto mask = bert::LengthMask(lengths, 16), valid = nn::greater(nn::squeeze(mask, {2}), 0);
  auto table     = p.get("token/W", {65, 8}, nn::init::normal(.15)),
       positions = p.get("position/W", {16, 8}, nn::init::normal(.02));
  auto x         = (nn::matmul(encoded, table) + positions) * mask;
  auto norm      = [&](const Tensor& z, const std::string& name) {
    return nn::layer_norm(p, name, z, 1e-5) * mask;
  };
  auto first = norm(x, "ln1");
  auto a     = attn::MultiHeadAttention(
      p,
      "attention",
      first,
      first,
      8,
      2,
      nn::expand_dims(attn::AttentionAllowed(lengths, lengths, 16, 16, true), 1));
  auto h      = (x + a.output) * mask;
  auto mi     = norm(h, "ln2");
  auto moe    = Switch(p, mi, valid, 5);
  h           = norm(h + moe.fields.at("output"), "ln_final");
  auto w      = p.get("head/W", {8, 65}, nn::init::normal(.15)),
       b      = p.get("head/b", {65}, nn::init::zeros());
  auto logits = (nn::matmul(h, w) + b) * mask;
  return {logits, h, a.weights, x, mi, moe};
}

} // namespace switch_lm

/// @brief vit source-equation components; no optimizer or training driver.
namespace vit {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief A: owning source-compatible native type spelling.
using A = std::map<std::string, Tensor>;
/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::vit::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief logits, patches, embedded, hidden: live source graph endpoint/storage; tensor copies
  /// share native storage.
  Tensor logits;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor patches;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor embedded; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor hidden;   ///< Live source endpoint; owning handle shares storage and derivatives.

  /// @brief attention, states: live source graph endpoint/storage; tensor copies share native
  /// storage.
  std::vector<Tensor>
      attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      states; ///< Live source endpoint; owning handle shares storage and derivatives.
};
/// @brief Apply Forward with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param images Live Float32 CPU[B,32,32,3] source-scaled RGB.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::vit::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path Forward(nn::ParameterStore& p, const Tensor& images) {
  constexpr int D = 8, T = 17;
  const auto    n = images.size(0);
  Path          out;
  out.patches = nn::reshape(
      nn::transpose(nn::reshape(images, {n, 4, 8, 4, 8, 3}), {0, 1, 3, 2, 4, 5}), {n, 16, 192});
  nn::DenseOptions dense;
  dense.init     = nn::init::normal(.1);
  auto projected = nn::dense(p, "patch", out.patches, D, dense);
  auto cls       = p.get("cls", {1, 1, D}, nn::init::zeros());
  auto position  = p.get("position", {1, T, D}, nn::init::normal(.02));
  auto h         = nn::add(nn::concat({nn::tile(cls, {n, 1, 1}), projected}, 1), position);
  out.embedded   = h;
  auto linear    = [&](const Tensor& x, const std::string& name, int width) {
    return nn::dense(p, name, x, width, dense);
  };
  for (int l = 0; l < 2; ++l) {
    auto name  = "block" + std::to_string(l);
    auto x     = nn::layer_norm(p, name + "/ln1", h, 1e-6);
    auto split = nn::split(linear(x, name + "/qkv", 3 * D), 3, 2);
    auto heads = [&](const Tensor& z) {
      return nn::transpose(nn::reshape(z, {n, T, 2, 4}), {0, 2, 1, 3});
    };
    auto q = heads(split[0]), k = heads(split[1]), v = heads(split[2]);
    auto attention = nn::softmax(nn::matmul(q, k, false, true) * .5, -1);
    out.attention.push_back(attention);
    auto mixed = nn::reshape(nn::transpose(nn::matmul(attention, v), {0, 2, 1, 3}), {n, T, D});
    h          = nn::add(h, linear(mixed, name + "/out", D));
    auto ff    = nn::layer_norm(p, name + "/ln2", h, 1e-6);
    ff         = linear(nn::gelu(linear(ff, name + "/ff1", 16), true), name + "/ff2", D);
    h          = nn::add(h, ff);
    out.states.push_back(h);
  }
  out.hidden    = nn::layer_norm(p, "final_ln", h, 1e-6);
  auto features = nn::squeeze(nn::slice(out.hidden, {0, 0, 0}, {n, 1, D}), {1});
  out.logits    = nn::dense(p, "head", features, 10, dense);
  return out;
}

} // namespace vit

/// @brief mae source-equation components; no optimizer or training driver.
namespace mae {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief A: owning source-compatible native type spelling.
using vit::A;
/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// masked=m::mae::Masked({{"images",n::zeros({1,32,32,3})}},3);auto
/// y=m::mae::Forward(p,masked.at("images"),masked.at("keep"),masked.at("restore"));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief patches, visible, latent, restored, prediction: live source graph endpoint/storage;
  /// tensor copies share native storage.
  Tensor patches;    ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor visible;    ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor latent;     ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor restored;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor prediction; ///< Live source endpoint; owning handle shares storage and derivatives.

  /// @brief attention, states: live source graph endpoint/storage; tensor copies share native
  /// storage.
  std::vector<Tensor>
      attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      states; ///< Live source endpoint; owning handle shares storage and derivatives.
};
/// @brief Apply Position with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param d Explicit source geometry/control value; defaults match the declaration.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// masked=m::mae::Masked({{"images",n::zeros({1,32,32,3})}},3);auto
/// y=m::mae::Forward(p,masked.at("images"),masked.at("keep"),masked.at("restore"));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor Position(int d) {
  std::vector<float> pos(17 * d);
  for (int patch = 0; patch < 16; ++patch)
    for (int axis = 0; axis < 2; ++axis)
      for (int j = 0; j < d / 4; ++j) {
        double omega                            = std::pow(10000., -double(j) / (d / 4)),
               x                                = (axis == 0 ? patch % 4 : patch / 4) * omega;
        pos[(patch + 1) * d + axis * d / 2 + j] = std::sin(x);
        pos[(patch + 1) * d + axis * d / 2 + d / 4 + j] = std::cos(x);
      }
  return nn::from_vector(pos, {1, 17, d});
}
/// @brief Apply Linear with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param x Live CPU source endpoint with the family geometry above.
/// @param name Explicit source geometry/control value; defaults match the declaration.
/// @param width Packed width2..512 for single or3..512 for pairs, default32; truncate longest side
/// first.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// masked=m::mae::Masked({{"images",n::zeros({1,32,32,3})}},3);auto
/// y=m::mae::Forward(p,masked.at("images"),masked.at("keep"),masked.at("restore"));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor Linear(nn::ParameterStore& p, const Tensor& x, const std::string& name, int width) {
  nn::DenseOptions o;
  o.init = nn::init::normal(.1);
  return nn::dense(p, name, x, width, o);
}
/// @brief Apply Block with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param h Live CPU source endpoint with the family geometry above.
/// @param name Explicit source geometry/control value; defaults match the declaration.
/// @param d Explicit source geometry/control value; defaults match the declaration.
/// @param t Explicit source geometry/control value; defaults match the declaration.
/// @param heads Explicit source geometry/control value; defaults match the declaration.
/// @param out Explicit source geometry/control value; defaults match the declaration.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// masked=m::mae::Masked({{"images",n::zeros({1,32,32,3})}},3);auto
/// y=m::mae::Forward(p,masked.at("images"),masked.at("keep"),masked.at("restore"));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor Block(
    nn::ParameterStore& p,
    Tensor              h,
    const std::string&  name,
    int                 d,
    int                 t,
    int                 heads,
    Path&               out) {
  const auto n     = h.size(0);
  auto       z     = nn::layer_norm(p, name + "/ln1", h, 1e-6);
  auto       parts = nn::split(Linear(p, z, name + "/qkv", 3 * d), 3, 2);
  auto       split = [&](const Tensor& x) {
    return nn::transpose(nn::reshape(x, {n, t, heads, d / heads}), {0, 2, 1, 3});
  };
  auto q = split(parts[0]), k = split(parts[1]), v = split(parts[2]);
  auto att =
      nn::softmax(nn::matmul(q, k, false, true) * float(1 / std::sqrt(double(d / heads))), -1);
  out.attention.push_back(att);
  auto mixed = nn::reshape(nn::transpose(nn::matmul(att, v), {0, 2, 1, 3}), {n, t, d});
  h          = h + Linear(p, mixed, name + "/out", d);
  auto ff    = nn::layer_norm(p, name + "/ln2", h, 1e-6);
  h          = h + Linear(p, nn::gelu(Linear(p, ff, name + "/ff1", 2 * d), true), name + "/ff2", d);
  out.states.push_back(h);
  return h;
}
inline Path
/// @brief Apply Forward with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param images Live Float32 CPU[B,32,32,3] source-scaled RGB.
/// @param keep Float32 CPU[B,4,16] visible-index selection, from Masked().
/// @param restore Float32 CPU[B,16,16] inverse permutation, from Masked().
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// masked=m::mae::Masked({{"images",n::zeros({1,32,32,3})}},3);auto
/// y=m::mae::Forward(p,masked.at("images"),masked.at("keep"),masked.at("restore"));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
Forward(nn::ParameterStore& p, const Tensor& images, const Tensor& keep, const Tensor& restore) {
  Path       out;
  const auto n = images.size(0);
  out.patches  = nn::reshape(
      nn::transpose(nn::reshape(images, {n, 4, 8, 4, 8, 3}), {0, 1, 3, 2, 4, 5}), {n, 16, 192});
  auto projected = Linear(p, out.patches, "patch", 8),
       position  = nn::slice(Position(8), {0, 1, 0}, {1, 16, 8});
  out.visible    = nn::matmul(keep, projected + position);
  auto cls       = p.get("cls", {1, 1, 8}, nn::init::normal(.02));
  auto h         = nn::concat({nn::tile(cls, {n, 1, 1}), out.visible}, 1);
  for (int i = 0; i < 2; ++i)
    h = Block(p, h, "encoder/block" + std::to_string(i), 8, 5, 2, out);
  out.latent     = nn::layer_norm(p, "encoder/ln", h, 1e-6);
  h              = Linear(p, out.latent, "decoder/embed", 4);
  auto mask      = p.get("mask_token", {1, 1, 4}, nn::init::normal(.02));
  auto shuffled  = nn::concat({nn::slice(h, {0, 1, 0}, {n, 4, 4}), nn::tile(mask, {n, 12, 1})}, 1);
  auto restored  = nn::matmul(restore, shuffled);
  out.restored   = nn::concat({nn::slice(h, {0, 0, 0}, {n, 1, 4}), restored}, 1) + Position(4);
  h              = Block(p, out.restored, "decoder/block0", 4, 17, 1, out);
  h              = nn::layer_norm(p, "decoder/ln", h, 1e-6);
  h              = Linear(p, h, "decoder/pred", 192);
  out.prediction = nn::slice(h, {0, 1, 0}, {n, 16, 192});
  return out;
}
/// @brief Apply Masked with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param b Explicit source geometry/control value; defaults match the declaration.
/// @param seed Caller-specified deterministic local mt19937 seed; no global RNG mutation.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// masked=m::mae::Masked({{"images",n::zeros({1,32,32,3})}},3);auto
/// y=m::mae::Forward(p,masked.at("images"),masked.at("keep"),masked.at("restore"));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline A Masked(A b, unsigned seed) {
  const auto                n = b.at("images").size(0);
  std::vector<float>        keep(n * 4 * 16), restore(n * 16 * 16), mask(n * 16, 1);
  std::vector<std::int32_t> shuffle(n * 16);
  std::mt19937              rng(seed);
  for (int row = 0; row < n; ++row) {
    std::vector<int> ids(16);
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), rng);
    for (int j = 0; j < 16; ++j) {
      int original                            = ids[j];
      shuffle[row * 16 + j]                   = original;
      restore[(row * 16 + original) * 16 + j] = 1;
      if (j < 4) {
        keep[(row * 4 + j) * 16 + original] = 1;
        mask[row * 16 + original]           = 0;
      }
    }
  }
  b.emplace("keep", nn::from_vector(keep, {n, 4, 16}));
  b.emplace("restore", nn::from_vector(restore, {n, 16, 16}));
  b.emplace("mask", nn::from_vector(mask, {n, 16}));
  b.emplace("shuffle", nn::from_vector(shuffle, {n, 16}));
  return b;
}

} // namespace mae

/// @brief deit source-equation components; no optimizer or training driver.
namespace deit {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief A: owning source-compatible native type spelling.
using vit::A;
/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::deit::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief logits, class_logits, distill_logits, patches, embedded, hidden: live source graph
  /// endpoint/storage; tensor copies share native storage.
  Tensor logits;         ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor class_logits;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor distill_logits; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor patches;        ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor embedded;       ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor hidden;         ///< Live source endpoint; owning handle shares storage and derivatives.

  /// @brief attention, states: live source graph endpoint/storage; tensor copies share native
  /// storage.
  std::vector<Tensor>
      attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      states; ///< Live source endpoint; owning handle shares storage and derivatives.
};
/// @brief Apply Forward with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param images Live Float32 CPU[B,32,32,3] source-scaled RGB.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::deit::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path Forward(nn::ParameterStore& p, const Tensor& images) {
  constexpr int D = 8, T = 18;
  const auto    n = images.size(0);
  Path          out;
  out.patches = nn::reshape(
      nn::transpose(nn::reshape(images, {n, 4, 8, 4, 8, 3}), {0, 1, 3, 2, 4, 5}), {n, 16, 192});
  nn::DenseOptions dense;
  dense.init     = nn::init::normal(.1);
  auto projected = nn::dense(p, "patch", out.patches, D, dense);
  auto cls       = p.get("cls", {1, 1, D}, nn::init::zeros());
  auto dist      = p.get("distillation_token", {1, 1, D}, nn::init::normal(.02));
  auto position  = p.get("position", {1, T, D}, nn::init::normal(.02));
  auto h         = nn::add(
      nn::concat({nn::tile(cls, {n, 1, 1}), nn::tile(dist, {n, 1, 1}), projected}, 1), position);
  out.embedded = h;
  auto linear  = [&](const Tensor& x, const std::string& name, int width) {
    return nn::dense(p, name, x, width, dense);
  };
  for (int l = 0; l < 2; ++l) {
    auto name  = "block" + std::to_string(l);
    auto x     = nn::layer_norm(p, name + "/ln1", h, 1e-6);
    auto split = nn::split(linear(x, name + "/qkv", 3 * D), 3, 2);
    auto heads = [&](const Tensor& z) {
      return nn::transpose(nn::reshape(z, {n, T, 2, 4}), {0, 2, 1, 3});
    };
    auto q = heads(split[0]), k = heads(split[1]), v = heads(split[2]);
    auto attention = nn::softmax(nn::matmul(q, k, false, true) * .5, -1);
    out.attention.push_back(attention);
    auto mixed = nn::reshape(nn::transpose(nn::matmul(attention, v), {0, 2, 1, 3}), {n, T, D});
    h          = nn::add(h, linear(mixed, name + "/out", D));
    auto ff    = nn::layer_norm(p, name + "/ln2", h, 1e-6);
    ff         = linear(nn::gelu(linear(ff, name + "/ff1", 16), true), name + "/ff2", D);
    h          = nn::add(h, ff);
    out.states.push_back(h);
  }
  out.hidden         = nn::layer_norm(p, "final_ln", h, 1e-6);
  auto features      = nn::squeeze(nn::slice(out.hidden, {0, 0, 0}, {n, 1, D}), {1});
  out.class_logits   = nn::dense(p, "head", features, 10, dense);
  auto dist_features = nn::squeeze(nn::slice(out.hidden, {0, 1, 0}, {n, 1, D}), {1});
  out.distill_logits = nn::dense(p, "distill_head", dist_features, 10, dense);
  out.logits         = (out.class_logits + out.distill_logits) * .5;
  return out;
}

} // namespace deit

/// @brief mixer source-equation components; no optimizer or training driver.
namespace mixer {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief A: owning source-compatible native type spelling.
using vit::A;

/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::mixer::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief patches, embedded, hidden, logits: live source graph endpoint/storage; tensor copies
  /// share native storage.
  Tensor patches;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor embedded; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor hidden;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor logits;   ///< Live source endpoint; owning handle shares storage and derivatives.

  /// @brief token_states, states: live source graph endpoint/storage; tensor copies share native
  /// storage.
  std::vector<Tensor>
      token_states; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      states; ///< Live source endpoint; owning handle shares storage and derivatives.
};
/// @brief Apply Forward with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param images Live Float32 CPU[B,32,32,3] source-scaled RGB.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::mixer::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path Forward(nn::ParameterStore& p, const Tensor& images) {
  const auto n = images.size(0);
  Path       out;
  out.patches = nn::reshape(
      nn::transpose(nn::reshape(images, {n, 4, 8, 4, 8, 3}), {0, 1, 3, 2, 4, 5}), {n, 16, 192});
  nn::DenseOptions opts;
  opts.init   = nn::init::normal(.1);
  auto linear = [&](const Tensor& x, const std::string& name, int width) {
    return nn::dense(p, name, x, width, opts);
  };
  auto h       = linear(out.patches, "patch", 8);
  out.embedded = h;
  for (int i = 0; i < 2; ++i) {
    auto name  = "block" + std::to_string(i);
    auto z     = nn::transpose(nn::layer_norm(p, name + "/ln1", h, 1e-6), {0, 2, 1});
    auto token = linear(nn::gelu(linear(z, name + "/token1", 8), true), name + "/token2", 16);
    h          = nn::add(h, nn::transpose(token, {0, 2, 1}));
    out.token_states.push_back(h);
    auto channel = nn::layer_norm(p, name + "/ln2", h, 1e-6);
    channel =
        linear(nn::gelu(linear(channel, name + "/channel1", 16), true), name + "/channel2", 8);
    h = nn::add(h, channel);
    out.states.push_back(h);
  }
  out.hidden = nn::layer_norm(p, "final_ln", h, 1e-6);
  out.logits = linear(nn::mean(out.hidden, {1}), "head", 10);
  return out;
}

} // namespace mixer

/// @brief convmixer source-equation components; no optimizer or training driver.
namespace convmixer {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief A: owning source-compatible native type spelling.
using vit::A;
/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Separate mutable projection/statistics storage is caller-owned. Caller owns fitting, objectives
/// and generation. Graph tensors retain storage and live derivatives; lazy parameter creation
/// mutates only borrowed stores. Synchronize calls against concurrent store mutation. Copies of
/// path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);n::ParameterStore stats(8);auto
/// y=m::convmixer::Forward(p,stats,n::zeros({1,32,32,3}),false);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief logits, stem, hidden: live source graph endpoint/storage; tensor copies share native
  /// storage.
  Tensor logits; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor stem;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor hidden; ///< Live source endpoint; owning handle shares storage and derivatives.

  /// @brief depthwise, states, means, variances: live source graph endpoint/storage; tensor copies
  /// share native storage.
  std::vector<Tensor>
      depthwise; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      states; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      means; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      variances; ///< Live source endpoint; owning handle shares storage and derivatives.

  /// @brief bn_names: live source graph endpoint/storage; tensor copies share native storage.
  std::vector<std::string> bn_names;
};
inline Path
/// @brief Apply Forward with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Separate mutable projection/statistics storage is caller-owned. Caller owns fitting, objectives
/// and generation. Graph tensors retain storage and live derivatives; lazy parameter creation
/// mutates only borrowed stores. Synchronize calls against concurrent store mutation. Copies of
/// path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param stats Borrowed batch-normalization running statistics; forward returns measurements,
/// caller owns updates.
/// @param image Live Float32 CPU[B,32,32,3] source-scaled RGB.
/// @param training True uses current batch moments, false uses borrowed running statistics; caller
/// applies updates.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);n::ParameterStore stats(8);auto
/// y=m::convmixer::Forward(p,stats,n::zeros({1,32,32,3}),false);
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
Forward(nn::ParameterStore& p, nn::ParameterStore& stats, const Tensor& image, bool training) {
  Path out;
  auto bn = [&](const Tensor& x, const std::string& name) {
    auto gamma    = p.get(name + "/gamma", {8}, nn::init::ones()),
         beta     = p.get(name + "/beta", {8}, nn::init::zeros());
    auto mean     = stats.get(name + "/mean", {8}, nn::init::zeros()).detach();
    auto variance = stats.get(name + "/variance", {8}, nn::init::ones()).detach();
    auto mu = nn::mean(x, {0, 1, 2}, true), var = nn::mean(nn::square(x - mu), {0, 1, 2}, true);
    out.means.push_back(nn::reshape(mu, {8}));
    out.variances.push_back(nn::reshape(var, {8}));
    out.bn_names.push_back(name);
    return ((x - (training ? mu : mean)) * nn::rsqrt((training ? var : variance) + 1e-5)) * gamma +
           beta;
  };
  auto conv = [&](const Tensor& x, const std::string& name, int kernel, int stride, bool dw) {
    nn::ConvLayerOptions o;
    o.init    = nn::init::normal(.1);
    o.stride  = stride;
    o.padding = stride == 8 ? nn::Padding::Valid : nn::Padding::Same;
    return dw ? nn::depthwise_conv2d(p, name, x, 1, kernel, o)
              : nn::conv2d(p, name, x, 8, kernel, o);
  };
  auto h   = bn(nn::gelu(conv(image, "patch", 8, 8, false)), "patch_bn");
  out.stem = h;
  for (int i = 0; i < 2; ++i) {
    auto name = "block" + std::to_string(i);
    auto dw   = bn(nn::gelu(conv(h, name + "/depthwise", 3, 1, true)), name + "/depthwise_bn");
    h         = h + dw;
    out.depthwise.push_back(h);
    h = bn(nn::gelu(conv(h, name + "/pointwise", 1, 1, false)), name + "/pointwise_bn");
    out.states.push_back(h);
  }
  out.hidden = h;
  nn::DenseOptions o;
  o.init     = nn::init::normal(.1);
  out.logits = nn::dense(p, "head", nn::mean(h, {1, 2}), 10, o);
  return out;
}

} // namespace convmixer

/// @brief hybrid source-equation components; no optimizer or training driver.
namespace hybrid {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief A: owning source-compatible native type spelling.
using vit::A;

/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::hybrid::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief conv0, conv1, features, hidden, logits: live source graph endpoint/storage; tensor
  /// copies share native storage.
  Tensor conv0;    ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor conv1;    ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor features; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor hidden;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor logits;   ///< Live source endpoint; owning handle shares storage and derivatives.

  /// @brief attention, states: live source graph endpoint/storage; tensor copies share native
  /// storage.
  std::vector<Tensor>
      attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      states; ///< Live source endpoint; owning handle shares storage and derivatives.
};
/// @brief Apply Forward with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param images Live Float32 CPU[B,32,32,3] source-scaled RGB.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::hybrid::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path Forward(nn::ParameterStore& p, const Tensor& images) {
  Path       out;
  const auto n    = images.size(0);
  auto       conv = [&](const Tensor& x, const std::string& name, int channels) {
    nn::ConvLayerOptions o;
    o.init       = nn::init::normal(.1);
    o.activation = nn::Activation::Relu;
    return nn::avg_pool2d(nn::conv2d(p, name, x, channels, 3, o), 2, 2, nn::Padding::Valid);
  };
  out.conv0          = conv(images, "cnn/conv0", 4);
  out.conv1          = conv(out.conv0, "cnn/conv1", 8);
  out.features       = nn::avg_pool2d(out.conv1, 2, 2, nn::Padding::Valid);
  auto             h = nn::reshape(out.features, {n, 16, 8});
  nn::DenseOptions o;
  o.init      = nn::init::normal(.1);
  auto linear = [&](const Tensor& x, const std::string& name, int width) {
    return nn::dense(p, name, x, width, o);
  };
  std::vector<std::int32_t> indices;
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j)
      indices.push_back((i / 4 - j / 4 + 3) * 7 + i % 4 - j % 4 + 3);
  auto index = nn::from_vector(indices, {16, 16});
  for (int i = 0; i < 2; ++i) {
    const auto name  = "block" + std::to_string(i);
    auto       z     = nn::layer_norm(p, name + "/ln1", h, 1e-6);
    auto       parts = nn::split(linear(z, name + "/qkv", 24), 3, 2);
    auto       heads = [&](const Tensor& x) {
      return nn::transpose(nn::reshape(x, {n, 16, 2, 4}), {0, 2, 1, 3});
    };
    auto q = heads(parts[0]), k = heads(parts[1]), v = heads(parts[2]);
    auto table = p.get(name + "/relative_bias", {49, 2}, nn::init::normal(.02));
    auto bias  = nn::transpose(nn::gather(table, index, 0), {2, 0, 1});
    auto att   = nn::softmax(nn::matmul(q, k, false, true) * .5 + bias, -1);
    out.attention.push_back(att);
    auto mixed = nn::reshape(nn::transpose(nn::matmul(att, v), {0, 2, 1, 3}), {n, 16, 8});
    h          = h + linear(mixed, name + "/out", 8);
    auto ff    = nn::layer_norm(p, name + "/ln2", h, 1e-6);
    h          = h + linear(nn::gelu(linear(ff, name + "/ff1", 16), true), name + "/ff2", 8);
    out.states.push_back(h);
  }
  out.hidden = nn::layer_norm(p, "final_ln", h, 1e-6);
  out.logits = linear(nn::mean(out.hidden, {1}), "head", 10);
  return out;
}

} // namespace hybrid

/// @brief pvt source-equation components; no optimizer or training driver.
namespace pvt {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief A: owning source-compatible native type spelling.
using vit::A;

/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::pvt::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief logits, patches, hidden: live source graph endpoint/storage; tensor copies share native
  /// storage.
  Tensor logits;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor patches; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor hidden;  ///< Live source endpoint; owning handle shares storage and derivatives.

  /// @brief embedded, reduced, attention, states: live source graph endpoint/storage; tensor copies
  /// share native storage.
  std::vector<Tensor>
      embedded; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      reduced; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      states; ///< Live source endpoint; owning handle shares storage and derivatives.
};
/// @brief Apply Patches with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param x Live CPU source endpoint with the family geometry above.
/// @param grid Explicit source geometry/control value; defaults match the declaration.
/// @param patch Explicit source geometry/control value; defaults match the declaration.
/// @param channels Explicit source geometry/control value; defaults match the declaration.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::pvt::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor Patches(const Tensor& x, int grid, int patch, int channels) {
  const auto n = x.size(0);
  return nn::reshape(
      nn::transpose(
          nn::reshape(x, {n, grid / patch, patch, grid / patch, patch, channels}),
          {0, 1, 3, 2, 4, 5}),
      {n, (grid / patch) * (grid / patch), patch * patch * channels});
}
/// @brief Apply Forward with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param images Live Float32 CPU[B,32,32,3] source-scaled RGB.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::pvt::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path Forward(nn::ParameterStore& p, const Tensor& images) {
  Path             out;
  auto             h             = images;
  const auto       n             = images.size(0);
  int              previous_grid = 32, previous_d = 3;
  const int        widths[] = {4, 8, 12, 16}, heads[] = {1, 2, 3, 4}, ratios[] = {4, 2, 2, 1};
  nn::DenseOptions o;
  o.init      = nn::init::normal(.1);
  auto linear = [&](const Tensor& x, const std::string& name, int width) {
    return nn::dense(p, name, x, width, o);
  };
  for (int stage = 0; stage < 4; ++stage) {
    const int patch = stage ? 2 : 4, grid = previous_grid / patch, d = widths[stage],
              spatial = grid * grid, t = spatial + (stage == 3), nh = heads[stage],
              r = ratios[stage], keys = stage == 3 ? t : (grid / r) * (grid / r);
    const auto name    = "stage" + std::to_string(stage);
    auto       patches = Patches(h, previous_grid, patch, previous_d);
    if (stage == 0)
      out.patches = patches;
    h = linear(patches, name + "/patch", d);
    h = nn::layer_norm(p, name + "/patch_ln", h, 1e-6);
    if (stage == 3) {
      auto cls = p.get("cls", {1, 1, d}, nn::init::normal(.02));
      h        = nn::concat({nn::tile(cls, {n, 1, 1}), h}, 1);
    }
    h = h + p.get(name + "/position", {1, t, d}, nn::init::normal(.02));
    out.embedded.push_back(h);
    auto normalized = nn::layer_norm(p, name + "/ln1", h, 1e-6);
    auto q = linear(normalized, name + "/q", d), reduced = normalized;
    if (r > 1) {
      auto patches = Patches(nn::reshape(normalized, {n, grid, grid, d}), grid, r, d);
      reduced      = linear(patches, name + "/sr", d);
      reduced      = nn::layer_norm(p, name + "/sr_ln", reduced, 1e-6);
    }
    out.reduced.push_back(reduced);
    auto parts   = nn::split(linear(reduced, name + "/kv", 2 * d), 2, 2);
    auto toheads = [&](const Tensor& x, int count) {
      return nn::transpose(nn::reshape(x, {n, count, nh, d / nh}), {0, 2, 1, 3});
    };
    auto qh = toheads(q, t), k = toheads(parts[0], keys), v = toheads(parts[1], keys);
    auto attention = nn::softmax(nn::matmul(qh, k, false, true) * .5, -1);
    out.attention.push_back(attention);
    auto mixed = nn::reshape(nn::transpose(nn::matmul(attention, v), {0, 2, 1, 3}), {n, t, d});
    h          = h + linear(mixed, name + "/out", d);
    auto ff    = nn::layer_norm(p, name + "/ln2", h, 1e-6);
    h          = h + linear(nn::gelu(linear(ff, name + "/ff1", 2 * d), true), name + "/ff2", d);
    out.states.push_back(h);
    if (stage < 3)
      h = nn::reshape(h, {n, grid, grid, d});
    previous_grid = grid;
    previous_d    = d;
  }
  out.hidden = nn::layer_norm(p, "final_ln", h, 1e-6);
  auto cls   = nn::squeeze(nn::slice(out.hidden, {0, 0, 0}, {n, 1, 16}), {1});
  out.logits = linear(cls, "head", 10);
  return out;
}

} // namespace pvt

/// @brief swin source-equation components; no optimizer or training driver.
namespace swin {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief A: owning source-compatible native type spelling.
using vit::A;

/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::swin::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief logits, patches, embedded, merged, hidden: live source graph endpoint/storage; tensor
  /// copies share native storage.
  Tensor logits;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor patches;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor embedded; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor merged;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor hidden;   ///< Live source endpoint; owning handle shares storage and derivatives.

  /// @brief attention, states, masks: live source graph endpoint/storage; tensor copies share
  /// native storage.
  std::vector<Tensor>
      attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      states; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      masks; ///< Live source endpoint; owning handle shares storage and derivatives.
};
/// @brief Apply Shift with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param x Live CPU source endpoint with the family geometry above.
/// @param grid Explicit source geometry/control value; defaults match the declaration.
/// @param shift Explicit source geometry/control value; defaults match the declaration.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::swin::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor Shift(const Tensor& x, int grid, int shift) {
  if (!shift)
    return x;
  std::vector<std::int32_t> ids(grid);
  for (int i = 0; i < grid; ++i)
    ids[i] = (i + shift + grid) % grid;
  auto index = nn::from_vector(ids, {grid});
  return nn::gather(nn::gather(x, index, 1), index, 2);
}
/// @brief Apply Windows with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param x Live CPU source endpoint with the family geometry above.
/// @param grid Explicit source geometry/control value; defaults match the declaration.
/// @param d Explicit source geometry/control value; defaults match the declaration.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::swin::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor Windows(const Tensor& x, int grid, int d) {
  return nn::reshape(
      nn::transpose(nn::reshape(x, {x.size(0), grid / 2, 2, grid / 2, 2, d}), {0, 1, 3, 2, 4, 5}),
      {x.size(0) * grid * grid / 4, 4, d});
}
/// @brief Apply Reverse with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param x Live CPU source endpoint with the family geometry above.
/// @param grid Explicit source geometry/control value; defaults match the declaration.
/// @param d Explicit source geometry/control value; defaults match the declaration.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::swin::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor Reverse(const Tensor& x, int grid, int d) {
  const auto n = x.size(0) / (grid * grid / 4);
  return nn::reshape(
      nn::transpose(nn::reshape(x, {n, grid / 2, grid / 2, 2, 2, d}), {0, 1, 3, 2, 4, 5}),
      {n, grid, grid, d});
}
/// @brief Apply Forward with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param images Live Float32 CPU[B,32,32,3] source-scaled RGB.
/// @param controls Explicit source geometry/control value; defaults match the declaration.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::swin::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path Forward(nn::ParameterStore& p, const Tensor& images, const A& controls = {}) {
  Path       out;
  const auto n = images.size(0);
  out.patches  = nn::reshape(
      nn::transpose(nn::reshape(images, {n, 4, 8, 4, 8, 3}), {0, 1, 3, 2, 4, 5}), {n, 16, 192});
  nn::DenseOptions o;
  o.init      = nn::init::normal(.1);
  auto linear = [&](const Tensor& x, const std::string& name, int width) {
    return nn::dense(p, name, x, width, o);
  };
  auto h       = nn::layer_norm(p, "patch_ln", linear(out.patches, "patch", 8), 1e-6);
  out.embedded = h;
  for (int stage = 0; stage < 2; ++stage) {
    const int grid = stage == 0 ? 4 : 2, d = stage == 0 ? 8 : 16, t = grid * grid, nw = t / 4;
    for (int block = 0; block < 2; ++block) {
      const int  shift      = stage == 0 && block == 1 ? 1 : 0;
      const auto name       = "stage" + std::to_string(stage) + "/block" + std::to_string(block);
      auto       normalized = nn::layer_norm(p, name + "/ln1", h, 1e-6);
      auto       windows =
          Windows(Shift(nn::reshape(normalized, {n, grid, grid, d}), grid, shift), grid, d);
      auto parts = nn::split(linear(windows, name + "/qkv", 3 * d), 3, 2);
      auto heads = [&](const Tensor& x) {
        return nn::transpose(nn::reshape(x, {n * nw, 4, 2, d / 2}), {0, 2, 1, 3});
      };
      auto q = heads(parts[0]), k = heads(parts[1]), v = heads(parts[2]);
      auto table = p.get(name + "/relative_bias", {9, 2}, nn::init::normal(.02));
      std::vector<std::int32_t> ids;
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
          ids.push_back((i / 2 - j / 2 + 1) * 3 + i % 2 - j % 2 + 1);
      auto bias   = nn::transpose(nn::gather(table, nn::from_vector(ids, {4, 4}), 0), {2, 0, 1});
      auto scores = nn::matmul(q, k, false, true) * float(1 / std::sqrt(d / 2.)) + bias;
      std::vector<float> mask(nw * 16);
      if (shift) {
        auto region = [](int y, int x) {
          auto id = [](int a) { return a < 2 ? 0 : (a < 3 ? 1 : 2); };
          return 3 * id(y) + id(x);
        };
        for (int w = 0; w < nw; ++w)
          for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
              int yi = w / (grid / 2) * 2 + i / 2, xi = w % (grid / 2) * 2 + i % 2,
                  yj = w / (grid / 2) * 2 + j / 2, xj = w % (grid / 2) * 2 + j % 2;
              mask[w * 16 + i * 4 + j] = region(yi, xi) == region(yj, xj) ? 0.f : -100.f;
            }
      }
      const auto key = "mask" + std::to_string(stage * 2 + block);
      out.masks.push_back(
          controls.count(key) ? controls.at(key) : nn::from_vector(mask, {1, nw, 1, 4, 4}));
      auto masked =
          nn::reshape(nn::reshape(scores, {n, nw, 2, 4, 4}) + out.masks.back(), {n * nw, 2, 4, 4});
      auto attention = nn::softmax(masked, -1);
      out.attention.push_back(attention);
      auto mixed =
          nn::reshape(nn::transpose(nn::matmul(attention, v), {0, 2, 1, 3}), {n * nw, 4, d});
      auto projected = linear(mixed, name + "/out", d),
           reversed  = Shift(Reverse(projected, grid, d), grid, -shift);
      h              = h + nn::reshape(reversed, {n, t, d});
      auto ff        = nn::layer_norm(p, name + "/ln2", h, 1e-6);
      h = h + linear(nn::gelu(linear(ff, name + "/ff1", 2 * d), true), name + "/ff2", d);
      out.states.push_back(h);
    }
    if (stage == 0) {
      auto gridh = nn::reshape(h, {n, 4, 4, 8});
      auto sub   = [&](int y, int x) {
        return nn::gather(
            nn::gather(gridh, nn::from_vector(std::vector<std::int32_t>{y, y + 2}, {2}), 1),
            nn::from_vector(std::vector<std::int32_t>{x, x + 2}, {2}),
            2);
      };
      auto joined =
          nn::reshape(nn::concat({sub(0, 0), sub(1, 0), sub(0, 1), sub(1, 1)}, 3), {n, 4, 32});
      auto norm  = nn::layer_norm(p, "merge/ln", joined, 1e-6),
           w     = p.get("merge/W", {32, 16}, nn::init::normal(.1));
      h          = nn::matmul(norm, w);
      out.merged = h;
    }
  }
  out.hidden = nn::layer_norm(p, "final_ln", h, 1e-6);
  out.logits = linear(nn::mean(out.hidden, {1}), "head", 10);
  return out;
}

} // namespace swin

/// @brief dino source-equation components; no optimizer or training driver.
namespace dino {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief A: owning source-compatible native type spelling.
using vit::A;
/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::dino::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief logits, patches, embedded, hidden, features, bottleneck: live source graph
  /// endpoint/storage; tensor copies share native storage.
  Tensor logits;     ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor patches;    ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor embedded;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor hidden;     ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor features;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor bottleneck; ///< Live source endpoint; owning handle shares storage and derivatives.

  /// @brief attention, states: live source graph endpoint/storage; tensor copies share native
  /// storage.
  std::vector<Tensor>
      attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      states; ///< Live source endpoint; owning handle shares storage and derivatives.
};
/// @brief Apply Forward with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param images Live Float32 CPU[B,32,32,3] source-scaled RGB.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::dino::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path Forward(nn::ParameterStore& p, const Tensor& images) {
  constexpr int D = 8, T = 17;
  const auto    n = images.size(0);
  Path          out;
  out.patches = nn::reshape(
      nn::transpose(nn::reshape(images, {n, 4, 8, 4, 8, 3}), {0, 1, 3, 2, 4, 5}), {n, 16, 192});
  nn::DenseOptions dense;
  dense.init     = nn::init::normal(.1);
  auto projected = nn::dense(p, "patch", out.patches, D, dense);
  auto cls       = p.get("cls", {1, 1, D}, nn::init::zeros());
  auto position  = p.get("position", {1, T, D}, nn::init::normal(.02));
  auto h         = nn::add(nn::concat({nn::tile(cls, {n, 1, 1}), projected}, 1), position);
  out.embedded   = h;
  auto linear    = [&](const Tensor& x, const std::string& name, int width) {
    return nn::dense(p, name, x, width, dense);
  };
  for (int l = 0; l < 2; ++l) {
    auto name  = "block" + std::to_string(l);
    auto x     = nn::layer_norm(p, name + "/ln1", h, 1e-6);
    auto split = nn::split(linear(x, name + "/qkv", 3 * D), 3, 2);
    auto heads = [&](const Tensor& z) {
      return nn::transpose(nn::reshape(z, {n, T, 2, 4}), {0, 2, 1, 3});
    };
    auto q = heads(split[0]), k = heads(split[1]), v = heads(split[2]);
    auto attention = nn::softmax(nn::matmul(q, k, false, true) * .5, -1);
    out.attention.push_back(attention);
    auto mixed = nn::reshape(nn::transpose(nn::matmul(attention, v), {0, 2, 1, 3}), {n, T, D});
    h          = nn::add(h, linear(mixed, name + "/out", D));
    auto ff    = nn::layer_norm(p, name + "/ln2", h, 1e-6);
    ff         = linear(nn::gelu(linear(ff, name + "/ff1", 16), true), name + "/ff2", D);
    h          = nn::add(h, ff);
    out.states.push_back(h);
  }
  out.hidden          = nn::layer_norm(p, "final_ln", h, 1e-6);
  auto features       = nn::squeeze(nn::slice(out.hidden, {0, 0, 0}, {n, 1, D}), {1});
  out.features        = features;
  auto projected_head = nn::gelu(nn::dense(p, "projection/0", features, 16, dense), true);
  projected_head      = nn::gelu(nn::dense(p, "projection/1", projected_head, 16, dense), true);
  projected_head      = nn::dense(p, "projection/2", projected_head, 4, dense);
  out.bottleneck =
      projected_head *
      nn::rsqrt(nn::maximum(nn::sum(nn::square(projected_head), {1}, true), nn::scalar(1e-12)));
  auto direction = p.get("projection/last_v", {4, 16}, nn::init::normal(.1));
  auto normalized =
      direction *
      nn::rsqrt(nn::maximum(nn::sum(nn::square(direction), {0}, true), nn::scalar(1e-12)));
  out.logits = nn::matmul(out.bottleneck, normalized);
  return out;
}
/// @brief Apply Views with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param b Explicit source geometry/control value; defaults match the declaration.
/// @param seed Caller-specified deterministic local mt19937 seed; no global RNG mutation.
/// @param original Defaultfalse; true selects the original-view diagnostic without stochastic
/// augmentation.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::dino::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline A Views(A b, unsigned seed, bool original = false) {
  const auto                            n      = b.at("images").size(0);
  const auto                            images = b.at("images").to_vector<float>();
  std::vector<float>                    views(4 * n * 3072), gains(4 * n);
  std::vector<std::int32_t>             metadata(4 * n * 4);
  std::mt19937                          rng(seed);
  std::uniform_real_distribution<float> uniform(.8f, 1.2f);
  for (int v = 0; v < 4; ++v)
    for (int row = 0; row < n; ++row) {
      int size = original ? 32 : (v < 2 ? 24 : 12) + rng() % 9,
          y = original ? 0 : rng() % (33 - size), x = original ? 0 : rng() % (33 - size),
          flip           = original ? 0 : rng() % 2;
      float gain         = original ? 1.f : uniform(rng);
      int   m            = (v * n + row) * 4;
      metadata[m]        = y;
      metadata[m + 1]    = x;
      metadata[m + 2]    = size;
      metadata[m + 3]    = flip;
      gains[v * n + row] = gain;
      for (int oy = 0; oy < 32; ++oy)
        for (int ox = 0; ox < 32; ++ox)
          for (int c = 0; c < 3; ++c) {
            int sy = y + oy * size / 32,
                sx = x + (flip ? size - 1 - ox * size / 32 : ox * size / 32);
            views[((v * n + row) * 32 * 32 + oy * 32 + ox) * 3 + c] =
                std::min(1.f, std::max(0.f, gain * images[(row * 32 * 32 + sy * 32 + sx) * 3 + c]));
          }
    }
  b.emplace("views", nn::from_vector(views, {4, n, 32, 32, 3}));
  b.emplace("crop_metadata", nn::from_vector(metadata, {4, n, 4}));
  b.emplace("crop_gains", nn::from_vector(gains, {4, n}));
  return b;
}

} // namespace dino

/// @brief linformer source-equation components; no optimizer or training driver.
namespace linformer {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief A: owning source-compatible native type spelling.
using vit::A;

/// @brief Own ContextPath source graph endpoints or configuration.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::linformer::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct ContextPath {
  /// @brief context, attention, keys, values: live source graph endpoint/storage; tensor copies
  /// share native storage.
  Tensor context;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor keys;      ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor values;    ///< Live source endpoint; owning handle shares storage and derivatives.
};
/// @brief Apply Context with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param q Live CPU source endpoint with the family geometry above.
/// @param k Live CPU source endpoint with the family geometry above.
/// @param v Live CPU source endpoint with the family geometry above.
/// @param e Live CPU source endpoint with the family geometry above.
/// @param f Live CPU source endpoint with the family geometry above.
/// @param scale Explicit source geometry/control value; defaults match the declaration.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::linformer::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline ContextPath Context(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& e,
    const Tensor& f,
    double        scale = .5) {
  ContextPath out;
  out.keys      = nn::matmul(nn::transpose(e, {1, 0}), k);
  out.values    = nn::matmul(nn::transpose(f, {1, 0}), v);
  out.attention = nn::softmax(nn::matmul(q, out.keys, false, true) * scale, -1);
  out.context   = nn::matmul(out.attention, out.values);
  return out;
}
/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::linformer::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief logits, patches, embedded, hidden: live source graph endpoint/storage; tensor copies
  /// share native storage.
  Tensor logits;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor patches;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor embedded; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor hidden;   ///< Live source endpoint; owning handle shares storage and derivatives.

  /// @brief attention, states, queries, keys, values, projected_keys, projected_values: live source
  /// graph endpoint/storage; tensor copies share native storage.
  std::vector<Tensor>
      attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      states; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      queries;              ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor> keys; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      values; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      projected_keys; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      projected_values; ///< Live source endpoint; owning handle shares storage and derivatives.
};
/// @brief Apply Forward with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param images Live Float32 CPU[B,32,32,3] source-scaled RGB.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::linformer::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path Forward(nn::ParameterStore& p, const Tensor& images) {
  constexpr int D = 8, T = 17;
  const auto    n = images.size(0);
  Path          out;
  out.patches = nn::reshape(
      nn::transpose(nn::reshape(images, {n, 4, 8, 4, 8, 3}), {0, 1, 3, 2, 4, 5}), {n, 16, 192});
  nn::DenseOptions dense;
  dense.init     = nn::init::normal(.1);
  auto projected = nn::dense(p, "patch", out.patches, D, dense);
  auto cls       = p.get("cls", {1, 1, D}, nn::init::zeros());
  auto position  = p.get("position", {1, T, D}, nn::init::normal(.02));
  auto h         = nn::add(nn::concat({nn::tile(cls, {n, 1, 1}), projected}, 1), position);
  out.embedded   = h;
  auto linear    = [&](const Tensor& x, const std::string& name, int width) {
    return nn::dense(p, name, x, width, dense);
  };
  for (int l = 0; l < 2; ++l) {
    auto name  = "block" + std::to_string(l);
    auto x     = nn::layer_norm(p, name + "/ln1", h, 1e-6);
    auto split = nn::split(linear(x, name + "/qkv", 3 * D), 3, 2);
    auto heads = [&](const Tensor& z) {
      return nn::transpose(nn::reshape(z, {n, T, 2, 4}), {0, 2, 1, 3});
    };
    auto q = heads(split[0]), k = heads(split[1]), v = heads(split[2]);
    auto e          = p.get(name + "/E", {T, 4}, nn::init::normal(.1)),
         f          = p.get(name + "/F", {T, 4}, nn::init::normal(.1));
    auto compressed = Context(q, k, v, e, f);
    out.queries.push_back(q);
    out.keys.push_back(k);
    out.values.push_back(v);
    out.projected_keys.push_back(compressed.keys);
    out.projected_values.push_back(compressed.values);
    out.attention.push_back(compressed.attention);
    auto mixed = nn::reshape(nn::transpose(compressed.context, {0, 2, 1, 3}), {n, T, D});
    h          = nn::add(h, linear(mixed, name + "/out", D));
    auto ff    = nn::layer_norm(p, name + "/ln2", h, 1e-6);
    ff         = linear(nn::gelu(linear(ff, name + "/ff1", 16), true), name + "/ff2", D);
    h          = nn::add(h, ff);
    out.states.push_back(h);
  }
  out.hidden    = nn::layer_norm(p, "final_ln", h, 1e-6);
  auto features = nn::squeeze(nn::slice(out.hidden, {0, 0, 0}, {n, 1, D}), {1});
  out.logits    = nn::dense(p, "head", features, 10, dense);
  return out;
}

} // namespace linformer

/// @brief longformer source-equation components; no optimizer or training driver.
namespace longformer {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief A: owning source-compatible native type spelling.
using vit::A;

/// @brief Own ContextPath source graph endpoints or configuration.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::longformer::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct ContextPath {
  /// @brief context, attention, global_attention: live source graph endpoint/storage; tensor copies
  /// share native storage.
  Tensor context;          ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor attention;        ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor global_attention; ///< Live source endpoint; owning handle shares storage and derivatives.
};
/// @brief Apply Context with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param q Live CPU source endpoint with the family geometry above.
/// @param k Live CPU source endpoint with the family geometry above.
/// @param v Live CPU source endpoint with the family geometry above.
/// @param gq Live CPU source endpoint with the family geometry above.
/// @param gk Live CPU source endpoint with the family geometry above.
/// @param gv Live CPU source endpoint with the family geometry above.
/// @param radius Explicit source geometry/control value; defaults match the declaration.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::longformer::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline ContextPath Context(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& gq,
    const Tensor& gk,
    const Tensor& gv,
    int           radius = 2) {
  const auto n = q.size(2);
  component_detail::Require(n > 1 && radius >= 0, "valid Longformer window");
  const int          width = 2 * radius + 2;
  std::vector<int>   ids(n * width);
  std::vector<float> bias(n * width);
  for (int i = 0; i < n; ++i)
    for (int slot = 0; slot < width; ++slot) {
      int  j                 = slot == 0 ? 0 : i + slot - 1 - radius;
      bool valid             = slot == 0 || (j > 0 && j < n);
      ids[i * width + slot]  = valid ? j : 0;
      bias[i * width + slot] = valid ? 0.f : -1e9f;
    }
  auto indices = nn::from_vector(ids, {n, width});
  auto gather  = [&](const Tensor& x) {
    return nn::transpose(nn::gather(nn::transpose(x, {2, 0, 1, 3}), indices, 0), {2, 3, 0, 1, 4});
  };
  auto        keys = gather(k), values = gather(v);
  ContextPath out;
  out.attention = nn::softmax(
      nn::sum(nn::expand_dims(q, 3) * keys, {4}) * .5 + nn::from_vector(bias, {n, width}), -1);
  auto local           = nn::sum(nn::expand_dims(out.attention, 4) * values, {3});
  auto first           = nn::slice(gq, {0, 0, 0, 0}, {gq.size(0), gq.size(1), 1, gq.size(3)});
  out.global_attention = nn::softmax(nn::matmul(first, gk, false, true) * .5, -1);
  auto global          = nn::matmul(out.global_attention, gv);
  out.context          = nn::concat(
      {global, nn::slice(local, {0, 0, 1, 0}, {q.size(0), q.size(1), n - 1, q.size(3)})}, 2);
  return out;
}
/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::longformer::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief logits, patches, embedded, hidden: live source graph endpoint/storage; tensor copies
  /// share native storage.
  Tensor logits;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor patches;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor embedded; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor hidden;   ///< Live source endpoint; owning handle shares storage and derivatives.

  /// @brief attention, states, queries, keys, values, global_queries, global_keys: live source
  /// graph endpoint/storage; tensor copies share native storage.
  std::vector<Tensor>
      attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      states; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      queries;              ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor> keys; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      values; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      global_queries; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      global_keys; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      global_values; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      global_attention; ///< Live source endpoint; owning handle shares storage and derivatives.
};
/// @brief Apply Forward with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param images Live Float32 CPU[B,32,32,3] source-scaled RGB.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::longformer::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path Forward(nn::ParameterStore& p, const Tensor& images) {
  constexpr int D = 8, T = 17;
  const auto    n = images.size(0);
  Path          out;
  out.patches = nn::reshape(
      nn::transpose(nn::reshape(images, {n, 4, 8, 4, 8, 3}), {0, 1, 3, 2, 4, 5}), {n, 16, 192});
  nn::DenseOptions dense;
  dense.init     = nn::init::normal(.1);
  auto projected = nn::dense(p, "patch", out.patches, D, dense);
  auto cls       = p.get("cls", {1, 1, D}, nn::init::zeros());
  auto position  = p.get("position", {1, T, D}, nn::init::normal(.02));
  auto h         = nn::add(nn::concat({nn::tile(cls, {n, 1, 1}), projected}, 1), position);
  out.embedded   = h;
  auto linear    = [&](const Tensor& x, const std::string& name, int width) {
    return nn::dense(p, name, x, width, dense);
  };
  for (int l = 0; l < 2; ++l) {
    auto name  = "block" + std::to_string(l);
    auto x     = nn::layer_norm(p, name + "/ln1", h, 1e-6);
    auto split = nn::split(linear(x, name + "/qkv", 3 * D), 3, 2);
    auto heads = [&](const Tensor& z) {
      return nn::transpose(nn::reshape(z, {n, T, 2, 4}), {0, 2, 1, 3});
    };
    auto q = heads(split[0]), k = heads(split[1]), v = heads(split[2]);
    auto gs = nn::split(linear(x, name + "/global_qkv", 3 * D), 3, 2);
    auto gq = heads(gs[0]), gk = heads(gs[1]), gv = heads(gs[2]);
    auto attended = Context(q, k, v, gq, gk, gv);
    out.queries.push_back(q);
    out.keys.push_back(k);
    out.values.push_back(v);
    out.global_queries.push_back(gq);
    out.global_keys.push_back(gk);
    out.global_values.push_back(gv);
    out.attention.push_back(attended.attention);
    out.global_attention.push_back(attended.global_attention);
    auto mixed = nn::reshape(nn::transpose(attended.context, {0, 2, 1, 3}), {n, T, D});
    h          = nn::add(h, linear(mixed, name + "/out", D));
    auto ff    = nn::layer_norm(p, name + "/ln2", h, 1e-6);
    ff         = linear(nn::gelu(linear(ff, name + "/ff1", 16), true), name + "/ff2", D);
    h          = nn::add(h, ff);
    out.states.push_back(h);
  }
  out.hidden    = nn::layer_norm(p, "final_ln", h, 1e-6);
  auto features = nn::squeeze(nn::slice(out.hidden, {0, 0, 0}, {n, 1, D}), {1});
  out.logits    = nn::dense(p, "head", features, 10, dense);
  return out;
}

} // namespace longformer

/// @brief bigbird source-equation components; no optimizer or training driver.
namespace bigbird {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief A: owning source-compatible native type spelling.
using vit::A;

/// @brief Own Pattern source graph endpoints or configuration.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::bigbird::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Pattern {
  /// @brief indices, globals: live source graph endpoint/storage; tensor copies share native
  /// storage.
  std::vector<int> indices; ///< Owned source configuration/control value.
  std::vector<int> globals; ///< Owned source configuration/control value.

  /// @brief bias: live source graph endpoint/storage; tensor copies share native storage.
  std::vector<float> bias;
  /// @brief first, last, width: live source graph endpoint/storage; tensor copies share native
  /// storage.
  int first; ///< Owned source configuration/control value.
  int last;  ///< Owned source configuration/control value.
  int width; ///< Owned source configuration/control value.
};
/// @brief Apply Slots with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param n Explicit source geometry/control value; defaults match the declaration.
/// @param seed Caller-specified deterministic local mt19937 seed; no global RNG mutation.
/// @param block Explicit source geometry/control value; defaults match the declaration.
/// @param random_blocks Explicit source geometry/control value; defaults match the declaration.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::bigbird::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Pattern Slots(int n, unsigned seed, int block = 2, int random_blocks = 2) {
  component_detail::Require(
      n > 2 * block && block > 0 && random_blocks >= 0, "valid BigBird blocks");
  int count = (n + block - 1) / block, last = (count - 1) * block,
      width = (5 + random_blocks) * block;
  Pattern p;
  p.first = block;
  p.last  = last;
  p.width = width;
  p.indices.resize(n * width);
  p.bias.resize(n * width);
  p.globals.resize(block + n - last);
  for (int i = 0; i < block; ++i)
    p.globals[i] = i;
  for (int i = last; i < n; ++i)
    p.globals[block + i - last] = i;
  for (int qb = 0; qb < count; ++qb) {
    std::vector<int> chosen{0, count - 1};
    for (int j = std::max(0, qb - 1); j <= std::min(count - 1, qb + 1); ++j)
      if (std::find(chosen.begin(), chosen.end(), j) == chosen.end())
        chosen.push_back(j);
    std::vector<int> candidates;
    for (int j = 0; j < count; ++j)
      if (std::find(chosen.begin(), chosen.end(), j) == chosen.end())
        candidates.push_back(j);
    uint32_t state = seed ^ (0x9e3779b9u * unsigned(qb + 1));
    if (!state)
      state = 1;
    for (int i = int(candidates.size()) - 1; i > 0; --i) {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      std::swap(candidates[i], candidates[state % unsigned(i + 1)]);
    }
    for (int i = 0; i < std::min(random_blocks, int(candidates.size())); ++i)
      chosen.push_back(candidates[i]);
    std::vector<int> keys;
    for (auto b : chosen)
      for (int j = b * block; j < std::min(n, (b + 1) * block); ++j)
        keys.push_back(j);
    component_detail::Require(int(keys.size()) <= width, "bounded BigBird slots");
    for (int i = qb * block; i < std::min(n, (qb + 1) * block); ++i)
      for (int j = 0; j < width; ++j) {
        bool valid               = j < int(keys.size());
        p.indices[i * width + j] = valid ? keys[j] : 0;
        p.bias[i * width + j]    = valid ? 0.f : -1e9f;
      }
  }
  return p;
}
/// @brief Own ContextPath source graph endpoints or configuration.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::bigbird::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct ContextPath {
  /// @brief context, attention, global_attention, indices, mask, global_indices: live source graph
  /// endpoint/storage; tensor copies share native storage.
  Tensor context;          ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor attention;        ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor global_attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor indices;          ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor mask;             ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor global_indices;   ///< Live source endpoint; owning handle shares storage and derivatives.
};
inline ContextPath
/// @brief Apply Context with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param q Live CPU source endpoint with the family geometry above.
/// @param k Live CPU source endpoint with the family geometry above.
/// @param v Live CPU source endpoint with the family geometry above.
/// @param seed Caller-specified deterministic local mt19937 seed; no global RNG mutation.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::bigbird::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
Context(const Tensor& q, const Tensor& k, const Tensor& v, unsigned seed = 50911) {
  auto        n = q.size(2);
  auto        p = Slots(n, seed);
  ContextPath out;
  out.indices        = nn::from_vector(p.indices, {n, p.width});
  out.mask           = nn::from_vector(p.bias, {n, p.width});
  out.global_indices = nn::from_vector(p.globals, {static_cast<long long>(p.globals.size())});
  auto gather        = [&](const Tensor& x) {
    return nn::transpose(
        nn::gather(nn::transpose(x, {2, 0, 1, 3}), out.indices, 0), {2, 3, 0, 1, 4});
  };
  auto keys = gather(k), values = gather(v);
  out.attention = nn::softmax(nn::sum(nn::expand_dims(q, 3) * keys, {4}) * .5 + out.mask, -1);
  auto local    = nn::sum(nn::expand_dims(out.attention, 4) * values, {3});
  auto gq       = nn::transpose(
      nn::gather(nn::transpose(q, {2, 0, 1, 3}), out.global_indices, 0), {1, 2, 0, 3});
  out.global_attention = nn::softmax(nn::matmul(gq, k, false, true) * .5, -1);
  auto global          = nn::matmul(out.global_attention, v);
  auto slice           = [&](const Tensor& x, int start, int length) {
    return nn::slice(x, {0, 0, start, 0}, {q.size(0), q.size(1), length, q.size(3)});
  };
  out.context = nn::concat(
      {slice(global, 0, p.first),
       slice(local, p.first, p.last - p.first),
       slice(global, p.first, n - p.last)},
      2);
  return out;
}
/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::bigbird::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief logits, patches, embedded, hidden: live source graph endpoint/storage; tensor copies
  /// share native storage.
  Tensor logits;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor patches;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor embedded; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor hidden;   ///< Live source endpoint; owning handle shares storage and derivatives.

  /// @brief attention, states, queries, keys, values, indices, masks, global_attention: live source
  /// graph endpoint/storage; tensor copies share native storage.
  std::vector<Tensor>
      attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      states; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      queries;              ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor> keys; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      values; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      indices; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      masks; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      global_attention; ///< Live source endpoint; owning handle shares storage and derivatives.
};
/// @brief Apply Forward with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Caller owns fitting, objectives and generation. Graph tensors retain storage and live
/// derivatives; lazy parameter creation mutates only borrowed stores. Synchronize calls
/// against concurrent store mutation. Copies of path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param images Live Float32 CPU[B,32,32,3] source-scaled RGB.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);auto
/// y=m::bigbird::Forward(p,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path Forward(nn::ParameterStore& p, const Tensor& images) {
  constexpr int D = 8, T = 17;
  const auto    n = images.size(0);
  Path          out;
  out.patches = nn::reshape(
      nn::transpose(nn::reshape(images, {n, 4, 8, 4, 8, 3}), {0, 1, 3, 2, 4, 5}), {n, 16, 192});
  nn::DenseOptions dense;
  dense.init     = nn::init::normal(.1);
  auto projected = nn::dense(p, "patch", out.patches, D, dense);
  auto cls       = p.get("cls", {1, 1, D}, nn::init::zeros());
  auto position  = p.get("position", {1, T, D}, nn::init::normal(.02));
  auto h         = nn::add(nn::concat({nn::tile(cls, {n, 1, 1}), projected}, 1), position);
  out.embedded   = h;
  auto linear    = [&](const Tensor& x, const std::string& name, int width) {
    return nn::dense(p, name, x, width, dense);
  };
  for (int l = 0; l < 2; ++l) {
    auto name  = "block" + std::to_string(l);
    auto x     = nn::layer_norm(p, name + "/ln1", h, 1e-6);
    auto split = nn::split(linear(x, name + "/qkv", 3 * D), 3, 2);
    auto heads = [&](const Tensor& z) {
      return nn::transpose(nn::reshape(z, {n, T, 2, 4}), {0, 2, 1, 3});
    };
    auto q = heads(split[0]), k = heads(split[1]), v = heads(split[2]);
    auto attended = Context(q, k, v, 50911 + l);
    out.queries.push_back(q);
    out.keys.push_back(k);
    out.values.push_back(v);
    out.attention.push_back(attended.attention);
    out.global_attention.push_back(attended.global_attention);
    out.indices.push_back(attended.indices);
    out.masks.push_back(attended.mask);
    auto mixed = nn::reshape(nn::transpose(attended.context, {0, 2, 1, 3}), {n, T, D});
    h          = nn::add(h, linear(mixed, name + "/out", D));
    auto ff    = nn::layer_norm(p, name + "/ln2", h, 1e-6);
    ff         = linear(nn::gelu(linear(ff, name + "/ff1", 16), true), name + "/ff2", D);
    h          = nn::add(h, ff);
    out.states.push_back(h);
  }
  out.hidden    = nn::layer_norm(p, "final_ln", h, 1e-6);
  auto features = nn::squeeze(nn::slice(out.hidden, {0, 0, 0}, {n, 1, D}), {1});
  out.logits    = nn::dense(p, "head", features, 10, dense);
  return out;
}

} // namespace bigbird

/// @brief performer source-equation components; no optimizer or training driver.
namespace performer {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief A: owning source-compatible native type spelling.
using vit::A;

/// @brief Apply Projection with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Separate mutable projection/statistics storage is caller-owned. Caller owns fitting, objectives
/// and generation. Graph tensors retain storage and live derivatives; lazy parameter creation
/// mutates only borrowed stores. Synchronize calls against concurrent store mutation. Copies of
/// path fields share storage.
/// @param rows Explicit source geometry/control value; defaults match the declaration.
/// @param dimensions Explicit source geometry/control value; defaults match the declaration.
/// @param seed Caller-specified deterministic local mt19937 seed; no global RNG mutation.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);n::ParameterStore state(8);auto
/// y=m::performer::Forward(p,state,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor Projection(int rows, int dimensions, unsigned seed) {
  std::vector<float>               result(rows * dimensions);
  std::mt19937                     rng(seed);
  std::normal_distribution<double> normal(0., 1.);
  for (int start = 0; start < rows; start += dimensions) {
    std::vector<std::vector<double>> basis;
    for (int i = 0; i < std::min(dimensions, rows - start); ++i) {
      std::vector<double> v(dimensions);
      for (auto& x : v)
        x = normal(rng);
      for (const auto& old : basis) {
        double dot = 0;
        for (int j = 0; j < dimensions; ++j)
          dot += v[j] * old[j];
        for (int j = 0; j < dimensions; ++j)
          v[j] -= dot * old[j];
      }
      double norm = 0;
      for (auto x : v)
        norm += x * x;
      norm = std::sqrt(norm);
      component_detail::Require(norm > 1e-10, "nondegenerate orthogonal random basis");
      for (auto& x : v)
        x /= norm;
      basis.push_back(v);
      double radius = 0;
      for (int j = 0; j < dimensions; ++j) {
        double x = normal(rng);
        radius += x * x;
      }
      radius = std::sqrt(radius);
      for (int j = 0; j < dimensions; ++j)
        result[(start + i) * dimensions + j] = float(v[j] * radius);
    }
  }
  return nn::from_vector(result, {rows, dimensions});
}
/// @brief Apply Features with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Separate mutable projection/statistics storage is caller-owned. Caller owns fitting, objectives
/// and generation. Graph tensors retain storage and live derivatives; lazy parameter creation
/// mutates only borrowed stores. Synchronize calls against concurrent store mutation. Copies of
/// path fields share storage.
/// @param x Live CPU source endpoint with the family geometry above.
/// @param omega Live CPU source endpoint with the family geometry above.
/// @param query Live query endpoint or true to use query-feature stabilization, according to the
/// declared type.
/// @param dimensions Explicit source geometry/control value; defaults match the declaration.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);n::ParameterStore state(8);auto
/// y=m::performer::Forward(p,state,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor Features(const Tensor& x, const Tensor& omega, bool query, int dimensions = 4) {
  auto normalized = x * float(std::pow(double(dimensions), -.25));
  auto projected  = nn::matmul(normalized, nn::transpose(omega, {1, 0}));
  auto logphi     = projected - nn::sum(nn::square(normalized), {3}, true) * .5;
  auto shift      = nn::stop_gradient(nn::max(logphi, query ? nn::Axes{3} : nn::Axes{2, 3}, true));
  return nn::exp(logphi - shift) * (1. / std::sqrt(double(omega.size(0))));
}
/// @brief Own ContextPath source graph endpoints or configuration.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Separate mutable projection/statistics storage is caller-owned. Caller owns fitting, objectives
/// and generation. Graph tensors retain storage and live derivatives; lazy parameter creation
/// mutates only borrowed stores. Synchronize calls against concurrent store mutation. Copies of
/// path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);n::ParameterStore state(8);auto
/// y=m::performer::Forward(p,state,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct ContextPath {
  /// @brief context, query_features, key_features, summary, denominator: live source graph
  /// endpoint/storage; tensor copies share native storage.
  Tensor context;        ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor query_features; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor key_features;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor summary;        ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor denominator;    ///< Live source endpoint; owning handle shares storage and derivatives.
};
/// @brief Apply Context with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Separate mutable projection/statistics storage is caller-owned. Caller owns fitting, objectives
/// and generation. Graph tensors retain storage and live derivatives; lazy parameter creation
/// mutates only borrowed stores. Synchronize calls against concurrent store mutation. Copies of
/// path fields share storage.
/// @param q Live CPU source endpoint with the family geometry above.
/// @param k Live CPU source endpoint with the family geometry above.
/// @param v Live CPU source endpoint with the family geometry above.
/// @param omega Live CPU source endpoint with the family geometry above.
/// @param dimensions Explicit source geometry/control value; defaults match the declaration.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);n::ParameterStore state(8);auto
/// y=m::performer::Forward(p,state,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline ContextPath Context(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& omega,
    int           dimensions = 4) {
  ContextPath out;
  out.query_features = Features(q, omega, true, dimensions);
  out.key_features   = Features(k, omega, false, dimensions);
  out.summary        = nn::matmul(out.key_features, v, true, false);
  out.denominator =
      nn::matmul(out.query_features, nn::expand_dims(nn::sum(out.key_features, {2}), 3));
  out.context =
      nn::matmul(out.query_features, out.summary) / nn::maximum(out.denominator, nn::scalar(1e-12));
  return out;
}
/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Separate mutable projection/statistics storage is caller-owned. Caller owns fitting, objectives
/// and generation. Graph tensors retain storage and live derivatives; lazy parameter creation
/// mutates only borrowed stores. Synchronize calls against concurrent store mutation. Copies of
/// path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);n::ParameterStore state(8);auto
/// y=m::performer::Forward(p,state,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief logits, patches, embedded, hidden: live source graph endpoint/storage; tensor copies
  /// share native storage.
  Tensor logits;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor patches;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor embedded; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor hidden;   ///< Live source endpoint; owning handle shares storage and derivatives.

  /// @brief states, queries, keys, values, query_features, key_features, summaries: live source
  /// graph endpoint/storage; tensor copies share native storage.
  std::vector<Tensor>
      states; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      queries;              ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor> keys; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      values; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      query_features; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      key_features; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      summaries; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      denominators; ///< Live source endpoint; owning handle shares storage and derivatives.
};
inline Path
/// @brief Apply Forward with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Separate mutable projection/statistics storage is caller-owned. Caller owns fitting, objectives
/// and generation. Graph tensors retain storage and live derivatives; lazy parameter creation
/// mutates only borrowed stores. Synchronize calls against concurrent store mutation. Copies of
/// path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param projection_state Borrowed random projection storage; source seed/order preserved, no
/// optimizer ownership.
/// @param images Live Float32 CPU[B,32,32,3] source-scaled RGB.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);n::ParameterStore state(8);auto
/// y=m::performer::Forward(p,state,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
Forward(nn::ParameterStore& p, nn::ParameterStore& projection_state, const Tensor& images) {
  constexpr int D = 8, T = 17;
  const auto    n = images.size(0);
  Path          out;
  out.patches = nn::reshape(
      nn::transpose(nn::reshape(images, {n, 4, 8, 4, 8, 3}), {0, 1, 3, 2, 4, 5}), {n, 16, 192});
  nn::DenseOptions dense;
  dense.init     = nn::init::normal(.1);
  auto projected = nn::dense(p, "patch", out.patches, D, dense);
  auto cls       = p.get("cls", {1, 1, D}, nn::init::zeros());
  auto position  = p.get("position", {1, T, D}, nn::init::normal(.02));
  auto h         = nn::add(nn::concat({nn::tile(cls, {n, 1, 1}), projected}, 1), position);
  out.embedded   = h;
  auto linear    = [&](const Tensor& x, const std::string& name, int width) {
    return nn::dense(p, name, x, width, dense);
  };
  for (int l = 0; l < 2; ++l) {
    auto name  = "block" + std::to_string(l);
    auto x     = nn::layer_norm(p, name + "/ln1", h, 1e-6);
    auto split = nn::split(linear(x, name + "/qkv", 3 * D), 3, 2);
    auto heads = [&](const Tensor& z) {
      return nn::transpose(nn::reshape(z, {n, T, 2, 4}), {0, 2, 1, 3});
    };
    auto q = heads(split[0]), k = heads(split[1]), v = heads(split[2]);
    auto omega = projection_state.get(
        name + "/omega", {16, 4}, [l](const nn::Shape&, nn::DType, nn::Generator&) {
          return Projection(16, 4, 50501 + l);
        });
    auto favored = Context(q, k, v, omega);
    out.queries.push_back(q);
    out.keys.push_back(k);
    out.values.push_back(v);
    out.query_features.push_back(favored.query_features);
    out.key_features.push_back(favored.key_features);
    out.summaries.push_back(favored.summary);
    out.denominators.push_back(favored.denominator);
    auto mixed = nn::reshape(nn::transpose(favored.context, {0, 2, 1, 3}), {n, T, D});
    h          = nn::add(h, linear(mixed, name + "/out", D));
    auto ff    = nn::layer_norm(p, name + "/ln2", h, 1e-6);
    ff         = linear(nn::gelu(linear(ff, name + "/ff1", 16), true), name + "/ff2", D);
    h          = nn::add(h, ff);
    out.states.push_back(h);
  }
  out.hidden    = nn::layer_norm(p, "final_ln", h, 1e-6);
  auto features = nn::squeeze(nn::slice(out.hidden, {0, 0, 0}, {n, 1, D}), {1});
  out.logits    = nn::dense(p, "head", features, 10, dense);
  return out;
}

} // namespace performer

/// @brief reformer source-equation components; no optimizer or training driver.
namespace reformer {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief A: owning source-compatible native type spelling.
using vit::A;

/// @brief Apply Rotation with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Separate mutable projection/statistics storage is caller-owned. Caller owns fitting, objectives
/// and generation. Graph tensors retain storage and live derivatives; lazy parameter creation
/// mutates only borrowed stores. Synchronize calls against concurrent store mutation. Copies of
/// path fields share storage.
/// @param d Explicit source geometry/control value; defaults match the declaration.
/// @param half Explicit source geometry/control value; defaults match the declaration.
/// @param seed Caller-specified deterministic local mt19937 seed; no global RNG mutation.
/// @return Live native graph endpoints or owned source-control values in the documented family
/// geometry.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);n::ParameterStore state(8);auto
/// y=m::reformer::Forward(p,state,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Tensor Rotation(int d, int half, unsigned seed) {
  std::mt19937                    rng(seed);
  std::normal_distribution<float> normal;
  std::vector<float>              x(d * half);
  for (auto& v : x)
    v = normal(rng);
  return nn::from_vector(x, {d, half});
}
/// @brief Own ContextPath source graph endpoints or configuration.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Separate mutable projection/statistics storage is caller-owned. Caller owns fitting, objectives
/// and generation. Graph tensors retain storage and live derivatives; lazy parameter creation
/// mutates only borrowed stores. Synchronize calls against concurrent store mutation. Copies of
/// path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);n::ParameterStore state(8);auto
/// y=m::reformer::Forward(p,state,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct ContextPath {
  /// @brief context, attention, buckets, order: live source graph endpoint/storage; tensor copies
  /// share native storage.
  Tensor context;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor buckets;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor order;     ///< Live source endpoint; owning handle shares storage and derivatives.
};
// Hashing/sorting is intentionally discrete; gathered QK and V retain their
// differentiable paths, matching the source's conditional LSH derivatives.
/// @brief Apply Context with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Separate mutable projection/statistics storage is caller-owned. Caller owns fitting, objectives
/// and generation. Graph tensors retain storage and live derivatives; lazy parameter creation
/// mutates only borrowed stores. Synchronize calls against concurrent store mutation. Copies of
/// path fields share storage.
/// @param qk Live CPU source endpoint with the family geometry above.
/// @param v Live CPU source endpoint with the family geometry above.
/// @param rotation Live CPU source endpoint with the family geometry above.
/// @param buckets Explicit source geometry/control value; defaults match the declaration.
/// @param chunk Explicit source geometry/control value; defaults match the declaration.
/// @param frozen_buckets Live CPU source endpoint with the family geometry above.
/// @param frozen_order Live CPU source endpoint with the family geometry above.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);n::ParameterStore state(8);auto
/// y=m::reformer::Forward(p,state,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline ContextPath Context(
    const Tensor& qk,
    const Tensor& v,
    const Tensor& rotation,
    int           buckets        = 4,
    int           chunk          = 4,
    const Tensor& frozen_buckets = {},
    const Tensor& frozen_order   = {}) {
  int n = qk.size(2), bh = qk.size(0) * qk.size(1), padded = ((n + chunk - 1) / chunk) * chunk,
      groups = padded / chunk;
  component_detail::Require(
      n > chunk && buckets >= 2 && buckets % 2 == 0 && chunk > 0, "valid Reformer dimensions");
  auto q         = nn::pad(qk, {{0, 0}, {0, 0}, {0, padded - n}, {0, 0}}),
       values    = nn::pad(v, {{0, 0}, {0, 0}, {0, padded - n}, {0, 0}});
  auto projected = nn::matmul(q.detach(), rotation.detach());
  auto hash      = nn::argmax(nn::concat({projected, -projected}, 3), 3).to_vector<std::int64_t>();
  std::vector<int> order(bh * padded), h(bh * padded), flat(bh * padded), inverse(bh * padded);
  for (int b = 0; b < bh; ++b) {
    std::vector<int> ids(padded);
    std::iota(ids.begin(), ids.end(), 0);
    for (int i = 0; i < padded; ++i)
      h[b * padded + i] = i < n ? hash[b * padded + i] : buckets;
    std::stable_sort(ids.begin(), ids.end(), [&](int a, int c) {
      return h[b * padded + a] < h[b * padded + c];
    });
    for (int i = 0; i < padded; ++i) {
      order[b * padded + i]        = ids[i];
      flat[b * padded + i]         = b * padded + ids[i];
      inverse[b * padded + ids[i]] = b * padded + i;
    }
  }
  if (frozen_buckets.defined())
    h = frozen_buckets.to_vector<int>();
  if (frozen_order.defined()) {
    order = frozen_order.to_vector<int>();
    for (int b = 0; b < bh; ++b)
      for (int i = 0; i < padded; ++i) {
        flat[b * padded + i]                        = b * padded + order[b * padded + i];
        inverse[b * padded + order[b * padded + i]] = b * padded + i;
      }
  }
  auto gather = [&](const Tensor& x) {
    return nn::reshape(
        nn::gather(nn::reshape(x, {bh * padded, 4}), nn::from_vector(flat, {bh * padded}), 0),
        {bh, groups, chunk, 4});
  };
  auto query = gather(q), val = gather(values);
  auto key =
      query / nn::sqrt(nn::maximum(nn::sum(nn::square(query), {3}, true), nn::scalar(1e-12)));
  auto previous = [&](const Tensor& x) {
    return nn::concat(
        {nn::slice(x, {0, groups - 1, 0, 0}, {bh, 1, chunk, 4}),
         nn::slice(x, {0, 0, 0, 0}, {bh, groups - 1, chunk, 4})},
        1);
  };
  auto keys = nn::concat({previous(key), key}, 2), vals = nn::concat({previous(val), val}, 2);
  std::vector<float> bias(bh * groups * chunk * 2 * chunk), active(bias.size());
  for (int b = 0; b < bh; ++b)
    for (int g = 0; g < groups; ++g)
      for (int i = 0; i < chunk; ++i)
        for (int j = 0; j < 2 * chunk; ++j) {
          int qi        = order[b * padded + g * chunk + i],
              kg        = j < chunk ? (g + groups - 1) % groups : g,
              kj        = order[b * padded + kg * chunk + j % chunk];
          int  index    = ((b * groups + g) * chunk + i) * 2 * chunk + j;
          bool allowed  = h[b * padded + qi] == h[b * padded + kj] && kj < n;
          bool self     = qi == kj;
          active[index] = allowed && !self ? 1.f : 0.f;
          bias[index]   = !allowed ? -1e9f : self ? -1e5f : 0.f;
        }
  auto logits = nn::matmul(query, keys, false, true) * .5;
  auto scores = logits * nn::from_vector(active, {bh, groups, chunk, 2 * chunk}) +
                nn::from_vector(bias, {bh, groups, chunk, 2 * chunk});
  ContextPath out;
  out.attention =
      nn::reshape(nn::softmax(scores, -1), {qk.size(0), qk.size(1), groups, chunk, 2 * chunk});
  auto context  = nn::reshape(nn::matmul(nn::softmax(scores, -1), vals), {bh * padded, 4});
  auto unsorted = nn::reshape(
      nn::gather(context, nn::from_vector(inverse, {bh * padded}), 0),
      {qk.size(0), qk.size(1), padded, 4});
  out.context = nn::slice(unsorted, {0, 0, 0, 0}, {qk.size(0), qk.size(1), n, 4});
  out.buckets = nn::from_vector(h, {qk.size(0), qk.size(1), padded});
  out.order   = nn::from_vector(order, {qk.size(0), qk.size(1), padded});
  return out;
}
/// @brief Own Path source graph endpoints or configuration.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Separate mutable projection/statistics storage is caller-owned. Caller owns fitting, objectives
/// and generation. Graph tensors retain storage and live derivatives; lazy parameter creation
/// mutates only borrowed stores. Synchronize calls against concurrent store mutation. Copies of
/// path fields share storage.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);n::ParameterStore state(8);auto
/// y=m::reformer::Forward(p,state,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
struct Path {
  /// @brief logits, patches, embedded, hidden: live source graph endpoint/storage; tensor copies
  /// share native storage.
  Tensor logits;   ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor patches;  ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor embedded; ///< Live source endpoint; owning handle shares storage and derivatives.
  Tensor hidden;   ///< Live source endpoint; owning handle shares storage and derivatives.

  /// @brief states, queries, values, attention, buckets, order: live source graph endpoint/storage;
  /// tensor copies share native storage.
  std::vector<Tensor>
      states; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      queries; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      values; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      attention; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      buckets; ///< Live source endpoint; owning handle shares storage and derivatives.
  std::vector<Tensor>
      order; ///< Live source endpoint; owning handle shares storage and derivatives.
};
/// @brief Apply Forward with the original source equations and parameter names.
///
/// Float32 CPU[B,32,32,3] NHWC images in the source scale;exact patch/head geometry is retained.
/// Separate mutable projection/statistics storage is caller-owned. Caller owns fitting, objectives
/// and generation. Graph tensors retain storage and live derivatives; lazy parameter creation
/// mutates only borrowed stores. Synchronize calls against concurrent store mutation. Copies of
/// path fields share storage.
/// @param p Borrowed Float32 CPU parameter store; missing source-named leaves initialize in source
/// order.
/// @param projection_state Borrowed random projection storage; source seed/order preserved, no
/// optimizer ownership.
/// @param images Live Float32 CPU[B,32,32,3] source-scaled RGB.
/// @param routing Explicit source geometry/control value; defaults match the declaration.
/// @throws std::invalid_argument For explicit argument guards or incompatible native operation
/// geometry.
/// @throws std::logic_error For unsupported backend/dtype primitive execution.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(7);n::ParameterStore state(8);auto
/// y=m::reformer::Forward(p,state,n::zeros({1,32,32,3}));
/// @endcode
/// @par References
/// - Source model equations and parameter names preserved from the frozen example
///   headers in examples/nn/common; complete editable experiments are in
///   docs/mepml/nn/.
/// - Vaswani et al.(2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see datamunge::nn::ParameterStore, datamunge::nn::Module::plot()
inline Path Forward(
    nn::ParameterStore& p,
    nn::ParameterStore& projection_state,
    const Tensor&       images,
    const A&            routing = {}) {
  constexpr int D = 8, T = 17;
  const auto    n = images.size(0);
  Path          out;
  out.patches = nn::reshape(
      nn::transpose(nn::reshape(images, {n, 4, 8, 4, 8, 3}), {0, 1, 3, 2, 4, 5}), {n, 16, 192});
  nn::DenseOptions dense;
  dense.init     = nn::init::normal(.1);
  auto projected = nn::dense(p, "patch", out.patches, D, dense);
  auto cls       = p.get("cls", {1, 1, D}, nn::init::zeros());
  auto position  = p.get("position", {1, T, D}, nn::init::normal(.02));
  auto h         = nn::add(nn::concat({nn::tile(cls, {n, 1, 1}), projected}, 1), position);
  out.embedded   = h;
  auto linear    = [&](const Tensor& x, const std::string& name, int width) {
    return nn::dense(p, name, x, width, dense);
  };
  for (int l = 0; l < 2; ++l) {
    auto name  = "block" + std::to_string(l);
    auto x     = nn::layer_norm(p, name + "/ln1", h, 1e-6);
    auto split = nn::split(linear(x, name + "/qkv", 2 * D), 2, 2);
    auto heads = [&](const Tensor& z) {
      return nn::transpose(nn::reshape(z, {n, T, 2, 4}), {0, 2, 1, 3});
    };
    auto q = heads(split[0]), v = heads(split[1]);
    auto rotation = projection_state.get(
        name + "/rotation", {4, 2}, [l](const nn::Shape&, nn::DType, nn::Generator&) {
          return Rotation(4, 2, 51111 + l);
        });
    auto suffix  = std::to_string(l);
    auto favored = Context(
        q,
        v,
        rotation,
        4,
        4,
        routing.count("buckets" + suffix) ? routing.at("buckets" + suffix) : Tensor{},
        routing.count("order" + suffix) ? routing.at("order" + suffix) : Tensor{});
    out.queries.push_back(q);
    out.values.push_back(v);
    out.attention.push_back(favored.attention);
    out.buckets.push_back(favored.buckets);
    out.order.push_back(favored.order);
    auto mixed = nn::reshape(nn::transpose(favored.context, {0, 2, 1, 3}), {n, T, D});
    h          = nn::add(h, linear(mixed, name + "/out", D));
    auto ff    = nn::layer_norm(p, name + "/ln2", h, 1e-6);
    ff         = linear(nn::gelu(linear(ff, name + "/ff1", 16), true), name + "/ff2", D);
    h          = nn::add(h, ff);
    out.states.push_back(h);
  }
  out.hidden    = nn::layer_norm(p, "final_ln", h, 1e-6);
  auto features = nn::squeeze(nn::slice(out.hidden, {0, 0, 0}, {n, 1, D}), {1});
  out.logits    = nn::dense(p, "head", features, 10, dense);
  return out;
}

} // namespace reformer

} // namespace datamunge::nn::models
