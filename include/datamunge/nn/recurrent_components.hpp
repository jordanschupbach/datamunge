#pragma once
/// @file recurrent_components.hpp
/// @brief Source-faithful masked recurrent paths, live gate diagnostics and explicit gradient
/// policies.
#include <cmath>
#include <cstdint>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/conv.hpp>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
#include <datamunge/nn/params.hpp>
#include <limits>
#include <string>
#include <vector>
namespace datamunge::nn::models {
/// @brief Internal CPU, geometry and source-initializer helpers; no fitting orchestration.
namespace recurrent_component_detail {
/// @brief Require a nonempty finite floating CPU tensor before shape or storage access.
/// @param x Candidate tensor, including scalar gradients.
/// @throws std::invalid_argument Undefined, nonfloating, non-CPU, empty or nonfinite input.
inline void Floating(const Tensor& x) {
  if (!x.defined() || x.device().backend != compute::Backend::CPU ||
      (x.dtype() != DType::Float32 && x.dtype() != DType::Float64) || x.numel() < 1)
    throw std::invalid_argument("recurrent finite floating CPU tensor required");
  for (auto axis : x.shape())
    if (axis <= 0)
      throw std::invalid_argument("recurrent positive axes");
  for (double v : x.to_vector<double>())
    if (!std::isfinite(v))
      throw std::invalid_argument("recurrent finite tensor required");
}
/// @brief Validate nonnegative integer CPU lengths aligned with a sequence batch.
/// @param lengths Int32/64 CPU[N] lengths in[0,steps].
/// @param rows Positive batch size.
/// @param steps Maximum valid length.
/// @throws std::invalid_argument Invalid geometry, dtype or length.
inline void Lengths(const Tensor& lengths, std::int64_t rows, std::int64_t steps) {
  if (!lengths.defined() || lengths.device().backend != compute::Backend::CPU ||
      lengths.dim() != 1 || lengths.size(0) != rows ||
      (lengths.dtype() != DType::Int32 && lengths.dtype() != DType::Int64))
    throw std::invalid_argument("recurrent integer length vector required");
  for (auto length : cast(lengths, DType::Int64).to_vector<std::int64_t>())
    if (length < 0 || length > steps)
      throw std::invalid_argument("recurrent length outside sequence");
}
/// @brief Validate one fixed-source sequence and all explicit initial states before creating
/// leaves.
/// @param p Caller store, same dtype as input.
/// @param input Floating[N,12,2], or spatial[N,4,4,4,1].
/// @param lengths Integer[N], in[0,T].
/// @param initial Initial hidden/cell tensors, matching dtype and geometry.
/// @param count Required number of initial tensors.
/// @param spatial True selects the4x4 moving-dot preset; false selects the12-unit vector preset.
/// @throws std::invalid_argument Invalid geometry, dtype or lengths.
inline void Sequence(
    const ParameterStore&      p,
    const Tensor&              input,
    const Tensor&              lengths,
    const std::vector<Tensor>& initial,
    std::size_t                count,
    bool                       spatial = false) {
  Floating(input);
  if (input.dim() != (spatial ? 5 : 3))
    throw std::invalid_argument("recurrent source input rank");
  if (input.dtype() != p.dtype() || initial.size() != count)
    throw std::invalid_argument("recurrent store/initial dtype or count");
  Shape shape = spatial ? Shape{input.size(0), 4, 4, 4, 1} : Shape{input.size(0), 12, 2};
  if (input.shape() != shape)
    throw std::invalid_argument("recurrent source input shape");
  Lengths(lengths, input.size(0), spatial ? 4 : 12);
  Shape state = spatial ? Shape{input.size(0), 4, 4, 2} : Shape{input.size(0), 12};
  for (const auto& x : initial) {
    Floating(x);
    if (x.shape() != state || x.dtype() != input.dtype())
      throw std::invalid_argument("recurrent initial state shape/dtype");
  }
}
/// @brief Create/reuse one source leaf and validate its consumed dtype/device/values.
/// @param p Caller-owned mutable store; existing leaves are not overwritten.
/// @param name Parameter name.
/// @param shape Required geometry.
/// @param initializer Source initializer, invoked only for a missing name.
/// @return Owning live parameter handle.
/// @throws std::invalid_argument Invalid leaf geometry, dtype, device or finite values.
inline Tensor Leaf(
    ParameterStore&    p,
    const std::string& name,
    const Shape&       shape,
    const Initializer& initializer) {
  auto value = p.get(name, shape, initializer);
  Floating(value);
  if (value.dtype() != p.dtype())
    throw std::invalid_argument("recurrent leaf dtype");
  return value;
}
/// @brief Validate generic vector-cell input, source dimensions and explicit hidden/cell states.
/// @param p Caller store with matching floating dtype.
/// @param x Finite floating CPU[N,incoming].
/// @param incoming Input width in[1,1024].
/// @param units Hidden width in[1,1024].
/// @param hidden Finite matching CPU[N,units].
/// @param cell Optional matching CPU[N,units], undefined for a GRU.
/// @throws std::invalid_argument Invalid dimensions, dtype or finite values.
inline void Cell(
    const ParameterStore& p,
    const Tensor&         x,
    std::int64_t          incoming,
    std::int64_t          units,
    const Tensor&         hidden,
    const Tensor&         cell = {}) {
  Floating(x);
  Floating(hidden);
  if (incoming < 1 || incoming > 1024 || units < 1 || units > 1024 || x.dim() != 2 ||
      x.size(1) != incoming || hidden.shape() != Shape{x.size(0), units} ||
      x.dtype() != p.dtype() || hidden.dtype() != x.dtype())
    throw std::invalid_argument("recurrent cell dimensions/dtype");
  if (cell.defined()) {
    Floating(cell);
    if (cell.shape() != hidden.shape() || cell.dtype() != x.dtype())
      throw std::invalid_argument("recurrent cell memory shape/dtype");
  }
}
/// @brief Construct the source initializer: zeros except the forget-gate block.
/// @param units Positive hidden width.
/// @param forget_bias Finite forget bias, default1.
/// @return Owning initializer closure with no RNG draws.
inline Initializer LstmBias(std::int64_t units, double forget_bias = 1.) {
  return [units, forget_bias](const Shape&, DType dtype, Generator&) {
    return concat(
        {zeros({units}, dtype), full({units}, forget_bias, dtype), zeros({2 * units}, dtype)}, 0);
  };
}
/// @brief Structural source Float32[N,1] valid-step mask.
/// @param lengths Previously validated integer CPU[N] lengths.
/// @param t Nonnegative structural time.
/// @return Detached Float32[N,1] zero/one tensor.
inline Tensor StepMask(const Tensor& lengths, int t) {
  return expand_dims(cast(greater(lengths, double(t)), DType::Float32), 1);
}
/// @brief Source boolean[N,1] valid-step mask.
/// @param lengths Previously validated integer CPU[N] lengths.
/// @param t Structural time.
/// @return Detached boolean[N,1] tensor.
inline Tensor StepValid(const Tensor& lengths, int t) {
  return expand_dims(greater(lengths, double(t)), 1);
}
/// @brief Preserve source arithmetic masking, mask*next+(1-mask)*previous.
/// @param mask Source Float32 valid-step mask, broadcastable to states.
/// @param next Candidate live state.
/// @param previous Previous live state.
/// @return Live arithmetic blend with native derivative semantics.
inline Tensor Blend(const Tensor& mask, const Tensor& next, const Tensor& previous) {
  return add(mul(mask, next), mul(sub(1., mask), previous));
}
/// @brief Source where-mask diagnostic tensors, zeroing invalid timesteps.
/// @param valid Structural broadcastable boolean mask.
/// @param x Live diagnostic tensor.
/// @return Live selected diagnostics or zeros.
inline Tensor MaskedDiagnostics(const Tensor& valid, const Tensor& x) {
  return where(valid, x, zeros_like(x));
}
/// @brief Source normal(.1) dense head with zero bias, reused after initialization.
/// @param p Caller store, same dtype as x.
/// @param name Head prefix.
/// @param x Live finite CPU[N,incoming].
/// @param incoming Input width.
/// @param outgoing Output width.
/// @return Live affine logits; no fitting or optimizer.
inline Tensor Head(
    ParameterStore&    p,
    const std::string& name,
    const Tensor&      x,
    std::int64_t       incoming,
    std::int64_t       outgoing) {
  const Tensor weight = Leaf(p, name + "/W", {incoming, outgoing}, init::normal(.1));
  const Tensor bias   = Leaf(p, name + "/b", {outgoing}, init::zeros());
  return bias_add(matmul(x, weight), bias);
}
} // namespace recurrent_component_detail
/// @brief Own live recurrent cell endpoints without a model or optimizer.
///
/// Hidden/cell/gate tensors own their graphs after caller/store collection. GRU cell and all
/// non-layer-normalized diagnostic fields are undefined. No detach, mutation or cached state.
///
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;m::RecurrentGateOutput out;
/// @endcode
/// @see MaskedGatedSequence(), RecurrentGradientControl()
/// @par References
/// - Hochreiter and Schmidhuber (1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 . Exact compact variants and their distinct gate
/// conventions are specified by the associated presets.
struct RecurrentGateOutput {
  Tensor hidden;                 ///< Live CPU[N,H] next hidden state.
  Tensor cell;                   ///< Live CPU[N,H] next memory, undefined for GRU.
  Tensor gates;                  ///< Live CPU[N,4H] i/f/candidate/o or[N,3H] z/r/candidate.
  Tensor normalized_gate_affine; ///< Live CPU[N,4H] pre-gain per-gate normalization, layer-norm
                                 ///< LSTM only.
  Tensor normalized_cell; ///< Live CPU[N,H] gained/shifted cell readout, layer-norm LSTM only.
};
/// @brief Select the source compact gated-cell family.
///
/// Discrete architecture choice; never optimized. The source preset is12 steps, two input
/// features,12 hidden units and a two-class head.
///
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto cell=m::MaskedGatedCell::Gru;
/// @endcode
/// @see MaskedGatedSequence(), RecurrentGradientControl()
/// @par References
/// - Hochreiter and Schmidhuber (1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 . Exact compact variants and their distinct gate
/// conventions are specified by the associated presets.
enum class MaskedGatedCell {
  Lstm,     ///< Standard source LSTM with unit forget bias.
  Gru,      ///< Reset-before GRU with zero update bias.
  Peephole, ///< Old-cell input/forget and new-cell output peepholes.
  LayerNorm ///< Per-example per-gate normalization and separate normalized cell readout.
};
/// @brief Own live masked-sequence outputs as flat tensor handles.
///
/// Every defined field owns its graph/storage beyond record/store collection. Vector presets have
/// states[N,13,12]; spatial states are[N,5,4,4,2]. Optional fields are undefined for families that
/// do not produce them. No model, optimizer or fit/sampling loop is retained.
///
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;m::MaskedRecurrentOutput out;
/// @endcode
/// @see MaskedGatedSequence(), RecurrentGradientControl()
/// @par References
/// - Hochreiter and Schmidhuber (1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 . Exact compact variants and their distinct gate
/// conventions are specified by the associated presets.
struct MaskedRecurrentOutput {
  Tensor logits;      ///< Live[N,2] logits, or spatial[N,4,4,1] logits.
  Tensor states;      ///< Live hidden history including explicit initial state, in traversal order.
  Tensor cell_states; ///< Live LSTM memory history including initial memory; otherwise undefined.
  Tensor final_state; ///< Live final hidden state; bidirectional concatenates forward/backward
                      ///< to[N,24].
  Tensor final_cell;  ///< Live final LSTM memory, otherwise undefined.
  Tensor gates;       ///< Live per-time valid-only gate diagnostics, otherwise undefined.
  Tensor normalized_gate_affine; ///< Layer-norm LSTM valid-only gate normalization, otherwise
                                 ///< undefined.
  Tensor normalized_cell;  ///< Layer-norm LSTM valid-only gained cell readout, otherwise undefined.
  Tensor backward_states;  ///< Bidirectional reverse-traversal hidden history, otherwise undefined.
  Tensor backward_aligned; ///< Bidirectional backward states aligned to original time, otherwise
                           ///< undefined.
  Tensor final_backward;   ///< Bidirectional final reverse hidden state, otherwise undefined.
};
/// @brief Internal owning-record finite-value validation after source tensor arithmetic.
namespace recurrent_component_detail {
/// @brief Validate all defined live cell endpoints before returning ownership.
/// @param value Owning cell record; optional diagnostics may be undefined.
/// @return Unchanged owning record, with all defined fields finite floating CPU tensors.
/// @throws std::invalid_argument Nonfinite computed arithmetic or invalid endpoint storage.
inline RecurrentGateOutput Checked(RecurrentGateOutput value) {
  for (const auto& field :
       {value.hidden, value.cell, value.gates, value.normalized_gate_affine, value.normalized_cell})
    if (field.defined())
      Floating(field);
  return value;
}
/// @brief Validate all defined live sequence endpoints before returning ownership.
/// @param value Owning sequence record; unused family fields may be undefined.
/// @return Unchanged owning record and autograd paths.
/// @throws std::invalid_argument Nonfinite computed arithmetic or invalid endpoint storage.
inline MaskedRecurrentOutput Checked(MaskedRecurrentOutput value) {
  for (const auto& field :
       {value.logits,
        value.states,
        value.cell_states,
        value.final_state,
        value.final_cell,
        value.gates,
        value.normalized_gate_affine,
        value.normalized_cell,
        value.backward_states,
        value.backward_aligned,
        value.final_backward})
    if (field.defined())
      Floating(field);
  return value;
}
} // namespace recurrent_component_detail
/// @brief Normalize one axis with the source-rounded population variance and epsilon.
///
/// No gain/bias, batch coupling, RNG or mutation; O(elements) work and applicable higher paths.
///
/// @param x Finite Float32/64 CPU tensor, positive axes and rank>=1.
/// @param axis Axis in[-rank,rank).
/// @return Owning live centered*rsqrt(mean(centered²)+Double(Float32(1e-5))) tensor.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype, domain or parameter shape.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto y=m::RecurrentLayerNormalize(n::zeros({2,3}),1);
/// @endcode
/// @par References
/// - Ba, Kiros and Hinton (2016), Layer Normalization, https://arxiv.org/abs/1607.06450. Source
/// gate-wise normalization and normalized readout are retained.
/// @see MaskedGatedSequence(), RecurrentGradientControl()
inline Tensor RecurrentLayerNormalize(const Tensor& x, std::int64_t axis) {
  recurrent_component_detail::Floating(x);
  if (x.dim() < 1 || axis < -x.dim() || axis >= x.dim())
    throw std::invalid_argument("layer normalization axis");
  const Tensor centered = sub(x, mean(x, {axis}, true));
  const Tensor variance = mean(square(centered), {axis}, true);
  return mul(centered, rsqrt(add(variance, double(1e-5f))));
}
/// @brief Evaluate one source LSTM step with live diagnostics.
///
/// No fitting, optimizer, sampling or hidden no-grad. Computed nonfinite endpoints reject. Warm
/// valid stores/RNG are unchanged; serialize creation/writes. Source initialization precision and
/// binary grouping are retained. Creation failures may leave earlier leaves present. Work
/// O(N*H*(incoming+H)); applicable first/higher paths retained.
///
/// @param p Caller-owned floating store; initialized source names are reused, missing leaves
/// created in source order.
/// @param name Shared parameter prefix; same name reuses weights.
/// @param x Finite matching floating CPU[N,incoming], positive N.
/// @param in Input feature width in[1,1024].
/// @param units Hidden width in[1,1024].
/// @param state Finite same-dtype hidden CPU[N,H] and cell CPU[N,H].
/// @param forget_bias Finite initial forget bias, default1; existing bias leaf is unchanged.
/// @return Owning live next states and gate/readout diagnostics.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype, domain or parameter shape.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(27003);auto
/// x=n::zeros({1,2}),h=n::zeros({1,12});auto y=m::LstmGateStep(p,"cell",x,2,12,n::LstmState{h,h});
/// @endcode
/// @par References
/// - Hochreiter and Schmidhuber (1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735. This source preset includes the later forget gate
/// and unit forget bias.
/// @see MaskedGatedSequence(), RecurrentGradientControl()
inline RecurrentGateOutput LstmGateStep(
    ParameterStore&    p,
    const std::string& name,
    const Tensor&      x,
    std::int64_t       in,
    std::int64_t       units,
    const LstmState&   state,
    double             forget_bias = 1.0) {
  recurrent_component_detail::Cell(p, x, in, units, state.h, state.c);
  if (!state.c.defined())
    throw std::invalid_argument("LSTM requires explicit cell memory");
  if (!std::isfinite(forget_bias))
    throw std::invalid_argument("finite forget bias");
  const Tensor wx = recurrent_component_detail::Leaf(
      p, name + "/Wx", {in, 4 * units}, init::uniform(std::sqrt(6.0 / double(in + 4 * units))));
  const Tensor wh = recurrent_component_detail::Leaf(
      p, name + "/Wh", {units, 4 * units}, init::uniform(std::sqrt(6.0 / double(5 * units))));
  const Tensor b = recurrent_component_detail::Leaf(
      p, name + "/b", {4 * units}, recurrent_component_detail::LstmBias(units, forget_bias));
  const auto   parts = split(bias_add(add(matmul(x, wx), matmul(state.h, wh)), b), 4, 1);
  const Tensor i = sigmoid(parts[0]), f = sigmoid(parts[1]), g = tanh(parts[2]),
               o = sigmoid(parts[3]);
  const Tensor c = add(mul(f, state.c), mul(i, g));
  return recurrent_component_detail::Checked(
      RecurrentGateOutput{mul(o, tanh(c)), c, concat({i, f, g, o}, 1), {}, {}});
}
/// @brief Evaluate one source reset-before GRU step with live diagnostics.
///
/// No fitting, optimizer, sampling or hidden no-grad. Computed nonfinite endpoints reject. Warm
/// valid stores/RNG are unchanged; serialize creation/writes. Source initialization precision and
/// binary grouping are retained. Creation failures may leave earlier leaves present. Work
/// O(N*H*(incoming+H)); applicable first/higher paths retained.
///
/// @param p Caller-owned floating store; initialized source names are reused, missing leaves
/// created in source order.
/// @param name Shared parameter prefix; same name reuses weights.
/// @param x Finite matching floating CPU[N,incoming], positive N.
/// @param in Input feature width in[1,1024].
/// @param u Hidden width in[1,1024].
/// @param h Finite same-dtype hidden CPU[N,H].
/// @param update_bias Finite initial update bias, default0; existing bias leaf is unchanged.
/// @return Owning live next states and gate/readout diagnostics.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype, domain or parameter shape.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(27003);auto
/// x=n::zeros({1,2}),h=n::zeros({1,12});auto y=m::GruGateStep(p,"cell",x,2,12,h);
/// @endcode
/// @par References
/// - Cho et al. (2014), Learning Phrase Representations using RNN Encoder-Decoder for Statistical
/// Machine Translation, https://arxiv.org/abs/1406.1078. This source uses reset-before candidate
/// arithmetic.
/// @see MaskedGatedSequence(), RecurrentGradientControl()
inline RecurrentGateOutput GruGateStep(
    ParameterStore&    p,
    const std::string& name,
    const Tensor&      x,
    std::int64_t       in,
    std::int64_t       u,
    const Tensor&      h,
    double             update_bias = 0.0) {
  recurrent_component_detail::Cell(p, x, in, u, h);
  if (!std::isfinite(update_bias))
    throw std::invalid_argument("finite update bias");
  const Tensor Wx = recurrent_component_detail::Leaf(
      p, name + "/Wx_zr", {in, 2 * u}, init::uniform(std::sqrt(6.0 / double(in + 2 * u))));
  const Tensor Wh = recurrent_component_detail::Leaf(
      p, name + "/Wh_zr", {u, 2 * u}, init::uniform(std::sqrt(6.0 / double(3 * u))));
  const Tensor b = recurrent_component_detail::Leaf(
      p, name + "/b_zr", {2 * u}, [u, update_bias](const Shape&, DType d, Generator&) {
        return concat({full({u}, update_bias, d), zeros({u}, d)}, 0);
      });
  const Tensor Wxn = recurrent_component_detail::Leaf(
      p, name + "/Wx_n", {in, u}, init::uniform(std::sqrt(6.0 / double(in + u))));
  const Tensor Whn = recurrent_component_detail::Leaf(
      p, name + "/Wh_n", {u, u}, init::uniform(std::sqrt(6.0 / double(2 * u))));
  const Tensor bn = recurrent_component_detail::Leaf(p, name + "/b_n", {u}, init::zeros());
  const auto   zr = split(sigmoid(bias_add(add(matmul(x, Wx), matmul(h, Wh)), b)), 2, 1);
  const Tensor n  = tanh(bias_add(add(matmul(x, Wxn), matmul(mul(zr[1], h), Whn)), bn));
  return recurrent_component_detail::Checked(
      RecurrentGateOutput{
          add(mul(sub(1.0, zr[0]), n), mul(zr[0], h)),
          Tensor(),
          concat({zr[0], zr[1], n}, 1),
          {},
          {}});
}
/// @brief Evaluate one source peephole LSTM step with live diagnostics.
///
/// No fitting, optimizer, sampling or hidden no-grad. Computed nonfinite endpoints reject. Warm
/// valid stores/RNG are unchanged; serialize creation/writes. Source initialization precision and
/// binary grouping are retained. Creation failures may leave earlier leaves present. Work
/// O(N*H*(incoming+H)); applicable first/higher paths retained.
///
/// @param p Caller-owned floating store; initialized source names are reused, missing leaves
/// created in source order.
/// @param name Shared parameter prefix; same name reuses weights.
/// @param x Finite matching floating CPU[N,incoming], positive N.
/// @param incoming Input feature width in[1,1024].
/// @param units Hidden width in[1,1024].
/// @param state Finite same-dtype hidden CPU[N,H] and cell CPU[N,H].
/// @return Owning live next states and gate/readout diagnostics.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype, domain or parameter shape.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(27003);auto
/// x=n::zeros({1,2}),h=n::zeros({1,12});auto
/// y=m::PeepholeGateStep(p,"cell",x,2,12,n::LstmState{h,h});
/// @endcode
/// @par References
/// - Gers and Schmidhuber (2000), Recurrent Nets that Time and Count,
/// https://sferics.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf. Old-cell input/forget and new-cell
/// output peepholes follow the repository source.
/// @see MaskedGatedSequence(), RecurrentGradientControl()
inline RecurrentGateOutput PeepholeGateStep(
    ParameterStore&    p,
    const std::string& name,
    const Tensor&      x,
    std::int64_t       incoming,
    std::int64_t       units,
    const LstmState&   state) {
  recurrent_component_detail::Cell(p, x, incoming, units, state.h, state.c);
  if (!state.c.defined())
    throw std::invalid_argument("LSTM requires explicit cell memory");
  const Tensor wx = recurrent_component_detail::Leaf(
      p,
      name + "/Wx",
      {incoming, 4 * units},
      init::uniform(std::sqrt(6.f / float(incoming + 4 * units))));
  const Tensor wh = recurrent_component_detail::Leaf(
      p, name + "/Wh", {units, 4 * units}, init::uniform(std::sqrt(6.f / float(5 * units))));
  const Tensor b = recurrent_component_detail::Leaf(
      p, name + "/b", {4 * units}, recurrent_component_detail::LstmBias(units));
  const Tensor pi =
      recurrent_component_detail::Leaf(p, name + "/peephole_i", {units}, init::normal(.05f));
  const Tensor pf =
      recurrent_component_detail::Leaf(p, name + "/peephole_f", {units}, init::normal(.05f));
  const Tensor po =
      recurrent_component_detail::Leaf(p, name + "/peephole_o", {units}, init::normal(.05f));
  const auto   parts = split(bias_add(add(matmul(x, wx), matmul(state.h, wh)), b), 4, 1);
  const Tensor i     = sigmoid(add(parts[0], mul(state.c, pi))),
               f     = sigmoid(add(parts[1], mul(state.c, pf)));
  const Tensor g     = tanh(parts[2]);
  const Tensor cell  = add(mul(f, state.c), mul(i, g));
  const Tensor o     = sigmoid(add(parts[3], mul(cell, po)));
  return recurrent_component_detail::Checked(
      RecurrentGateOutput{mul(o, tanh(cell)), cell, concat({i, f, g, o}, 1), {}, {}});
}
/// @brief Evaluate one source layer-normalized LSTM step with live diagnostics.
///
/// No fitting, optimizer, sampling or hidden no-grad. Computed nonfinite endpoints reject. Warm
/// valid stores/RNG are unchanged; serialize creation/writes. Source initialization precision and
/// binary grouping are retained. Creation failures may leave earlier leaves present. Work
/// O(N*H*(incoming+H)); applicable first/higher paths retained.
///
/// @param p Caller-owned floating store; initialized source names are reused, missing leaves
/// created in source order.
/// @param name Shared parameter prefix; same name reuses weights.
/// @param x Finite matching floating CPU[N,incoming], positive N.
/// @param incoming Input feature width in[1,1024].
/// @param units Hidden width in[1,1024].
/// @param state Finite same-dtype hidden CPU[N,H] and cell CPU[N,H].
/// @return Owning live next states and gate/readout diagnostics.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype, domain or parameter shape.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(27003);auto
/// x=n::zeros({1,2}),h=n::zeros({1,12});auto
/// y=m::LayerNormGateStep(p,"cell",x,2,12,n::LstmState{h,h});
/// @endcode
/// @par References
/// - Ba, Kiros and Hinton (2016), Layer Normalization, https://arxiv.org/abs/1607.06450. Source
/// gate-wise normalization and normalized readout are retained.
/// @see MaskedGatedSequence(), RecurrentGradientControl()
inline RecurrentGateOutput LayerNormGateStep(
    ParameterStore&    p,
    const std::string& name,
    const Tensor&      x,
    std::int64_t       incoming,
    std::int64_t       units,
    const LstmState&   state) {
  recurrent_component_detail::Cell(p, x, incoming, units, state.h, state.c);
  if (!state.c.defined())
    throw std::invalid_argument("LSTM requires explicit cell memory");
  const Tensor wx = recurrent_component_detail::Leaf(
      p,
      name + "/Wx",
      {incoming, 4 * units},
      init::uniform(std::sqrt(6.f / float(incoming + 4 * units))));
  const Tensor wh = recurrent_component_detail::Leaf(
      p, name + "/Wh", {units, 4 * units}, init::uniform(std::sqrt(6.f / float(5 * units))));
  const Tensor bias = recurrent_component_detail::Leaf(
      p, name + "/b", {4 * units}, recurrent_component_detail::LstmBias(units));
  const Tensor gate_gamma =
      recurrent_component_detail::Leaf(p, name + "/gate_gamma", {4 * units}, init::ones());
  const Tensor cell_gamma =
      recurrent_component_detail::Leaf(p, name + "/cell_gamma", {units}, init::ones());
  const Tensor cell_beta =
      recurrent_component_detail::Leaf(p, name + "/cell_beta", {units}, init::zeros());
  const Tensor affine = add(matmul(x, wx), matmul(state.h, wh));
  const Tensor normalized =
      reshape(RecurrentLayerNormalize(reshape(affine, {-1, 4, units}), 2), {-1, 4 * units});
  const auto   parts = split(add(mul(normalized, gate_gamma), bias), 4, 1);
  const Tensor i = sigmoid(parts[0]), f = sigmoid(parts[1]), g = tanh(parts[2]),
               o               = sigmoid(parts[3]);
  const Tensor cell            = add(mul(f, state.c), mul(i, g));
  const Tensor normalized_cell = add(mul(RecurrentLayerNormalize(cell, 1), cell_gamma), cell_beta);
  return recurrent_component_detail::Checked(
      RecurrentGateOutput{
          mul(o, tanh(normalized_cell)),
          cell,
          concat({i, f, g, o}, 1),
          normalized,
          normalized_cell});
}
/// @brief Construct the source clockwork structural mask.
///
/// Own Float32[12,12] slow-to-fast connection mask; from/to units are in period1,2,4,8 modules of
/// three.
///
/// @return Owning detached Float32 CPU[12,12] mask; no RNG or mutation.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto mask=m::ClockworkRecurrentConnectivity();
/// @endcode
/// @par References
/// - Koutnik, Greff, Gomez and Schmidhuber (2014), A Clockwork RNN,
/// https://arxiv.org/abs/1402.3511. This preset has four three-unit modules at periods1,2,4,8.
/// @see MaskedGatedSequence(), RecurrentGradientControl()
inline Tensor ClockworkRecurrentConnectivity() {
  std::vector<float> mask(144);
  for (int from = 0; from < 12; ++from)
    for (int to = 0; to < 12; ++to)
      mask[from * 12 + to] = ((1 << (from / 3)) >= (1 << (to / 3))) ? 1.f : 0.f;
  return from_vector(mask, {12, 12});
}
/// @brief Construct the source clockwork structural mask.
///
/// Own Float32[12,12] t/unit activity mask for source integer times0..11 and periods1,2,4,8.
///
/// @return Owning detached Float32 CPU[12,12] mask; no RNG or mutation.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto mask=m::ClockworkRecurrentSchedule();
/// @endcode
/// @par References
/// - Koutnik, Greff, Gomez and Schmidhuber (2014), A Clockwork RNN,
/// https://arxiv.org/abs/1402.3511. This preset has four three-unit modules at periods1,2,4,8.
/// @see MaskedGatedSequence(), RecurrentGradientControl()
inline Tensor ClockworkRecurrentSchedule() {
  std::vector<float> mask(144);
  for (int t = 0; t < 12; ++t)
    for (int j = 0; j < 12; ++j)
      mask[t * 12 + j] = t % (1 << (j / 3)) == 0 ? 1.f : 0.f;
  return from_vector(mask, {12, 12});
}
/// @brief Evaluate the original masked tanh RNN classifier path.
///
/// This is the exact compact source preset, not a fitting harness. No optimizer, training loop,
/// projection, labels, RNG bank or cached hidden state. Source parameter initialization and
/// primitive operation grouping are preserved. Zero lengths retain initial states; diagnostics zero
/// invalid steps. Computed nonfinite endpoints reject. Warm stores/RNG are unchanged; serialize
/// leaf creation/writes; creation failure may leave earlier leaves present. BPTT and applicable
/// higher paths remain live after parent/store collection. Work O(N*T*H*(input_width+H)) for vector
/// cells; ConvLSTM uses the fixed3x3 stencil.
///
/// @param p Caller-owned floating source store; missing leaves are initialized in source order,
/// warm leaves reused.
/// @param input Finite Float32/64 CPU[N,12,2], same dtype as store.
/// @param lengths Int32/64 CPU[N] valid lengths in[0,12].
/// @param initial 1 explicit finite same-dtype CPU[N,12] initial states; GRU retains the source
/// unused second slot.
/// @return Owning live logits, complete hidden/state histories and family diagnostics.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype, domain or parameter shape.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(27003);auto
/// x=n::zeros({1,12,2}),h=n::zeros({1,12});auto
/// y=m::MaskedTanhSequence(p,x,n::zeros({1},n::DType::Int32),{h});
/// @endcode
/// @par References
/// - Hochreiter and Schmidhuber (1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735. This source preset includes the later forget gate
/// and unit forget bias.
/// @see MaskedGatedSequence(), RecurrentGradientControl()
inline MaskedRecurrentOutput MaskedTanhSequence(
    ParameterStore&            p,
    const Tensor&              input,
    const Tensor&              lengths,
    const std::vector<Tensor>& initial) {
  recurrent_component_detail::Sequence(p, input, lengths, initial, 1);
  const auto          steps = time_steps(input);
  Tensor              h     = initial[0];
  std::vector<Tensor> states{h};
  for (int t = 0; t < 12; ++t) {
    h = recurrent_component_detail::Blend(
        recurrent_component_detail::StepMask(lengths, t), rnn_step(p, "cell", steps[t], h), h);
    states.push_back(h);
  }
  return recurrent_component_detail::Checked(
      MaskedRecurrentOutput{
          .logits      = recurrent_component_detail::Head(p, "head", h, 12, 2),
          .states      = stack_time(states),
          .final_state = h});
}
/// @brief Evaluate the original masked bidirectional RNN classifier path.
///
/// This is the exact compact source preset, not a fitting harness. No optimizer, training loop,
/// projection, labels, RNG bank or cached hidden state. Source parameter initialization and
/// primitive operation grouping are preserved. Zero lengths retain initial states; diagnostics zero
/// invalid steps. Computed nonfinite endpoints reject. Warm stores/RNG are unchanged; serialize
/// leaf creation/writes; creation failure may leave earlier leaves present. BPTT and applicable
/// higher paths remain live after parent/store collection. Work O(N*T*H*(input_width+H)) for vector
/// cells; ConvLSTM uses the fixed3x3 stencil.
///
/// @param p Caller-owned floating source store; missing leaves are initialized in source order,
/// warm leaves reused.
/// @param input Finite Float32/64 CPU[N,12,2], same dtype as store.
/// @param lengths Int32/64 CPU[N] valid lengths in[0,12].
/// @param initial 2 explicit finite same-dtype CPU[N,12] initial states; GRU retains the source
/// unused second slot.
/// @return Owning live logits, complete hidden/state histories and family diagnostics.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype, domain or parameter shape.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(27003);auto
/// x=n::zeros({1,12,2}),h=n::zeros({1,12});auto
/// y=m::MaskedBidirectionalSequence(p,x,n::zeros({1},n::DType::Int32),{h,h});
/// @endcode
/// @par References
/// - Schuster and Paliwal (1997), Bidirectional Recurrent Neural Networks,
/// https://doi.org/10.1109/78.650093.
/// @see MaskedGatedSequence(), RecurrentGradientControl()
inline MaskedRecurrentOutput MaskedBidirectionalSequence(
    ParameterStore&            p,
    const Tensor&              input,
    const Tensor&              lengths,
    const std::vector<Tensor>& initial) {
  recurrent_component_detail::Sequence(p, input, lengths, initial, 2);
  const auto          xs = time_steps(input);
  Tensor              h = initial[0], b = initial[1];
  std::vector<Tensor> forward{h}, backward{b}, aligned(12);
  for (int t = 0; t < 12; ++t)
    h = recurrent_component_detail::Blend(
        recurrent_component_detail::StepMask(lengths, t), rnn_step(p, "forward", xs[t], h), h),
    forward.push_back(h);
  for (int t = 11; t >= 0; --t) {
    b = recurrent_component_detail::Blend(
        recurrent_component_detail::StepMask(lengths, t), rnn_step(p, "backward", xs[t], b), b);
    backward.push_back(b);
    aligned[t] = b;
  }
  const Tensor joined = concat({h, b}, 1);
  return recurrent_component_detail::Checked(
      MaskedRecurrentOutput{
          .logits           = recurrent_component_detail::Head(p, "head", joined, 24, 2),
          .states           = stack_time(forward),
          .final_state      = joined,
          .backward_states  = stack_time(backward),
          .backward_aligned = stack_time(aligned),
          .final_backward   = b});
}
/// @brief Evaluate the original masked clockwork RNN classifier path.
///
/// This is the exact compact source preset, not a fitting harness. No optimizer, training loop,
/// projection, labels, RNG bank or cached hidden state. Source parameter initialization and
/// primitive operation grouping are preserved. Zero lengths retain initial states; diagnostics zero
/// invalid steps. Computed nonfinite endpoints reject. Warm stores/RNG are unchanged; serialize
/// leaf creation/writes; creation failure may leave earlier leaves present. BPTT and applicable
/// higher paths remain live after parent/store collection. Work O(N*T*H*(input_width+H)) for vector
/// cells; ConvLSTM uses the fixed3x3 stencil.
///
/// @param p Caller-owned floating source store; missing leaves are initialized in source order,
/// warm leaves reused.
/// @param input Finite Float32/64 CPU[N,12,2], same dtype as store.
/// @param lengths Int32/64 CPU[N] valid lengths in[0,12].
/// @param initial 1 explicit finite same-dtype CPU[N,12] initial states; GRU retains the source
/// unused second slot.
/// @return Owning live logits, complete hidden/state histories and family diagnostics.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype, domain or parameter shape.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(27003);auto
/// x=n::zeros({1,12,2}),h=n::zeros({1,12});auto
/// y=m::MaskedClockworkSequence(p,x,n::zeros({1},n::DType::Int32),{h});
/// @endcode
/// @par References
/// - Koutnik, Greff, Gomez and Schmidhuber (2014), A Clockwork RNN,
/// https://arxiv.org/abs/1402.3511. This preset has four three-unit modules at periods1,2,4,8.
/// @see MaskedGatedSequence(), RecurrentGradientControl()
inline MaskedRecurrentOutput MaskedClockworkSequence(
    ParameterStore&            p,
    const Tensor&              input,
    const Tensor&              lengths,
    const std::vector<Tensor>& initial) {
  recurrent_component_detail::Sequence(p, input, lengths, initial, 1);
  const auto   steps = time_steps(input);
  const Tensor wx    = recurrent_component_detail::Leaf(p, "cell/Wx", {2, 12}, init::normal(.1)),
               wh    = recurrent_component_detail::Leaf(p, "cell/Wh", {12, 12}, init::normal(.1)),
               bias  = recurrent_component_detail::Leaf(p, "cell/b", {12}, init::zeros());
  const Tensor        masked_wh = mul(wh, ClockworkRecurrentConnectivity());
  const Tensor        schedule  = ClockworkRecurrentSchedule();
  Tensor              h         = initial[0];
  std::vector<Tensor> states{h};
  for (int t = 0; t < 12; ++t) {
    const Tensor candidate = tanh(add(add(matmul(steps[t], wx), matmul(h, masked_wh)), bias));
    const Tensor clock     = slice(schedule, {t, 0}, {1, 12});
    h                      = recurrent_component_detail::Blend(
        mul(recurrent_component_detail::StepMask(lengths, t), clock), candidate, h);
    states.push_back(h);
  }
  return recurrent_component_detail::Checked(
      MaskedRecurrentOutput{
          .logits      = recurrent_component_detail::Head(p, "head", h, 12, 2),
          .states      = stack_time(states),
          .final_state = h});
}
/// @brief Evaluate the original masked independent ReLU recurrence classifier path.
///
/// This is the exact compact source preset, not a fitting harness. No optimizer, training loop,
/// projection, labels, RNG bank or cached hidden state. Source parameter initialization and
/// primitive operation grouping are preserved. Zero lengths retain initial states; diagnostics zero
/// invalid steps. Computed nonfinite endpoints reject. Warm stores/RNG are unchanged; serialize
/// leaf creation/writes; creation failure may leave earlier leaves present. BPTT and applicable
/// higher paths remain live after parent/store collection. Work O(N*T*H*(input_width+H)) for vector
/// cells; ConvLSTM uses the fixed3x3 stencil.
///
/// @param p Caller-owned floating source store; missing leaves are initialized in source order,
/// warm leaves reused.
/// @param input Finite Float32/64 CPU[N,12,2], same dtype as store.
/// @param lengths Int32/64 CPU[N] valid lengths in[0,12].
/// @param initial 1 explicit finite same-dtype CPU[N,12] initial states; GRU retains the source
/// unused second slot.
/// @return Owning live logits, complete hidden/state histories and family diagnostics.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype, domain or parameter shape.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(27003);auto
/// x=n::zeros({1,12,2}),h=n::zeros({1,12});auto
/// y=m::MaskedIndRnnSequence(p,x,n::zeros({1},n::DType::Int32),{h});
/// @endcode
/// @par References
/// - Li et al. (2018), Independently Recurrent Neural Network (IndRNN): Building A Longer and
/// Deeper RNN, https://arxiv.org/abs/1803.04831. This source preset carries a separate explicit
/// post-update coefficient projection.
/// @see MaskedGatedSequence(), RecurrentGradientControl()
inline MaskedRecurrentOutput MaskedIndRnnSequence(
    ParameterStore&            p,
    const Tensor&              input,
    const Tensor&              lengths,
    const std::vector<Tensor>& initial) {
  recurrent_component_detail::Sequence(p, input, lengths, initial, 1);
  const Tensor wx =
      recurrent_component_detail::Leaf(p, "cell/Wx", {2, 12}, init::uniform(std::sqrt(6.f / 14)));
  const Tensor u = recurrent_component_detail::Leaf(p, "cell/u", {12}, init::constant(.9f)),
               b = recurrent_component_detail::Leaf(p, "cell/b", {12}, init::constant(.05f));
  const auto features =
      time_steps(reshape(bias_add(matmul(reshape(input, {-1, 2}), wx), b), {-1, 12, 12}));
  Tensor              h = initial[0];
  std::vector<Tensor> states{h};
  for (int t = 0; t < 12; ++t) {
    const Tensor candidate = relu(add(features[t], mul(h, u)));
    h                      = recurrent_component_detail::Blend(
        recurrent_component_detail::StepMask(lengths, t), candidate, h);
    states.push_back(h);
  }
  return recurrent_component_detail::Checked(
      MaskedRecurrentOutput{
          .logits      = recurrent_component_detail::Head(p, "head", h, 12, 2),
          .states      = stack_time(states),
          .final_state = h});
}
/// @brief Evaluate the original masked minimal gated unit classifier path.
///
/// This is the exact compact source preset, not a fitting harness. No optimizer, training loop,
/// projection, labels, RNG bank or cached hidden state. Source parameter initialization and
/// primitive operation grouping are preserved. Zero lengths retain initial states; diagnostics zero
/// invalid steps. Computed nonfinite endpoints reject. Warm stores/RNG are unchanged; serialize
/// leaf creation/writes; creation failure may leave earlier leaves present. BPTT and applicable
/// higher paths remain live after parent/store collection. Work O(N*T*H*(input_width+H)) for vector
/// cells; ConvLSTM uses the fixed3x3 stencil.
///
/// @param p Caller-owned floating source store; missing leaves are initialized in source order,
/// warm leaves reused.
/// @param input Finite Float32/64 CPU[N,12,2], same dtype as store.
/// @param lengths Int32/64 CPU[N] valid lengths in[0,12].
/// @param initial 1 explicit finite same-dtype CPU[N,12] initial states; GRU retains the source
/// unused second slot.
/// @return Owning live logits, complete hidden/state histories and family diagnostics.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype, domain or parameter shape.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(27003);auto
/// x=n::zeros({1,12,2}),h=n::zeros({1,12});auto
/// y=m::MaskedMguSequence(p,x,n::zeros({1},n::DType::Int32),{h});
/// @endcode
/// @par References
/// - Zhou, Wu, Zhang and Zhou (2016), Minimal Gated Unit for Recurrent Neural Networks,
/// https://arxiv.org/abs/1603.09420.
/// @see MaskedGatedSequence(), RecurrentGradientControl()
inline MaskedRecurrentOutput MaskedMguSequence(
    ParameterStore&            p,
    const Tensor&              input,
    const Tensor&              lengths,
    const std::vector<Tensor>& initial) {
  recurrent_component_detail::Sequence(p, input, lengths, initial, 1);
  const Tensor wxf =
      recurrent_component_detail::Leaf(p, "cell/Wxf", {2, 12}, init::uniform(std::sqrt(6.f / 14)));
  const Tensor whf =
      recurrent_component_detail::Leaf(p, "cell/Whf", {12, 12}, init::uniform(std::sqrt(6.f / 24)));
  const Tensor bf = recurrent_component_detail::Leaf(p, "cell/bf", {12}, init::zeros());
  const Tensor wxn =
      recurrent_component_detail::Leaf(p, "cell/Wxn", {2, 12}, init::uniform(std::sqrt(6.f / 14)));
  const Tensor whn =
      recurrent_component_detail::Leaf(p, "cell/Whn", {12, 12}, init::uniform(std::sqrt(6.f / 24)));
  const Tensor        bn = recurrent_component_detail::Leaf(p, "cell/bn", {12}, init::zeros());
  const auto          xs = time_steps(input);
  Tensor              h  = initial[0];
  std::vector<Tensor> states{h}, gates;
  for (int t = 0; t < 12; ++t) {
    const Tensor f         = sigmoid(bias_add(add(matmul(xs[t], wxf), matmul(h, whf)), bf));
    const Tensor candidate = tanh(bias_add(add(matmul(xs[t], wxn), matmul(mul(f, h), whn)), bn));
    const Tensor next      = add(mul(sub(1.0, f), h), mul(f, candidate));
    const Tensor valid     = recurrent_component_detail::StepValid(lengths, t);
    h                      = where(valid, next, h);
    gates.push_back(
        recurrent_component_detail::MaskedDiagnostics(valid, concat({f, candidate}, 1)));
    states.push_back(h);
  }
  return recurrent_component_detail::Checked(
      MaskedRecurrentOutput{
          .logits      = recurrent_component_detail::Head(p, "head", h, 12, 2),
          .states      = stack_time(states),
          .final_state = h,
          .gates       = stack_time(gates)});
}
/// @brief Evaluate the original masked gated-cell sequence classifier path.
///
/// This is the exact compact source preset, not a fitting harness. No optimizer, training loop,
/// projection, labels, RNG bank or cached hidden state. Source parameter initialization and
/// primitive operation grouping are preserved. Zero lengths retain initial states; diagnostics zero
/// invalid steps. Computed nonfinite endpoints reject. Warm stores/RNG are unchanged; serialize
/// leaf creation/writes; creation failure may leave earlier leaves present. BPTT and applicable
/// higher paths remain live after parent/store collection. Work O(N*T*H*(input_width+H)) for vector
/// cells; ConvLSTM uses the fixed3x3 stencil.
///
/// @param p Caller-owned floating source store; missing leaves are initialized in source order,
/// warm leaves reused.
/// @param input Finite Float32/64 CPU[N,12,2], same dtype as store.
/// @param lengths Int32/64 CPU[N] valid lengths in[0,12].
/// @param initial 2 explicit finite same-dtype CPU[N,12] initial states; GRU retains the source
/// unused second slot.
/// @param kind Discrete Lstm/Gru/Peephole/LayerNorm architecture.
/// @return Owning live logits, complete hidden/state histories and family diagnostics.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype, domain or parameter shape.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(27003);auto
/// x=n::zeros({1,12,2}),h=n::zeros({1,12});auto
/// y=m::MaskedGatedSequence(p,x,n::zeros({1},n::DType::Int32),{h,h},m::MaskedGatedCell::Lstm);
/// @endcode
/// @par References
/// - Hochreiter and Schmidhuber (1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735. This source preset includes the later forget gate
/// and unit forget bias.
/// @see MaskedGatedSequence(), RecurrentGradientControl()
inline MaskedRecurrentOutput MaskedGatedSequence(
    ParameterStore&            p,
    const Tensor&              input,
    const Tensor&              lengths,
    const std::vector<Tensor>& initial,
    MaskedGatedCell            kind) {
  recurrent_component_detail::Sequence(p, input, lengths, initial, 2);
  if (int(kind) < 0 || int(kind) > 3)
    throw std::invalid_argument("gated cell kind");
  const auto xs = time_steps(input);
  Tensor     h = initial[0], c = kind != MaskedGatedCell::Gru ? initial[1] : mul(initial[0], 0.0);
  std::vector<Tensor> states{h}, cells{c}, gates, normed_gates, normed_cells;
  for (int t = 0; t < 12; ++t) {
    const Tensor        mask  = recurrent_component_detail::StepMask(lengths, t);
    const Tensor        valid = greater(mask, 0.0);
    RecurrentGateOutput next;
    switch (kind) {
      case MaskedGatedCell::Lstm:
        next = LstmGateStep(p, "cell", xs[t], 2, 12, {h, c}, 1.0);
        break;
      case MaskedGatedCell::Peephole:
        next = PeepholeGateStep(p, "cell", xs[t], 2, 12, {h, c});
        break;
      case MaskedGatedCell::LayerNorm:
        next = LayerNormGateStep(p, "cell", xs[t], 2, 12, {h, c});
        normed_gates.push_back(
            recurrent_component_detail::MaskedDiagnostics(valid, next.normalized_gate_affine));
        normed_cells.push_back(
            recurrent_component_detail::MaskedDiagnostics(valid, next.normalized_cell));
        break;
      case MaskedGatedCell::Gru:
        next      = GruGateStep(p, "cell", xs[t], 2, 12, h, 0.0);
        next.cell = c;
        break;
    }
    h = recurrent_component_detail::Blend(mask, next.hidden, h);
    c = recurrent_component_detail::Blend(mask, next.cell, c);
    states.push_back(h);
    cells.push_back(c);
    gates.push_back(recurrent_component_detail::MaskedDiagnostics(valid, next.gates));
  }
  MaskedRecurrentOutput out{
      .logits      = recurrent_component_detail::Head(p, "head", h, 12, 2),
      .states      = stack_time(states),
      .cell_states = stack_time(cells),
      .final_state = h,
      .final_cell  = c,
      .gates       = stack_time(gates)};
  if (!normed_gates.empty()) {
    out.normalized_gate_affine = stack_time(normed_gates);
    out.normalized_cell        = stack_time(normed_cells);
  }
  return recurrent_component_detail::Checked(out);
}
/// @brief Evaluate the original masked spatial ConvLSTM classifier path.
///
/// This is the exact compact source preset, not a fitting harness. No optimizer, training loop,
/// projection, labels, RNG bank or cached hidden state. Source parameter initialization and
/// primitive operation grouping are preserved. Zero lengths retain initial states; diagnostics zero
/// invalid steps. Computed nonfinite endpoints reject. Warm stores/RNG are unchanged; serialize
/// leaf creation/writes; creation failure may leave earlier leaves present. BPTT and applicable
/// higher paths remain live after parent/store collection. Work O(N*T*H*(input_width+H)) for vector
/// cells; ConvLSTM uses the fixed3x3 stencil.
///
/// @param p Caller-owned floating source store; missing leaves are initialized in source order,
/// warm leaves reused.
/// @param input Finite Float32/64 CPU[N,4,4,4,1].
/// @param lengths Int32/64 CPU[N] valid lengths in[0,4].
/// @param initial 2 explicit finite same-dtype initial tensors, each[N,4,4,2].
/// @return Owning live logits, complete hidden/state histories and family diagnostics.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype, domain or parameter shape.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(27003);auto
/// x=n::zeros({1,4,4,4,1}),h=n::zeros({1,4,4,2});auto
/// y=m::MaskedConvLstmSequence(p,x,n::zeros({1},n::DType::Int32),{h,h});
/// @endcode
/// @par References
/// - Shi et al. (2015), Convolutional LSTM Network: A Machine Learning Approach for Precipitation
/// Nowcasting, https://arxiv.org/abs/1506.04214. This is the compact original moving-dot preset,
/// not a precipitation benchmark.
/// @see MaskedGatedSequence(), RecurrentGradientControl()
inline MaskedRecurrentOutput MaskedConvLstmSequence(
    ParameterStore&            p,
    const Tensor&              input,
    const Tensor&              lengths,
    const std::vector<Tensor>& initial) {
  recurrent_component_detail::Sequence(p, input, lengths, initial, 2, true);
  const Tensor wx = recurrent_component_detail::Leaf(
      p, "cell/Wx", {3, 3, 1, 8}, init::uniform(std::sqrt(6.f / 81)));
  const Tensor wh = recurrent_component_detail::Leaf(
      p, "cell/Wh", {3, 3, 2, 8}, init::uniform(std::sqrt(6.f / 90)));
  const Tensor b =
      recurrent_component_detail::Leaf(p, "cell/b", {8}, [](const Shape&, DType d, Generator&) {
        return concat({zeros({2}, d), full({2}, 1.0, d), zeros({4}, d)}, 0);
      });
  const auto          xs = time_steps(input);
  Tensor              h = initial[0], c = initial[1];
  std::vector<Tensor> states{h}, cells{c}, gates;
  for (int t = 0; t < static_cast<int>(xs.size()); ++t) {
    const Tensor pre   = bias_add(add(conv2d(xs[t], wx), conv2d(h, wh)), b);
    const auto   split = datamunge::nn::split(pre, 4, 3);
    const Tensor i = sigmoid(split[0]), f = sigmoid(split[1]), o = sigmoid(split[3]),
                 z      = tanh(split[2]);
    const Tensor next_c = add(mul(f, c), mul(i, z));
    const Tensor next_h = mul(o, tanh(next_c));
    const Tensor valid  = reshape(greater(lengths, double(t)), {-1, 1, 1, 1});
    h                   = where(valid, next_h, h);
    c                   = where(valid, next_c, c);
    gates.push_back(recurrent_component_detail::MaskedDiagnostics(valid, concat({i, f, z, o}, 3)));
    states.push_back(h);
    cells.push_back(c);
  }
  const Tensor head_w = recurrent_component_detail::Leaf(
      p, "head/W", {1, 1, 2, 1}, init::uniform(std::sqrt(6.f / 3)));
  const Tensor head_b = recurrent_component_detail::Leaf(p, "head/b", {1}, init::zeros());
  return recurrent_component_detail::Checked(
      MaskedRecurrentOutput{
          .logits      = bias_add(conv2d(h, head_w), head_b),
          .states      = stack_time(states),
          .cell_states = stack_time(cells),
          .final_state = h,
          .final_cell  = c,
          .gates       = stack_time(gates)});
}
/// @brief Compute the source-ordered live global gradient norm.
///
/// No host-double accumulation, fitting, mutation or detach. O(total elements); zero norm has
/// native sqrt derivative conventions.
///
/// @param gradients Nonempty list of finite nonempty Float32/64 CPU tensors, all same dtype;
/// arbitrary positive shapes or scalars.
/// @return Live scalar sqrt(sum_k sum(g_k²)), preserving source tensor reduction order.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype, domain or parameter shape.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto
/// norm=m::RecurrentGradientNorm({n::tensor({3.f,4.f})});
/// @endcode
/// @par References
/// - Pascanu, Mikolov and Bengio (2012/2013), On the difficulty of training Recurrent Neural
/// Networks, https://arxiv.org/abs/1211.5063. The source Float32 reduction and scaling order are
/// preserved.
/// @see MaskedGatedSequence(), RecurrentGradientControl()
inline Tensor RecurrentGradientNorm(const std::vector<Tensor>& gradients) {
  if (gradients.empty())
    throw std::invalid_argument("empty recurrent gradients");
  for (const auto& g : gradients) {
    recurrent_component_detail::Floating(g);
    if (g.dtype() != gradients[0].dtype())
      throw std::invalid_argument("gradient dtype mismatch");
  }
  Tensor total;
  for (const auto& g : gradients) {
    const Tensor s = sum(square(g));
    total          = total.defined() ? add(total, s) : s;
  }
  auto norm = sqrt(total);
  recurrent_component_detail::Floating(norm);
  return norm;
}
/// @brief Own live controlled gradients and norm diagnostics.
///
/// All tensor/list handles own their graphs after parent collection. No optimizer or model
/// retained; input tensors are not modified.
///
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;m::RecurrentGradientControlOutput out;
/// @endcode
/// @see MaskedGatedSequence(), RecurrentGradientControl()
/// @par References
/// - Hochreiter and Schmidhuber (1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 . Exact compact variants and their distinct gate
/// conventions are specified by the associated presets.
struct RecurrentGradientControlOutput {
  std::vector<Tensor> values;       ///< Live applied gradients in input order, same shapes/dtype.
  Tensor              raw_norm;     ///< Live source-ordered scalar norm before control.
  Tensor              applied_norm; ///< Live scalar norm after control.
  Tensor scale; ///< Live uniform scale for global mode; constant1 for none/value modes.
};
/// @brief Apply one explicit source gradient policy without an optimizer.
///
/// Global scale=min(1,limit/max(norm,Float32(1e-30))); coordinate mode clips each value and
/// generally changes direction. No updates, RNG or detach. Native min/max/clip/sqrt conventions
/// apply at kinks; O(total elements).
///
/// @param gradients Finite same-dtype CPU gradients, as for RecurrentGradientNorm.
/// @param policy 0 none,1 global norm,2 coordinate clipping; other values reject.
/// @param norm_limit Finite nonnegative Float32 global limit, default.05f.
/// @param value_limit Finite nonnegative Float32 coordinate limit, default.01f.
/// @return Owning live applied gradients, raw/applied scalar norms and uniform scale.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype, domain or parameter shape.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto
/// y=m::RecurrentGradientControl({n::tensor({3.f,4.f})},1);
/// @endcode
/// @par References
/// - Pascanu, Mikolov and Bengio (2012/2013), On the difficulty of training Recurrent Neural
/// Networks, https://arxiv.org/abs/1211.5063. The source Float32 reduction and scaling order are
/// preserved.
/// @see MaskedGatedSequence(), RecurrentGradientControl()
inline RecurrentGradientControlOutput RecurrentGradientControl(
    const std::vector<Tensor>& gradients,
    int                        policy,
    float                      norm_limit  = .05f,
    float                      value_limit = .01f) {
  if (policy < 0 || policy > 2 || !std::isfinite(norm_limit) || !std::isfinite(value_limit) ||
      norm_limit < 0 || value_limit < 0)
    throw std::invalid_argument("gradient control policy/limits");
  const Tensor        norm  = RecurrentGradientNorm(gradients);
  const Tensor        scale = policy == 1 ? minimum(
                                         full({}, 1.0, norm.dtype()),
                                         div(full({}, double(norm_limit), norm.dtype()),
                                             maximum(norm, full({}, double(1e-30f), norm.dtype()))))
                                          : full({}, 1.0, norm.dtype());
  std::vector<Tensor> result;
  for (const auto& g : gradients)
    result.push_back(
        policy == 2 ? clip(g, -double(value_limit), double(value_limit)) : mul(g, scale));
  return {result, norm, RecurrentGradientNorm(result), scale};
}

} // namespace datamunge::nn::models
