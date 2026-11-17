#pragma once
/// @file recurrent_extensions.hpp
/// @brief Explicit stacked, truncated, dropped, time-gated and input-projected source recurrent
/// paths.
#include <datamunge/nn/recurrent_components.hpp>
#include <random>
namespace datamunge::nn::models {
/// @brief Flat live endpoints of the compact recurrent extension presets.
///
/// Tensor handles own storage and differentiable parent paths. Family-inapplicable
/// fields are undefined. No optimizer, cached state or implicit training mode.
/// @par Example
/// @code
/// dm::nn::models::ExtendedRecurrentOutput result;
/// @endcode
/// @see MaskedStackedSequence(), MaskedPhasedSequence()
/// @par References
/// - Hochreiter and Schmidhuber (1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 ; Bradbury et al. (2017), Quasi-Recurrent Neural
/// Networks, https://arxiv.org/abs/1611.01576 . Family-specific conventions are documented by the
/// preset functions.
struct ExtendedRecurrentOutput {
  Tensor logits;         ///< CPU[N,2] final-state head scores.
  Tensor states;         ///< CPU[N,13,12] hidden history, including explicit initial state.
  Tensor cell_states;    ///< CPU[N,13,12] memory history for gated families.
  Tensor final_state;    ///< CPU[N,12] final classifier hidden state.
  Tensor final_cell;     ///< CPU[N,12] final gated memory state.
  Tensor gates;          ///< CPU[N,12,36 or48] valid-step gate diagnostics.
  Tensor upper_states;   ///< CPU[N,13,12] upper stacked-layer history.
  Tensor final_lower;    ///< CPU[N,12] final lower stacked state.
  Tensor boundary_state; ///< CPU[N,12] last-window pre-detach boundary value.
  Tensor recurrent_mask; ///< Caller-supplied live CPU[N,12,12] recurrent multipliers.
  Tensor time_gates;     ///< CPU[N,12,12] valid triangular/leak gate history.
  Tensor phase;          ///< CPU[N,12,12] valid wrapped timestamp phases.
};
/// @brief Evaluate the source StackedRnn path with explicit controls.
///
/// Exact source twelve-step preset, with no fitting, optimizer, labels, hidden RNG draws or cached
/// state. Warm-store inference is immutable; synchronize missing-leaf creation/writes.
/// Initialization failure may leave earlier leaves. All state histories own storage and retain
/// applicable first/higher derivative paths after store collection. Invalid timesteps hold previous
/// states and zero gate diagnostics. Work O(N*12*12*(2+12)); input-only projections are batched
/// before pooling.
///
/// @param p Caller-owned Float32/64 CPU store; missing source leaves initialized, existing leaves
/// reused.
/// @param input Finite CPU[N,12,2] observations, matching store dtype.
/// @param lengths Int32/64 CPU[N] valid lengths in[0,12].
/// @param initial Explicit finite matching CPU[N,12] initial state tensors; two for stacked/gated
/// paths, one for truncated/dropout.
/// @return Owning flat live endpoints; inapplicable family fields undefined.
/// @throws std::invalid_argument Invalid finite CPU dtype, geometry or option domain.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);auto
/// x=n::zeros({1,12,2}),h=n::zeros({1,12});auto
/// lengths=n::from_vector(std::vector<std::int32_t>{4});auto
/// y=m::MaskedStackedSequence(p,x,lengths,{h,h});
/// @endcode
/// @par References
/// - Schmidhuber (1992), Learning complex, extended sequences using the principle of history
/// compression, https://doi.org/10.1162/neco.1992.4.2.234. This compact preset stacks two tanh
/// cells with current lower-state input.
/// @see MaskedTanhSequence()
inline ExtendedRecurrentOutput MaskedStackedSequence(
    ParameterStore&            p,
    const Tensor&              input,
    const Tensor&              lengths,
    const std::vector<Tensor>& initial) {
  recurrent_component_detail::Sequence(p, input, lengths, initial, 2);

  const auto          steps = time_steps(input);
  Tensor              lower = initial[0], upper = initial[1];
  std::vector<Tensor> lower_trace{lower}, upper_trace{upper};
  for (int t = 0; t < 12; ++t) {
    const Tensor mask = recurrent_component_detail::StepMask(lengths, t);
    lower = recurrent_component_detail::Blend(mask, rnn_step(p, "lower", steps[t], lower), lower);
    upper = recurrent_component_detail::Blend(mask, rnn_step(p, "upper", lower, upper), upper);
    lower_trace.push_back(lower);
    upper_trace.push_back(upper);
  }
  ExtendedRecurrentOutput result;
  result.logits       = recurrent_component_detail::Head(p, "head", upper, 12, 2);
  result.states       = stack_time(lower_trace);
  result.upper_states = stack_time(upper_trace);
  result.final_state  = upper;
  result.final_lower  = lower;
  for (const auto& value : std::vector<Tensor>{
           result.logits,
           result.states,
           result.upper_states,
           result.final_state,
           result.final_lower})
    recurrent_component_detail::Floating(value);
  return result;
}
/// @brief Evaluate the source TruncatedBptt path with explicit controls.
///
/// Exact source twelve-step preset, with no fitting, optimizer, labels, hidden RNG draws or cached
/// state. Warm-store inference is immutable; synchronize missing-leaf creation/writes.
/// Initialization failure may leave earlier leaves. All state histories own storage and retain
/// applicable first/higher derivative paths after store collection. Invalid timesteps hold previous
/// states and zero gate diagnostics. Work O(N*12*12*(2+12)); input-only projections are batched
/// before pooling. stop_gradient affects only backward paths at max(length-window,0); all forward
/// values remain unchanged. Boundary selection is structural.
///
/// @param p Caller-owned Float32/64 CPU store; missing source leaves initialized, existing leaves
/// reused.
/// @param input Finite CPU[N,12,2] observations, matching store dtype.
/// @param lengths Int32/64 CPU[N] valid lengths in[0,12].
/// @param initial Explicit finite matching CPU[N,12] initial state tensors; two for stacked/gated
/// paths, one for truncated/dropout.
/// @param window Integer final valid backward window in[0,12];0 retains full BPTT.
/// @return Owning flat live endpoints; inapplicable family fields undefined.
/// @throws std::invalid_argument Invalid finite CPU dtype, geometry or option domain.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);auto
/// x=n::zeros({1,12,2}),h=n::zeros({1,12});auto
/// lengths=n::from_vector(std::vector<std::int32_t>{4});auto
/// y=m::MaskedTruncatedSequence(p,x,lengths,{h},3);
/// @endcode
/// @par References
/// - Williams and Peng (1990), An Efficient Gradient-Based Algorithm for On-Line Training of
/// Recurrent Network Trajectories, https://doi.org/10.1162/neco.1990.2.4.490. The source uses a
/// final-window boundary.
/// @see MaskedTanhSequence()
inline ExtendedRecurrentOutput MaskedTruncatedSequence(
    ParameterStore&            p,
    const Tensor&              input,
    const Tensor&              lengths,
    const std::vector<Tensor>& initial,
    int                        window) {
  recurrent_component_detail::Sequence(p, input, lengths, initial, 1);
  if (window < 0 || window > 12)
    throw std::invalid_argument("backward window in[0,12]");

  const auto          steps = time_steps(input);
  Tensor              h = initial[0], boundary = initial[0];
  std::vector<Tensor> states{h};
  const Tensor        shifted        = sub(lengths, double(window));
  const Tensor        boundary_index = maximum(shifted, zeros_like(shifted));
  for (int t = 0; t < 12; ++t) {
    if (window > 0) {
      const Tensor selected = expand_dims(equal(boundary_index, double(t)), 1);
      boundary              = where(selected, h, boundary);
      if (t > 0)
        h = where(selected, stop_gradient(h), h);
    }
    h = recurrent_component_detail::Blend(
        recurrent_component_detail::StepMask(lengths, t), rnn_step(p, "cell", steps[t], h), h);
    states.push_back(h);
  }
  ExtendedRecurrentOutput result;
  result.logits         = recurrent_component_detail::Head(p, "head", h, 12, 2);
  result.states         = stack_time(states);
  result.final_state    = h;
  result.boundary_state = boundary;
  for (const auto& value :
       std::vector<Tensor>{result.logits, result.states, result.final_state, result.boundary_state})
    recurrent_component_detail::Floating(value);
  return result;
}
/// @brief Evaluate the source RecurrentDropout path with explicit controls.
///
/// Exact source twelve-step preset, with no fitting, optimizer, labels, hidden RNG draws or cached
/// state. Warm-store inference is immutable; synchronize missing-leaf creation/writes.
/// Initialization failure may leave earlier leaves. All state histories own storage and retain
/// applicable first/higher derivative paths after store collection. Invalid timesteps hold previous
/// states and zero gate diagnostics. Work O(N*12*12*(2+12)); input-only projections are batched
/// before pooling.
///
/// @param p Caller-owned Float32/64 CPU store; missing source leaves initialized, existing leaves
/// reused.
/// @param input Finite CPU[N,12,2] observations, matching store dtype.
/// @param lengths Int32/64 CPU[N] valid lengths in[0,12].
/// @param initial Explicit finite matching CPU[N,12] initial state tensors; two for stacked/gated
/// paths, one for truncated/dropout.
/// @param recurrent_mask Finite matching floating CPU[N,12,12] explicit multipliers; all ones
/// disables dropout.
/// @return Owning flat live endpoints; inapplicable family fields undefined.
/// @throws std::invalid_argument Invalid finite CPU dtype, geometry or option domain.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);auto
/// x=n::zeros({1,12,2}),h=n::zeros({1,12});auto
/// lengths=n::from_vector(std::vector<std::int32_t>{4});auto
/// y=m::MaskedDropoutSequence(p,x,lengths,{h},n::ones({1,12,12}));
/// @endcode
/// @par References
/// - Gal and Ghahramani (2016), A Theoretically Grounded Application of Dropout in Recurrent Neural
/// Networks, https://arxiv.org/abs/1512.05287. The caller chooses shared or per-time masks.
/// @see MaskedTanhSequence()
inline ExtendedRecurrentOutput MaskedDropoutSequence(
    ParameterStore&            p,
    const Tensor&              input,
    const Tensor&              lengths,
    const std::vector<Tensor>& initial,
    const Tensor&              recurrent_mask) {
  recurrent_component_detail::Sequence(p, input, lengths, initial, 1);
  recurrent_component_detail::Floating(recurrent_mask);
  if (recurrent_mask.shape() != Shape{input.size(0), 12, 12} ||
      recurrent_mask.dtype() != input.dtype())
    throw std::invalid_argument("explicit recurrent mask geometry/dtype");

  const auto          steps = time_steps(input), masks = time_steps(recurrent_mask);
  Tensor              h = initial[0];
  std::vector<Tensor> states{h};
  for (int t = 0; t < 12; ++t) {
    h = recurrent_component_detail::Blend(
        recurrent_component_detail::StepMask(lengths, t),
        rnn_step(p, "cell", steps[t], mul(h, masks[t])),
        h);
    states.push_back(h);
  }
  ExtendedRecurrentOutput result;
  result.logits         = recurrent_component_detail::Head(p, "head", h, 12, 2);
  result.states         = stack_time(states);
  result.final_state    = h;
  result.recurrent_mask = recurrent_mask;
  for (const auto& value :
       std::vector<Tensor>{result.logits, result.states, result.final_state, result.recurrent_mask})
    recurrent_component_detail::Floating(value);
  return result;
}
/// @brief Evaluate the source PhasedLstm path with explicit controls.
///
/// Exact source twelve-step preset, with no fitting, optimizer, labels, hidden RNG draws or cached
/// state. Warm-store inference is immutable; synchronize missing-leaf creation/writes.
/// Initialization failure may leave earlier leaves. All state histories own storage and retain
/// applicable first/higher derivative paths after store collection. Invalid timesteps hold previous
/// states and zero gate diagnostics. Work O(N*12*12*(2+12)); input-only projections are batched
/// before pooling. floor is stop-gradient; timestamp, period and shift paths are live except native
/// nonsmooth phase-boundary conventions. Nonfinite computed endpoints reject.
///
/// @param p Caller-owned Float32/64 CPU store; missing source leaves initialized, existing leaves
/// reused.
/// @param input Finite CPU[N,12,2] observations, matching store dtype.
/// @param lengths Int32/64 CPU[N] valid lengths in[0,12].
/// @param initial Explicit finite matching CPU[N,12] initial state tensors; two for stacked/gated
/// paths, one for truncated/dropout.
/// @param timestamps Finite matching floating CPU[N,12] observation times; gradients retained away
/// from phase kinks.
/// @param leak Finite floating CPU scalar in[0,1];0 at inference, source training .001.
/// @return Owning flat live endpoints; inapplicable family fields undefined.
/// @throws std::invalid_argument Invalid finite CPU dtype, geometry or option domain.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);auto
/// x=n::zeros({1,12,2}),h=n::zeros({1,12});auto
/// lengths=n::from_vector(std::vector<std::int32_t>{4});auto
/// y=m::MaskedPhasedSequence(p,x,lengths,{h,h},n::zeros({1,12}),n::scalar(0.f));
/// @endcode
/// @par References
/// - Neil, Pfeiffer and Liu (2016), Phased LSTM, https://arxiv.org/abs/1610.09513. Source fixed
/// r_on=.5 and explicit closed-phase leak.
/// @see MaskedTanhSequence()
inline ExtendedRecurrentOutput MaskedPhasedSequence(
    ParameterStore&            p,
    const Tensor&              input,
    const Tensor&              lengths,
    const std::vector<Tensor>& initial,
    const Tensor&              timestamps,
    const Tensor&              leak) {
  recurrent_component_detail::Sequence(p, input, lengths, initial, 2);
  recurrent_component_detail::Floating(timestamps);
  recurrent_component_detail::Floating(leak);
  if (timestamps.shape() != Shape{input.size(0), 12} || timestamps.dtype() != input.dtype() ||
      leak.dim() != 0 || leak.item() < 0 || leak.item() > 1)
    throw std::invalid_argument("phased timestamps/leak");

  const Tensor log_period  = p.get("time/log_period", {12}, [](const Shape&, DType d, Generator&) {
    std::vector<float> v(12);
    for (int j = 0; j < 12; ++j)
      v[j] = std::log(2.f + .5f * j);
    return cast(from_vector(v), d);
  });
  const Tensor phase_shift = p.get("time/shift", {12}, [](const Shape&, DType d, Generator&) {
    std::vector<float> v(12);
    for (int j = 0; j < 12; ++j)
      v[j] = .137f * (2.f + .5f * j) * (j + .2f);
    return cast(from_vector(v), d);
  });
  const Tensor period      = exp(log_period);
  const auto   xs          = time_steps(input);
  const auto   ts          = unstack(timestamps, 1);
  Tensor       h = initial[0], c = initial[1];
  std::vector<Tensor> states{h}, cells{c}, gates, time_gates, phases;
  for (int t = 0; t < 12; ++t) {
    const auto   proposed = LstmGateStep(p, "cell", xs[t], 2, 12, {h, c}, 1.0);
    const Tensor raw      = div(sub(expand_dims(ts[t], 1), phase_shift), period);
    const Tensor phase    = sub(raw, stop_gradient(floor(raw)));
    const Tensor rising   = mul(phase, 4.0);
    const Tensor falling  = sub(2.0, rising);
    const Tensor gate =
        where(less(phase, .25), rising, where(less(phase, .5), falling, mul(leak, phase)));
    const Tensor next_h = add(mul(gate, proposed.hidden), mul(sub(1.0, gate), h));
    const Tensor next_c = add(mul(gate, proposed.cell), mul(sub(1.0, gate), c));
    const Tensor valid  = recurrent_component_detail::StepValid(lengths, t);
    h                   = where(valid, next_h, h);
    c                   = where(valid, next_c, c);
    gates.push_back(recurrent_component_detail::MaskedDiagnostics(valid, proposed.gates));
    time_gates.push_back(recurrent_component_detail::MaskedDiagnostics(valid, gate));
    phases.push_back(recurrent_component_detail::MaskedDiagnostics(valid, phase));
    states.push_back(h);
    cells.push_back(c);
  }
  ExtendedRecurrentOutput result;
  result.logits      = recurrent_component_detail::Head(p, "head", h, 12, 2);
  result.states      = stack_time(states);
  result.cell_states = stack_time(cells);
  result.final_state = h;
  result.final_cell  = c;
  result.gates       = stack_time(gates);
  result.time_gates  = stack_time(time_gates);
  result.phase       = stack_time(phases);
  for (const auto& value : std::vector<Tensor>{
           result.logits,
           result.states,
           result.cell_states,
           result.final_state,
           result.final_cell,
           result.gates,
           result.time_gates,
           result.phase})
    recurrent_component_detail::Floating(value);
  return result;
}
/// @brief Evaluate the source Qrnn path with explicit controls.
///
/// Exact source twelve-step preset, with no fitting, optimizer, labels, hidden RNG draws or cached
/// state. Warm-store inference is immutable; synchronize missing-leaf creation/writes.
/// Initialization failure may leave earlier leaves. All state histories own storage and retain
/// applicable first/higher derivative paths after store collection. Invalid timesteps hold previous
/// states and zero gate diagnostics. Work O(N*12*12*(2+12)); input-only projections are batched
/// before pooling.
///
/// @param p Caller-owned Float32/64 CPU store; missing source leaves initialized, existing leaves
/// reused.
/// @param input Finite CPU[N,12,2] observations, matching store dtype.
/// @param lengths Int32/64 CPU[N] valid lengths in[0,12].
/// @param initial Explicit finite matching CPU[N,12] initial state tensors; two for stacked/gated
/// paths, one for truncated/dropout.
/// @return Owning flat live endpoints; inapplicable family fields undefined.
/// @throws std::invalid_argument Invalid finite CPU dtype, geometry or option domain.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);auto
/// x=n::zeros({1,12,2}),h=n::zeros({1,12});auto
/// lengths=n::from_vector(std::vector<std::int32_t>{4});auto
/// y=m::MaskedQrnnSequence(p,x,lengths,{h,h});
/// @endcode
/// @par References
/// - Bradbury et al. (2017), Quasi-Recurrent Neural Networks, https://arxiv.org/abs/1611.01576.
/// Width-two causal source filters and fo pooling.
/// @see MaskedTanhSequence()
inline ExtendedRecurrentOutput MaskedQrnnSequence(
    ParameterStore&            p,
    const Tensor&              input,
    const Tensor&              lengths,
    const std::vector<Tensor>& initial) {
  recurrent_component_detail::Sequence(p, input, lengths, initial, 2);

  const Tensor prior   = p.get("filter/prior", {2, 36}, init::uniform(std::sqrt(6.f / 40)));
  const Tensor current = p.get("filter/current", {2, 36}, init::uniform(std::sqrt(6.f / 40)));
  const Tensor bias    = p.get("filter/b", {36}, init::zeros());
  // All temporal features are computed together BEFORE the sequential pooling.
  const Tensor shifted = slice(pad(input, {{0, 0}, {1, 0}, {0, 0}}), {0, 0, 0}, {-1, 12, 2});
  const Tensor affine  = reshape(
      add(add(matmul(reshape(shifted, {-1, 2}), prior), matmul(reshape(input, {-1, 2}), current)),
          bias),
      {-1, 12, 36});
  const auto          features = time_steps(affine);
  Tensor              h = initial[0], c = initial[1];
  std::vector<Tensor> states{h}, cells{c}, gates;
  for (int t = 0; t < 12; ++t) {
    const auto   parts = split(features[t], 3, 1);
    const Tensor z     = tanh(parts[0]);
    const Tensor f = sigmoid(parts[1]), o = sigmoid(parts[2]);
    const Tensor next_c = add(mul(f, c), mul(sub(1.0, f), z));
    const Tensor next_h = mul(o, next_c);
    const Tensor mask   = recurrent_component_detail::StepMask(lengths, t);
    h                   = recurrent_component_detail::Blend(mask, next_h, h);
    c                   = recurrent_component_detail::Blend(mask, next_c, c);
    gates.push_back(
        recurrent_component_detail::MaskedDiagnostics(
            recurrent_component_detail::StepValid(lengths, t), concat({z, f, o}, 1)));
    states.push_back(h);
    cells.push_back(c);
  }
  ExtendedRecurrentOutput result;
  result.logits      = recurrent_component_detail::Head(p, "head", h, 12, 2);
  result.states      = stack_time(states);
  result.cell_states = stack_time(cells);
  result.final_state = h;
  result.final_cell  = c;
  result.gates       = stack_time(gates);
  for (const auto& value : std::vector<Tensor>{
           result.logits,
           result.states,
           result.cell_states,
           result.final_state,
           result.final_cell,
           result.gates})
    recurrent_component_detail::Floating(value);
  return result;
}
/// @brief Evaluate the source Sru path with explicit controls.
///
/// Exact source twelve-step preset, with no fitting, optimizer, labels, hidden RNG draws or cached
/// state. Warm-store inference is immutable; synchronize missing-leaf creation/writes.
/// Initialization failure may leave earlier leaves. All state histories own storage and retain
/// applicable first/higher derivative paths after store collection. Invalid timesteps hold previous
/// states and zero gate diagnostics. Work O(N*12*12*(2+12)); input-only projections are batched
/// before pooling.
///
/// @param p Caller-owned Float32/64 CPU store; missing source leaves initialized, existing leaves
/// reused.
/// @param input Finite CPU[N,12,2] observations, matching store dtype.
/// @param lengths Int32/64 CPU[N] valid lengths in[0,12].
/// @param initial Explicit finite matching CPU[N,12] initial state tensors; two for stacked/gated
/// paths, one for truncated/dropout.
/// @return Owning flat live endpoints; inapplicable family fields undefined.
/// @throws std::invalid_argument Invalid finite CPU dtype, geometry or option domain.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);auto
/// x=n::zeros({1,12,2}),h=n::zeros({1,12});auto
/// lengths=n::from_vector(std::vector<std::int32_t>{4});auto
/// y=m::MaskedSruSequence(p,x,lengths,{h,h});
/// @endcode
/// @par References
/// - Lei et al. (2018), Simple Recurrent Units for Highly Parallelizable Recurrence,
/// https://arxiv.org/abs/1709.02755. This preset uses input-only gates and projected skip.
/// @see MaskedTanhSequence()
inline ExtendedRecurrentOutput MaskedSruSequence(
    ParameterStore&            p,
    const Tensor&              input,
    const Tensor&              lengths,
    const std::vector<Tensor>& initial) {
  recurrent_component_detail::Sequence(p, input, lengths, initial, 2);

  const Tensor weights      = p.get("project/W", {2, 36}, init::uniform(std::sqrt(6.f / 38)));
  const Tensor bias         = p.get("project/b", {24}, init::zeros());
  const Tensor skip_weights = p.get("skip/W", {2, 12}, init::uniform(std::sqrt(6.f / 14)));
  // Input-only transforms are batched across time; the recurrence is channelwise.
  const Tensor        flat      = reshape(input, {-1, 2});
  const auto          features  = time_steps(reshape(matmul(flat, weights), {-1, 12, 36}));
  const auto          skip      = time_steps(reshape(matmul(flat, skip_weights), {-1, 12, 12}));
  const auto          gate_bias = split(bias, 2, 0);
  Tensor              h = initial[0], c = initial[1];
  std::vector<Tensor> states{h}, cells{c}, gates;
  for (int t = 0; t < 12; ++t) {
    const auto   parts = split(features[t], 3, 1);
    const Tensor z     = parts[0];
    const Tensor f = sigmoid(add(parts[1], gate_bias[0])), o = sigmoid(add(parts[2], gate_bias[1]));
    const Tensor next_c = add(mul(f, c), mul(sub(1.0, f), z));
    const Tensor next_h = add(mul(o, tanh(next_c)), mul(sub(1.0, o), skip[t]));
    const Tensor mask   = recurrent_component_detail::StepMask(lengths, t);
    h                   = recurrent_component_detail::Blend(mask, next_h, h);
    c                   = recurrent_component_detail::Blend(mask, next_c, c);
    gates.push_back(
        recurrent_component_detail::MaskedDiagnostics(
            recurrent_component_detail::StepValid(lengths, t), concat({z, f, o, skip[t]}, 1)));
    states.push_back(h);
    cells.push_back(c);
  }
  ExtendedRecurrentOutput result;
  result.logits      = recurrent_component_detail::Head(p, "head", h, 12, 2);
  result.states      = stack_time(states);
  result.cell_states = stack_time(cells);
  result.final_state = h;
  result.final_cell  = c;
  result.gates       = stack_time(gates);
  for (const auto& value : std::vector<Tensor>{
           result.logits,
           result.states,
           result.cell_states,
           result.final_state,
           result.final_cell,
           result.gates})
    recurrent_component_detail::Floating(value);
  return result;
}
/// @brief Construct the source nonuniform observation-time fixture.
///
/// No RNG, model state or fitting. Work/storage O(12*N).
///
/// @param n Positive row count at most1000000.
/// @return Owning detached Float32 CPU[N,12] repeated time rows.
/// @throws std::invalid_argument Invalid finite CPU dtype, geometry or option domain.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto t=m::RecurrentObservationTimes(2);
/// @endcode
/// @par References
/// - Neil, Pfeiffer and Liu (2016), Phased LSTM, https://arxiv.org/abs/1610.09513. Source fixed
/// r_on=.5 and explicit closed-phase leak.
/// @see MaskedTanhSequence()
inline Tensor RecurrentObservationTimes(std::int64_t n) {
  if (n < 1 || n > 1000000)
    throw std::invalid_argument("time fixture count");
  const float values[12] = {
      .13f, .81f, 1.59f, 1.93f, 2.8f, 3.14f, 4.4f, 5.1f, 5.6f, 7.f, 8.2f, 10.1f};
  std::vector<float> t(static_cast<std::size_t>(n) * 12);
  for (std::size_t i = 0; i < t.size(); ++i)
    t[i] = values[i % 12];
  return from_vector(t, {n, 12});
}
/// @brief Owning detached explicit recurrent dropout masks and their complete draw bank.
/// @par Example
/// @code
/// auto masks=dm::nn::models::MakeRecurrentDropoutMask(2,7,true);
/// @endcode
/// @see MakeRecurrentDropoutMask()
/// @par References
/// - Hochreiter and Schmidhuber (1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 ; Bradbury et al. (2017), Quasi-Recurrent Neural
/// Networks, https://arxiv.org/abs/1611.01576 . Family-specific conventions are documented by the
/// preset functions.
struct RecurrentDropoutMaskOutput {
  Tensor mask; ///< Float32 CPU[N,12,12] multipliers0 or1.25.
  Tensor
      uniforms; ///< Double CPU[N,12,12] midpoint-MT uniforms, including unused shared-time draws.
};
/// @brief Generate explicit source inverted-dropout multipliers.
///
/// Consumes144 midpoint Double uniforms per row, keeps1.25 when u>=.2, else0. No hidden sampling in
/// inference. Global/store RNG untouched; O(144*N).
///
/// @param n Row count in[1,1000000].
/// @param seed Local UInt32 MT19937 seed.
/// @param shared True reuses time0 draws for all twelve steps; false uses each time.
/// @return Owning mask and full bank; generator state is local.
/// @throws std::invalid_argument Invalid finite CPU dtype, geometry or option domain.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto y=m::MakeRecurrentDropoutMask(2,7,true);
/// @endcode
/// @par References
/// - Gal and Ghahramani (2016), A Theoretically Grounded Application of Dropout in Recurrent Neural
/// Networks, https://arxiv.org/abs/1512.05287. The caller chooses shared or per-time masks.
/// @see MaskedTanhSequence()
inline RecurrentDropoutMaskOutput MakeRecurrentDropoutMask(int n, std::uint32_t seed, bool shared) {
  if (n < 1 || n > 1000000)
    throw std::invalid_argument("dropout fixture count");
  std::vector<double> uniform(static_cast<std::size_t>(n) * 144);
  std::vector<float>  mask(uniform.size());
  std::mt19937        rng(static_cast<std::mt19937::result_type>(seed));
  for (auto& u : uniform)
    u = (double(rng()) + .5) / 4294967296.;
  for (int b = 0; b < n; ++b)
    for (int t = 0; t < 12; ++t)
      for (int j = 0; j < 12; ++j)
        mask[(static_cast<std::size_t>(b) * 12 + t) * 12 + j] =
            uniform[(static_cast<std::size_t>(b) * 12 + (shared ? 0 : t)) * 12 + j] >= .2 ? 1.25f
                                                                                          : 0.f;
  return {from_vector(mask, {n, 12, 12}), from_vector(uniform, {n, 12, 12})};
}
} // namespace datamunge::nn::models
