#pragma once
/// @file dnc.hpp
/// @brief Explicit differentiable neural computer allocation, temporal links and controller.
#include <datamunge/nn/memory_networks.hpp>
#include <map>
namespace datamunge::nn::models {
/// @brief Compute softplus-strength cosine-content addressing.
///
/// The compact preset uses four slots of width four and Float32 CPU tensors.
/// All returned handles own storage and retain applicable derivatives. No optimizer or training.
/// @param memory Float32 CPU[N,4,4] slot contents.
/// @param key Float32 CPU[N,4] search key.
/// @param strength Float32 CPU[N,1] unconstrained strength.
/// @return Live CPU[N,4] normalized content weights.
/// @throws std::invalid_argument Invalid explicit bounds or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto
/// y=m::DncContent(n::zeros({1,4,4}),n::zeros({1,4}),n::zeros({1,1}));
/// @endcode
/// @par References
/// - Graves et al. (2016), Hybrid computing using a neural network with dynamic external memory,
/// https://doi.org/10.1038/nature20101 . This is the compact four-slot teaching preset.
/// @see DncSequence(), DncAllocation()
inline Tensor DncContent(const Tensor& memory, const Tensor& key, const Tensor& strength) {
  const Tensor dot = sum(mul(memory, expand_dims(key, 1)), {2});
  const Tensor mn  = sqrt(add(sum(square(memory), {2}), 1e-6));
  const Tensor kn  = sqrt(add(sum(square(key), {1}, true), 1e-6));
  return softmax(mul(softplus(strength), div(dot, add(mul(mn, kn), 1e-6))), -1);
}

/// @brief Allocate to least-used slots with stable ties and differentiable usage values.
///
/// The compact preset uses four slots of width four and Float32 CPU tensors.
/// All returned handles own storage and retain applicable derivatives. No optimizer or training.
/// @param usage Float32 CPU[N,4] finite usage in[0,1].
/// @return Live CPU[N,4] allocation; sorted ranks are stopped, usage is mapped to1e-6+(1-1e-6)u.
/// @throws std::invalid_argument Invalid explicit bounds or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto y=m::DncAllocation(n::zeros({1,4}));
/// @endcode
/// @par References
/// - Graves et al. (2016), Hybrid computing using a neural network with dynamic external memory,
/// https://doi.org/10.1038/nature20101 . This is the compact four-slot teaching preset.
/// @see DncSequence(), DncAllocation()
inline Tensor DncAllocation(const Tensor& usage) {
  recurrent_component_detail::Floating(usage);
  if (usage.dim() != 2 || usage.size(1) != 4 || usage.dtype() != DType::Float32)
    throw std::invalid_argument("DNC usage requires Float32[N,4]");
  for (float v : usage.to_vector<float>())
    if (v < 0 || v > 1)
      throw std::invalid_argument("DNC usage outside [0,1]");
  const Tensor        u       = add(mul(usage, 1.0 - 1e-6), 1e-6);
  const Tensor        stopped = stop_gradient(u);
  std::vector<Tensor> rows;
  for (int i = 0; i < 4; ++i) {
    const Tensor current = memory_network_detail::Cols(stopped, i, 1);
    Tensor       rank    = zeros_like(current);
    for (int j = 0; j < 4; ++j) {
      const Tensor other  = memory_network_detail::Cols(stopped, j, 1);
      Tensor       before = less(other, current);
      if (j < i)
        before = logical_or(before, equal(other, current));
      rank = add(rank, cast(before, DType::Float32));
    }
    rows.push_back(reshape(one_hot(cast(rank, DType::Int32), 4), {-1, 4}));
  }
  const Tensor        permutation = stack(rows, 1); // [B, slot, rank]
  const Tensor        sorted      = sum(mul(expand_dims(u, 2), permutation), {1});
  Tensor              product     = add(mul(memory_network_detail::Cols(sorted, 0, 1), 0.0), 1.0);
  std::vector<Tensor> values;
  for (int j = 0; j < 4; ++j) {
    const Tensor value = memory_network_detail::Cols(sorted, j, 1);
    values.push_back(mul(sub(1.0, value), product));
    product = mul(product, value);
  }
  return sum(mul(permutation, expand_dims(concat(values, 1), 1)), {2});
}

/// @brief Own the live DncLink endpoints of the four-slot DNC preset.
///
/// Handles retain storage and graph independently of their caller; no optimizer or cached state.
/// @par Example
/// @code
/// dm::nn::models::DncLink value;
/// @endcode
/// @par References
/// - Graves et al. (2016), Hybrid computing using a neural network with dynamic external memory,
/// https://doi.org/10.1038/nature20101 . This is the compact four-slot teaching preset.
/// @see DncSequence()
struct DncLink {
  Tensor links;      ///< Float32 CPU[N,4,4] temporal links with zero diagonal.
  Tensor precedence; ///< Float32 CPU[N,4] temporal precedence.
};

/// @brief Update temporal links and precedence with a zero link diagonal.
///
/// The compact preset uses four slots of width four and Float32 CPU tensors.
/// All returned handles own storage and retain applicable derivatives. No optimizer or training.
/// @param previous_links Float32 CPU[N,4,4] previous links.
/// @param previous_precedence Float32 CPU[N,4] prior precedence.
/// @param write Float32 CPU[N,4] write weights.
/// @return Live new links and precedence.
/// @throws std::invalid_argument Invalid explicit bounds or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto
/// y=m::DncTemporal(n::zeros({1,4,4}),n::zeros({1,4}),n::zeros({1,4}));
/// @endcode
/// @par References
/// - Graves et al. (2016), Hybrid computing using a neural network with dynamic external memory,
/// https://doi.org/10.1038/nature20101 . This is the compact four-slot teaching preset.
/// @see DncSequence(), DncAllocation()
inline DncLink
DncTemporal(const Tensor& previous_links, const Tensor& previous_precedence, const Tensor& write) {
  const Tensor wi = expand_dims(write, 2), wj = expand_dims(write, 1);
  const Tensor links =
      add(mul(sub(sub(1.0, wi), wj), previous_links), mul(wi, expand_dims(previous_precedence, 1)));
  const Tensor no_diagonal =
      from_vector(std::vector<float>{0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0}, {1, 4, 4});
  const Tensor precedence = add(mul(sub(1.0, sum(write, {1}, true)), previous_precedence), write);
  return {mul(links, no_diagonal), precedence};
}

/// @brief Own the live DncState endpoints of the four-slot DNC preset.
///
/// Handles retain storage and graph independently of their caller; no optimizer or cached state.
/// @par Example
/// @code
/// dm::nn::models::DncState value;
/// @endcode
/// @par References
/// - Graves et al. (2016), Hybrid computing using a neural network with dynamic external memory,
/// https://doi.org/10.1038/nature20101 . This is the compact four-slot teaching preset.
/// @see DncSequence()
struct DncState {
  Tensor h;             ///< Float32 CPU[N,8] controller hidden state.
  Tensor c;             ///< Float32 CPU[N,8] controller cell state.
  Tensor memory;        ///< Float32 CPU[N,4,4] slot contents.
  Tensor read;          ///< Float32 CPU[N,4] read vector.
  Tensor read_weights;  ///< Float32 CPU[N,4] read weights.
  Tensor write_weights; ///< Float32 CPU[N,4] write weights.
  Tensor usage;         ///< Float32 CPU[N,4] slot usage in[0,1].
  Tensor precedence;    ///< Float32 CPU[N,4] temporal precedence.
  Tensor links;         ///< Float32 CPU[N,4,4] temporal links with zero diagonal.

  /// @brief Return the nine state handles in stable h/c/memory/read/address/usage/link order.
  /// @return Owning live handles sharing the state storage.
  std::vector<Tensor> Values() const {
    return {h, c, memory, read, read_weights, write_weights, usage, precedence, links};
  }
  /// @brief Reassemble nine state handles in Values() order.
  /// @param v Exactly nine state handles.
  /// @return State sharing the supplied storage and graph.
  /// @throws std::invalid_argument Handle count is not nine.
  /// @par Example
  /// @code
  /// auto state=dm::nn::models::DncState::From(std::vector<dm::nn::Tensor>(9));
  /// @endcode
  /// @see Values()
  static DncState From(const std::vector<Tensor>& v) {
    if (v.size() != 9)
      throw std::invalid_argument("nine DNC states required");
    return {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]};
  }
};

/// @brief Return stable names for the nine DNC state traces.
///
/// The compact preset uses four slots of width four and Float32 CPU tensors.
/// All returned handles own storage and retain applicable derivatives. No optimizer or training.
/// @return Reference to immutable process-lifetime names; concurrent reads are safe.
/// @throws std::invalid_argument Invalid explicit bounds or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;const auto& names=m::DncNames();
/// @endcode
/// @par References
/// - Graves et al. (2016), Hybrid computing using a neural network with dynamic external memory,
/// https://doi.org/10.1038/nature20101 . This is the compact four-slot teaching preset.
/// @see DncSequence(), DncAllocation()
inline const std::vector<std::string>& DncNames() {
  static const std::vector<std::string> names{
      "h", "c", "memory", "read", "read_weights", "write_weights", "usage", "precedence", "links"};
  return names;
}

/// @brief Own the live DncAccess endpoints of the four-slot DNC preset.
///
/// Handles retain storage and graph independently of their caller; no optimizer or cached state.
/// @par Example
/// @code
/// dm::nn::models::DncAccess value;
/// @endcode
/// @par References
/// - Graves et al. (2016), Hybrid computing using a neural network with dynamic external memory,
/// https://doi.org/10.1038/nature20101 . This is the compact four-slot teaching preset.
/// @see DncSequence()
struct DncAccess {
  DncState state;      ///< Live new state, owning all nine endpoint handles.
  Tensor   allocation; ///< Float32 CPU[N,4] least-used-slot allocation.
  Tensor   forward;    ///< Float32 CPU[N,4] forward temporal traversal.
  Tensor   backward;   ///< Float32 CPU[N,4] backward temporal traversal.
};

/// @brief Access four-slot DNC memory with allocation, freeing and temporal traversal.
///
/// The compact preset uses four slots of width four and Float32 CPU tensors.
/// All returned handles own storage and retain applicable derivatives. No optimizer or training.
/// @param interface Float32 CPU[N,24] write key/strength/erase/add/gates, read key/strength/modes.
/// @param previous Aligned Float32 CPU state; h/c are forwarded unchanged.
/// @return Live new state and allocation/forward/backward endpoints; prior write weights are
/// stopped in usage.
/// @throws std::invalid_argument Invalid explicit bounds or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;m::DncState
/// s{n::zeros({1,8}),n::zeros({1,8}),n::zeros({1,4,4}),n::zeros({1,4}),n::zeros({1,4}),n::zeros({1,4}),n::zeros({1,4}),n::zeros({1,4}),n::zeros({1,4,4})};auto
/// y=m::DncMemory(n::zeros({1,24}),s);
/// @endcode
/// @par References
/// - Graves et al. (2016), Hybrid computing using a neural network with dynamic external memory,
/// https://doi.org/10.1038/nature20101 . This is the compact four-slot teaching preset.
/// @see DncSequence(), DncAllocation()
inline DncAccess DncMemory(const Tensor& interface, const DncState& previous) {
  auto part = [&](std::int64_t b, std::int64_t w) {
    return memory_network_detail::Cols(interface, b, w);
  };
  // Match the reference DNC: usage updates do not backpropagate into previous write addresses.
  const Tensor prior_write = stop_gradient(previous.write_weights);
  const Tensor used        = add(previous.usage, mul(sub(1.0, previous.usage), prior_write));
  const Tensor usage       = mul(used, sub(1.0, mul(sigmoid(part(15, 1)), previous.read_weights)));
  const Tensor allocation  = DncAllocation(usage);
  const Tensor content     = DncContent(previous.memory, part(0, 4), part(4, 1));
  const Tensor ga = sigmoid(part(13, 1)), gw = sigmoid(part(14, 1));
  const Tensor write       = mul(gw, add(mul(ga, allocation), mul(sub(1.0, ga), content)));
  const Tensor weights     = expand_dims(write, 2);
  const Tensor erase       = sigmoid(part(5, 4));
  const Tensor write_value = part(9, 4);
  const Tensor memory =
      add(mul(previous.memory, sub(1.0, mul(weights, expand_dims(erase, 1)))),
          mul(weights, expand_dims(write_value, 1)));
  const auto   temporal     = DncTemporal(previous.links, previous.precedence, write);
  const Tensor forward      = sum(mul(temporal.links, expand_dims(previous.read_weights, 1)), {2});
  const Tensor backward     = sum(mul(temporal.links, expand_dims(previous.read_weights, 2)), {1});
  const Tensor read_content = DncContent(memory, part(16, 4), part(20, 1));
  const Tensor modes        = softmax(part(21, 3), -1);
  const Tensor read_weights =
      add(add(mul(backward, memory_network_detail::Cols(modes, 0, 1)),
              mul(forward, memory_network_detail::Cols(modes, 1, 1))),
          mul(read_content, memory_network_detail::Cols(modes, 2, 1)));
  const Tensor read = sum(mul(memory, expand_dims(read_weights, 2)), {1});
  return {
      {previous.h,
       previous.c,
       memory,
       read,
       read_weights,
       write,
       usage,
       temporal.precedence,
       temporal.links},
      allocation,
      forward,
      backward};
}

/// @brief Own the live DncStep endpoints of the four-slot DNC preset.
///
/// Handles retain storage and graph independently of their caller; no optimizer or cached state.
/// @par Example
/// @code
/// dm::nn::models::DncStep value;
/// @endcode
/// @par References
/// - Graves et al. (2016), Hybrid computing using a neural network with dynamic external memory,
/// https://doi.org/10.1038/nature20101 . This is the compact four-slot teaching preset.
/// @see DncSequence()
struct DncStep {
  DncState state;      ///< Live new state, owning all nine endpoint handles.
  Tensor   logits;     ///< Float32 CPU[N,3] output logits.
  Tensor   interface;  ///< Float32 CPU[N,24] unactivated access interface.
  Tensor   allocation; ///< Float32 CPU[N,4] least-used-slot allocation.
  Tensor   forward;    ///< Float32 CPU[N,4] forward temporal traversal.
  Tensor   backward;   ///< Float32 CPU[N,4] backward temporal traversal.
};

/// @brief Advance an LSTM8 controller and four-slot DNC access by one step.
///
/// The compact preset uses four slots of width four and Float32 CPU tensors.
/// All returned handles own storage and retain applicable derivatives. No optimizer or training.
/// @param p Caller-owned Float32 CPU store, lazy799 parameters in seven tensors.
/// @param input Float32 CPU[N,4] input row.
/// @param previous Aligned explicit Float32 CPU state.
/// @return Live new state, logits and access endpoints; caller retains every recurrent state.
/// @throws std::invalid_argument Invalid explicit bounds or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);m::DncState
/// s{n::zeros({1,8}),n::zeros({1,8}),n::zeros({1,4,4}),n::zeros({1,4}),n::zeros({1,4}),n::zeros({1,4}),n::zeros({1,4}),n::zeros({1,4}),n::zeros({1,4,4})};auto
/// y=m::DncCell(p,n::zeros({1,4}),s);
/// @endcode
/// @par References
/// - Graves et al. (2016), Hybrid computing using a neural network with dynamic external memory,
/// https://doi.org/10.1038/nature20101 . This is the compact four-slot teaching preset.
/// @see DncSequence(), DncAllocation()
inline DncStep DncCell(ParameterStore& p, const Tensor& input, const DncState& previous) {
  recurrent_component_detail::Floating(input);
  if (input.dim() != 2 || input.size(1) != 4 || input.dtype() != DType::Float32 ||
      p.dtype() != DType::Float32)
    throw std::invalid_argument("DNC cell Float32[N,4] input/store required");
  auto rows = input.size(0);
  memory_network_detail::Geometry(previous.h, {rows, 8});
  memory_network_detail::Geometry(previous.c, {rows, 8});
  memory_network_detail::Geometry(previous.memory, {rows, 4, 4});
  memory_network_detail::Geometry(previous.links, {rows, 4, 4});
  for (const auto& value : std::vector<Tensor>{
           previous.read,
           previous.read_weights,
           previous.write_weights,
           previous.usage,
           previous.precedence})
    memory_network_detail::Geometry(value, {rows, 4});
  const auto controller =
      lstm_step(p, "controller", concat({input, previous.read}, 1), {previous.h, previous.c});
  const Tensor interface = dense(p, "interface", controller.h, 24);
  auto         access    = DncMemory(interface, previous);
  access.state.h         = controller.h;
  access.state.c         = controller.c;
  const Tensor logits    = dense(p, "output", concat({controller.h, access.state.read}, 1), 3);
  return {access.state, logits, interface, access.allocation, access.forward, access.backward};
}

/// @brief Unroll nine copy-task steps with explicit length masking.
///
/// The compact preset uses four slots of width four and Float32 CPU tensors.
/// All returned handles own storage and retain applicable derivatives. No optimizer or training.
/// @param p Caller-owned Float32 CPU store; lazy799 parameters.
/// @param input Float32 CPU[N,9,4] input, delimiter and recall steps.
/// @param lengths Int32 CPU[N] original bit-sequence lengths in1..4.
/// @return Live logits/interface/allocation/forward/backward[N,9,...] and nine states[N,10,...];
/// padding keeps state and zeros outputs.
/// @throws std::invalid_argument Invalid explicit bounds or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);auto
/// y=m::DncSequence(p,n::zeros({1,9,4}),n::from_vector(std::vector<int>{2}));
/// @endcode
/// @par References
/// - Graves et al. (2016), Hybrid computing using a neural network with dynamic external memory,
/// https://doi.org/10.1038/nature20101 . This is the compact four-slot teaching preset.
/// @see DncSequence(), DncAllocation()
inline std::map<std::string, Tensor>
DncSequence(ParameterStore& p, const Tensor& input, const Tensor& lengths) {
  recurrent_component_detail::Floating(input);
  if (input.dim() != 3 || input.size(1) != 9 || input.size(2) != 4 ||
      input.dtype() != DType::Float32 || lengths.dtype() != DType::Int32 ||
      lengths.shape() != Shape{input.size(0)})
    throw std::invalid_argument("DNC sequence requires Float32[N,9,4] and Int32[N]");
  for (int v : lengths.to_vector<int>())
    if (v < 1 || v > 4)
      throw std::invalid_argument("DNC lengths must be1..4");
  const std::int64_t batch = input.shape()[0];
  auto               zero  = [&](std::int64_t w) { return zeros({batch, w}); };
  DncState           state{
      zero(8),
      zero(8),
      zeros({batch, 4, 4}),
      zero(4),
      zero(4),
      zero(4),
      zero(4),
      zero(4),
      zeros({batch, 4, 4})};
  std::vector<std::vector<Tensor>> traces;
  for (const auto& v : state.Values())
    traces.push_back({v});
  std::vector<Tensor> logits, interfaces, allocation, forward, backward;
  const auto          steps = time_steps(input);
  const Tensor        twice = mul(cast(lengths, DType::Float32), 2.0);
  for (int t = 0; t < 9; ++t) {
    const auto   next     = DncCell(p, steps[t], state);
    const Tensor valid    = expand_dims(greater_equal(twice, double(t)), 1);
    auto         previous = state.Values(), values = next.state.Values();
    for (int j = 0; j < 9; ++j) {
      values[j] = where((j == 2 || j == 8) ? expand_dims(valid, 2) : valid, values[j], previous[j]);
      traces[j].push_back(values[j]);
    }
    state     = DncState::From(values);
    auto mask = [&](const Tensor& v) { return where(valid, v, zeros_like(v)); };
    logits.push_back(mask(next.logits));
    interfaces.push_back(mask(next.interface));
    allocation.push_back(mask(next.allocation));
    forward.push_back(mask(next.forward));
    backward.push_back(mask(next.backward));
  }
  std::map<std::string, Tensor> path{
      {"logits", stack_time(logits)},
      {"interface", stack_time(interfaces)},
      {"allocation", stack_time(allocation)},
      {"forward", stack_time(forward)},
      {"backward", stack_time(backward)}};
  for (int j = 0; j < 9; ++j)
    path[DncNames()[j]] = stack(traces[j], 1);
  return path;
}

} // namespace datamunge::nn::models
