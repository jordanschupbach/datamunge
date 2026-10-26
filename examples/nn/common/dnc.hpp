// Differentiable neural computer allocation, freeing, temporal links and memory
// access. Port of tflow's common/nn/dnc.h (4 slots x 4 wide memory, LSTM(8)
// controller, one write head with allocation + content addressing, one read
// head mixing backward / forward temporal links and content; 799 parameters).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common/ntm.hpp"

namespace memory_nets {

/// Softmax over slots of softplus(strength) * cosine(memory rows, key), with
/// tflow's 1e-6 guards. memory [B, 4, 4], key [B, 4], strength [B, 1].
inline Tensor DncContent(const Tensor& memory, const Tensor& key, const Tensor& strength) {
  const Tensor dot = nn::sum(nn::mul(memory, nn::expand_dims(key, 1)), {2});
  const Tensor mn  = nn::sqrt(nn::add(nn::sum(nn::square(memory), {2}), 1e-6));
  const Tensor kn  = nn::sqrt(nn::add(nn::sum(nn::square(key), {1}, true), 1e-6));
  return nn::softmax(nn::mul(nn::softplus(strength), nn::div(dot, nn::add(nn::mul(mn, kn), 1e-6))), -1);
}

/// Allocation weights from usage [B, 4]: slots sorted by (stop-gradient) usage
/// with stable ties, a_sorted[j] = (1 - u_j) prod_{k<j} u_k, mapped back to
/// slot order. Gradients flow through the sorted usage values only.
inline Tensor DncAllocation(const Tensor& usage) {
  const Tensor        u       = nn::add(nn::mul(usage, 1.0 - 1e-6), 1e-6);
  const Tensor        stopped = nn::stop_gradient(u);
  std::vector<Tensor> rows;
  for (int i = 0; i < 4; ++i) {
    const Tensor current = Cols(stopped, i, 1);
    Tensor       rank    = nn::zeros_like(current);
    for (int j = 0; j < 4; ++j) {
      const Tensor other  = Cols(stopped, j, 1);
      Tensor       before = nn::less(other, current);
      if (j < i) before = nn::logical_or(before, nn::equal(other, current));
      rank = nn::add(rank, nn::cast(before, nn::DType::Float32));
    }
    rows.push_back(nn::reshape(nn::one_hot(nn::cast(rank, nn::DType::Int32), 4), {-1, 4}));
  }
  const Tensor permutation = nn::stack(rows, 1);  // [B, slot, rank]
  const Tensor sorted      = nn::sum(nn::mul(nn::expand_dims(u, 2), permutation), {1});
  Tensor       product     = nn::add(nn::mul(Cols(sorted, 0, 1), 0.0), 1.0);
  std::vector<Tensor> values;
  for (int j = 0; j < 4; ++j) {
    const Tensor value = Cols(sorted, j, 1);
    values.push_back(nn::mul(nn::sub(1.0, value), product));
    product = nn::mul(product, value);
  }
  return nn::sum(nn::mul(permutation, nn::expand_dims(nn::concat(values, 1), 1)), {2});
}

struct DncLink {
  Tensor links, precedence;
};

/// L' = (1 - w_i - w_j) L + w_i p_j without the diagonal; p' = (1 - sum w) p + w.
inline DncLink DncTemporal(const Tensor& previous_links, const Tensor& previous_precedence, const Tensor& write) {
  const Tensor wi = nn::expand_dims(write, 2), wj = nn::expand_dims(write, 1);
  const Tensor links = nn::add(nn::mul(nn::sub(nn::sub(1.0, wi), wj), previous_links),
                               nn::mul(wi, nn::expand_dims(previous_precedence, 1)));
  const Tensor no_diagonal = nn::from_vector(
      std::vector<float>{0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0}, {1, 4, 4});
  const Tensor precedence =
      nn::add(nn::mul(nn::sub(1.0, nn::sum(write, {1}, true)), previous_precedence), write);
  return {nn::mul(links, no_diagonal), precedence};
}

struct DncState {
  Tensor h, c, memory, read, read_weights, write_weights, usage, precedence, links;
  std::vector<Tensor> Values() const { return {h, c, memory, read, read_weights, write_weights, usage, precedence, links}; }
  static DncState From(const std::vector<Tensor>& v) { return {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]}; }
};

inline const std::vector<std::string>& DncNames() {
  static const std::vector<std::string> names{"h",     "c",          "memory", "read", "read_weights", "write_weights",
                                              "usage", "precedence", "links"};
  return names;
}

struct DncAccess {
  DncState state;
  Tensor   allocation, forward, backward;
};

/// Memory access from a 24-wide interface: write key(4) strength(1) erase(4)
/// add(4) allocation gate(1) write gate(1) free gate(1), read key(4)
/// strength(1) read modes(3).
inline DncAccess DncMemory(const Tensor& interface, const DncState& previous) {
  auto part = [&](std::int64_t b, std::int64_t w) { return Cols(interface, b, w); };
  // Match the reference DNC: usage updates do not backpropagate into previous write addresses.
  const Tensor prior_write = nn::stop_gradient(previous.write_weights);
  const Tensor used        = nn::add(previous.usage, nn::mul(nn::sub(1.0, previous.usage), prior_write));
  const Tensor usage       = nn::mul(used, nn::sub(1.0, nn::mul(nn::sigmoid(part(15, 1)), previous.read_weights)));
  const Tensor allocation  = DncAllocation(usage);
  const Tensor content     = DncContent(previous.memory, part(0, 4), part(4, 1));
  const Tensor ga = nn::sigmoid(part(13, 1)), gw = nn::sigmoid(part(14, 1));
  const Tensor write   = nn::mul(gw, nn::add(nn::mul(ga, allocation), nn::mul(nn::sub(1.0, ga), content)));
  const Tensor weights = nn::expand_dims(write, 2);
  const Tensor erase   = nn::sigmoid(part(5, 4));
  const Tensor add     = part(9, 4);
  const Tensor memory  = nn::add(nn::mul(previous.memory, nn::sub(1.0, nn::mul(weights, nn::expand_dims(erase, 1)))),
                                 nn::mul(weights, nn::expand_dims(add, 1)));
  const auto   temporal = DncTemporal(previous.links, previous.precedence, write);
  const Tensor forward  = nn::sum(nn::mul(temporal.links, nn::expand_dims(previous.read_weights, 1)), {2});
  const Tensor backward = nn::sum(nn::mul(temporal.links, nn::expand_dims(previous.read_weights, 2)), {1});
  const Tensor read_content = DncContent(memory, part(16, 4), part(20, 1));
  const Tensor modes        = nn::softmax(part(21, 3), -1);
  const Tensor read_weights = nn::add(nn::add(nn::mul(backward, Cols(modes, 0, 1)), nn::mul(forward, Cols(modes, 1, 1))),
                                      nn::mul(read_content, Cols(modes, 2, 1)));
  const Tensor read = nn::sum(nn::mul(memory, nn::expand_dims(read_weights, 2)), {1});
  return {{previous.h, previous.c, memory, read, read_weights, write, usage, temporal.precedence, temporal.links},
          allocation,
          forward,
          backward};
}

struct DncStep {
  DncState state;
  Tensor   logits, interface, allocation, forward, backward;
};

inline DncStep DncCell(nn::ParameterStore& p, const Tensor& input, const DncState& previous) {
  const auto controller = nn::lstm_step(p, "controller", nn::concat({input, previous.read}, 1), {previous.h, previous.c});
  const Tensor interface = nn::dense(p, "interface", controller.h, 24);
  auto         access    = DncMemory(interface, previous);
  access.state.h         = controller.h;
  access.state.c         = controller.c;
  const Tensor logits    = nn::dense(p, "output", nn::concat({controller.h, access.state.read}, 1), 3);
  return {access.state, logits, interface, access.allocation, access.forward, access.backward};
}

/// Nine-step unroll from a zero state with valid steps t <= 2 * length
/// (tflow's nn::Dnc). Fields: logits, interface, allocation, forward, backward
/// [B, 9, .] (zero on invalid steps) and the nine state traces [B, 10, ...].
inline Outputs Dnc(nn::ParameterStore& p, const Tensor& input, const Tensor& lengths) {
  const std::int64_t batch = input.shape()[0];
  auto zero = [&](std::int64_t w) { return nn::zeros({batch, w}); };
  DncState state{zero(8), zero(8), nn::zeros({batch, 4, 4}), zero(4), zero(4), zero(4), zero(4), zero(4),
                 nn::zeros({batch, 4, 4})};
  std::vector<std::vector<Tensor>> traces;
  for (const auto& v : state.Values()) traces.push_back({v});
  std::vector<Tensor> logits, interfaces, allocation, forward, backward;
  const auto          steps = nn::time_steps(input);
  const Tensor        twice = nn::mul(nn::cast(lengths, nn::DType::Float32), 2.0);
  for (int t = 0; t < 9; ++t) {
    const auto   next  = DncCell(p, steps[t], state);
    const Tensor valid = nn::expand_dims(nn::greater_equal(twice, double(t)), 1);
    auto previous = state.Values(), values = next.state.Values();
    for (int j = 0; j < 9; ++j) {
      values[j] = nn::where((j == 2 || j == 8) ? nn::expand_dims(valid, 2) : valid, values[j], previous[j]);
      traces[j].push_back(values[j]);
    }
    state     = DncState::From(values);
    auto mask = [&](const Tensor& v) { return nn::where(valid, v, nn::zeros_like(v)); };
    logits.push_back(mask(next.logits));
    interfaces.push_back(mask(next.interface));
    allocation.push_back(mask(next.allocation));
    forward.push_back(mask(next.forward));
    backward.push_back(mask(next.backward));
  }
  Outputs path{{"logits", nn::stack_time(logits)},
               {"interface", nn::stack_time(interfaces)},
               {"allocation", nn::stack_time(allocation)},
               {"forward", nn::stack_time(forward)},
               {"backward", nn::stack_time(backward)}};
  for (int j = 0; j < 9; ++j) path[DncNames()[j]] = nn::stack(traces[j], 1);
  return path;
}

/// tflow's dnc Model: seed 45203, Adam .005.
inline std::unique_ptr<MemoryModel> DncModel(bool training, const Batch& example) { return CopyModel(45203, Dnc, training, example); }

}  // namespace memory_nets
