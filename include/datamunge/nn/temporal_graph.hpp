/// @file
/// @brief Score-before-update recurrent memory for temporal link prediction.
#pragma once

#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Own temporal link logits and post-event memory snapshots.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(7);
/// const auto event = n::from_vector(std::vector<int>{0,1,2}, {1,1,3});
/// const auto path = n::temporal_graph_memory(p, "links", event, n::ones({1,1}),
///                                            n::zeros({3}), 4);
/// // path.memory.shape() == {1,1,3,4}.
/// @endcode
/// @see temporal_graph_memory()
struct TemporalGraphPath {
  Tensor logits; ///< [B,T,2] positive and sampled negative scores before each event update.
  Tensor memory; ///< [B,T,N,H] node memories after each event update.
};

/// @brief Unroll a temporal graph link predictor on independently reset streams.
///
/// Scores both destinations using pre-event memory, then updates only the
/// positive endpoints with a shared GRU. Messages include each endpoint's
/// log(1 + elapsed time) and a signed source role. Memory and last-event
/// times start at zero. Output tensors preserve gradients across events.
/// Synchronize concurrent mutation of the shared parameter store.
///
/// @param store Parameter store owning the link, message and memory weights.
/// @param name Layer path for learned parameters.
/// @param events Int32/64 [B,T,3] distinct source, positive, and negative node IDs.
/// @param times [B,T] finite, nonnegative, nondecreasing event times.
/// @param roles [N] finite scalar source roles. Both this and times match the store dtype.
/// @param width Positive node memory width.
/// @return Positive and negative link logits and post-event states.
/// @throws std::invalid_argument For invalid event IDs, timestamps, or incompatible shapes.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(7);
/// const auto event = n::from_vector(std::vector<int>{0,1,2}, {1,1,3});
/// const auto path = n::temporal_graph_memory(p, "links", event, n::ones({1,1}),
///                                            n::zeros({3}), 4);
/// // The two first-step logits are equal because both destinations have zero memory.
/// @endcode
/// @par References
/// - Rossi et al. (2020), Temporal Graph Networks for Deep Learning on Dynamic Graphs,
///   https://arxiv.org/abs/2006.10637
/// @see gru_step(), dense()
inline TemporalGraphPath temporal_graph_memory(ParameterStore &store, const std::string &name,
                                                const Tensor &events, const Tensor &times,
                                                const Tensor &roles, std::int64_t width) {
  if (events.dim() != 3 || events.size(0) <= 0 || events.size(1) <= 0 || events.size(2) != 3 ||
      (events.dtype() != DType::Int32 && events.dtype() != DType::Int64) ||
      times.shape() != Shape{events.size(0), events.size(1)} || roles.dim() != 1 ||
      roles.size(0) < 3 || times.dtype() != store.dtype() || roles.dtype() != store.dtype() ||
      width <= 0 || !all(is_finite(roles)).item<bool>())
    throw std::invalid_argument("temporal_graph_memory: incompatible streams or roles");

  const auto batch = events.size(0), steps = events.size(1), nodes = roles.size(0);
  const auto ids = events.to_vector<std::int64_t>();
  const auto timestamps = times.to_vector<double>();
  for (std::size_t index = 0; index < timestamps.size(); ++index) {
    const auto a = ids[3 * index], b = ids[3 * index + 1], c = ids[3 * index + 2];
    const auto time = timestamps[index];
    if (a < 0 || a >= nodes || b < 0 || b >= nodes || c < 0 || c >= nodes ||
        a == b || a == c || b == c || !std::isfinite(time) || time < 0. ||
        (index % steps != 0 && time < timestamps[index - 1]))
      throw std::invalid_argument("temporal_graph_memory: invalid event or timestamp");
  }

  auto memory = zeros({batch, nodes, width}, times.dtype(), times.context());
  auto last = zeros({batch, nodes}, times.dtype(), times.context());
  std::vector<Tensor> scores, states;
  for (std::int64_t step = 0; step < steps; ++step) {
    const auto event = select(events, 1, step);
    const auto now = expand_dims(select(times, 1, step), 1);
    const auto mask = [&](int col) { return one_hot(select(event, 1, col), nodes, 1., 0., times.dtype()); };
    const auto a = mask(0), b = mask(1), negative = mask(2);
    const auto read = [&](const Tensor &sel) { return sum(expand_dims(sel, 2) * memory, {1}); };
    const auto ha = read(a), hb = read(b);
    const auto role = sum(a * roles, {1}, true);
    const auto predict = [&](const Tensor &other) {
      const auto input = concat({role, ha, other, ha * other}, 1);
      const auto hidden = dense(store, name + "/link_hidden", input, 16, Activation::Tanh);
      return dense(store, name + "/link", hidden, 1);
    };
    scores.push_back(concat({predict(hb), predict(read(negative))}, 1));

    const auto update = [&](const Tensor &self, const Tensor &other, const Tensor &sel,
                            double sign) {
      const auto elapsed = log1p(now - sum(last * sel, {1}, true));
      const auto raw = concat({self, other, elapsed, sign * role}, 1);
      const auto message = dense(store, name + "/message", raw, width, Activation::Tanh);
      return gru_step(store, name + "/memory", message, self, 0.);
    };
    const auto next_a = update(ha, hb, a, 1.), next_b = update(hb, ha, b, -1.);
    const auto touched = a + b, keep = 1. - touched;
    memory = expand_dims(keep, 2) * memory +
             expand_dims(a, 2) * expand_dims(next_a, 1) +
             expand_dims(b, 2) * expand_dims(next_b, 1);
    last = keep * last + touched * now;
    states.push_back(memory);
  }
  return {stack_time(scores), stack_time(states)};
}

} // namespace datamunge::nn
