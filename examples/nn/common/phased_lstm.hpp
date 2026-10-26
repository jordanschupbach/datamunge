// Timestamp-driven triangular time gates blend proposed LSTM hidden/cell
// states with the previous states. Port of tflow's common/nn/phased_lstm.h
// (path) and common/ml/phased_lstm.h (model: 770 parameters in 7 tensors,
// Adam .01, seed 36003).
//
// tflow's extra placeholders (timestamps [B, 12] and the closed-phase leak) are
// the model's side inputs "timestamps" (default: PhasedTimestamps) and "leak"
// (default 0, inference); training sets leak .001. Evaluate also reports
// timestamp_gradient, the chosen logit's gradient w.r.t. the timestamps.
#pragma once

#include <cmath>
#include <memory>

#include "common/gated_sequence.hpp"
#include "common/recurrent_model.hpp"
#include "common/rnn.hpp"

namespace rnn_cells {

/// The fixture's shared nonuniform observation times, repeated for n rows.
inline Tensor PhasedTimestamps(std::int64_t n) {
  const float        values[12] = {.13f, .81f, 1.59f, 1.93f, 2.8f, 3.14f, 4.4f, 5.1f, 5.6f, 7.f, 8.2f, 10.1f};
  std::vector<float> t(static_cast<std::size_t>(n) * 12);
  for (std::size_t i = 0; i < t.size(); ++i) t[i] = values[i % 12];
  return nn::from_vector(t, {n, 12});
}

inline Outputs PhasedLstm(nn::ParameterStore& p, const Tensor& input, const Tensor& lengths, const std::vector<Tensor>& initial,
                          const Tensor& timestamps, const Tensor& leak) {
  const Tensor log_period = p.get("time/log_period", {12}, [](const nn::Shape&, nn::DType d, nn::Generator&) {
    std::vector<float> v(12);
    for (int j = 0; j < 12; ++j) v[j] = std::log(2.f + .5f * j);
    return nn::cast(nn::from_vector(v), d);
  });
  const Tensor phase_shift = p.get("time/shift", {12}, [](const nn::Shape&, nn::DType d, nn::Generator&) {
    std::vector<float> v(12);
    for (int j = 0; j < 12; ++j) v[j] = .137f * (2.f + .5f * j) * (j + .2f);
    return nn::cast(nn::from_vector(v), d);
  });
  const Tensor period = nn::exp(log_period);
  const auto   xs     = nn::time_steps(input);
  const auto   ts     = nn::unstack(timestamps, 1);
  Tensor       h = initial[0], c = initial[1];
  std::vector<Tensor> states{h}, cells{c}, gates, time_gates, phases;
  for (int t = 0; t < 12; ++t) {
    const GatedStep proposed = LstmGatedStep(p, "cell", xs[t], 2, 12, {h, c}, 1.0);
    const Tensor    raw      = nn::div(nn::sub(nn::expand_dims(ts[t], 1), phase_shift), period);
    const Tensor    phase    = nn::sub(raw, nn::stop_gradient(nn::floor(raw)));
    const Tensor    rising   = nn::mul(phase, 4.0);
    const Tensor    falling  = nn::sub(2.0, rising);
    const Tensor    gate     = nn::where(nn::less(phase, .25), rising, nn::where(nn::less(phase, .5), falling, nn::mul(leak, phase)));
    const Tensor    next_h   = nn::add(nn::mul(gate, proposed.state.h), nn::mul(nn::sub(1.0, gate), h));
    const Tensor    next_c   = nn::add(nn::mul(gate, proposed.state.c), nn::mul(nn::sub(1.0, gate), c));
    const Tensor    valid    = StepValid(lengths, t);
    h = nn::where(valid, next_h, h);
    c = nn::where(valid, next_c, c);
    gates.push_back(MaskedDiagnostics(valid, proposed.gates));
    time_gates.push_back(MaskedDiagnostics(valid, gate));
    phases.push_back(MaskedDiagnostics(valid, phase));
    states.push_back(h);
    cells.push_back(c);
  }
  return {{"logits", Head(p, "head", h, 12, 2)}, {"states", nn::stack_time(states)},         {"cell_states", nn::stack_time(cells)},
          {"final_state", h},                     {"final_cell", c},                           {"gates", nn::stack_time(gates)},
          {"time_gates", nn::stack_time(time_gates)}, {"phase", nn::stack_time(phases)}};
}

inline std::unique_ptr<RecurrentModel> PhasedLstmModel(bool training = true) {
  auto m = std::make_unique<RecurrentModel>(
      36003, nullptr, nn::Shape{12, 2}, std::vector<nn::Shape>{{12}, {12}},
      std::vector<std::string>{"initial_gradient", "initial_cell_gradient"},
      std::vector<std::string>{"logits", "probability", "states", "final_state", "cell_states", "final_cell", "gates", "time_gates", "phase"});
  RecurrentModel* model = m.get();
  m->forward = [model](nn::ParameterStore& p, const Tensor& x, const Tensor& l, const std::vector<Tensor>& s) {
    const auto& side = model->side_inputs;
    const auto  t = side.find("timestamps"), k = side.find("leak");
    return PhasedLstm(p, x, l, s, t != side.end() ? t->second : PhasedTimestamps(x.shape()[0]),
                      k != side.end() ? k->second : nn::scalar(0.0));
  };
  m->evaluate_extra = [model](Outputs& result, const Feed& x) {
    auto&  side  = model->side_inputs;
    const auto it = side.find("timestamps");
    Tensor times = (it != side.end() ? it->second : PhasedTimestamps(x.inputs.shape()[0])).detach().clone();
    times.requires_grad_(true);
    const auto saved = side;
    side["timestamps"] = times;
    const Outputs out  = model->forward(model->store, x.inputs, x.lengths, model->Initials(x.inputs.shape()[0]));
    side               = saved;
    nn::GradOptions options;
    options.allow_unused       = true;
    const auto g               = nn::grad(std::vector<Tensor>{model->loss_fn(out, x.labels).second}, {times}, {}, options);
    result["timestamp_gradient"] = g[0].defined() ? g[0].detach() : nn::zeros_like(times.detach());
  };
  m->Build(training);
  return m;
}

}  // namespace rnn_cells
