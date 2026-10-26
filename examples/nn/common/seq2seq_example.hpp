// Shared driver of tflow's small encoder-decoder examples (seq2seq, luong,
// translation, pointer_generator, pointer_network; the same program as
// bahdanau.cpp with the model swapped). For each task:
//   1. TF parity (when tflow's recorded run output/models/<name>/<task> holds
//      initial_arrays.npz): load TF's initial weights, compare the initial test
//      outputs incl. every BPTT partial, the first-batch forward and parameter
//      gradients, replay the 48 Adam updates in TF's batch order and compare
//      the validation history, the final weights and the greedy decodes.
//   2. Own-initialisation run with every audit of the tflow example: restore
//      bit-exact, standalone one-step decoder vs the teacher-forced unroll,
//      deterministic label-free greedy decoding (teacher/target/lengths cannot
//      leak), padding invariance of every output and partial, the empty
//      source, inference immutability; outputs written like tflow's.
//
// A Model provides: `explicit Model(seed)` (creates every parameter), member
// `nn::ParameterStore p`, `Map Evaluate(const Map&)`, `std::vector<Tensor>
// Gradients(const Map&)`, `seq2seq::DecoderStep StepAt(const Map& batch, const
// Map& evaluated, int t)` (one standalone step from the teacher token and
// decoder state at t) and `Tensor Greedy(const Map&, Tensor* attention)`.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <datamunge/nn/nn.hpp>

#include "common/checks.hpp"
#include "common/paths.hpp"
#include "common/seq2seq.hpp"
#include "common/tf_replay.hpp"

namespace seq2seq_example {

namespace nn = datamunge::nn;
using nn::Tensor;
using replay::Map;
using seq2seq::kSteps;
using seq2seq::kUnits;

struct Spec {
  std::string              name;               // output/nn/models/<name>, tflow's output/models/<name>
  std::string              dataset;            // tflow dataset directory name
  std::vector<std::string> tasks;
  bool                     flat_data = false;  // data.npz directly under the dataset (translation)
  std::vector<std::string> assets{"data.npz", "manifest.json"};
  std::int64_t             params = 0, tensors = 0;
  int                      vocab = 14;         // output classes
  std::uint64_t            seed  = 0;          // own-initialisation seed (tflow's Params seed)
  int                      corrupt = 13;       // token written into teacher/target/padding
  bool                     pointer = false;    // pointer network: float points, STOP 5, BOS -1
  bool                     attention = true;   // decoded.npz carries the greedy attention
  std::string              exact_key = "greedy_exact_accuracy", exact_label = "greedyExact";
  std::string              tf_recorded;        // printed TF reference line
  // Replay tolerances after 48 Adam updates (float32 rounding differences grow
  // along the trajectory; the pointer models amplify them most).
  double history_tol = 1e-4, weights_tol = 1e-3;
};

struct Parity {
  bool   ran = false;
  double outputs = 0, forward = 0, gradients = 0, first_after = -1, history = 0, weights = -1, final_outputs = -1,
         final_gradients = -1;
  int    greedy_same = -1, greedy_total = 0;
};

/// Sets every padded input/teacher position of `b` to the spec's corrupt values.
inline void Pad(const Spec& spec, Map& b) {
  const auto n  = b.at("input").shape()[0];
  const auto il = b.at("input_lengths").to_vector<int>(), ol = b.at("output_lengths").to_vector<int>();
  auto       te = b.at("teacher").to_vector<std::int32_t>();
  for (std::int64_t i = 0; i < n; ++i)
    for (int t = 0; t < kSteps; ++t)
      if (t >= ol[i]) te[i * kSteps + t] = spec.corrupt;
  b.at("teacher") = nn::from_vector(te, {n, kSteps});
  if (spec.pointer) {
    auto       in    = b.at("input").to_vector<float>();
    const auto shape = b.at("input").shape();
    for (std::int64_t i = 0; i < n; ++i)
      for (int t = 0; t < shape[1]; ++t)
        if (t >= il[i])
          for (int k = 0; k < shape[2]; ++k) in[(i * shape[1] + t) * shape[2] + k] = 13.f;
    b.at("input") = nn::from_vector(in, shape);
  } else {
    auto in = b.at("input").to_vector<std::int32_t>();
    for (std::int64_t i = 0; i < n; ++i)
      for (int t = 0; t < kSteps; ++t)
        if (t >= il[i]) in[i * kSteps + t] = 13;
    b.at("input") = nn::from_vector(in, {n, kSteps});
  }
}

struct Result {
  Parity parity;
  double test_ce = 0, teacher_accuracy = 0;
  int    exact = 0;
  std::vector<double> history;
};

/// main(): options epochs=3 data=PATH tf=PATH.

/// Zero LSTM state [n, 8].
inline Tensor Zero(std::int64_t n) { return nn::zeros({n, kUnits}); }

/// Evaluate() bookkeeping shared by the models: the forward fields plus one
/// partial per `wrt` (zeros where the loss does not depend on it).
inline Map Fields(const Tensor& loss, const std::vector<std::pair<std::string, Tensor>>& fields, const std::vector<Tensor>& wrt) {
  nn::GradOptions o;
  o.allow_unused = true;
  const auto g   = nn::grad(std::vector<Tensor>{loss}, wrt, {}, o);
  Map        out{{"loss", loss.detach()}};
  for (const auto& [k, v] : fields) out[k] = v.detach();
  for (std::size_t i = 0; i < g.size(); ++i) out["partial" + std::to_string(i)] = g[i].defined() ? g[i] : nn::zeros_like(wrt[i]).detach();
  return out;
}

}  // namespace seq2seq_example
