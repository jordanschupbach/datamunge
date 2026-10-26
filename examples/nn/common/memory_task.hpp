// Shared model/data helpers for tflow's 4.3 memory-augmented networks (ntm, dnc, memnet,
// key_value). tflow repeats the same Model class and three-epoch main() in
// each .cc: placeholders for a batch, a hand-built memory network, a loss,
// symbolic gradients of every parameter and of the (encoded) inputs, Adam over
// every parameter, a SaveV2 checkpoint, and npz records of the initial / first
// update / final state. Here the graph is a `Forward` function, session fetches
// are eager calls, gradients come from nn::grad and checkpoints are .npz files
// of the parameter store.
//
// Flow (identical to tflow): copy output/datasets/<dataset>/{data.npz,
// manifest.json} into the run directory, 128 fit / 32 validation / 32 test rows,
// batch 8 with tflow's BatchOrder(128, 8, seed), three epochs = 48 updates,
// records initial_arrays / initial_outputs / first_batch / initial_gradients /
// first_after / final_gradients, save -> restore into a fresh model (bit-exact),
// label-free prediction equals evaluation, weights immutable under inference.
// `init=PATH` (tflow's initial_arrays.npz) replays tflow's run; with the tflow
// run directory present (`tf=`) every replayed record is compared.
#pragma once

#include <chrono>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/marked_sequence_demo.hpp"
#include "common/paths.hpp"
#include "common/recurrent_model.hpp"

namespace memory_nets {

namespace nn = datamunge::nn;
namespace fs = std::filesystem;
using nn::Tensor;
using Batch   = std::map<std::string, Tensor>;
using Outputs = std::map<std::string, Tensor>;
using rnn_cells::MaxAbsDiff;
using rnn_cells::SameBits;

/// x[:, begin:begin+width] for a [B, N] tensor (tflow's Slice(x, {0, begin}, {-1, width})).
inline Tensor Cols(const Tensor& x, std::int64_t begin, std::int64_t width) { return nn::slice(x, {0, begin}, {-1, width}); }

/// The rows of every array of a batch (tflow's Rows / ml::OcrRows).
inline Batch Rows(const Batch& a, const std::vector<std::int64_t>& rows) {
  Batch b;
  for (const auto& [k, v] : a) b[k] = nn::take_rows(v, rows);
  return b;
}

/// "<split>_<key>" arrays of data.npz under their short keys.
inline Batch Split(const nn::NamedTensors& all, const std::string& split, const std::vector<std::string>& keys) {
  Batch b;
  for (const auto& k : keys) b[k] = rnn_cells::Find(all, split + "_" + k);
  return b;
}

inline void SaveMap(const fs::path& file, const std::map<std::string, Tensor>& m) {
  nn::save_npz(file.string(), nn::NamedTensors(m.begin(), m.end()));
}

/// Bit-exact equality of every array two maps share (and the same key sets).
inline bool SameMaps(const Outputs& a, const Outputs& b) {
  if (a.size() != b.size()) return false;
  for (const auto& [k, v] : a)
    if (!b.count(k) || !SameBits(v, b.at(k))) return false;
  return true;
}

/// A memory network: the parameters plus a forward pass over a batch.
class MemoryModel {
 public:
  /// (store, differentiable leaves, batch) -> named outputs incl. "logits".
  using Forward = std::function<Outputs(nn::ParameterStore&, const Batch& leaves, const Batch& batch)>;
  nn::ParameterStore store;
  Forward            forward;
  /// Float leaves built from a batch (tflow's x placeholder or OneHot encodings).
  std::function<Batch(const Batch&)> leaves;
  /// (outputs, batch with labels) -> training loss.
  std::function<Tensor(const Outputs&, const Batch&)> loss;
  /// leaf name -> Evaluate field of its loss gradient ("input" -> "input_gradient").
  std::vector<std::pair<std::string, std::string>> leaf_gradients;
  std::unique_ptr<nn::Optimizer> optimizer;

  MemoryModel(std::uint64_t seed, Forward f) : store(seed), forward(std::move(f)) {}

  /// Creates every parameter by running `example` through the forward pass;
  /// with `training`, adds Adam at `learning_rate`.
  void Build(const Batch& example, bool training, double learning_rate) {
    {
      nn::NoGradGuard guard;
      forward(store, leaves(example), example);
    }
    if (training) {
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      optimizer   = std::make_unique<nn::Optimizer>(store, learning_rate, config);
    }
  }

  /// Logits from the inputs and lengths only (no labels are read).
  Tensor Predict(const Batch& b) {
    nn::NoGradGuard guard;
    return forward(store, leaves(b), b).at("logits");
  }

  /// Every forward field, "loss" and the loss gradient of each leaf.
  Outputs Evaluate(const Batch& b) {
    Batch leaf = leaves(b);
    for (auto& [k, v] : leaf) {
      v = v.detach().clone();
      v.requires_grad_(true);
    }
    Outputs             out = forward(store, leaf, b);
    const Tensor        l   = loss(out, b);
    std::vector<Tensor> wrt;
    for (const auto& [name, field] : leaf_gradients) wrt.push_back(leaf.at(name));
    nn::GradOptions options;
    options.allow_unused = true;
    const auto g         = nn::grad(std::vector<Tensor>{l}, wrt, {}, options);
    Outputs    result;
    for (const auto& [k, v] : out) result[k] = v.detach();
    result["loss"] = l.detach();
    for (std::size_t i = 0; i < wrt.size(); ++i)
      result[leaf_gradients[i].second] = g[i].defined() ? g[i].detach() : nn::zeros_like(wrt[i].detach());
    return result;
  }

  std::vector<Tensor> RawGradients(const Batch& b) {
    const Outputs out    = forward(store, leaves(b), b);
    const auto    params = store.parameters();
    nn::GradOptions options;
    options.allow_unused = true;
    auto g = nn::grad(std::vector<Tensor>{loss(out, b)}, params, {}, options);
    for (std::size_t i = 0; i < g.size(); ++i)
      if (!g[i].defined()) g[i] = nn::zeros_like(params[i].detach());
    return g;
  }

  /// Raw loss gradient of every parameter, by name (tflow's Arrays(&batch)).
  nn::NamedTensors Gradient(const Batch& b) {
    const auto       g = RawGradients(b);
    nn::NamedTensors result;
    for (std::size_t i = 0; i < g.size(); ++i) result.emplace_back(store.names()[i], g[i].detach());
    return result;
  }

  void Update(const Batch& b) { optimizer->step(RawGradients(b)); }

  nn::NamedTensors Arrays() const {
    nn::NamedTensors out;
    for (const auto& name : store.names()) out.emplace_back(name, store.at(name).detach().clone());
    return out;
  }

  void Save(const fs::path& file) const { store.save(file.string()); }
  void Restore(const fs::path& file) { store.load(file.string(), true); }

  /// Overwrites every parameter from tflow's initial_arrays.npz.
  void LoadArrays(const fs::path& file) {
    const auto arrays = nn::load_npz(file.string());
    check::Require(arrays.size() == store.size(), "init arrays cover exactly the model's parameters");
    for (const auto& [name, value] : arrays) {
      check::Require(store.contains(name), "init array " + name + " is a model parameter");
      store.set(name, nn::cast(value, store.at(name).dtype()));
    }
  }
};

/// One tflow 4.3 example's fixed configuration.
struct TaskSpec {
  std::string              name, dataset;
  std::vector<std::string> keys;                 // per-split arrays of data.npz
  std::string              history_column;       // "validation_bit_ce" or "validation_ce"
  std::string              print_name;           // "validationBitCE" or "validationCE"
  std::string              test_key;             // metrics.json key compared with tflow
  std::uint64_t            order_seed = 0;
  std::int64_t             parameters = 0, tensors = 0;
  std::string              count_message;
  bool                     first_forward = false;  // DNC also records first_forward / final_first_forward
  std::function<std::unique_ptr<MemoryModel>(bool training, const Batch& example)> make;
};

/// The state of a run after training and restore.
struct TaskRun {
  fs::path                     root;
  rnn_cells::DemoOptions       options;
  Batch                        fit, validation, test, first;
  int                          updates = 0;
  std::unique_ptr<MemoryModel> model, deployed;
  nn::NamedTensors             weights;  // deployed arrays after restore
  Outputs                      out;      // deployed->Evaluate(test)
  Tensor                       prediction;
  std::chrono::steady_clock::time_point start;
};

/// Data, training, records, restore and the label-free prediction checks.

/// outputs.npz, model_arrays.npz, the immutability check, metrics.json and
/// the comparison with tflow's run directory.

}  // namespace memory_nets
