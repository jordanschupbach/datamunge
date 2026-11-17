// The three-epoch marked-token memory experiment shared by tflow's 4.1
// recurrent-cell examples (bidirectional_rnn, clockwork_rnn, gradient_clipping,
// indrnn, mgu and, through tflow's common/ml/gated_sequence_demo.h, lstm, gru,
// peephole_lstm and layer_norm_lstm). tflow repeats this Run() in every .cc;
// here it is written once with hooks for each example's extra checks.
//
// Flow (identical to tflow): 512/128/128 marked sequences, Adam .01 (or the
// example's optimizer), batch 32 with tflow's BatchOrder(512, 32, 26019), three
// epochs = 48 updates; npz records of the initial/first/epoch/final state;
// save -> restore into a fresh model (bit-exact); label-free prediction equals
// evaluation; repeat and single-row parity; arbitrary padding cannot change any
// output or gradient; flipping the last valid token cannot change earlier
// states; zero-length sequences keep the zero state; weights immutable under
// inference. Checkpoints are .npz files of the parameter store.
//
// Parity with tflow: the data and batch order are bit-identical to tflow's.
// Parameters initialise from nn's generator, so a default run matches tflow
// only statistically; `init=PATH` (tflow's initial_arrays.npz) replays tflow's
// run, and when the tflow run directory (`tf=`) has the same files the port
// prints and checks the differences.
#pragma once

#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/marked_sequence_data.hpp"
#include "common/paths.hpp"
#include "common/recurrent_model.hpp"

namespace rnn_cells {

namespace fs = std::filesystem;

struct DemoOptions {
  int      epochs = 3;
  fs::path out;   // output directory
  fs::path init;  // optional initial arrays (tflow's initial_arrays.npz)
  fs::path tf;    // tflow's run directory for comparisons (skipped when missing)
};

/// tflow's output/models/<name> (where its examples write their runs).
inline fs::path TflowRunDir(const std::string& name) {
  const char* home = std::getenv("HOME");
  return fs::path(home ? home : ".") / "projects" / "tflow_examples" / "output" / "models" / name;
}

/// Parses epochs=N out=PATH init=PATH tf=PATH.
inline DemoOptions ParseDemoOptions(int argc, char** argv, const std::string& name) {
  DemoOptions o;
  o.out = paths::OutputRoot() / "models" / name;
  o.tf  = TflowRunDir(name);
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a.rfind("epochs=", 0) == 0) o.epochs = std::stoi(a.substr(7));
    else if (a.rfind("out=", 0) == 0) o.out = a.substr(4);
    else if (a.rfind("init=", 0) == 0) o.init = a.substr(5);
    else if (a.rfind("tf=", 0) == 0) o.tf = a.substr(3);
    else throw std::invalid_argument("options: epochs=N out=PATH init=PATH tf=PATH");
  }
  check::Require(o.epochs > 0 && o.epochs <= 100, "threeCheapDefaultEpochs");
  return o;
}

struct DemoContext {
  fs::path                      root;
  std::vector<MarkedSequences>& pools;
  RecurrentModel&               model;     // the trained model
  RecurrentModel*               deployed;  // restored, inference only (null before restore)
  Outputs                       output, last_output, cue_output;
  nn::NamedTensors              initial_arrays;
  std::ostringstream            metrics;  // extra ",\"key\":value" metrics.json entries
};

struct MarkedDemo {
  std::string  name;
  std::int64_t parameters = 0, tensors = 0;
  std::string  count_message, restore_message;
  std::function<std::unique_ptr<RecurrentModel>(bool training)> make;
  /// Extra leading metrics.json entries ("\"key\":value," each).
  std::string metrics_prefix;
  std::string title, description, pass_tag = "fullBPTT/masking/causality/zeroLength/restore/labelFree/immutable";
  /// Called for every update with the batch, before Update (gradient_clipping's per-step records).
  std::function<void(RecurrentModel&, const MarkedSequences&, const fs::path&, int step)> before_update, after_update;
  std::function<void(DemoContext&)> after_training;  // after the last epoch, before saving
  /// Called at the start of every batch, before the first-batch records and the
  /// update (per-step dropout masks, training-mode switches).
  std::function<void(RecurrentModel&, const MarkedSequences&, const fs::path&, int step)> before_batch;
  std::function<void(DemoContext&)> after_outputs;   // right after outputs.npz (structure checks)
  std::function<void(DemoContext&)> after_counterfactuals;  // after padding/flip evaluations
  std::function<void(DemoContext&)> after_predictions;      // after predictions.tsv
};

inline void SaveOutputs(const fs::path& file, const Outputs& out) { nn::save_npz(file.string(), Named(out)); }

/// Max |ours - tflow| over the arrays both npz files contain; -1 when tflow's file is missing.
inline double CompareNpz(const fs::path& ours, const fs::path& theirs) {
  if (!fs::exists(theirs)) return -1;
  const auto a = nn::load_npz(ours.string()), b = nn::load_npz(theirs.string());
  double     worst = 0;
  for (const auto& [name, value] : a)
    for (const auto& [other, tf_value] : b)
      if (name == other) worst = std::max(worst, MaxAbsDiff(value, tf_value));
  return worst;
}

inline std::string ReadText(const fs::path& file) {
  std::ifstream f(file, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

/// Compares this run with tflow's run directory: data and batch order always
/// (they must be identical); with init= also every replayed record.
/// The number after "key": in a flat JSON object (NaN when absent).
inline double JsonNumber(const std::string& json, const std::string& key) {
  const auto at = json.find("\"" + key + "\":");
  return at == std::string::npos ? NAN : std::stod(json.substr(at + key.size() + 3));
}

inline void CompareWithTflow(const fs::path& root, const DemoOptions& options, const std::vector<std::string>& records,
                             const std::string& test_key = "test_ce") {
  if (options.tf.empty() || !fs::exists(options.tf / "data.npz")) {
    std::cout << "tflow run " << options.tf << " not found; skipping the comparison\n";
    return;
  }
  check::Near(CompareNpz(root / "data.npz", options.tf / "data.npz"), 0, 0, "dataBitIdenticalToTflow");
  if (fs::exists(options.tf / "batches.tsv")) {
    // tflow ran three epochs; with another epochs= budget the shorter record
    // must be a prefix of the longer one (same BatchOrder stream).
    const std::string ours = ReadText(root / "batches.tsv"), theirs = ReadText(options.tf / "batches.tsv");
    check::Require(ours.rfind(theirs, 0) == 0 || theirs.rfind(ours, 0) == 0, "batchOrderIdenticalToTflow");
  }
  std::cout << "vs tflow: data.npz and batches.tsv identical";
  if (options.init.empty()) {
    std::cout << " (pass init=" << (options.tf / "initial_arrays.npz").string() << " to replay tflow's run)\n";
    return;
  }
  double worst_initial = 0, worst_final = 0;
  for (const auto& file : records) {
    const double d = CompareNpz(root / file, options.tf / file);
    if (d < 0) continue;
    std::cout << "\n  " << file << " max|diff|=" << d;
    double& worst = file.rfind("initial", 0) == 0 || file.rfind("first", 0) == 0 ? worst_initial : worst_final;
    worst         = std::max(worst, d);
  }
  std::cout << "\n";
  check::Require(worst_initial <= 1e-5, "replayedInitialForwardGradientsAndFirstUpdateMatchTflow");
  // Later records drift with float32 rounding (Adam divides by tiny second
  // moments), so the trained run is checked through its test loss.
  const double ours = JsonNumber(ReadText(root / "metrics.json"), test_key), theirs = JsonNumber(ReadText(options.tf / "metrics.json"), test_key);
  if (!std::isnan(theirs)) {
    std::cout << "  " << test_key << " ours=" << ours << " tflow=" << theirs << " (trained records max|diff|=" << worst_final << ")\n";
    check::Near(ours, theirs, 5e-3, "replayedTrainedTestLossMatchesTflow");
  }
}

}  // namespace rnn_cells
