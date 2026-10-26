// CMD: just run-nn early_stopping_checkpoints
// Early stopping, checkpoint-best and exact optimizer resume.
//
// Port of tflow's early_stopping_checkpoints. A controlled validation example
// trains a scalar toward a training target of 4 while clean validation
// prefers 2. Validation improves and then worsens, making checkpoint-best and
// patience visible without a dataset. The best checkpoint includes model
// weights, Adam moments and its int64 step.
//
// In a separate experiment, checkpoint after 17 updates, continue for 23,
// then restore into a fresh model/optimizer and reproduce those same 23
// updates. Compare every slot and the final counter against the
// uninterrupted run. A weights-only restore demonstrates why resetting
// optimizer state differs. This deterministic training loop has no sampler or
// random-op state; an application with those states must persist them (e.g.
// nn::BatchOrder::save_state) and the validation monitor separately.
//
// TF's SaveV2 bundles and fresh graph/session are replaced by one .npz archive
// holding the parameter store ("params/<name>") and nn::Optimizer::state()
// ("optimizer/<slot>"), and by a newly constructed ParameterStore/Optimizer
// whose weight starts as NaN, so the checkpoint must provide every value.
// Files are written to output/nn/checkpoints/early_stopping/.

#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

struct Model {
  nn::ParameterStore store;
  Tensor             w;
  nn::Optimizer      optimizer;

  static Tensor MakeWeight(nn::ParameterStore& store, double initial) {
    return store.get("w", {}, nn::init::constant(initial));
  }
  explicit Model(double initial = 0)
      : w(MakeWeight(store, initial)), optimizer(store, .1, nn::OptimizerConfig{nn::OptimizerKind::Adam}) {}

  Tensor Loss() const { return nn::square(w - 4.0); }
  double Validation() const {
    nn::NoGradGuard guard;
    return nn::square(w - 2.0).item();
  }
  void Train(int count) {
    for (int i = 0; i < count; ++i) {
      optimizer.set_learning_rate(nn::schedule::step_decay(.1, .5, 25, optimizer.iterations()));
      optimizer.minimize(Loss());
    }
  }

  /// Weights, optimizer slots and step as (name, value) pairs.
  nn::NamedTensors State() const {
    nn::NamedTensors state;
    for (const auto& name : store.names()) state.emplace_back("params/" + name, store.at(name).detach().clone());
    for (const auto& [name, value] : optimizer.state()) state.emplace_back("optimizer/" + name, value);
    return state;
  }
  void Save(const std::string& path) const { nn::save_npz(path, State()); }

  /// Restores everything (or only the weights). The archive is read and
  /// checked completely before anything changes, so a failed restore leaves
  /// the model untouched.
  void Restore(const std::string& path, bool weights_only = false) {
    const auto                    archive = nn::load_npz(path);
    std::map<std::string, Tensor> params;
    nn::NamedTensors              slots;
    for (const auto& [name, value] : archive) {
      if (name.rfind("params/", 0) == 0)
        params[name.substr(7)] = value;
      else if (name.rfind("optimizer/", 0) == 0)
        slots.emplace_back(name.substr(10), value);
      else
        throw std::runtime_error("unexpected checkpoint entry " + name);
    }
    for (const auto& name : store.names())
      if (!params.count(name)) throw std::runtime_error("checkpoint is missing parameter " + name);
    if (params.size() != store.size()) throw std::runtime_error("checkpoint has unexpected parameters");
    if (!weights_only && slots.empty()) throw std::runtime_error("checkpoint has no optimizer state");
    if (!weights_only) {
      // This example has one scalar Adam parameter; load_state deliberately
      // defers general slot-shape checks until step(), so check our schema here.
      for (const auto& [name, value] : slots) {
        if (name != "step" && name != "nadam_product" && name != "m/0" && name != "v/0")
          throw std::runtime_error("unexpected Adam checkpoint slot " + name);
        if (!value.shape().empty()) throw std::runtime_error("Adam checkpoint slots must be scalar");
      }
    }
    // Validate on independent tensors before either real weights or moments change.
    Model probe;
    for (const auto& [name, value] : params) probe.store.set(name, value);
    if (!weights_only) probe.optimizer.load_state(slots);
    for (const auto& [name, value] : params) store.set(name, value);
    if (!weights_only) optimizer.load_state(slots);
  }
};

void MonitorChecks() {
  nn::EarlyStopping stop(2, .1);
  check::Require(stop.observe(1, 0).improved, "first validation is best");
  check::Require(!stop.observe(.95, 1).improved, "min_delta must be exceeded");
  check::Require(stop.observe(.8, 2).improved && stop.misses() == 0, "improvement resets patience");
  check::Require(!stop.observe(.8, 3).stop, "one miss is below patience");
  check::Require(stop.observe(.85, 4).stop, "patience boundary");
  nn::EarlyStopping maximize(0, 0, true);
  check::Require(!maximize.observe(.4, 0).stop && maximize.observe(.5, 1).improved, "maximize mode");
  check::Require(maximize.observe(.5, 2).stop, "zero patience stops at first tie");
  check::Throws([&] { maximize.observe(.6, 2); }, "epochs must increase");
  std::cout << "  patience, min_delta, tie, maximize and epoch-order checks passed\n";
}

void BestCheckpoint(const std::string& path, int epochs) {
  Model             model;
  nn::EarlyStopping monitor(6, 1e-6);
  bool              stopped    = false;
  double            last_loss  = 0;
  int               last_epoch = 0;
  for (int epoch = 0; epoch < epochs; ++epoch) {
    model.Train(1);
    last_loss           = model.Validation();
    last_epoch          = epoch;
    const auto decision = monitor.observe(last_loss, epoch);
    if (decision.improved) model.Save(path);
    if (decision.stop) {
      stopped    = true;
      last_epoch = epoch;
      break;
    }
  }
  if (epochs >= 200) check::Require(stopped, "validation must trigger early stopping");
  model.Restore(path);
  const double restored = model.Validation();
  check::Near(restored, monitor.best(), 1e-7, "restore exact best validation loss");
  check::Require(restored <= last_loss, "best checkpoint cannot worsen held-out loss");
  if (stopped) check::Require(restored < .01 && monitor.best() < last_loss, "best checkpoint improves held-out loss");
  check::Require(model.optimizer.iterations() == monitor.best_epoch() + 1, "best checkpoint restores optimizer counter");
  std::printf("  %s after epoch %d; best epoch %lld; validation %.6g -> restored %.6g\n", stopped ? "stopped" : "budget exhausted", last_epoch,
              static_cast<long long>(monitor.best_epoch()), last_loss, restored);
  std::printf("  restored w=%.6f (clean validation target 2), optimizer step %lld\n", model.w.item(),
              static_cast<long long>(model.optimizer.iterations()));
}

void Resume(const std::string& path, int prefix, int suffix) {
  Model original;
  original.Train(prefix);
  original.Save(path);
  original.Train(suffix);
  const auto expected = original.State();

  // Starts from NaN: the checkpoint must provide EVERY value we need.
  Model restored(std::numeric_limits<double>::quiet_NaN());
  restored.Restore(path);
  restored.Train(suffix);
  const auto actual = restored.State();
  check::Require(actual.size() == expected.size(), "restored state size");
  double worst = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    check::Require(actual[i].first == expected[i].first, "restored state names");
    if (actual[i].first == "optimizer/step") {
      check::Require(actual[i].second.item<std::int64_t>() == prefix + suffix && expected[i].second.item<std::int64_t>() == prefix + suffix,
                     "resume counter");
      continue;
    }
    const auto a = actual[i].second.to_vector<double>(), e = expected[i].second.to_vector<double>();
    check::Require(a.size() == e.size(), "restored slot size");
    for (std::size_t j = 0; j < a.size(); ++j) {
      worst = std::max(worst, std::abs(a[j] - e[j]));
      check::Near(a[j], e[j], 1e-7, "resume all optimizer state (" + actual[i].first + ")");
    }
  }
  std::printf("  %d + save/restore + %d updates = %d uninterrupted updates; max state error %.3g\n", prefix, suffix, prefix + suffix, worst);

  Model reset_optimizer;
  reset_optimizer.Restore(path, /*weights_only=*/true);
  reset_optimizer.Train(suffix);
  const double weights_result = reset_optimizer.w.item(), correct_result = restored.w.item();
  check::Require(std::abs(weights_result - correct_result) > 1e-7, "weights-only restore should expose missing state");
  std::printf("  weights-only resume w=%.6f; full-state resume w=%.6f\n", weights_result, correct_result);
  check::Throws([&] { restored.Restore(path + "_missing.npz"); }, "missing checkpoint must report failure");
  check::Near(restored.w.item(), correct_result, 0, "failed restore must not change weights");
  auto broken = original.State();
  for (auto& [name, value] : broken) {
    if (name == "params/w") value = nn::scalar(99);
    if (name == "optimizer/m/0") value = nn::zeros({2});
  }
  nn::save_npz(path + "_broken.npz", broken);
  const auto before = restored.State();
  check::Throws([&] { restored.Restore(path + "_broken.npz"); }, "bad moment shape must fail before changing weights");
  const auto after = restored.State();
  for (std::size_t i = 0; i < before.size(); ++i)
    check::Require(before[i].first == after[i].first && before[i].second.to_vector<double>() == after[i].second.to_vector<double>(),
                   "failed restore preserves every model and optimizer value");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int epochs = 200, prefix = 17, suffix = 23;
    for (int i = 1; i < argc; ++i) {
      const std::string argument(argv[i]);
      if (argument.rfind("epochs=", 0) == 0) {
        epochs = std::stoi(argument.substr(7));
        prefix = std::max(1, epochs / 2);
        suffix = epochs - prefix;
      }
    }
    check::Require(epochs >= 2, "at least two updates required for checkpoint resume");
    const auto directory = paths::OutputDir("checkpoints/early_stopping");
    std::cout << "== Validation monitor semantics\n";
    MonitorChecks();
    std::cout << "\n== Save and restore the best validation checkpoint\n";
    BestCheckpoint((directory / "best.npz").string(), epochs);
    std::cout << "\n== Resume model, Adam moments and learning-rate counter\n";
    Resume((directory / "resume.npz").string(), prefix, suffix);
    std::cout << "All early stopping and checkpoint checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
