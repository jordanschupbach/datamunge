// CMD: just run-nn restricted_boltzmann
// Sampled Bernoulli CD1 with exact likelihood diagnostics on a tiny synthetic corpus.
//
// Port of tflow's restricted_boltzmann (common/{nn,ml}/restricted_boltzmann.h and
// common/ml/binary_energy_data.h -> examples/nn/common/{restricted_boltzmann,
// binary_energy_data}.hpp). Data: synthetic 2x4 single bars with 5% bit flips
// (512 fit / 128 validation / 128 test), generated in-process from tflow's
// mt19937 uniform banks, so no dataset is needed. 8 visible x 4 hidden Bernoulli
// units, 44 parameters. Three epochs of 32-row SGD(.05) updates (tflow's
// BatchOrder(512, 32, 21019)) on the sampled CD1 stopped-chain surrogate
// mean(F(v0) - F(stop(v1))); exact normalized NLL by enumerating all 256 visible
// states. Every Gibbs uniform is tflow's explicit bank (seeds 2100051+step, ...),
// so given the same weights our chains are TF's chains.
//
// Checks (as tflow): 44 parameters / 3 tensors; exact visible distribution sums to 1;
// fresh restore bit-exact; fixed-uniform Gibbs chain repeats bit-exactly; inference
// leaves the weights unchanged; all records written. TF parity: init=auto (default)
// replays TF's initial_arrays.npz from tflow's output/models/restricted_boltzmann,
// then the first forward/gradients/update and the final weights are compared with
// TF's; independently TF's final weights must reproduce TF's test outputs and
// 100-sweep chains. Data and uniform banks are checked bit-identical to TF's.
// Deviations: final.npz instead of a SaveV2 bundle; no independent_audit.json.
// Options: epochs=3 out=DIR init=auto|tf|nn|PATH tf=DIR threads=4.
// Output: output/nn/models/restricted_boltzmann.
#include <chrono>

#include "common/restricted_boltzmann.hpp"

namespace {

using namespace ebm;

Map Sample(RestrictedBoltzmannModel& model, int sweeps = 100) {
  const std::int64_t n = 1024;
  Tensor             state = nn::cast(nn::less(BinaryUniform(n, 8, 21201), .5), nn::DType::Float32);
  Map                saved{{"initial", state}};
  std::vector<Tensor> path{state};
  std::map<std::string, std::vector<Tensor>> steps;
  for (int k = 0; k < sweeps; ++k) {
    const auto fields = model.Gibbs(state, BinaryUniform(n, 4, 2120051 + k), BinaryUniform(n, 8, 3120051 + k), false);
    for (const auto& [key, value] : fields) steps[key].push_back(value);
    state = fields.at("visible");
    path.push_back(state);
  }
  for (const auto& [key, values] : steps) saved[key] = nn::stack(values, 0);
  saved["path"]    = nn::stack(path, 0);
  saved["samples"] = state;
  return saved;
}

void Run(const Options& o) {
  const fs::path root = o.out, tf = o.tf;
  const auto     started = std::chrono::steady_clock::now();
  Map            data;
  std::vector<Tensor> pools;
  for (int k = 0; k < 3; ++k) {
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    const auto generated = BinaryBars(k ? 128 : 512, 21001 + 2 * k, 10000 * k);
    for (const auto& [key, value] : generated) data[split + "_" + key] = value;
    pools.push_back(generated.at("points"));
  }
  Save(root / "data.npz", data);
  CheckData(tf, data, {"fit_points", "validation_points", "test_points", "fit_uniform", "test_components"});

  RestrictedBoltzmannModel model;
  check::Require(model.store.count() == 44 && model.store.size() == 3, "44 Bernoulli RBM parameters 3 tensors");
  const fs::path init      = bpix::InitFile(o.init, tf);
  const bool     replaying = !init.empty();
  if (replaying) {
    replay::LoadInto(model.store, init);
    std::cout << "replaying TF's initial arrays " << init.string() << "\n";
  } else {
    std::cout << "weights from nn's generator (statistical parity with TF only)\n";
  }
  Save(root / "initial_arrays.npz", model.Arrays());
  Save(root / "initial_outputs.npz", model.Evaluate(pools[2]));
  nn::BatchOrder order(512, 32, 21019);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_exact_nll\n";
  batches << "epoch\tstep\trow\n";
  Map randomness;
  int updates = 0;
  for (int epoch = 1; epoch <= o.epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const auto   n = std::int64_t(rows.size());
      const Tensor x = Rows(pools[0], rows), u = BinaryUniform(n, 4, 2100051 + updates), v = BinaryUniform(n, 8, 3100051 + updates);
      randomness["step" + std::to_string(updates) + "_hidden"]  = BinaryUniform(n, 4, 2100051 + updates, true);
      randomness["step" + std::to_string(updates) + "_visible"] = BinaryUniform(n, 8, 3100051 + updates, true);
      if (!updates) {
        Save(root / "first_batch.npz", {{"points", x}, {"hidden_uniform", u}, {"visible_uniform", v}});
        Save(root / "first_forward.npz", model.Gibbs(x, u, v));
        Save(root / "initial_gradients.npz", model.Gradient(x, u, v));
        Save(root / "initial_exact_gradients.npz", model.Gradient(x, u, v, true));
      }
      model.Update(x, u, v);
      if (!updates) Save(root / "first_after.npz", model.Arrays());
      for (auto row : rows) batches << epoch << '\t' << updates << '\t' << row << '\n';
      ++updates;
    }
    const auto valid = model.Evaluate(pools[1]);
    history << epoch << '\t' << Scalar(valid, "nll") << '\n';
    Save(root / ("epoch" + std::to_string(epoch) + "_arrays.npz"), model.Arrays());
    Save(root / ("epoch" + std::to_string(epoch) + "_validation.npz"), valid);
    std::cout << "epoch" << epoch << " exactValidationNLL=" << Scalar(valid, "nll") << std::endl;
  }
  history.flush();
  batches.flush();
  const double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  Save(root / "training_randomness.npz", randomness);
  check::Require(SameAsTf(tf, "training_randomness.npz", randomness), "Gibbs uniform banks bit-identical to TF's training_randomness.npz");
  const auto first = Load(root / "first_batch.npz");
  const auto &fx = first.at("points"), &fu = first.at("hidden_uniform"), &fv = first.at("visible_uniform");
  Save(root / "final_gradients.npz", model.Gradient(fx, fu, fv));
  Save(root / "final_exact_gradients.npz", model.Gradient(fx, fu, fv, true));
  Save(root / "final_first_gibbs.npz", model.Gibbs(fx, fu, fv));
  model.store.save((root / "final.npz").string());
  RestrictedBoltzmannModel deployed(false);
  deployed.store.load((root / "final.npz").string(), true);
  check::Require(model.Snapshot() == deployed.Snapshot(), "all 44 weights bitexact restore");
  const auto before = deployed.Snapshot();
  Save(root / "model_arrays.npz", deployed.Arrays());
  const auto output = deployed.Evaluate(pools[2]);
  Save(root / "outputs.npz", output);
  const Tensor states       = BinaryStates(8);
  auto         distribution = deployed.Evaluate(states);
  distribution["states"]    = states;
  double mass               = 0;
  for (float lp : distribution.at("log_probability").to_vector<float>()) mass += std::exp(double(lp));
  check::Near(mass, 1., 2e-6, "exact enumerated visible distribution normalizes");
  Save(root / "exact_distribution.npz", distribution);
  const auto sample = Sample(deployed), repeat = Sample(deployed);
  for (const auto& [key, value] : sample) check::Require(Bits(value) == Bits(repeat.at(key)), "fixed uniform Gibbs chain bitexact repeat");
  Save(root / "samples.npz", sample);
  BinaryTiles(root / "samples.png", Rows(sample.at("samples"), Range(64)), 8);
  const Tensor truth = Rows(pools[2], Range(32)), u = BinaryUniform(32, 4, 21101), v = BinaryUniform(32, 8, 31101);
  auto         rec  = deployed.Gibbs(truth, u, v);
  rec["truth"]           = truth;
  rec["hidden_uniform"]  = u;
  rec["visible_uniform"] = v;
  std::vector<Tensor> cells;
  for (std::int64_t b = 0; b < 32; ++b)
    for (const Tensor& x : {truth, rec.at("visible_probability"), rec.at("visible")}) cells.push_back(nn::slice_axis(x, 0, b, b + 1));
  const Tensor gallery = nn::concat(cells, 0);
  rec["gallery"]       = gallery;
  Save(root / "reconstruction.npz", rec);
  BinaryTiles(root / "reconstruction.png", gallery, 12);
  const auto          fit = pools[0].to_vector<float>(), test = pools[2].to_vector<float>();
  std::vector<double> mean(8);
  double              baseline_nll = 0;
  for (int j = 0; j < 8; ++j) {
    double m = 0;
    for (int b = 0; b < 512; ++b) m += double(fit[b * 8 + j]) / 512;
    mean[j] = m;
    check::Require(m > 0 && m < 1, "fit Bernoulli baseline finite");
    for (int b = 0; b < 128; ++b) baseline_nll -= (test[b * 8 + j] != 0 ? std::log(m) : std::log1p(-m)) / 128;
  }
  Save(root / "baseline.npz", {{"fit_visible_mean", nn::from_vector(mean, {8})}});
  check::Require(before == deployed.Snapshot(), "all RBM likelihood/sampling/inference weights immutable");
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":44,\"parameter_tensors\":3,\"visible\":8,\"hidden\":4,\"epochs\":" << o.epochs
       << ",\"optimizer_updates\":" << updates << ",\"training\":\"sampledCD1SGD0.05\",\"test_exact_nll\":" << Scalar(output, "nll")
       << ",\"fit_independent_bernoulli_nll\":" << baseline_nll << ",\"log_partition\":" << Scalar(output, "log_partition")
       << ",\"sampling_sweeps\":100,\"samples\":1024,\"replayed_tf_initial_arrays\":" << (replaying ? "true" : "false")
       << ",\"training_seconds\":" << train_seconds << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Restricted Boltzmann machine</title><style>body{font:16px system-ui;"
          "margin:30px;max-width:1100px}img{max-width:100%;image-rendering:pixelated}</style><h1>Bernoulli RBM: sampled "
          "contrastive divergence versus exact likelihood</h1><p>Synthetic 2x4 single bars: uniformly one of 4 vertical or 2 "
          "horizontal bars, each bit flipped with probability .05. 512 fit / 128 validation / 128 test. 8 visible, 4 hidden "
          "Bernoulli units, 44 parameters, E(v,h) = -vWh - bv - ch. Three epochs, 48 SGD .05 updates on the sampled CD1 "
          "stopped-chain surrogate mean(F(v0) - F(stop(v1))) (not the exact NLL gradient). Exact log Z by enumerating all 256 "
          "visible states. Datamunge port of tflow's restricted_boltzmann.</p><h2>First 64 cold-start chains after 100 Gibbs "
          "sweeps</h2><img src='samples.png'><h2>Heldout one-sweep reconstruction</h2><p>Triplets truth, visible probabilities "
          "after sampled hidden, sampled visible.</p><img src='reconstruction.png'><p>See metrics.json and "
          "exact_distribution.npz. <a href='https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf'>Hinton 2010 practical "
          "guide</a>.</p>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete RBM records");
  std::cout << "exactTestNLL=" << Scalar(output, "nll") << " fitBernoulli=" << baseline_nll << " visibleMass=" << mass
            << ";CD1/exactZ/restore/GibbsRepeat/immutable PASS (training " << train_seconds << " s)\n";

  // ---- TF parity ----
  if (!fs::exists(tf / "model_arrays.npz")) {
    std::cout << "no TF record at " << tf.string() << "; parity checks skipped\nArtifacts: " << root.string() << "\n";
    return;
  }
  std::cout << "TF parity (" << tf.string() << "):\n";
  if (replaying) {
    const double fwd = bpix::Compare(tf, "first_forward.npz", Load(root / "first_forward.npz"));
    const double grd = bpix::Compare(tf, "initial_gradients.npz", Load(root / "initial_gradients.npz"));
    const double exg = bpix::Compare(tf, "initial_exact_gradients.npz", Load(root / "initial_exact_gradients.npz"));
    const double aft = bpix::Compare(tf, "first_after.npz", Load(root / "first_after.npz"));
    check::Require(fwd < 1e-5 && grd < 1e-5 && exg < 1e-5 && aft < 1e-5, "replayed first forward/gradients/update match TF");
    if (o.epochs == 3) {
      const double fin = bpix::Compare(tf, "model_arrays.npz", deployed.Arrays());
      bpix::Compare(tf, "outputs.npz", output);
      check::Require(fin < 1e-4, "replayed run's final weights match TF's");
      bpix::Compare(tf, "samples.npz", sample);
      std::printf("  sampled states (path/hidden/visible) agreement with TF's chains: %.6f\n",
                  Agreement(tf, "samples.npz", sample, {"path", "hidden", "visible", "samples"}));
    }
    CompareHistory(root, tf, 1, "validation exact NLL");
  }
  RestrictedBoltzmannModel tf_model(false);
  replay::LoadInto(tf_model.store, tf / "model_arrays.npz");
  const double out_diff = bpix::Compare(tf, "outputs.npz", tf_model.Evaluate(pools[2]));
  const double dst_diff = bpix::Compare(tf, "exact_distribution.npz", tf_model.Evaluate(states));
  const auto   chains   = Sample(tf_model);
  const double prb_diff = bpix::Compare(tf, "samples.npz", chains);
  const double agree    = Agreement(tf, "samples.npz", chains, {"path", "hidden", "visible", "samples"});
  std::printf("  TF final weights: 100-sweep sampled states agreement with TF's samples.npz %.6f\n", agree);
  check::Require(out_diff < 1e-5 && dst_diff < 1e-5 && prb_diff < 1e-5 && agree > .999, "TF's final weights reproduce TF's outputs and Gibbs chains");
  std::cout << "Artifacts: " << root.string() << "\nAll restricted_boltzmann checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Run(ParseOptions(argc, argv, "restricted_boltzmann", {}, "epochs=N out=PATH init=auto|tf|nn|PATH tf=DIR threads=N"));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
