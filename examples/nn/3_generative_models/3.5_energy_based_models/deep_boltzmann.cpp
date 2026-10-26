// CMD: just run-nn deep_boltzmann
// Fully undirected tiny DBM: exact NLL training, mean-field diagnostic, actual
// blocked Gibbs generation.
//
// Port of tflow's deep_boltzmann (common/{nn,ml}/deep_boltzmann.h ->
// examples/nn/common/deep_boltzmann.hpp). Data: synthetic 2x4 single bars + 5% bit
// flips (512 fit / 128 validation / 128 test) from tflow's mt19937 banks (seeds
// 23001+2k), generated in-process. 8 visible - 4 hidden1 - 3 hidden2 Bernoulli units,
// 59 parameters. h2 summed analytically and 16 h1 / 256 visible states enumerated
// for exact log Z / posterior / NLL; three epochs of 48 SGD(.05) updates on the
// EXACT NLL gradient (BatchOrder(512, 32, 23019)). Clamped 20-iteration mean field
// is a separate diagnostic (negative ELBO upper bound on the NLL).
//
// Checks (as tflow): 59 parameters / 5 tensors; restore bit-exact; mean-field NLL
// upper bound >= exact NLL; visible marginal sums to 1; fixed-uniform blocked chain
// repeats bit-exactly; forcing h2 to zeros/ones with the same visible noise keeps
// the visible state but changes p(h1|v,h2) (top-down feedback); inference
// immutable. TF parity: init=auto (default) replays TF's initial_arrays.npz from
// tflow's output/models/deep_boltzmann; the first forward/gradients/update, history,
// final weights and outputs are compared with TF's; independently TF's final
// weights must reproduce TF's outputs and Gibbs chains. Deviations: final.npz
// instead of a SaveV2 bundle; no independent_audit.json. Options: epochs=3 out=DIR
// init=auto|tf|nn|PATH tf=DIR threads=4. Output: output/nn/models/deep_boltzmann.
#include <chrono>

#include "common/deep_boltzmann.hpp"

namespace {

using namespace ebm;

Map Sample(DeepBoltzmannModel& model) {
  const std::int64_t n     = 1024;
  Tensor             state = nn::cast(nn::less(BinaryUniform(n, 4, 23201), .5), nn::DType::Float32), visible;
  Map                saved{{"initial_h1", state}};
  std::vector<Tensor> path{state};
  std::map<std::string, std::vector<Tensor>> steps;
  for (int k = 0; k < 100; ++k) {
    const auto fields =
        model.Step(state, BinaryUniform(n, 8, 2320051 + k), BinaryUniform(n, 3, 2330051 + k), BinaryUniform(n, 4, 2340051 + k));
    for (const auto& [key, value] : fields) steps[key].push_back(value);
    state   = fields.at("hidden1");
    visible = fields.at("visible");
    path.push_back(state);
  }
  for (const auto& [key, values] : steps) saved[key] = nn::stack(values, 0);
  saved["h1_path"] = nn::stack(path, 0);
  saved["samples"] = visible;
  return saved;
}

void Run(const Options& o) {
  const fs::path root = o.out, tf = o.tf;
  const auto     started = std::chrono::steady_clock::now();
  Map            data;
  std::vector<Tensor> pools;
  for (int k = 0; k < 3; ++k) {
    const std::string split     = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    const auto        generated = BinaryBars(k ? 128 : 512, 23001 + 2 * k, 10000 * k);
    for (const auto& [key, value] : generated) data[split + "_" + key] = value;
    pools.push_back(generated.at("points"));
  }
  Save(root / "data.npz", data);
  CheckData(tf, data, {"fit_points", "validation_points", "test_points", "fit_uniform", "test_components"});
  DeepBoltzmannModel model;
  check::Require(model.store.count() == 59 && model.store.size() == 5, "59 fully undirected DBM parameters 5 tensors");
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
  nn::BatchOrder order(512, 32, 23019);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_exact_nll\n";
  batches << "epoch\tstep\trow\n";
  int updates = 0;
  for (int epoch = 1; epoch <= o.epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const Tensor x = Rows(pools[0], rows);
      if (!updates) {
        Save(root / "first_batch.npz", {{"points", x}});
        Save(root / "first_forward.npz", model.Evaluate(x));
        Save(root / "initial_gradients.npz", model.Gradient(x));
      }
      model.Update(x);
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
  const auto   first         = Load(root / "first_batch.npz");
  Save(root / "final_gradients.npz", model.Gradient(first.at("points")));
  model.store.save((root / "final.npz").string());
  DeepBoltzmannModel deployed(false);
  deployed.store.load((root / "final.npz").string(), true);
  check::Require(model.Snapshot() == deployed.Snapshot(), "all 59 weights bitexact restore");
  const auto before = deployed.Snapshot();
  const auto output = deployed.Evaluate(pools[2]);
  Save(root / "model_arrays.npz", deployed.Arrays());
  Save(root / "outputs.npz", output);
  double     min_gap = INFINITY, mean_gap = 0;
  const auto upper = output.at("mean_field_nll_upper").to_vector<float>(), logp = output.at("log_probability").to_vector<float>();
  for (int b = 0; b < 128; ++b) {
    const double gap = double(upper[b]) + logp[b];
    min_gap          = std::min(min_gap, gap);
    mean_gap += gap / 128;
  }
  check::Require(min_gap >= -3e-6, "clamped mean-field NLL upper bound, not exact inference");
  const Tensor states = BinaryStates(8);
  auto         all    = deployed.Evaluate(states);
  all["states"]       = states;
  double mass         = 0;
  for (float lp : all.at("log_probability").to_vector<float>()) mass += std::exp(double(lp));
  check::Near(mass, 1., 2e-6, "normalized fully undirected DBM visible marginal");
  Save(root / "exact_distribution.npz", all);
  const auto sample = Sample(deployed), repeat = Sample(deployed);
  for (const auto& [key, value] : sample) check::Require(Bits(value) == Bits(repeat.at(key)), "fixed uniform DBM blocked chain bitexact repeat");
  Save(root / "samples.npz", sample);
  BinaryTiles(root / "samples.png", Rows(sample.at("samples"), Range(64)), 8);
  const Tensor conditional_means = nn::slice_axis(nn::reshape(nn::slice_axis(sample.at("visible_probability"), 0, 99, 100), {1024, 8}), 0, 0, 64);
  Save(root / "gallery.npz", {{"conditional_means", conditional_means}});
  BinaryTiles(root / "conditional_means.png", conditional_means, 8);

  // Same old h1 and SAME visible noise; force h2 zeros vs ones to prove top-down feedback in p(h1|v,h2).
  const Tensor h1 = Rows(sample.at("initial_h1"), Range(8));
  const Tensor vnoise = BinaryUniform(8, 8, 23501), h1noise = BinaryUniform(8, 4, 23503);
  const auto   ones = deployed.Step(h1, vnoise, nn::zeros({8, 3}), h1noise), zeros = deployed.Step(h1, vnoise, nn::ones({8, 3}), h1noise);
  check::Require(Bits(ones.at("visible")) == Bits(zeros.at("visible")), "feedback control keeps visible state identical");
  const double effect = replay::MaxAbsDiff(ones.at("hidden1_probability"), zeros.at("hidden1_probability"));
  check::Require(effect > 1e-4, "actual top-down h2 changes conditional h1");
  Save(root / "feedback_ones.npz", ones);
  Save(root / "feedback_zeros.npz", zeros);
  Save(root / "feedback_inputs.npz", {{"h1", h1}, {"visible_uniform", vnoise}, {"hidden1_uniform", h1noise}});
  const auto          fit = pools[0].to_vector<float>(), test = pools[2].to_vector<float>();
  std::vector<double> mean(8);
  double              baseline = 0;
  for (int j = 0; j < 8; ++j) {
    double m = 0;
    for (int b = 0; b < 512; ++b) m += double(fit[b * 8 + j]) / 512;
    mean[j] = m;
    for (int b = 0; b < 128; ++b) baseline -= (test[b * 8 + j] != 0 ? std::log(m) : std::log1p(-m)) / 128;
  }
  Save(root / "baseline.npz", {{"fit_visible_mean", nn::from_vector(mean, {8})}});
  check::Require(before == deployed.Snapshot(), "DBM exact/mean-field/Gibbs inference weights immutable");
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":59,\"parameter_tensors\":5,\"visible\":8,\"hidden1\":4,\"hidden2\":3,\"epochs\":" << o.epochs
       << ",\"optimizer_updates\":" << updates << ",\"training\":\"tinyExactNLLSGD0.05\",\"test_exact_nll\":" << Scalar(output, "nll")
       << ",\"fit_independent_bernoulli_nll\":" << baseline << ",\"mean_field_iterations\":20,\"mean_field_nll_gap_mean\":" << mean_gap
       << ",\"top_down_feedback_max_effect\":" << effect << ",\"sampling_sweeps\":100,\"samples\":1024,\"replayed_tf_initial_arrays\":"
       << (replaying ? "true" : "false") << ",\"training_seconds\":" << train_seconds << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Deep Boltzmann machine</title><style>body{font:16px system-ui;margin:30px;"
          "max-width:1100px}img{max-width:100%;image-rendering:pixelated}</style><h1>Fully undirected deep Boltzmann machine</h1>"
          "<p>8 visible - 4 hidden1 - 3 hidden2 Bernoulli units, 59 parameters, E = -vW1h1 - h1W2h2 - bv v - b1 h1 - b2 h2. "
          "Tiny exact-likelihood adaptation of <a href='https://proceedings.mlr.press/v5/salakhutdinov09a.html'>Salakhutdinov-"
          "Hinton 2009</a>: h2 summed analytically, 16 h1 / 256 visible states enumerated; three epochs of 48 SGD .05 updates on "
          "the exact NLL gradient. Clamped 20-iteration mean field is a diagnostic. Datamunge port of tflow's deep_boltzmann.</p>"
          "<h2>First 64 sampled visibles after 100 blocked Gibbs sweeps</h2><img src='samples.png'><h2>Corresponding visible "
          "conditional probabilities</h2><img src='conditional_means.png'><p>See metrics.json.</p>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete DBM records");
  std::cout << "exactTestNLL=" << Scalar(output, "nll") << " fitBernoulli=" << baseline << " meanFieldGap=" << mean_gap
            << " topDownEffect=" << effect << ";exactZ/variationalBound/feedback/restore/GibbsRepeat/immutable PASS (training "
            << train_seconds << " s)\n";

  // ---- TF parity ----
  if (!fs::exists(tf / "model_arrays.npz")) {
    std::cout << "no TF record at " << tf.string() << "; parity checks skipped\nArtifacts: " << root.string() << "\n";
    return;
  }
  std::cout << "TF parity (" << tf.string() << "):\n";
  if (replaying) {
    const double fwd = bpix::Compare(tf, "first_forward.npz", Load(root / "first_forward.npz"));
    const double grd = bpix::Compare(tf, "initial_gradients.npz", Load(root / "initial_gradients.npz"));
    const double aft = bpix::Compare(tf, "first_after.npz", Load(root / "first_after.npz"));
    check::Require(fwd < 1e-5 && grd < 1e-5 && aft < 1e-5, "replayed first forward/gradients/update match TF");
    if (o.epochs == 3) {
      const double fin = bpix::Compare(tf, "model_arrays.npz", deployed.Arrays());
      bpix::Compare(tf, "outputs.npz", output);
      check::Require(fin < 1e-4, "replayed run's final weights match TF's");
    }
    CompareHistory(root, tf, 1, "validation exact NLL");
  }
  DeepBoltzmannModel tf_model(false);
  replay::LoadInto(tf_model.store, tf / "model_arrays.npz");
  const double out_diff = bpix::Compare(tf, "outputs.npz", tf_model.Evaluate(pools[2]));
  const auto   chains   = Sample(tf_model);
  const double prb_diff = bpix::Compare(tf, "samples.npz", chains);
  const double agree    = Agreement(tf, "samples.npz", chains, {"h1_path", "hidden1", "hidden2", "visible", "samples"});
  std::printf("  TF final weights: blocked-chain sampled states agreement with TF's samples.npz %.6f\n", agree);
  check::Require(out_diff < 1e-5 && prb_diff < 1e-5 && agree > .999, "TF's final weights reproduce TF's outputs and Gibbs chains");
  std::cout << "Artifacts: " << root.string() << "\nAll deep_boltzmann checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Run(ParseOptions(argc, argv, "deep_boltzmann", {}, "epochs=N out=PATH init=auto|tf|nn|PATH tf=DIR threads=N"));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
