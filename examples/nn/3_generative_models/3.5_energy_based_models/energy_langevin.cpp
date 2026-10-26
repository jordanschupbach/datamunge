// CMD: just run-nn energy_langevin
// Persistent-negative energy learning with actual-force ULA; three cheap epochs
// and numerical normalizer checks.
//
// Port of tflow's energy_langevin (common/{nn,ml}/energy_langevin.h ->
// examples/nn/common/energy_langevin.hpp). Data: synthetic 2-D 8-Gaussian mixture
// (centers on radius 2, std .15; 512 fit / 128 validation / 128 test) built from
// tflow's mt19937 Gaussian banks (seeds 25001+2k), so no dataset is needed. Energy
// E(x) = .5 ||x||^2 + .5 sum_j v_j tanh(x W_j + b_j), 64 parameters. Three epochs of
// 48 Adam(.01) updates (BatchOrder(512, 32, 25019)) on mean E(data) - mean
// E(stop(negative)); 32 persistent negative chains from N(0, I) (seed 25101), five
// ULA steps per update at alpha .04 with tflow's innovation banks (2500051 + 5 step
// + k). The ULA force is the actual autograd input gradient (nn::grad w.r.t. the
// state; first order only: the training loss stops gradients through the chains,
// as tflow). The analytic Hessian rows are a diagnostic. Normalizer by midpoint
// quadrature on [-8, 8]^2 (128^2 and 256^2 grids).
//
// Checks (as tflow): 64 parameters / 3 tensors; negative chains receive no gradient
// from the training loss; restore bit-exact; fixed-seed ULA repeats bit-exactly;
// inference immutable; complete records. Added: the analytic Hessian rows match
// central finite differences of the autograd force. TF parity: init=auto (default)
// replays TF's initial_arrays.npz from tflow's output/models/energy_langevin; the
// per-update training trace (weights before/after, every ULA state/force, every
// gradient), history and final weights are compared with TF's; independently TF's
// final weights must reproduce TF's test outputs and 128-sweep samples. Data and
// persistent starts are checked bit-identical. Deviations: final.npz instead of a
// SaveV2 bundle; no independent_audit.json. Options: epochs=3 out=DIR
// init=auto|tf|nn|PATH tf=DIR threads=4. Output: output/nn/models/energy_langevin.
#include <array>
#include <chrono>

#include "common/energy_langevin.hpp"

namespace {

using namespace ebm;

Map Sample(const EnergyLangevinModel& model, float alpha, int sweeps, std::uint32_t seed) {
  const std::int64_t n       = 1024;
  Tensor             current = Gaussian(n, seed);
  Map                saved{{"initial", current}};
  std::vector<Tensor> path{current};
  std::map<std::string, std::vector<Tensor>> steps;
  for (int k = 0; k < sweeps; ++k) {
    const auto fields = model.Step(current, Gaussian(n, seed * 100 + 51 + k), alpha);
    for (const auto& [key, value] : fields) steps[key].push_back(nn::reshape(value, {n, -1}));
    current = fields.at("next");
    path.push_back(current);
  }
  for (const auto& [key, values] : steps) saved[key] = nn::stack(values, 0);
  saved["samples"] = current;
  saved["path"]    = nn::stack(path, 0);
  return saved;
}

struct Weights {
  std::vector<float> W, b, v;
  explicit Weights(const Map& w)
      : W(w.at("hidden/W").to_vector<float>()), b(w.at("hidden/b").to_vector<float>()), v(w.at("output/W").to_vector<float>()) {}
};

double Energy(double x, double y, const Weights& w) {
  double value = .5 * (x * x + y * y);
  for (int j = 0; j < 16; ++j) value += .5 * double(w.v[j]) * std::tanh(x * w.W[j] + y * w.W[16 + j] + w.b[j]);
  return value;
}

Map Quadrature(const Map& arrays, int side) {
  const Weights       w(arrays);
  std::vector<double> grid(side * side * 2), energy(side * side);
  double              total = 0;
  for (int y = 0; y < side; ++y)
    for (int x = 0; x < side; ++x) {
      const int    i = y * side + x;
      const double a = -8 + (x + .5) * 16 / side, b = -8 + (y + .5) * 16 / side;
      grid[i * 2] = a, grid[i * 2 + 1] = b;
      energy[i] = Energy(a, b, w);
      total += std::exp(-energy[i]);
    }
  const double logz = std::log(total * std::pow(16. / side, 2));
  return {{"grid", nn::from_vector(grid, {side * side, 2})},
          {"energy", nn::from_vector(energy, {side * side})},
          {"log_partition_estimate", nn::from_vector(std::vector<double>{logz}, {})}};
}

double Nll(const Tensor& x, const Map& arrays, double logz) {
  const Weights w(arrays);
  const auto    v = x.to_vector<float>();
  const auto    n = x.size(0);
  double        value = 0;
  for (std::int64_t b = 0; b < n; ++b) value += (Energy(v[b * 2], v[b * 2 + 1], w) + logz) / double(n);
  return value;
}

void Plot(const fs::path& path, const Tensor& heldout, const std::vector<Tensor>& samples) {
  const std::int64_t W = std::int64_t(samples.size()) * 384;
  std::vector<float> rgb(static_cast<std::size_t>(384 * W * 3), 245.f);
  auto put = [&](std::int64_t y, std::int64_t x, std::array<int, 3> c) {
    for (int k = 0; k < 3; ++k) rgb[static_cast<std::size_t>((y * W + x) * 3 + k)] = float(c[k]);
  };
  for (std::size_t p = 0; p < samples.size(); ++p) {
    for (int j = 0; j < 384; ++j) {
      put(192, p * 384 + j, {190, 190, 190});
      put(j, p * 384 + 192, {190, 190, 190});
    }
    auto draw = [&](const Tensor& x, int radius, std::array<int, 3> color) {
      const auto v = x.to_vector<float>();
      for (std::int64_t b = 0; b < x.size(0); ++b) {
        const int px = int(std::floor((double(v[b * 2]) + 3) * 64)), py = int(std::floor((3 - double(v[b * 2 + 1])) * 64));
        if (px < 0 || px >= 384 || py < 0 || py >= 384) continue;
        for (int dy = -radius; dy <= radius; ++dy)
          for (int dx = -radius; dx <= radius; ++dx)
            if (px + dx >= 0 && px + dx < 384 && py + dy >= 0 && py + dy < 384) put(py + dy, std::int64_t(p) * 384 + px + dx, color);
      }
    };
    draw(samples[p], 1, {30, 90, 190});
    draw(heldout, 2, {230, 100, 20});
  }
  bpix::PngBytes(path, rgb, 384, W);
}

void Run(const Options& o) {
  const fs::path root = o.out, tf = o.tf;
  const auto     started = std::chrono::steady_clock::now();
  Map            data;
  std::vector<Tensor> pools;
  for (int k = 0; k < 3; ++k) {
    const int         n     = k ? 128 : 512;
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    const Tensor      base  = Gaussian(n, 25001 + 2 * k, true, 3);
    const auto        bv    = base.to_vector<double>();
    std::vector<float>        x(n * 2);
    std::vector<std::int32_t> component(n), ids(n);
    for (int b = 0; b < n; ++b) {
      const int c  = std::min(7, int(4 * std::erfc(-bv[b * 3 + 2] / std::sqrt(2.))));
      component[b] = c;
      ids[b]       = 10000 * k + b;
      for (int j = 0; j < 2; ++j)
        x[b * 2 + j] = float(2 * (j ? std::sin(2 * M_PI * c / 8) : std::cos(2 * M_PI * c / 8)) + .15 * bv[b * 3 + j]);
    }
    data[split + "_base"]       = base;
    data[split + "_points"]     = nn::from_vector(x, {n, 2});
    data[split + "_components"] = Ints(component);
    data[split + "_source_ids"] = Ints(ids);
    pools.push_back(data.at(split + "_points"));
  }
  Save(root / "data.npz", data);
  CheckData(tf, data, {"fit_points", "validation_points", "test_points", "fit_base", "test_components"});
  EnergyLangevinModel model;
  check::Require(model.store.count() == 64 && model.store.size() == 3, "64 confining energy parameters, no output bias");
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
  nn::BatchOrder order(512, 32, 25019);
  Tensor         persistent = Gaussian(32, 25101);
  Save(root / "persistent_initial.npz", {{"states", persistent}});
  check::Require(SameAsTf(tf, "persistent_initial.npz", {{"states", persistent}}), "persistent chain starts bit-identical to TF's");
  check::Require(model.NegativeStopped(Rows(pools[0], Range(32)), persistent), "negative chains have no gradient through training loss");
  Map           training;
  std::ofstream history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_quadrature_nll\tlog_partition_128\n";
  batches << "epoch\tstep\trow\n";
  int updates = 0;
  for (int epoch = 1; epoch <= o.epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const Tensor      points = Rows(pools[0], rows);
      const std::string step   = "step" + std::to_string(updates) + "_";
      for (const auto& [key, value] : model.Arrays()) training[step + "before_" + key] = value;
      std::vector<Tensor> negative_path{persistent};
      for (int k = 0; k < 5; ++k) {
        const auto fields = model.Step(persistent, Gaussian(32, 2500051 + updates * 5 + k), .04f);
        for (const auto& [key, value] : fields) training[step + "ula" + std::to_string(k) + "_" + key] = value;
        persistent = fields.at("next");
        negative_path.push_back(persistent);
      }
      training[step + "negative_path"] = nn::stack(negative_path, 0);
      if (!updates) {
        Save(root / "first_batch.npz", {{"positive", points}, {"negative", persistent}});
        Save(root / "first_forward.npz", model.Contrastive(points, persistent));
        Save(root / "initial_gradients.npz", model.Gradient(points, persistent));
      }
      for (const auto& [key, value] : model.Gradient(points, persistent)) training[step + "gradient_" + key] = value;
      model.Update(points, persistent);
      if (!updates) Save(root / "first_after.npz", model.Arrays());
      for (const auto& [key, value] : model.Arrays()) training[step + "after_" + key] = value;
      for (auto row : rows) batches << epoch << '\t' << updates << '\t' << row << '\n';
      ++updates;
    }
    const auto   w    = model.Arrays();
    const auto   q    = Quadrature(w, 128);
    const double logz = Scalar(q, "log_partition_estimate"), nll = Nll(pools[1], w, logz);
    history << epoch << '\t' << nll << '\t' << logz << '\n';
    Save(root / ("epoch" + std::to_string(epoch) + "_arrays.npz"), w);
    Save(root / ("epoch" + std::to_string(epoch) + "_validation.npz"), model.Evaluate(pools[1]));
    std::cout << "epoch" << epoch << " validationQuadratureNLL=" << nll << std::endl;
  }
  history.flush();
  batches.flush();
  const double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  Save(root / "training_trace.npz", training);
  const auto first = Load(root / "first_batch.npz");
  Save(root / "final_gradients.npz", model.Gradient(first.at("positive"), first.at("negative")));
  model.store.save((root / "final.npz").string());
  EnergyLangevinModel deployed(false), oracle(false, true);
  deployed.store.load((root / "final.npz").string(), true);
  check::Require(model.Snapshot() == deployed.Snapshot(), "all 64 energy weights bitexact restore");
  const auto before = deployed.Snapshot();
  const auto w      = deployed.Arrays();
  Save(root / "model_arrays.npz", w);
  const auto output = deployed.Evaluate(pools[2]);
  Save(root / "outputs.npz", output);
  Save(root / "force_probe.npz", deployed.Step(pools[2], Gaussian(128, 25103), .04f));
  // Added check: analytic Hessian rows vs central differences of the autograd force (float64 copy).
  {
    const Tensor x  = nn::cast(Rows(pools[2], Range(16)), nn::DType::Float64);
    nn::ParameterStore p64(0, nn::DType::Float64);
    ConfiningEnergyParameters(p64);
    for (const auto& name : p64.names()) p64.set(name, nn::cast(w.at(name), nn::DType::Float64));
    auto force = [&](const Tensor& at) {
      const Tensor leaf = at.detach().clone().requires_grad_(true);
      return nn::grad(nn::sum(ConfiningEnergy(&p64, leaf).energy), {leaf})[0].detach();
    };
    const auto rows   = ConfiningEnergy(&p64, x).hessian_rows;
    double     fd_err = 0;
    for (int a = 0; a < 2; ++a) {
      std::vector<double> e(32, 0.);
      for (int b = 0; b < 16; ++b) e[b * 2 + a] = 1e-5;
      const Tensor d  = nn::from_vector(e, {16, 2});
      const Tensor fd = nn::div(nn::sub(force(nn::add(x, d)), force(nn::sub(x, d))), 2e-5);
      fd_err          = std::max(fd_err, replay::MaxAbsDiff(fd, rows[a].detach()));
    }
    std::printf("analytic Hessian vs finite differences of the autograd force: max err %.3g\n", fd_err);
    check::Require(fd_err < 1e-6, "analytic Hessian diagnostic matches finite differences of the actual force");
  }
  std::vector<Tensor> draws;
  Map                 sample4;
  for (float alpha : {.01f, .04f, .1f}) {
    const int  label  = int(std::round(alpha * 100));
    const auto sample = Sample(deployed, alpha, 128, 25201);
    Save(root / ("samples_alpha" + std::to_string(label) + ".npz"), sample);
    draws.push_back(sample.at("samples"));
    if (label == 4) {
      sample4           = sample;
      const auto repeat = Sample(deployed, alpha, 128, 25201);
      for (const auto& [key, value] : sample) check::Require(Bits(value) == Bits(repeat.at(key)), "fixed seed energy ULA repeat bitexact");
    }
  }
  Plot(root / "samples.png", pools[2], draws);
  std::vector<Tensor> oracle_draws;
  for (float alpha : {.01f, .04f, .1f}) {
    const int  label  = int(std::round(alpha * 100));
    const auto sample = Sample(oracle, alpha, 128, 25301);
    Save(root / ("oracle_alpha" + std::to_string(label) + ".npz"), sample);
    oracle_draws.push_back(sample.at("samples"));
  }
  const Tensor       oracle_base = Gaussian(128, 25303, true);
  const auto         ob          = oracle_base.to_vector<double>();
  std::vector<float> truth(256);
  for (int b = 0; b < 128; ++b) truth[b * 2] = float(.3 + .7 * ob[b * 2]), truth[b * 2 + 1] = float(-.2 + .5 * ob[b * 2 + 1]);
  const Tensor oracle_truth = nn::from_vector(truth, {128, 2});
  Save(root / "oracle_target.npz", {{"base", oracle_base}, {"points", oracle_truth}});
  Plot(root / "oracle_samples.png", oracle_truth, oracle_draws);
  const auto   q128 = Quadrature(w, 128), q256 = Quadrature(w, 256);
  Save(root / "quadrature128.npz", q128);
  Save(root / "quadrature256.npz", q256);
  const double logz = Scalar(q256, "log_partition_estimate"), nll = Nll(pools[2], w, logz);
  double       residual_bound = 0;
  for (float v : w.at("output/W").to_vector<float>()) residual_bound += .5 * std::abs(double(v));
  const double tail_integral_upper = std::exp(residual_bound) * 2 * M_PI * 2 * std::erfc(8 / std::sqrt(2.));
  const auto   fit = pools[0].to_vector<float>(), test = pools[2].to_vector<float>();
  double       mean[2] = {0, 0}, cov[2][2] = {{0, 0}, {0, 0}};
  for (int b = 0; b < 512; ++b)
    for (int j = 0; j < 2; ++j) mean[j] += double(fit[b * 2 + j]) / 512;
  for (int b = 0; b < 512; ++b)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 2; ++k) cov[j][k] += (double(fit[b * 2 + j]) - mean[j]) * (double(fit[b * 2 + k]) - mean[k]) / 512;
  const double det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
  double       baseline = 0;
  for (int b = 0; b < 128; ++b) {
    const double x = test[b * 2] - mean[0], y = test[b * 2 + 1] - mean[1];
    baseline += (std::log(2 * M_PI) + .5 * std::log(det) + .5 * (x * x * cov[1][1] - 2 * x * y * cov[0][1] + y * y * cov[0][0]) / det) / 128;
  }
  Save(root / "baseline.npz", {{"fit_mean", nn::from_vector(std::vector<double>{mean[0], mean[1]}, {2})},
                               {"fit_covariance", nn::from_vector(std::vector<double>{cov[0][0], cov[0][1], cov[1][0], cov[1][1]}, {2, 2})}});
  check::Require(before == deployed.Snapshot(), "all energy/force/ULA/quadrature inference weights immutable");
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":64,\"parameter_tensors\":3,\"epochs\":" << o.epochs << ",\"optimizer_updates\":" << updates
       << ",\"negative_chains\":32,\"ula_steps_per_update\":5,\"training_ula_step_size\":0.04,\"sample_ula_steps\":128,"
          "\"samples_per_sampler\":1024,\"test_quadrature_nll\":"
       << nll << ",\"fit_full_gaussian_nll\":" << baseline << ",\"log_partition_128_estimate\":" << Scalar(q128, "log_partition_estimate")
       << ",\"log_partition_256_estimate\":" << logz << ",\"bounded_residual_magnitude\":" << residual_bound
       << ",\"outside_box_unnormalized_mass_upper_bound\":" << tail_integral_upper
       << ",\"replayed_tf_initial_arrays\":" << (replaying ? "true" : "false") << ",\"training_seconds\":" << train_seconds << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Energy model and Langevin</title><style>body{font:16px system-ui;margin:30px;"
          "max-width:1250px}img{max-width:100%}</style><h1>Learned confining energy with persistent Langevin negatives</h1><p>"
          "Synthetic 8-Gaussian mixture in 2-D (centers radius 2, std .15), 512 fit / 128 validation / 128 test. E(x) = .5||x||^2 + "
          ".5 sum_j v_j tanh(x W_j + b_j), 64 parameters. Three epochs, 48 Adam .01 updates on mean E(data) - mean E(stopped "
          "negative); 32 persistent chains, five ULA steps per update at alpha .04. The ULA force x_next = x - .5 alpha grad E(x) + "
          "sqrt(alpha) z is the actual autograd input gradient. Datamunge port of tflow's energy_langevin.</p><h2>Learned-energy "
          "128-sweep samples, alpha .01 / .04 / .10</h2><p>Blue generated, orange heldout, axes [-3, 3].</p><img src='samples.png'>"
          "<h2>Gaussian oracle ULA control</h2><img src='oracle_samples.png'><p>Numerical partition by 128^2 / 256^2 midpoint grids "
          "on [-8, 8]^2. <a href='https://arxiv.org/abs/1903.08689'>Du-Mordatch EBM</a>.</p>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete energy Langevin records");
  std::cout << "testQuadratureNLL=" << nll << " fitGaussian=" << baseline << " logZ128/256=" << Scalar(q128, "log_partition_estimate")
            << '/' << logz << ";confinement/actualForce/stoppedNegatives/persistentChains/restore/repeat/immutable PASS (training "
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
    check::Require(fwd < 1e-5 && grd < 1e-5 && aft < 1e-5, "replayed first contrastive forward/gradients/update match TF");
    if (o.epochs == 3) {
      bpix::Compare(tf, "training_trace.npz", training);
      const double fin = bpix::Compare(tf, "model_arrays.npz", w);
      bpix::Compare(tf, "outputs.npz", output);
      bpix::Compare(tf, "samples_alpha4.npz", sample4);
      check::Require(fin < 1e-4, "replayed run's final weights match TF's");
    }
    CompareHistory(root, tf, 1, "validation quadrature NLL");
  }
  EnergyLangevinModel tf_model(false);
  replay::LoadInto(tf_model.store, tf / "model_arrays.npz");
  const double out_diff = bpix::Compare(tf, "outputs.npz", tf_model.Evaluate(pools[2]));
  const double smp_diff = bpix::Compare(tf, "samples_alpha4.npz", Sample(tf_model, .04f, 128, 25201));
  const double orc_diff = bpix::Compare(tf, "oracle_alpha4.npz", Sample(oracle, .04f, 128, 25301));
  const double q_diff   = bpix::Compare(tf, "quadrature256.npz", Quadrature(Load(tf / "model_arrays.npz"), 256));
  check::Require(out_diff < 1e-5 && smp_diff < 1e-4 && orc_diff < 1e-4 && q_diff < 1e-9,
                 "TF's final weights reproduce TF's outputs, ULA samples and quadrature");
  std::cout << "Artifacts: " << root.string() << "\nAll energy_langevin checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Run(ParseOptions(argc, argv, "energy_langevin", {}, "epochs=N out=PATH init=auto|tf|nn|PATH tf=DIR threads=N"));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
