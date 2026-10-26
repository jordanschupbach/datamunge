// CMD: just run-nn flow_matching
// Independent-coupling straight-path flow matching / first rectified flow, three
// cheap epochs.
//
// Port of tflow's flow_matching (common/{nn,ml}/flow_matching.h ->
// examples/nn/common/flow_matching.hpp). Data: synthetic, no dataset needed: 8
// equally weighted 2-D Gaussians (centers on radius 2, std .15), 512 fit / 128
// validation / 128 test from tflow's mt19937 Box-Muller banks. Independent
// standard Gaussian z paired with each data endpoint x, uniform t in [0, 1],
// x_t = (1 - t) z + t x, target velocity x - z; regress v(x_t, t) (3 -> 32 SiLU
// -> 32 SiLU -> 2, 1250 parameters) by MSE, three epochs of 32-point Adam(.003)
// batches in tflow's BatchOrder(512, 32, 20019): training is simulation free.
// Sampling solves dx/dt = v(x, t), t 0 -> 1 from 1024 common N(0, I) starts with
// Euler or RK4 (1, 8, 16, 32 steps), saving every stage and the full 2x2 step
// Jacobian; an analytic mixture-velocity "oracle" (no weights) is integrated with
// RK4 16/32/64 as a solver control.
//
// Checks (as tflow): parameter census; finite fields; fresh restore bit-exact;
// deterministic 32-step samplers repeat bit-exactly; straight-path endpoint
// identities (t = 0 -> z, t = 1 -> x) bit-exact; time conditioning changes the
// velocity; inference immutable. Data, noise and time banks are checked
// bit-identical to TF's data.npz. TF parity: init=auto (default) replays TF's
// initial_arrays.npz from tflow's output/models/flow_matching, so the run follows
// TF's and every record is compared; TF's final weights (model_arrays.npz) loaded
// into a fresh model must reproduce TF's test outputs and samplers. Deviations:
// final.npz instead of a SaveV2 bundle; no independent_audit.json. Options:
// epochs=3 out=DIR init=auto|tf|nn|PATH tf=DIR threads=4. Output:
// output/nn/models/flow_matching.
#include <chrono>

#include "common/flow_matching.hpp"

namespace {

namespace nn = datamunge::nn;
namespace fs = std::filesystem;
using nn::Tensor;
using Map = replay::Map;
using flowm::FlowMatchingModel;

Tensor Gaussian(std::int64_t n, std::uint32_t seed, bool wide = false, std::int64_t dimensions = 2) {
  const Tensor raw = nn::reshape(ae::GaussianNoise({n, dimensions}, 1, seed), {n, dimensions});
  return wide ? raw : nn::cast(raw, nn::DType::Float32);
}
Tensor Time(std::int64_t n, float value) { return nn::full({n}, value); }
Tensor Times(std::int64_t n, std::uint32_t seed, bool cyclic = false) {
  std::vector<float> out(static_cast<std::size_t>(n));
  std::mt19937       rng(seed);
  for (std::int64_t b = 0; b < n; ++b) out[b] = float(cyclic ? (b + .5) / n : (double(rng()) + .5) / 4294967296.);
  return nn::from_vector(out, {n});
}
void Finite(const Map& fields) {
  for (const auto& [key, x] : fields)
    if (x.dtype() == nn::DType::Float32)
      for (float v : x.contiguous().to_vector<float>()) check::Require(std::isfinite(v), "finite flow matching field:" + key);
}
using diffusion::Bits;
using diffusion::Load;
using diffusion::Save;

Map Sample(const FlowMatchingModel& model, int steps, bool rk4, std::int64_t n = 1024) {
  Tensor                                     current = Gaussian(n, 20201);
  std::vector<Tensor>                        path{current};
  std::map<std::string, std::vector<Tensor>> parts;
  Map                                        saved{{"initial", current}};
  for (int k = 0; k < steps; ++k) {
    auto fields = model.StepFields(current, Time(n, float(k) / steps), 1.f / steps, rk4);
    Finite(fields);
    for (const auto& [key, value] : fields) parts[key].push_back(value);
    current = fields.at("next");
    path.push_back(current);
  }
  for (const auto& [key, v] : parts) saved[key] = nn::stack(v, 0);
  saved["samples"] = current;
  saved["path"]    = nn::stack(path, 0);
  return saved;
}

/// tflow's Plot: one 384x384 panel per sampler, axes [-3, 3], blue samples, orange heldout.
void Plot(const fs::path& path, const Tensor& heldout, const std::vector<Tensor>& samples) {
  const std::int64_t W = std::int64_t(samples.size()) * 384;
  std::vector<float> rgb(static_cast<std::size_t>(384 * W * 3), 245.f);
  auto at = [&](std::int64_t y, std::int64_t x, int c) -> float& { return rgb[std::size_t((y * W + x) * 3 + c)]; };
  for (std::size_t p = 0; p < samples.size(); ++p) {
    for (int j = 0; j < 384; ++j)
      for (int c = 0; c < 3; ++c) at(192, p * 384 + j, c) = 190, at(j, p * 384 + 192, c) = 190;
    auto draw = [&](const Tensor& x, int radius, std::array<int, 3> color) {
      const auto v = x.contiguous().to_vector<float>();
      for (std::int64_t b = 0; b < x.size(0); ++b) {
        const int px = int(std::floor((double(v[2 * b]) + 3) * 64)), py = int(std::floor((3 - double(v[2 * b + 1])) * 64));
        if (px < 0 || px >= 384 || py < 0 || py >= 384) continue;
        for (int dy = -radius; dy <= radius; ++dy)
          for (int dx = -radius; dx <= radius; ++dx)
            if (px + dx >= 0 && px + dx < 384 && py + dy >= 0 && py + dy < 384)
              for (int c = 0; c < 3; ++c) at(py + dy, p * 384 + px + dx, c) = float(color[c]);
      }
    };
    draw(samples[p], 1, {30, 90, 190});
    draw(heldout, 2, {230, 100, 20});
  }
  bpix::PngBytes(path, rgb, 384, W);
}

double Scalar(const Map& m, const std::string& k) { return m.at(k).item<float>(); }

void Run(const diffusion::Options& o) {
  const fs::path root = o.out, tf = o.tf;
  const auto     started = std::chrono::steady_clock::now();
  Map            data;
  std::vector<Tensor> pools;
  for (int k = 0; k < 3; ++k) {
    const int          n     = k ? 128 : 512;
    const std::string  split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    const Tensor       base  = Gaussian(n, 20001 + 2 * k, true, 3);
    const auto         bv    = base.to_vector<double>();
    std::vector<float> x(std::size_t(n) * 2);
    std::vector<std::int32_t> ids(n), component(n);
    for (int b = 0; b < n; ++b) {
      const int c  = std::min(7, int(8 * .5 * std::erfc(-bv[b * 3 + 2] / std::sqrt(2.))));
      component[b] = c;
      ids[b]       = k * 10000 + b;
      for (int j = 0; j < 2; ++j)
        x[b * 2 + j] = float(2 * (j ? std::sin(2 * M_PI * c / 8) : std::cos(2 * M_PI * c / 8)) + .15 * bv[b * 3 + j]);
    }
    data[split + "_base"]       = base;
    data[split + "_points"]     = nn::from_vector(x, {n, 2});
    data[split + "_components"] = diffusion::Ints(component);
    data[split + "_source_ids"] = diffusion::Ints(ids);
    pools.push_back(data[split + "_points"]);
  }
  const Tensor vz = Gaussian(128, 20101), tz = Gaussian(128, 20103), vt = Times(128, 0, true), tt = Times(128, 0, true);
  data["validation_noise"] = vz;
  data["test_noise"]       = tz;
  data["validation_time"]  = vt;
  data["test_time"]        = tt;
  Save(root / "data.npz", data);
  if (fs::exists(tf / "data.npz")) {
    const auto theirs = Load(tf / "data.npz");
    for (const auto& [k, v] : data)
      check::Require(theirs.count(k) && Bits(theirs.at(k)) == Bits(v), "data/noise/time bit-identical to TF's data.npz: " + k);
    std::cout << "data: mixture pools, noise and time banks bit-identical to TF's data.npz\n";
  }

  FlowMatchingModel model;
  check::Require(model.store.count() == 1250 && model.store.size() == 6, "1250 velocity parameters 6 tensors");
  const fs::path init      = bpix::InitFile(o.init, tf);
  const bool     replaying = !init.empty();
  if (replaying) {
    replay::LoadInto(model.store, init);
    std::cout << "replaying TF's initial arrays " << init.string() << "\n";
  } else {
    std::cout << "weights from nn's generator (statistical parity with TF only)\n";
  }
  Save(root / "initial_arrays.npz", model.Arrays());
  Save(root / "initial_outputs.npz", model.Evaluate(pools[2], tz, tt));
  std::ofstream history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_velocity_mse\n";
  batches << "epoch\tstep\trow\n";
  nn::BatchOrder order(512, 32, 20019);
  Map            randomness;
  int            updates = 0;
  for (int epoch = 1; epoch <= o.epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const auto   n = std::int64_t(rows.size());
      const Tensor x = nn::take_rows(pools[0], rows), z = Gaussian(n, 2000051 + updates), t = Times(n, 3000021 + updates);
      randomness["step" + std::to_string(updates) + "_noise"] = Gaussian(n, 2000051 + updates, true);
      randomness["step" + std::to_string(updates) + "_time"]  = t;
      if (!updates) {
        Save(root / "first_batch.npz", {{"points", x}, {"noise", z}, {"time", t}});
        Save(root / "first_forward.npz", model.Evaluate(x, z, t));
        Save(root / "initial_gradients.npz", model.Gradient(x, z, t));
      }
      model.Update(x, z, t);
      if (!updates) Save(root / "first_after.npz", model.Arrays());
      for (auto row : rows) batches << epoch << '\t' << updates << '\t' << row << '\n';
      ++updates;
    }
    const auto eval = model.Evaluate(pools[1], vz, vt);
    Finite(eval);
    history << epoch << '\t' << Scalar(eval, "loss") << '\n';
    Save(root / ("epoch" + std::to_string(epoch) + "_arrays.npz"), model.Arrays());
    Save(root / ("epoch" + std::to_string(epoch) + "_validation.npz"), eval);
    std::cout << "epoch" << epoch << " velocityMSE=" << Scalar(eval, "loss") << std::endl;
  }
  const double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  Save(root / "training_randomness.npz", randomness);
  const auto batch = Load(root / "first_batch.npz");
  Save(root / "final_gradients.npz", model.Gradient(batch.at("points"), batch.at("noise"), batch.at("time")));
  model.store.save((root / "final.npz").string());
  FlowMatchingModel deployed(false), oracle(false, true);
  deployed.store.load((root / "final.npz").string(), true);
  check::Require(model.Snapshot() == deployed.Snapshot(), "all flow weights BITEXACT restore");
  const auto before = deployed.Snapshot();
  const auto output = deployed.Evaluate(pools[2], tz, tt), oracle_output = oracle.Evaluate(pools[2], tz, tt);
  Finite(output);
  Finite(oracle_output);
  Save(root / "model_arrays.npz", deployed.Arrays());
  Save(root / "outputs.npz", output);
  Save(root / "oracle_outputs.npz", oracle_output);
  Save(root / "euler_probe.npz", deployed.StepFields(output.at("interpolated"), tt, .03125f, false));
  Save(root / "rk4_probe.npz", deployed.StepFields(output.at("interpolated"), tt, .03125f, true));
  std::vector<Tensor> euler_samples, rk4_samples;
  for (int steps : {1, 8, 16, 32})
    for (bool rk4 : {false, true}) {
      const auto        sample = Sample(deployed, steps, rk4);
      const std::string name   = std::string(rk4 ? "rk4" : "euler") + std::to_string(steps);
      Save(root / (name + ".npz"), sample);
      if (steps != 1) (rk4 ? rk4_samples : euler_samples).push_back(sample.at("samples"));
      if (steps == 32) {
        const auto again = Sample(deployed, steps, rk4);
        for (const auto& [key, value] : sample)
          check::Require(Bits(value) == Bits(again.at(key)), "deterministic flow sampler repeat BITEXACT");
      }
    }
  Plot(root / "euler_samples.png", pools[2], euler_samples);
  Plot(root / "rk4_samples.png", pools[2], rk4_samples);
  std::vector<Tensor> oracle_samples;
  for (int steps : {16, 32, 64}) {
    const auto sample = Sample(oracle, steps, true);
    Save(root / ("oracle_rk4" + std::to_string(steps) + ".npz"), sample);
    oracle_samples.push_back(sample.at("samples"));
  }
  Plot(root / "oracle_samples.png", pools[2], oracle_samples);
  // Endpoints of interpolation are exact algebraic checks, not training losses.
  const auto start = deployed.Evaluate(pools[2], tz, Time(128, 0)), end = deployed.Evaluate(pools[2], tz, Time(128, 1));
  check::Require(Bits(start.at("interpolated")) == Bits(tz) && Bits(end.at("interpolated")) == Bits(pools[2]),
                 "straight-path endpoint identities BITEXACT");
  Save(root / "endpoint_controls.npz", {{"start", start.at("interpolated")}, {"end", end.at("interpolated")}});
  const auto   shifted     = deployed.StepFields(output.at("interpolated"), Time(128, .73f), .03125f, false);
  const auto   probe       = deployed.StepFields(output.at("interpolated"), tt, .03125f, false);
  const double time_effect = diffusion::MaxAbs(shifted.at("velocity0"), probe.at("velocity0"));
  check::Require(time_effect > 1e-5, "time conditioning actually changes velocity");
  Save(root / "time_control.npz", shifted);

  // Fit-only correlated Gaussian: analytic marginal velocity baseline (float64).
  const auto          fit = pools[0].to_vector<float>();
  std::vector<double> fitmean(2, 0.), cov(4, 0.), base_velocity(256);
  for (int b = 0; b < 512; ++b)
    for (int j = 0; j < 2; ++j) fitmean[j] += double(fit[b * 2 + j]) / 512;
  for (int b = 0; b < 512; ++b)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 2; ++k) cov[j * 2 + k] += (double(fit[b * 2 + j]) - fitmean[j]) * (double(fit[b * 2 + k]) - fitmean[k]) / 512;
  const auto ip = output.at("interpolated").to_vector<float>(), tg = output.at("target").to_vector<float>();
  const auto tv = tt.to_vector<float>();
  double     bmse = 0, zero = 0;
  for (int b = 0; b < 128; ++b) {
    const double t = tv[b], a = (1 - t) * (1 - t) + t * t * cov[0], d = (1 - t) * (1 - t) + t * t * cov[3], c = t * t * cov[1],
                 det = a * d - c * c, u = ip[b * 2] - t * fitmean[0], v = ip[b * 2 + 1] - t * fitmean[1], p = (d * u - c * v) / det,
                 q = (a * v - c * u) / det;
    for (int j = 0; j < 2; ++j) {
      const double value = fitmean[j] + (t * cov[j * 2] - (j == 0 ? 1 - t : 0)) * p + (t * cov[j * 2 + 1] - (j == 1 ? 1 - t : 0)) * q;
      base_velocity[b * 2 + j] = value;
      bmse += std::pow(value - tg[b * 2 + j], 2) / 256;
      zero += std::pow(double(tg[b * 2 + j]), 2) / 256;
    }
  }
  Save(root / "baseline.npz", {{"fit_mean", nn::from_vector(fitmean, {2})},
                               {"fit_covariance", nn::from_vector(cov, {2, 2})},
                               {"velocity", nn::from_vector(base_velocity, {128, 2})}});
  check::Require(before == deployed.Snapshot(), "all flow matching inference weights immutable");
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":1250,\"parameter_tensors\":6,\"epochs\":" << o.epochs << ",\"optimizer_updates\":" << updates
       << ",\"test_velocity_mse\":" << Scalar(output, "loss") << ",\"fit_gaussian_velocity_mse\":" << bmse
       << ",\"zero_velocity_mse\":" << zero << ",\"oracle_conditional_velocity_mse\":" << Scalar(oracle_output, "loss")
       << ",\"time_conditioning_max_effect\":" << time_effect << ",\"samples_per_sampler\":1024,\"replayed_tf_initial_arrays\":"
       << (replaying ? "true" : "false") << ",\"training_seconds\":" << train_seconds << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Flow matching</title><style>body{font:16px system-ui;margin:30px;"
          "max-width:1250px}img{max-width:100%}</style><h1>Straight-path conditional flow matching / first rectified flow</h1>"
          "<p>Synthetic 8 equally weighted 2D Gaussians, centers radius 2, std .15. 512 fit/128 validation/128 test. Independent "
          "standard Gaussian z paired with each data endpoint x. Uniform time t in [0,1], interpolation x_t=(1-t)z+t*x, target velocity "
          "x-z. Regress v_theta(x_t,t) with mean squared error; training is SIMULATION FREE. 1250 parameter 3->32 SiLU->32 SiLU->2 MLP, "
          "three epochs 48 Adam .003 updates, fixed final. Datamunge port of tflow's flow_matching.</p><p><a "
          "href='https://arxiv.org/abs/2210.02747'>Flow Matching</a> and <a href='https://arxiv.org/abs/2209.03003'>Rectified Flow</a> "
          "motivate this independent-coupling straight path. Sampling solves dx/dt=v(x,t), t 0->1 from N(0,I), deterministically. One "
          "Euler step is NOT guaranteed to reproduce data.</p><h2>Euler 8, 16, 32 steps</h2><p>1024 common starts, blue generated/orange "
          "128 heldout, axes [-3,3].</p><img src='euler_samples.png'><h2>RK4 8, 16, 32 steps</h2><img src='rk4_samples.png'><h2>Analytic "
          "mixture velocity control, RK4 16, 32, 64</h2><p>Known conditional mean velocity from Gaussian mixture responsibilities: a "
          "solver control with no learned weights, not a trained-model quality claim.</p><img src='oracle_samples.png'><p>No "
          "likelihood, FID, straightness, speed or quality superiority claim.</p>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete flow matching records");
  std::cout << "testVelocityMSE=" << Scalar(output, "loss") << " fitGaussian=" << bmse << " exactConditional=" << Scalar(oracle_output, "loss")
            << ";endpoints/timeControl/deterministicODE/restore/immutable PASS (training " << train_seconds << " s)\n";

  // ---- TF parity ----
  if (!fs::exists(tf / "model_arrays.npz")) {
    std::cout << "no TF record at " << tf.string() << "; parity checks skipped\nArtifacts: " << root.string() << "\n";
    return;
  }
  std::cout << "TF parity (" << tf.string() << "):\n";
  const double oracle_diff = bpix::Compare(tf, "oracle_outputs.npz", oracle_output);
  const double oracle_smp  = bpix::Compare(tf, "oracle_rk464.npz", Load(root / "oracle_rk464.npz"));
  bpix::Compare(tf, "baseline.npz", Load(root / "baseline.npz"));
  check::Require(oracle_diff < 1e-5 && oracle_smp < 1e-3, "analytic mixture velocity and its RK4 samples match TF");
  if (replaying) {
    const double fwd = bpix::Compare(tf, "first_forward.npz", Load(root / "first_forward.npz"));
    const double grd = bpix::Compare(tf, "initial_gradients.npz", Load(root / "initial_gradients.npz"));
    const double aft = bpix::Compare(tf, "first_after.npz", Load(root / "first_after.npz"));
    check::Require(fwd < 1e-5 && grd < 1e-5 && aft < 1e-5, "replayed first forward/gradients/update match TF");
    bpix::Compare(tf, "model_arrays.npz", deployed.Arrays());
    bpix::Compare(tf, "outputs.npz", output);
    bpix::Compare(tf, "rk432.npz", Load(root / "rk432.npz"));
    history.flush();
    const auto ours = replay::History(root / "history.tsv"), theirs = replay::History(tf / "history.tsv");
    for (std::size_t i = 0; i < std::min(ours.size(), theirs.size()); ++i)
      std::printf("  epoch %zu validation velocity MSE ours %.6f TF %.6f\n", i + 1, ours[i], theirs[i]);
  }
  FlowMatchingModel tf_model(false);
  replay::LoadInto(tf_model.store, tf / "model_arrays.npz");
  const auto   tf_out   = tf_model.Evaluate(pools[2], tz, tt);
  const double out_diff = bpix::Compare(tf, "outputs.npz", tf_out);
  const double prb_diff = bpix::Compare(tf, "rk4_probe.npz", tf_model.StepFields(tf_out.at("interpolated"), tt, .03125f, true));
  const double eul_diff = bpix::Compare(tf, "euler32.npz", Sample(tf_model, 32, false));
  const double rk4_diff = bpix::Compare(tf, "rk432.npz", Sample(tf_model, 32, true));
  check::Require(out_diff < 1e-4 && prb_diff < 1e-4 && eul_diff < 1e-3 && rk4_diff < 1e-3,
                 "TF's final weights reproduce TF's test outputs, step Jacobians and samplers");
  std::cout << "TF final weights: test velocity MSE " << Scalar(tf_out, "loss") << " (TF record " << Scalar(Load(tf / "outputs.npz"), "loss")
            << ")\nArtifacts: " << root.string() << "\nAll flow_matching checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto o = diffusion::ParseOptions(argc, argv, "flow_matching", {}, "epochs=N out=PATH init=auto|tf|nn|PATH tf=DIR threads=N");
    Run(o);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
