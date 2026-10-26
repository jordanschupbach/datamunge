// CMD: just run-nn noise_conditional_score
// NCSN: weighted denoising score matching and annealed Langevin dynamics on
// continuous 8x8 MNIST.
//
// Port of tflow's noise_conditional_score (common/{nn,ml}/noise_conditional_score.h
// -> examples/nn/common/noise_conditional_score.hpp). Data: tflow's saliency MNIST
// pools (paths::Dataset("saliency")): 1000 fit / 200 validation / 200 test,
// half-pixel bilinear 28 -> 8, scaled to [-1, 1]. Epsilon MLP 72 -> 32 SiLU -> 32
// SiLU -> 64 (5504 parameters / 6 tensors) with time features over 15 levels; 16
// geometric sigmas 1 -> .05; VE corruption x + sigma eps; score = -eps / sigma;
// loss .5 mean ||sigma score + eps||^2 on uniform random levels, three epochs of
// 32-image Adam(.003) batches in tflow's BatchOrder(1000, 32, 12019). Annealed
// Langevin sampling: 16 levels x 4 updates, alpha_i = 2e-5 (sigma_i / .05)^2,
// starts uniform [-1, 1] via the Gaussian CDF, fresh innovations every update
// (including the last), optional final Tweedie x + .05^2 score shown separately.
//
// Checks (as tflow): Gaussian-oracle Langevin fixture (drift, diffusion and
// state/innovation Jacobians); parameter census; finite fields; the last Langevin
// step still adds noise; fresh restore bit-exact; fixed-seed trajectories repeat
// bit-exactly; noise conditioning changes epsilon; inference immutable. Data,
// noise and level banks are checked bit-identical to TF's data.npz. TF parity:
// init=auto (default) replays TF's initial_arrays.npz from tflow's
// output/models/noise_conditional_score and every record is compared with TF's;
// TF's final weights (model_arrays.npz) loaded into a fresh model must reproduce
// TF's test outputs and samples. Deviations: final.npz instead of a SaveV2
// bundle; no independent_audit.json. Options: epochs=3 data=DIR out=DIR
// init=auto|tf|nn|PATH tf=DIR threads=4. Output: output/nn/models/noise_conditional_score.
#include <chrono>

#include "common/noise_conditional_score.hpp"

namespace {

namespace nn = datamunge::nn;
namespace fs = std::filesystem;
using nn::Tensor;
using namespace diffusion;
using ncsn::NoiseConditionalScoreModel;

/// tflow's Times over 16 levels: i % 16 or int(16 (u + .5) / 2^32).
Tensor Levels(std::int64_t n, std::uint32_t seed, bool cyclic = false) {
  std::vector<std::int32_t> out(static_cast<std::size_t>(n));
  std::mt19937              rng(seed);
  for (std::int64_t i = 0; i < n; ++i) out[i] = cyclic ? int(i % 16) : int(16 * (double(rng()) + .5) / 4294967296.);
  return Ints(out);
}

/// tflow's Png: clamp [-1, 1], floor((v + 1) .5 255 + .5).
void Png(const fs::path& path, const Tensor& images, int columns) {
  const auto         n = images.size(0);
  const auto         v = images.contiguous().to_vector<float>();
  const std::int64_t H = (n + columns - 1) / columns * 64, W = columns * 64;
  std::vector<float> out(static_cast<std::size_t>(H * W * 3), 220.f);
  for (std::int64_t b = 0; b < n; ++b)
    for (int y = 0; y < 64; ++y)
      for (int x = 0; x < 64; ++x) {
        const double value = std::clamp(double(v[std::size_t(b * 64 + (y / 8) * 8 + x / 8)]), -1., 1.);
        for (int c = 0; c < 3; ++c) out[std::size_t(((b / columns * 64 + y) * W + b % columns * 64 + x) * 3 + c)] = float(std::floor((value + 1) * .5 * 255 + .5));
      }
  bpix::PngBytes(path, out, H, W);
}

/// Elementwise float32 x + (s * s) * score, as tflow's host loops.
Tensor Tweedie(const Tensor& x, float s, const Tensor& score) {
  auto       xv = x.contiguous().to_vector<float>();
  const auto sv = score.contiguous().to_vector<float>();
  for (std::size_t i = 0; i < xv.size(); ++i) xv[i] = xv[i] + s * s * sv[i];
  return nn::from_vector(xv, x.shape());
}

Map GaussianFixture() {
  const Tensor input = Gaussian8(16, 12301), noise = Gaussian8(16, 12303), levels = Levels(16, 0, true);
  Tensor       x = input.clone(), z = noise.clone();
  x.requires_grad_();
  z.requires_grad_();
  const Tensor score  = nn::div(nn::sub(nn::scalar_like(.3, x), x), .49);
  const auto   update = ncsn::AnnealedLangevin(x, levels, score, z);
  const auto   g      = nn::grad(nn::sum(update.value), {x, z});
  return {{"input", input},          {"innovation", noise},          {"level", levels},
          {"score", score.detach()}, {"next", update.value.detach()}, {"mean", update.mean.detach()},
          {"alpha", update.alpha},   {"state_gradient", g[0]},       {"innovation_gradient", g[1]}};
}

Map Sample(const NoiseConditionalScoreModel& model, std::int64_t n, std::uint32_t seed) {
  const Tensor initial_gaussian = Gaussian8(n, seed, true);
  std::vector<float> iv;
  for (double g : initial_gaussian.to_vector<double>()) iv.push_back(float(std::erfc(-g / std::sqrt(2.)) - 1));
  const Tensor initial = nn::from_vector(iv, {n, 8, 8, 1});
  Tensor       current = initial;
  StepStack    steps;
  std::vector<Tensor> trajectory{initial}, innovations;
  for (int step = 0; step < 64; ++step) {
    const Tensor t = FullInt(n, step / 4), z = Gaussian8(n, seed + 1 + step);
    innovations.push_back(Gaussian8(n, seed + 1 + step, true));
    const auto out = model.Reverse(current, t, z);
    Finite(out);
    steps.add({{"score", out.at("score")}, {"epsilon", out.at("epsilon")}, {"mean", out.at("mean")}, {"alpha", out.at("alpha")}});
    if (step == 63) {
      const auto other = model.Reverse(current, t, nn::full(z.shape(), 123.0));
      check::Require(MaxAbs(other.at("next"), out.at("next")) > .01, "Langevin LAST step STILL adds noise (unlike DDPM)");
    }
    current = out.at("next");
    trajectory.push_back(current);
  }
  const auto last = model.Reverse(current, FullInt(n, 15), nn::zeros({n, 8, 8, 1}));
  Map        result = steps.stacked();
  result["initial_gaussian"] = initial_gaussian;
  result["initial"]          = initial;
  result["images"]           = current;
  result["tweedie_images"]   = Tweedie(current, .05f, last.at("score"));
  result["tweedie_score"]    = last.at("score");
  result["trajectory"]       = nn::stack(trajectory, 0);
  result["innovations"]      = nn::stack(innovations, 0);
  return result;
}

double Scalar(const Map& m, const std::string& k) { return m.at(k).item<float>(); }

void Run(const Options& o) {
  const fs::path root = o.out, tf = o.tf;
  const auto     started = std::chrono::steady_clock::now();
  const auto     fixture = GaussianFixture();
  Save(root / "gaussian_langevin_fixture.npz", fixture);
  auto        d      = LoadDigits(o.data);
  const auto& images = d.images;
  Map         data   = d.record;
  const Tensor validation_noise = Gaussian8(200, 12101), test_noise = Gaussian8(200, 12103);
  const Tensor validation_time = Levels(200, 0, true), test_time = Levels(200, 0, true);
  data["validation_noise"] = validation_noise;
  data["test_noise"]       = test_noise;
  data["validation_time"]  = validation_time;
  data["test_time"]        = test_time;
  Save(root / "data.npz", data);
  if (fs::exists(tf / "data.npz")) {
    const auto theirs = Load(tf / "data.npz");
    for (const auto& [k, v] : data)
      check::Require(theirs.count(k) && Bits(theirs.at(k)) == Bits(v), "data/noise/level banks bit-identical to TF's data.npz: " + k);
    std::cout << "data: 8x8 pools, noise and level banks bit-identical to TF's data.npz\n";
  }
  if (fs::exists(o.data / "manifest.json")) fs::copy_file(o.data / "manifest.json", root / "data_manifest.json", fs::copy_options::overwrite_existing);
  const auto& sigmas = ncsn::ScoreSigmas();
  Save(root / "schedule.npz", {{"sigma", nn::from_vector(sigmas, {16})}});

  NoiseConditionalScoreModel model;
  check::Require(model.store.count() == 5504 && model.store.size() == 6, "5504 epsilon MLP parameters 6 tensors");
  const fs::path init      = bpix::InitFile(o.init, tf);
  const bool     replaying = !init.empty();
  if (replaying) {
    replay::LoadInto(model.store, init);
    std::cout << "replaying TF's initial arrays " << init.string() << "\n";
  } else {
    std::cout << "weights from nn's generator (statistical parity with TF only)\n";
  }
  Save(root / "initial_arrays.npz", model.Arrays());
  Save(root / "initial_outputs.npz", model.Evaluate(images[2], test_noise, test_time));
  nn::BatchOrder order(1000, 32, 12019);
  Map            training_randomness;
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_weighted_dsm_loss\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = d.pools[0].source.to_vector<int>();
  int        updates    = 0;
  for (int epoch = 1; epoch <= o.epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const auto   n = std::int64_t(rows.size());
      const Tensor x = Rows(images[0], rows), e = Gaussian8(n, 1200051 + updates), t = Levels(n, 2200021 + updates);
      training_randomness["step" + std::to_string(updates) + "_noise"] = Gaussian8(n, 1200051 + updates, true);
      training_randomness["step" + std::to_string(updates) + "_time"]  = t;
      if (!updates) {
        Save(root / "first_batch.npz", {{"images", x}, {"noise", e}, {"time", t}});
        Save(root / "first_forward.npz", model.Evaluate(x, e, t));
        Save(root / "initial_gradients.npz", model.Gradient(x, e, t));
      }
      model.Update(x, e, t);
      if (!updates) Save(root / "first_after.npz", model.Arrays());
      for (auto row : rows) batches << epoch << '\t' << updates << '\t' << row << '\t' << fit_source[row] << '\n';
      ++updates;
    }
    const auto validation = model.Evaluate(images[1], validation_noise, validation_time);
    Finite(validation);
    history << epoch << '\t' << Scalar(validation, "loss") << '\n';
    Save(root / ("epoch" + std::to_string(epoch) + "_arrays.npz"), model.Arrays());
    Save(root / ("epoch" + std::to_string(epoch) + "_validation.npz"), validation);
    const auto sample = Sample(model, 16, 12201);
    Save(root / ("epoch" + std::to_string(epoch) + "_samples.npz"), sample);
    Png(root / ("epoch" + std::to_string(epoch) + "_samples.png"), sample.at("images"), 8);
    std::cout << "epoch" << epoch << ":validationWeightedDSM=" << Scalar(validation, "loss") << std::endl;
  }
  const double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  Save(root / "training_randomness.npz", training_randomness);
  model.store.save((root / "final.npz").string());
  const auto first = Load(root / "first_batch.npz");
  Save(root / "final_gradients.npz", model.Gradient(first.at("images"), first.at("noise"), first.at("time")));
  NoiseConditionalScoreModel deployed(false);
  deployed.store.load((root / "final.npz").string(), true);
  check::Require(model.Snapshot() == deployed.Snapshot(), "all NCSN weights BITEXACT restore");
  const auto before = deployed.Snapshot();
  const auto output = deployed.Evaluate(images[2], test_noise, test_time);
  Finite(output);
  Save(root / "outputs.npz", output);
  Save(root / "model_arrays.npz", deployed.Arrays());
  const auto samples = Sample(deployed, 32, 12201), repeat = Sample(deployed, 32, 12201);
  for (const auto& [key, value] : samples)
    check::Require(Bits(value) == Bits(repeat.at(key)), "fixed seed NCSN trajectory BITEXACT repeat");
  Save(root / "samples.npz", samples);
  Png(root / "samples.png", samples.at("images"), 8);
  Png(root / "tweedie.png", samples.at("tweedie_images"), 8);
  std::vector<std::vector<Tensor>> cells(8);
  for (int b = 0; b < 8; ++b)
    for (int j = 0; j < 5; ++j) cells[b].push_back(Row(At(samples.at("trajectory"), j * 16), b));
  const Tensor trajectory_gallery = Gallery(cells);
  Save(root / "trajectory_gallery.npz", {{"images", trajectory_gallery}});
  Png(root / "trajectory.png", trajectory_gallery, 5);

  const Tensor probe_noise = Gaussian8(4, 12107);
  const Tensor cases       = Rows(images[2], {0, 1, 2, 3});
  Map          denoise;
  std::vector<std::vector<Tensor>> dcells(4);
  for (int b = 0; b < 4; ++b) dcells[b].push_back(Row(cases, b));
  const int ids[] = {0, 5, 10, 15};
  for (int level = 0; level < 4; ++level) {
    const Tensor t        = FullInt(4, ids[level]);
    const auto   eval     = deployed.Evaluate(cases, probe_noise, t);
    const auto   reverse  = deployed.Reverse(eval.at("noised"), t, nn::zeros({4, 8, 8, 1}));
    const Tensor estimate = Tweedie(eval.at("noised"), float(sigmas[ids[level]]), reverse.at("score"));
    denoise["time" + std::to_string(level)]    = t;
    denoise["noised" + std::to_string(level)]  = eval.at("noised");
    denoise["epsilon" + std::to_string(level)] = eval.at("epsilon");
    denoise["x0" + std::to_string(level)]      = estimate;
    for (int b = 0; b < 4; ++b) {
      dcells[b].push_back(Row(eval.at("noised"), b));
      dcells[b].push_back(Row(estimate, b));
    }
  }
  const Tensor denoise_gallery = Gallery(dcells);
  denoise["truth"]   = cases;
  denoise["noise"]   = probe_noise;
  denoise["gallery"] = denoise_gallery;
  Save(root / "denoising.npz", denoise);
  Png(root / "denoising.png", denoise_gallery, 9);

  const Tensor state   = output.at("noised");
  const auto   reverse = deployed.Reverse(state, test_time, test_noise);
  Finite(reverse);
  Save(root / "reverse_probe.npz", reverse);
  std::vector<std::int32_t> alt(200);
  const auto                tt = test_time.to_vector<std::int32_t>();
  for (int b = 0; b < 200; ++b) alt[b] = (tt[b] + 7) % 16;
  const Tensor alternate_time = Ints(alt);
  const auto   alternate      = deployed.Reverse(state, alternate_time, test_noise);
  const double time_effect    = MaxAbs(reverse.at("epsilon"), alternate.at("epsilon"));
  check::Require(time_effect > 1e-6, "time conditioning changes actual epsilon prediction");
  Save(root / "time_control.npz", {{"time", alternate_time}, {"epsilon", alternate.at("epsilon")}});

  // Fit-only independent Gaussian pixel prior: exact smoothed-score baseline.
  const auto          fit = images[0].to_vector<float>();
  std::vector<double> fit_mean(64), fit_variance(64), baseline(200 * 64);
  for (int j = 0; j < 64; ++j) {
    double mean = 0, var = 0;
    for (int b = 0; b < 1000; ++b) mean += double(fit[b * 64 + j]) / 1000;
    for (int b = 0; b < 1000; ++b) var += std::pow(double(fit[b * 64 + j]) - mean, 2) / 1000;
    fit_mean[j] = mean, fit_variance[j] = var;
  }
  const auto sv = state.to_vector<float>(), nv = test_noise.to_vector<float>(), rv = reverse.at("score").to_vector<float>(),
             iv = images[2].to_vector<float>();
  double zero_loss = 0, baseline_loss = 0, tweedie_mse = 0;
  for (int b = 0; b < 200; ++b) {
    const double sigma = sigmas[tt[b]];
    for (int j = 0; j < 64; ++j) {
      const int    k              = b * 64 + j;
      const double expected_score = -(double(sv[k]) - fit_mean[j]) / (fit_variance[j] + sigma * sigma);
      baseline[k]                 = expected_score;
      zero_loss += .5 * std::pow(double(nv[k]), 2) / (200 * 64);
      baseline_loss += .5 * std::pow(sigma * expected_score + nv[k], 2) / (200 * 64);
      const double estimate = double(sv[k]) + sigma * sigma * rv[k];
      tweedie_mse += std::pow(estimate - iv[k], 2) / (200 * 64);
    }
  }
  Save(root / "baseline.npz", {{"fit_mean", nn::from_vector(fit_mean, {64})},
                               {"fit_variance", nn::from_vector(fit_variance, {64})},
                               {"score", nn::from_vector(baseline, {200, 8, 8, 1})}});
  check::Require(before == deployed.Snapshot(), "all NCSN forward-score-Langevin-gradient-sampling weights immutable");
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":5504,\"parameter_tensors\":6,\"noise_levels\":16,\"sigma_max\":1,\"sigma_min\":0.05,\"langevin_inner_steps\":4,"
          "\"langevin_updates\":64,\"epochs\":"
       << o.epochs << ",\"optimizer_updates\":" << updates << ",\"test_weighted_dsm_loss\":" << Scalar(output, "loss")
       << ",\"zero_score_weighted_dsm_loss\":" << zero_loss << ",\"fit_gaussian_pixel_prior_weighted_dsm_loss\":" << baseline_loss
       << ",\"test_fixed_noise_tweedie_mse\":" << tweedie_mse << ",\"noise_conditioning_epsilon_max_effect\":" << time_effect
       << ",\"samples\":32,\"selected_epoch_fixed_final\":" << o.epochs << ",\"replayed_tf_initial_arrays\":" << (replaying ? "true" : "false")
       << ",\"training_seconds\":" << train_seconds << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Noise-conditional score matching</title><style>body{font:16px system-ui;"
          "margin:30px;max-width:1100px}img{max-width:100%;image-rendering:pixelated}</style><h1>NCSN: weighted denoising score matching "
          "and annealed Langevin sampling</h1><p><a href='https://arxiv.org/abs/1907.05600'>Noise-conditional score networks</a>, compact "
          "72->32 SiLU->32 SiLU->64 time MLP, 5504 parameters / 6 tensors. 16 geometric Gaussian sigmas 1 to .05. VE corruption "
          "x_t=x0+sigma*epsilon. Score=-epsilonPredictor/sigma; loss .5 mean||sigma*score+epsilon||^2 (weighted denoising score "
          "matching). Real continuous 8x8 MNIST 1000 fit/200 val/200 test, three epochs 96 Adam .003 steps, fixed final. Datamunge "
          "port of tflow's noise_conditional_score.</p><p>Annealed Langevin: 16 noise levels, 4 updates per level, "
          "alpha_i=2e-5*(sigma_i/.05)^2; update x&lt;-x+.5alpha_i*score+sqrt(alpha_i)*z. Start uniform[-1,1] via Gaussian CDF, "
          "independent innovations every update, INCLUDING last. Optional final Tweedie x+.05^2*score shown separately.</p>"
          "<h2>First 32 raw annealed Langevin samples</h2><img src='samples.png'><h2>Same samples after optional final Tweedie "
          "correction</h2><img src='tweedie.png'><h2>First 8 trajectories, updates 0/16/32/48/64</h2><img src='trajectory.png'>"
          "<h2>First 4 heldout noisy-input probes</h2><p>Truth, then noisy/Tweedie pairs at sigma indices 0, 5, 10, 15.</p>"
          "<img src='denoising.png'><p>Short-run noisy samples are not evidence of learned digits or equilibrium.</p>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete NCSN records");
  std::cout << "testWeightedDSM=" << Scalar(output, "loss") << ",zero=" << zero_loss << ",fitGaussianScore=" << baseline_loss
            << ",tweedieMSE=" << tweedie_mse << ";VEcorruption/16sigmas/64Langevin/noisyLast/oracle/timeControl/restore/immutable PASS"
            << " (training " << train_seconds << " s)\n";

  // ---- TF parity ----
  if (!fs::exists(tf / "model_arrays.npz")) {
    std::cout << "no TF record at " << tf.string() << "; parity checks skipped\nArtifacts: " << root.string() << "\n";
    return;
  }
  std::cout << "TF parity (" << tf.string() << "):\n";
  check::Require(bpix::Compare(tf, "gaussian_langevin_fixture.npz", fixture) < 1e-6, "Gaussian Langevin oracle fixture matches TF");
  if (replaying) {
    const double fwd = bpix::Compare(tf, "first_forward.npz", Load(root / "first_forward.npz"));
    const double grd = bpix::Compare(tf, "initial_gradients.npz", Load(root / "initial_gradients.npz"));
    const double aft = bpix::Compare(tf, "first_after.npz", Load(root / "first_after.npz"));
    check::Require(fwd < 1e-5 && grd < 1e-5 && aft < 1e-5, "replayed first forward/gradients/update match TF");
    bpix::Compare(tf, "model_arrays.npz", deployed.Arrays());
    bpix::Compare(tf, "outputs.npz", output);
    bpix::Compare(tf, "baseline.npz", Load(root / "baseline.npz"));
    history.flush();
    const auto ours = replay::History(root / "history.tsv"), theirs = replay::History(tf / "history.tsv");
    for (std::size_t i = 0; i < std::min(ours.size(), theirs.size()); ++i)
      std::printf("  epoch %zu validation weighted DSM ours %.6f TF %.6f\n", i + 1, ours[i], theirs[i]);
  }
  NoiseConditionalScoreModel tf_model(false);
  replay::LoadInto(tf_model.store, tf / "model_arrays.npz");
  const auto   tf_out   = tf_model.Evaluate(images[2], test_noise, test_time);
  const double out_diff = bpix::Compare(tf, "outputs.npz", tf_out);
  const double prb_diff = bpix::Compare(tf, "reverse_probe.npz", tf_model.Reverse(tf_out.at("noised"), test_time, test_noise));
  const double smp_diff = bpix::Compare(tf, "samples.npz", Sample(tf_model, 32, 12201));
  check::Require(out_diff < 1e-4 && prb_diff < 1e-3 && smp_diff < 1e-3, "TF's final weights reproduce TF's test outputs, probe and samples");
  std::cout << "TF final weights: test weighted DSM " << Scalar(tf_out, "loss") << " (TF record " << Scalar(Load(tf / "outputs.npz"), "loss")
            << ")\nArtifacts: " << root.string() << "\nAll noise_conditional_score checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto o = diffusion::ParseOptions(argc, argv, "noise_conditional_score", {},
                                           "epochs=N data=PATH out=PATH init=auto|tf|nn|PATH tf=DIR threads=N");
    Run(o);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
