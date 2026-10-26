// CMD: just run-nn ddpm
// DDPM on real continuous grayscale 8x8 MNIST, three cheap epochs and replayable
// 32-step ancestral generation.
//
// Port of tflow's ddpm (common/{nn,ml}/ddpm.h -> examples/nn/common/ddpm.hpp).
// Data: tflow's saliency MNIST pools (paths::Dataset("saliency"), `just
// autoencoder-assets` in tflow_examples): 1000 fit / 200 validation / 200 test,
// half-pixel bilinear 28 -> 8, scaled to [-1, 1], NO threshold. Epsilon MLP: flat
// 64 pixels + 8 sin/cos time features, 72 -> 32 SiLU -> 32 SiLU -> 64, 5504
// parameters / 6 tensors. 32 cosine timesteps (s = .008, beta cap .999). Train on
// uniform random timesteps and independent Gaussian epsilon with mean epsilon
// MSE, three epochs of 32-image Adam(.003) batches in tflow's BatchOrder(1000,
// 32, 11019); fixed final. Sampling feeds only state, time and innovation:
// predict epsilon, derive and clip x0, posterior mean c1 x0 + c2 x_t, fixed
// posterior variance (zero at the last step).
//
// Checks (as tflow): parameter census; finite fields; last transition adds no
// noise (bit-exact); fresh restore bit-exact; fixed-seed trajectories repeat
// bit-exactly; time conditioning changes epsilon; inference immutable. Noise
// banks, timesteps and batch order are tflow's exactly. TF parity: init=auto
// (default) replays TF's initial_arrays.npz from tflow's output/models/ddpm so the
// run follows TF's, and every record is compared with TF's; independently, TF's
// final weights (model_arrays.npz) are loaded into a fresh model which must
// reproduce TF's test outputs and samples. Deviations: final.npz instead of a
// SaveV2 bundle; no independent_audit.json. Options: epochs=3 data=DIR out=DIR
// init=auto|tf|nn|PATH tf=DIR threads=4. Output: output/nn/models/ddpm.
#include <chrono>

#include "common/ddpm.hpp"

namespace {

namespace nn = datamunge::nn;
namespace fs = std::filesystem;
using nn::Tensor;
using namespace diffusion;

Map Sample(DdpmModel& model, std::int64_t n, std::uint32_t seed) {
  const Tensor initial = Gaussian8(n, seed);
  Tensor       current = initial;
  StepStack    steps;
  std::vector<Tensor> trajectory{initial}, innovations;
  for (int step = 0; step < 32; ++step) {
    const Tensor t = FullInt(n, 31 - step), innovation = Gaussian8(n, seed + 1 + step);
    innovations.push_back(Gaussian8(n, seed + 1 + step, true));
    auto out = model.Reverse(current, t, innovation);
    Finite(out);
    steps.add({{"epsilon", out.at("epsilon")}, {"mean", out.at("mean")}, {"x0_raw", out.at("x0_raw")},
               {"x0_clipped", out.at("x0_clipped")}});
    if (step == 31) {
      const auto other = model.Reverse(current, t, nn::full(innovation.shape(), 123.0));
      check::Require(Bits(other.at("next")) == Bits(out.at("next")), "last DDPM transition adds NO noise");
    }
    current = out.at("next");
    trajectory.push_back(current);
  }
  Map result = steps.stacked();
  result["initial"]     = initial;
  result["images"]      = current;
  result["trajectory"]  = nn::stack(trajectory, 0);
  result["innovations"] = nn::stack(innovations, 0);
  return result;
}

double Scalar(const Map& m, const std::string& k) { return m.at(k).item<float>(); }

void Run(const Options& o) {
  const fs::path root = o.out, tf = o.tf;
  const auto     started = std::chrono::steady_clock::now();
  auto           d       = LoadDigits(o.data);
  const auto&    images  = d.images;
  Map            data    = d.record;
  const Tensor validation_noise = Gaussian8(200, 11101), test_noise = Gaussian8(200, 11103);
  const Tensor validation_time = Times(200, 0, true), test_time = Times(200, 0, true);
  data["validation_noise"] = validation_noise;
  data["test_noise"]       = test_noise;
  data["validation_time"]  = validation_time;
  data["test_time"]        = test_time;
  Save(root / "data.npz", data);
  if (fs::exists(tf / "data.npz")) {
    const auto theirs = Load(tf / "data.npz");
    for (const std::string k : {"fit_images", "validation_images", "test_images", "test_noise", "test_time"})
      check::Require(Bits(theirs.at(k)) == Bits(data.at(k)), "data/noise/time bit-identical to TF's data.npz: " + k);
    std::cout << "data: 8x8 pools, noise and time banks bit-identical to TF's data.npz\n";
  }
  ae::CopyManifests(o.data, root);
  const auto schedule = ScheduleArrays();
  Save(root / "schedule.npz", schedule);
  if (fs::exists(tf / "schedule.npz")) bpix::Compare(tf, "schedule.npz", schedule);

  DdpmModel model;
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
  nn::BatchOrder order(1000, 32, 11019);
  Map            training_randomness;
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_epsilon_mse\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = d.pools[0].source.to_vector<int>();
  int        updates    = 0;
  for (int epoch = 1; epoch <= o.epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const auto   n = std::int64_t(rows.size());
      const Tensor x = Rows(images[0], rows), e = Gaussian8(n, 1100051 + updates), t = Times(n, 2100021 + updates);
      training_randomness["step" + std::to_string(updates) + "_noise"] = Gaussian8(n, 1100051 + updates, true);
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
    const auto sample = Sample(model, 16, 11201);
    Save(root / ("epoch" + std::to_string(epoch) + "_samples.npz"), sample);
    Png(root / ("epoch" + std::to_string(epoch) + "_samples.png"), sample.at("images"), 8);
    std::cout << "epoch" << epoch << ":validationepsilonMSE=" << Scalar(validation, "loss") << std::endl;
  }
  const double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  Save(root / "training_randomness.npz", training_randomness);
  model.store.save((root / "final.npz").string());
  const auto first = Load(root / "first_batch.npz");
  Save(root / "final_gradients.npz", model.Gradient(first.at("images"), first.at("noise"), first.at("time")));
  DdpmModel deployed(false);
  deployed.store.load((root / "final.npz").string(), true);
  check::Require(model.Snapshot() == deployed.Snapshot(), "all DDPM weights BITEXACT restore");
  const auto before = deployed.Snapshot();
  const auto output = deployed.Evaluate(images[2], test_noise, test_time);
  Finite(output);
  Save(root / "outputs.npz", output);
  Save(root / "model_arrays.npz", deployed.Arrays());
  const auto samples = Sample(deployed, 32, 11201), repeat = Sample(deployed, 32, 11201);
  for (const auto& [key, value] : samples)
    check::Require(Bits(value) == Bits(repeat.at(key)), "fixed seed DDPM trajectory BITEXACT repeat");
  Save(root / "samples.npz", samples);
  Png(root / "samples.png", samples.at("images"), 8);
  std::vector<std::vector<Tensor>> cells(8);
  for (int b = 0; b < 8; ++b)
    for (int j = 0; j < 5; ++j) cells[b].push_back(Row(At(samples.at("trajectory"), j * 8), b));
  const Tensor trajectory_gallery = Gallery(cells);
  Save(root / "trajectory_gallery.npz", {{"images", trajectory_gallery}});
  Png(root / "trajectory.png", trajectory_gallery, 5);

  const Tensor probe_noise = Gaussian8(4, 11107);
  const Tensor cases       = Rows(images[2], {0, 1, 2, 3});
  Map          denoise;
  std::vector<std::vector<Tensor>> dcells(4);
  for (int b = 0; b < 4; ++b) dcells[b].push_back(Row(cases, b));
  const int ids[] = {1, 7, 15, 31};
  for (int level = 0; level < 4; ++level) {
    const Tensor t       = FullInt(4, ids[level]);
    const auto   eval    = deployed.Evaluate(cases, probe_noise, t);
    const auto   reverse = deployed.Reverse(eval.at("noised"), t, nn::zeros({4, 8, 8, 1}));
    denoise["time" + std::to_string(level)]    = t;
    denoise["noised" + std::to_string(level)]  = eval.at("noised");
    denoise["epsilon" + std::to_string(level)] = eval.at("epsilon");
    denoise["x0" + std::to_string(level)]      = reverse.at("x0_clipped");
    for (int b = 0; b < 4; ++b) {
      dcells[b].push_back(Row(eval.at("noised"), b));
      dcells[b].push_back(Row(reverse.at("x0_clipped"), b));
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
  for (int b = 0; b < 200; ++b) alt[b] = (tt[b] + 13) % 32;
  const Tensor alternate_time = Ints(alt);
  const auto   alternate      = deployed.Reverse(state, alternate_time, test_noise);
  const double time_effect    = MaxAbs(reverse.at("epsilon"), alternate.at("epsilon"));
  check::Require(time_effect > 1e-6, "time conditioning changes actual epsilon prediction");
  Save(root / "time_control.npz", {{"time", alternate_time}, {"epsilon", alternate.at("epsilon")}});

  // Fit-only independent Gaussian pixel prior: analytic epsilon baseline.
  const auto          fit = images[0].to_vector<float>();
  std::vector<double> fit_mean(64), fit_variance(64), baseline(200 * 64);
  for (int j = 0; j < 64; ++j) {
    double mean = 0, var = 0;
    for (int b = 0; b < 1000; ++b) mean += double(fit[b * 64 + j]) / 1000;
    for (int b = 0; b < 1000; ++b) var += std::pow(double(fit[b * 64 + j]) - mean, 2) / 1000;
    fit_mean[j] = mean, fit_variance[j] = var;
  }
  const auto&  bars = DdpmSchedule().at("alpha_bar");
  const auto   sv = state.to_vector<float>(), nv = test_noise.to_vector<float>(), xv = reverse.at("x0_clipped").to_vector<float>(),
             iv = images[2].to_vector<float>();
  double zero_mse = 0, baseline_mse = 0, x0_mse = 0;
  for (int b = 0; b < 200; ++b) {
    const double bar = bars[tt[b]];
    for (int j = 0; j < 64; ++j) {
      const int    k        = b * 64 + j;
      const double expected = std::sqrt(1 - bar) / (bar * fit_variance[j] + 1 - bar) * (double(sv[k]) - std::sqrt(bar) * fit_mean[j]);
      baseline[k]           = expected;
      zero_mse += std::pow(double(nv[k]), 2) / (200 * 64);
      baseline_mse += std::pow(expected - nv[k], 2) / (200 * 64);
      x0_mse += std::pow(double(xv[k]) - iv[k], 2) / (200 * 64);
    }
  }
  Save(root / "baseline.npz", {{"fit_mean", nn::from_vector(fit_mean, {64})},
                               {"fit_variance", nn::from_vector(fit_variance, {64})},
                               {"epsilon", nn::from_vector(baseline, {200, 8, 8, 1})}});
  check::Require(before == deployed.Snapshot(), "all DDPM forward-reverse-gradient-sampling weights immutable");
  const double terminal_bar = bars[31];
  double       terminal_kl  = 0;
  for (float x : fit) terminal_kl += .5 * (terminal_bar * std::pow(double(x), 2) - terminal_bar - std::log1p(-terminal_bar)) / 1000;
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":5504,\"parameter_tensors\":6,\"timesteps\":32,\"epochs\":" << o.epochs
       << ",\"optimizer_updates\":" << updates << ",\"test_epsilon_mse\":" << Scalar(output, "loss")
       << ",\"zero_epsilon_mse\":" << zero_mse << ",\"fit_gaussian_pixel_prior_epsilon_mse\":" << baseline_mse
       << ",\"test_fixed_noise_x0_clipped_mse\":" << x0_mse << ",\"time_conditioning_max_effect\":" << time_effect
       << ",\"terminal_alpha_bar\":" << terminal_bar << ",\"fit_average_conditional_terminal_prior_kl_per_image\":" << terminal_kl
       << ",\"samples\":32,\"selected_epoch_fixed_final\":" << o.epochs << ",\"replayed_tf_initial_arrays\":" << (replaying ? "true" : "false")
       << ",\"training_seconds\":" << train_seconds << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>DDPM</title><style>body{font:16px system-ui;margin:30px;max-width:1100px}"
          "img{max-width:100%;image-rendering:pixelated}</style><h1>DDPM: forward noise and ancestral reverse diffusion</h1>"
          "<p><a href='https://arxiv.org/abs/2006.11239'>DDPM</a>, compact 8x8 grayscale MNIST epsilon predictor: flat 64 pixels + 8 "
          "sin/cos time features, 72->32 SiLU->32 SiLU->64, 5504 parameters / 6 tensors. 32 cosine timesteps (s=.008, beta cap .999). "
          "Three epochs, 96 Adam .003 steps, fixed final. Datamunge port of tflow's ddpm.</p><h2>First 32 fixed-seed ancestral samples</h2>"
          "<img src='samples.png'><h2>First 8 reverse trajectories</h2><p>Columns physical levels 32, 24, 16, 8, 0.</p>"
          "<img src='trajectory.png'><h2>First 4 heldout denoising cases</h2><p>Columns truth, then noisy/predicted x0 pairs at "
          "physical levels 2, 8, 16, 32.</p><img src='denoising.png'><p>Short training may produce noise or poor digits. See "
          "metrics.json. No quality/likelihood benchmark claim.</p>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete DDPM records");
  std::cout << "testepsilonMSE=" << Scalar(output, "loss") << ",zero=" << zero_mse << ",fitGaussianPrior=" << baseline_mse
            << ",terminalbar=" << terminal_bar << ";forward/schedule/32reverse/zeroLastNoise/timeControl/restore/immutable PASS"
            << " (training " << train_seconds << " s)\n";

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
    bpix::Compare(tf, "model_arrays.npz", deployed.Arrays());
    bpix::Compare(tf, "outputs.npz", output);
    history.flush();
    const auto ours = replay::History(root / "history.tsv"), theirs = replay::History(tf / "history.tsv");
    for (std::size_t i = 0; i < std::min(ours.size(), theirs.size()); ++i)
      std::printf("  epoch %zu validation epsilon MSE ours %.6f TF %.6f\n", i + 1, ours[i], theirs[i]);
  }
  DdpmModel tf_model(false);
  replay::LoadInto(tf_model.store, tf / "model_arrays.npz");
  const auto tf_out     = tf_model.Evaluate(images[2], test_noise, test_time);
  const double out_diff = bpix::Compare(tf, "outputs.npz", tf_out);
  const double smp_diff = bpix::Compare(tf, "samples.npz", Sample(tf_model, 32, 11201));
  check::Require(out_diff < 1e-4 && smp_diff < 1e-3, "TF's final weights reproduce TF's test outputs and samples");
  std::cout << "TF final weights: test epsilon MSE " << Scalar(tf_out, "loss") << " (TF record "
            << Scalar(Load(tf / "outputs.npz"), "loss") << ")\nArtifacts: " << root.string() << "\nAll ddpm checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto o = diffusion::ParseOptions(argc, argv, "ddpm", {}, "epochs=N data=PATH out=PATH init=auto|tf|nn|PATH tf=DIR threads=N");
    Run(o);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
