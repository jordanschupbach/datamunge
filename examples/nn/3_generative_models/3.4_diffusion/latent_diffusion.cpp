// CMD: just run-nn latent_diffusion
// Frozen real-MNIST VAE posterior means + small continuous latent DDPM, three epochs.
//
// Port of tflow's latent_diffusion (common/{nn,ml}/latent_diffusion.h ->
// examples/nn/common/latent_diffusion.hpp; the first stage is ae::
// VariationalAutoencoderModel from common/autoencoder.hpp). The frozen
// 101652-parameter VAE and its MNIST pools (data.npz: 1000 fit / 200 validation /
// 200 test, statically binarized at 128) come from a variational_autoencoder run:
// vae=auto (default: tflow's recorded output/models/variational_autoencoder when
// present, so the run replays TF's exactly; else our port's
// output/nn/models/variational_autoencoder), vae=tf, vae=ours or vae=DIR
// (best.npz or model_arrays.npz + data.npz). tflow's own default is its
// output/models/variational_autoencoder, i.e. the TF VAE run. Encode
// deterministic posterior MEANS (zero reparameterization noise), standardize with
// fit-only mean/std, train a 1474-parameter 10 -> 32 SiLU -> 32 SiLU -> 2 epsilon
// MLP (8 sin/cos time features) for three epochs of 32-row Adam(.003) batches in
// tflow's BatchOrder(1000, 32, 17019), 32 cosine levels. Sampling: N(0, I) start,
// independent innovations, UNCLIPPED latent x0; undo the scaling and decode
// sigmoid means with the frozen decoder.
//
// Checks (as tflow): frozen first-stage census and immutability; nondegenerate
// fit scales; latent prior census; finite fields; last step ignores innovation
// (bit-exact); restore bit-exact; fixed-seed trajectory repeat; decoder outputs in
// [0, 1]; inference immutable. Added TF parity: source images/noise/time banks
// bit-identical to TF's data.npz and the encoded codes within 1e-5; init=auto
// replays TF's initial_arrays.npz and the first forward/gradients/update must
// match TF; TF's final weights (model_arrays.npz) loaded into a fresh model must
// reproduce TF's test outputs and samples. Deviations: final.npz instead of a
// SaveV2 bundle; no independent_audit.json. Options: epochs=3 vae=auto|tf|ours|DIR
// out=DIR init=auto|tf|nn|PATH tf=DIR threads=4. Output: output/nn/models/latent_diffusion.
#include <chrono>

#include "common/latent_diffusion.hpp"

namespace {

namespace nn = datamunge::nn;
namespace fs = std::filesystem;
using nn::Tensor;
using namespace diffusion;

Tensor Gaussian(std::int64_t n, std::uint32_t seed, bool wide = false) {
  const Tensor raw = nn::reshape(ae::GaussianNoise({n, 2}, 1, seed), {n, 2});
  return wide ? raw : nn::cast(raw, nn::DType::Float32);
}

void FiniteLatent(const Map& fields) {
  for (const auto& [key, x] : fields)
    if (x.dtype() == nn::DType::Float32)
      for (float v : x.contiguous().to_vector<float>()) check::Require(std::isfinite(v), "finiteLatentField:" + key);
}

/// (x - mean) / scale, or x * scale + mean (inverse), in double, stored float32.
Tensor Transform(const Tensor& x, const std::vector<double>& mean, const std::vector<double>& scale, bool inverse) {
  const auto         v = x.contiguous().to_vector<float>();
  std::vector<float> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i)
    out[i] = float(inverse ? double(v[i]) * scale[i % 2] + mean[i % 2] : (double(v[i]) - mean[i % 2]) / scale[i % 2]);
  return nn::from_vector(out, x.shape());
}

/// tflow's Png: every 784-pixel probability image as a 112x112 tile (4x nearest).
void Png(const fs::path& path, const Tensor& images, int columns) {
  const auto         n = images.size(0);
  const auto         v = images.contiguous().to_vector<float>();
  const std::int64_t H = (n + columns - 1) / columns * 112, W = columns * 112;
  std::vector<float> out(std::size_t(H * W * 3), 220.f);
  for (std::int64_t b = 0; b < n; ++b)
    for (int y = 0; y < 112; ++y)
      for (int x = 0; x < 112; ++x) {
        const double value = v[std::size_t(b * 784 + (y / 4) * 28 + x / 4)];
        check::Require(value >= 0 && value <= 1, "decoderProbabilitiesInUnitInterval");
        for (int c = 0; c < 3; ++c) out[std::size_t(((b / columns * 112 + y) * W + b % columns * 112 + x) * 3 + c)] = float(std::floor(value * 255 + .5));
      }
  bpix::PngBytes(path, out, H, W);
}

Map Sample(LatentDiffusionModel& model, std::int64_t n, std::uint32_t seed) {
  Tensor              current = Gaussian(n, seed);
  Map                 saved{{"initial", current}};
  StepStack           steps;
  std::vector<Tensor> trajectory{current}, innovations;
  for (int step = 0; step < 32; ++step) {
    const Tensor t = FullInt(n, 31 - step), z = Gaussian(n, seed + 1 + step);
    auto         out = model.Step(current, t, z);
    FiniteLatent(out);
    steps.add(out);
    innovations.push_back(Gaussian(n, seed + 1 + step, true));
    if (step == 31)
      check::Require(Bits(model.Step(current, t, Gaussian(n, 90001)).at("next")) == Bits(out.at("next")), "lastLatentStepIgnoresInnovation");
    current = out.at("next");
    trajectory.push_back(current);
  }
  for (auto& [k, v] : steps.stacked()) saved[k] = nn::reshape(v, {32, n, v.numel() / (32 * n)});
  saved["innovations"] = nn::stack(innovations, 0);
  saved["trajectory"]  = nn::stack(trajectory, 0);
  saved["latents"]     = current;
  return saved;
}

/// Exact epsilon posterior mean under a fit-only full-covariance Gaussian latent prior.
Tensor GaussianBaseline(const Tensor& state, const Tensor& time, const std::vector<double>& mean, const std::vector<double>& cov) {
  const auto&         bars = DdpmSchedule().at("alpha_bar");
  const auto          s = state.contiguous().to_vector<float>();
  const auto          t = time.to_vector<std::int32_t>();
  std::vector<double> out(s.size());
  for (std::size_t b = 0; b < t.size(); ++b) {
    const double bar = bars[std::size_t(t[b])], sq = std::sqrt(bar), no = std::sqrt(1 - bar), a = bar * cov[0] + 1 - bar,
                 d = bar * cov[3] + 1 - bar, c = bar * cov[1], det = a * d - c * c, u = s[2 * b] - sq * mean[0],
                 v = s[2 * b + 1] - sq * mean[1];
    out[2 * b]     = no * (d * u - c * v) / det;
    out[2 * b + 1] = no * (a * v - c * u) / det;
  }
  return nn::from_vector(out, state.shape());
}

std::vector<double> Doubles(const Tensor& t) { return nn::cast(t, nn::DType::Float64).contiguous().to_vector<double>(); }
double              Scalar(const Map& m, const std::string& k) { return m.at(k).item<float>(); }

void Run(const Options& o) {
  const fs::path root = o.out, tf = o.tf;
  const auto     started = std::chrono::steady_clock::now();
  // ---- frozen first stage ----
  const fs::path tf_vae = replay::TfModels("variational_autoencoder"), our_vae = paths::OutputRoot() / "models" / "variational_autoencoder";
  const std::string choice = o.extra.at("vae");
  fs::path          source = choice == "tf" ? tf_vae : choice == "ours" ? our_vae : choice == "auto" ? (fs::exists(tf_vae / "data.npz") ? tf_vae : our_vae) : fs::path(choice);
  ae::VariationalAutoencoderModel vae(false);
  const auto                      vae_file = ae::LoadFrozen(vae.parameters, source, "frozen first-stage VAE");
  std::error_code                 ec;
  const bool                      tf_source = fs::equivalent(source, tf_vae, ec);
  std::cout << "frozen first stage: " << vae_file.string() << (tf_source ? " (TF's recorded VAE)" : "") << "\n";
  const auto frozen = ae::Snapshot(vae.parameters);
  check::Require(vae.parameters.count() == 101652, "frozen101652ParameterFirstStage");
  Save(root / "first_stage_arrays.npz", replay::Arrays(vae.parameters));
  const auto original = Load(source / "data.npz");
  for (const std::string name : {"data_manifest.json", "metrics.json"})
    if (fs::exists(source / name)) fs::copy_file(source / name, root / ("source_" + name), fs::copy_options::overwrite_existing);
  Map                 data;
  std::vector<Tensor> codes, images;
  for (const std::string split : {"fit", "validation", "test"}) {
    const Tensor x   = original.at(split + "_images");
    const auto   out = vae.Evaluate(x, nn::zeros({x.size(0), 2}));
    codes.push_back(out[0]);
    images.push_back(x);
    data[split + "_images"]          = x;
    data[split + "_original_images"] = original.at(split + "_original_images");
    data[split + "_source_ids"]      = original.at(split + "_source_ids");
    data[split + "_mu"]              = out[0];
    data[split + "_log_variance"]    = out[1];
    data[split + "_reconstruction"]  = out[3];
  }
  data["test_labels"] = original.at("test_labels");
  std::vector<double> mean(2), scale(2);
  const auto          fit_codes = codes[0].to_vector<float>();
  for (int j = 0; j < 2; ++j) {
    double m = 0, v = 0;
    for (int b = 0; b < 1000; ++b) m += double(fit_codes[2 * b + j]) / 1000;
    for (int b = 0; b < 1000; ++b) v += std::pow(double(fit_codes[2 * b + j]) - m, 2) / 1000;
    mean[j] = m, scale[j] = std::sqrt(v);
    check::Require(scale[j] > 1e-6, "fitOnlyNondegenerateLatentScales");
  }
  data["fit_code_mean"]  = nn::from_vector(mean, {2});
  data["fit_code_scale"] = nn::from_vector(scale, {2});
  std::vector<Tensor> standardized;
  int                 split = 0;
  for (const std::string name : {"fit", "validation", "test"}) {
    standardized.push_back(Transform(codes[split++], mean, scale, false));
    data[name + "_latents"] = standardized.back();
  }
  std::vector<double> zmean(2, 0.), cov(4, 0.);
  const auto          fs0 = standardized[0].to_vector<float>();
  for (int j = 0; j < 2; ++j)
    for (int b = 0; b < 1000; ++b) zmean[j] += double(fs0[b * 2 + j]) / 1000;
  for (int j = 0; j < 2; ++j)
    for (int k = 0; k < 2; ++k)
      for (int b = 0; b < 1000; ++b) cov[j * 2 + k] += (double(fs0[b * 2 + j]) - zmean[j]) * (double(fs0[b * 2 + k]) - zmean[k]) / 1000;
  data["fit_standardized_mean"]       = nn::from_vector(zmean, {2});
  data["fit_standardized_covariance"] = nn::from_vector(cov, {2, 2});
  const Tensor vn = Gaussian(200, 17101), tn = Gaussian(200, 17103), vt = Times(200, 0, true), tt = Times(200, 0, true);
  data["validation_noise"] = vn;
  data["test_noise"]       = tn;
  data["validation_time"]  = vt;
  data["test_time"]        = tt;
  Save(root / "data.npz", data);
  if (fs::exists(tf / "data.npz")) {
    const auto theirs = Load(tf / "data.npz");
    for (const std::string k : {"fit_images", "validation_images", "test_images", "test_labels", "fit_source_ids", "validation_noise",
                                "test_noise", "validation_time", "test_time"})
      check::Require(Bits(theirs.at(k)) == Bits(data.at(k)), "images/noise/time bit-identical to TF's data.npz: " + k);
    double code_diff = 0;
    for (const std::string k : {"fit_mu", "test_mu", "fit_latents", "validation_latents", "test_latents", "test_reconstruction"})
      code_diff = std::max(code_diff, MaxAbs(theirs.at(k), data.at(k)));
    std::cout << "data: images, noise and time banks bit-identical to TF's data.npz; encoded codes/latents max |ours - TF| = "
              << code_diff << "\n";
    if (tf_source) check::Require(code_diff < 1e-5, "TF's frozen VAE encodes TF's codes (< 1e-5)");
  }
  const auto schedule = ScheduleArrays();
  Save(root / "schedule.npz", schedule);
  if (fs::exists(tf / "schedule.npz")) bpix::Compare(tf, "schedule.npz", schedule);

  // ---- latent prior training ----
  LatentDiffusionModel model;
  check::Require(model.store.count() == 1474 && model.store.size() == 6, "1474LatentPriorParameters6tensors");
  const fs::path init      = bpix::InitFile(o.init, tf);
  const bool     replaying = !init.empty();
  if (replaying) {
    replay::LoadInto(model.store, init);
    std::cout << "replaying TF's initial arrays " << init.string() << "\n";
  } else {
    std::cout << "weights from nn's generator (statistical parity with TF only)\n";
  }
  Save(root / "initial_arrays.npz", model.Arrays());
  Save(root / "initial_outputs.npz", model.Evaluate(standardized[2], tn, tt));
  nn::BatchOrder order(1000, 32, 17019);
  int            updates = 0;
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_epsilon_mse\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  Map        randomness;
  const auto fit_source = data.at("fit_source_ids").to_vector<std::int32_t>();
  for (int epoch = 1; epoch <= o.epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const auto   n = std::int64_t(rows.size());
      const Tensor x = Rows(standardized[0], rows), e = Gaussian(n, 1700051 + updates), t = Times(n, 2700021 + updates);
      randomness["step" + std::to_string(updates) + "_noise"] = Gaussian(n, 1700051 + updates, true);
      randomness["step" + std::to_string(updates) + "_time"]  = t;
      if (!updates) {
        Save(root / "first_batch.npz", {{"latents", x}, {"noise", e}, {"time", t}});
        Save(root / "first_forward.npz", model.Evaluate(x, e, t));
        Save(root / "initial_gradients.npz", model.Gradient(x, e, t));
      }
      model.Update(x, e, t);
      if (!updates) Save(root / "first_after.npz", model.Arrays());
      for (auto row : rows) batches << epoch << '\t' << updates << '\t' << row << '\t' << fit_source[row] << '\n';
      ++updates;
    }
    const auto valid = model.Evaluate(standardized[1], vn, vt);
    FiniteLatent(valid);
    history << epoch << '\t' << Scalar(valid, "loss") << '\n';
    Save(root / ("epoch" + std::to_string(epoch) + "_arrays.npz"), model.Arrays());
    Save(root / ("epoch" + std::to_string(epoch) + "_validation.npz"), valid);
    auto sample = Sample(model, 16, 17201);
    sample["codes"]  = Transform(sample.at("latents"), mean, scale, true);
    sample["images"] = vae.Decode(sample.at("codes"));
    Save(root / ("epoch" + std::to_string(epoch) + "_samples.npz"), sample);
    Png(root / ("epoch" + std::to_string(epoch) + "_samples.png"), sample.at("images"), 8);
    std::cout << "epoch" << epoch << " latent validation epsilon MSE=" << Scalar(valid, "loss") << std::endl;
  }
  const double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  Save(root / "training_randomness.npz", randomness);
  model.store.save((root / "final.npz").string());
  const auto first = Load(root / "first_batch.npz");
  Save(root / "final_gradients.npz", model.Gradient(first.at("latents"), first.at("noise"), first.at("time")));
  LatentDiffusionModel deployed(false);
  deployed.store.load((root / "final.npz").string(), true);
  check::Require(model.Snapshot() == deployed.Snapshot(), "all1474LatentWeightsBitexactRestore");
  const auto snapshot = deployed.Snapshot();
  Save(root / "model_arrays.npz", deployed.Arrays());
  const auto output = deployed.Evaluate(standardized[2], tn, tt);
  FiniteLatent(output);
  Save(root / "outputs.npz", output);
  const auto probe = deployed.Step(output.at("noised"), tt, tn);
  FiniteLatent(probe);
  Save(root / "reverse_probe.npz", probe);
  auto sample = Sample(deployed, 32, 17201);
  const auto repeat = Sample(deployed, 32, 17201);
  for (const auto& [key, value] : sample) check::Require(Bits(value) == Bits(repeat.at(key)), "fixedSeedLatentTrajectoryBitexact");
  sample["codes"]  = Transform(sample.at("latents"), mean, scale, true);
  sample["images"] = vae.Decode(sample.at("codes"));
  Save(root / "samples.npz", sample);
  Png(root / "samples.png", sample.at("images"), 8);

  // ---- prior baselines ----
  const Tensor standard = Gaussian(32, 17201);
  const auto   sv       = standard.to_vector<float>();
  const double l00 = std::sqrt(cov[0]), l10 = cov[2] / l00, l11 = std::sqrt(cov[3] - l10 * l10);
  std::vector<float> agg(64);
  for (int b = 0; b < 32; ++b) {
    agg[2 * b]     = float(zmean[0] + l00 * sv[2 * b]);
    agg[2 * b + 1] = float(zmean[1] + l10 * sv[2 * b] + l11 * sv[2 * b + 1]);
  }
  const Tensor aggregate = nn::from_vector(agg, {32, 2}), aggregate_codes = Transform(aggregate, mean, scale, true);
  const Tensor prior_images = vae.Decode(standard), aggregate_images = vae.Decode(aggregate_codes);
  Save(root / "prior_baselines.npz", {{"standard_gaussian", standard},
                                      {"aggregate_standardized", aggregate},
                                      {"aggregate_codes", aggregate_codes},
                                      {"vae_prior_images", prior_images},
                                      {"aggregate_images", aggregate_images}});
  Png(root / "vae_prior.png", prior_images, 8);
  Png(root / "aggregate_prior.png", aggregate_images, 8);
  const Tensor baseline = GaussianBaseline(output.at("noised"), tt, zmean, cov);
  Save(root / "baseline.npz", {{"epsilon", baseline}});

  // ---- heldout known-noise denoising at level 16 ----
  std::vector<std::int64_t> selected;
  const auto                labels = data.at("test_labels").to_vector<std::int32_t>();
  for (int label = 0; label < 10; ++label)
    for (int b = 0; b < 200; ++b)
      if (labels[b] == label) {
        selected.push_back(b);
        break;
      }
  check::Require(selected.size() == 10, "heldoutOneRealSourcePerDigit");
  const Tensor clean = Rows(standardized[2], selected), noise = Gaussian(10, 17107), t15 = FullInt(10, 15);
  const auto   eval = deployed.Evaluate(clean, noise, t15);
  const auto   den  = deployed.Step(eval.at("noised"), t15, nn::zeros({10, 2}));
  const Tensor bg   = GaussianBaseline(eval.at("noised"), t15, zmean, cov);
  const auto&  sch  = DdpmSchedule();
  const auto   nv = eval.at("noised").to_vector<float>();
  const auto   bv = bg.to_vector<double>();
  std::vector<float> bgc(20);
  for (int i = 0; i < 20; ++i) bgc[i] = float((nv[i] - sch.at("sqrt_one_minus_alpha_bar")[15] * bv[i]) / sch.at("sqrt_alpha_bar")[15]);
  const Tensor bgcode         = nn::from_vector(bgc, {10, 2});
  const Tensor learned_images = vae.Decode(Transform(den.at("x0_raw"), mean, scale, true));
  const Tensor base_images    = vae.Decode(Transform(bgcode, mean, scale, true));
  const Tensor truth = nn::reshape(Rows(images[2], selected), {10, 784}), reconstruction = Rows(data.at("test_reconstruction"), selected);
  std::vector<Tensor> cells;
  for (std::int64_t b = 0; b < 10; ++b)
    for (const Tensor* src : {&truth, &reconstruction, &learned_images, &base_images}) cells.push_back(Row(*src, b));
  const Tensor              gallery = nn::concat(cells, 0);
  std::vector<std::int32_t> idx(selected.begin(), selected.end());
  Save(root / "denoising.npz", {{"indices", Ints(idx)},
                                {"clean", clean},
                                {"noise", noise},
                                {"time", t15},
                                {"noised", eval.at("noised")},
                                {"epsilon", eval.at("epsilon")},
                                {"x0", den.at("x0_raw")},
                                {"gaussian_epsilon", bg},
                                {"gaussian_x0", bgcode},
                                {"gallery", gallery}});
  Png(root / "denoising.png", gallery, 4);
  double     zero_mse = 0, base_mse = 0, x0_mse = 0, reconstruction_mse = 0;
  const auto tv = tn.to_vector<float>(), xr = probe.at("x0_raw").to_vector<float>(), st = standardized[2].to_vector<float>();
  const auto bl = baseline.to_vector<double>();
  for (int i = 0; i < 400; ++i) {
    zero_mse += std::pow(double(tv[i]), 2) / 400;
    base_mse += std::pow(bl[i] - tv[i], 2) / 400;
    x0_mse += std::pow(double(xr[i]) - st[i], 2) / 400;
  }
  const auto rv = data.at("test_reconstruction").contiguous().to_vector<float>(), iv = images[2].contiguous().to_vector<float>();
  for (int i = 0; i < 200 * 784; ++i) reconstruction_mse += std::pow(double(rv[i]) - iv[i], 2) / (200 * 784);
  check::Require(frozen == ae::Snapshot(vae.parameters), "allFrozenFirstStageWeightsUnchangedAcrossEncodingTrainingDecoding");
  check::Require(snapshot == deployed.Snapshot(), "latentPriorInferenceWeightsImmutable");
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":1474,\"parameter_tensors\":6,\"frozen_first_stage_parameters\":101652,\"epochs\":" << o.epochs
       << ",\"optimizer_updates\":" << updates << ",\"test_epsilon_mse\":" << Scalar(output, "loss") << ",\"zero_epsilon_mse\":" << zero_mse
       << ",\"fit_full_covariance_gaussian_epsilon_mse\":" << base_mse << ",\"test_raw_x0_latent_mse\":" << x0_mse
       << ",\"frozen_mean_reconstruction_pixel_mse\":" << reconstruction_mse << ",\"samples\":32,\"first_stage\":\""
       << vae_file.string() << "\",\"replayed_tf_initial_arrays\":" << (replaying ? "true" : "false")
       << ",\"training_seconds\":" << train_seconds << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Latent diffusion</title><style>body{font:16px system-ui;margin:30px;max-width:1100px}"
          "img{max-width:100%;image-rendering:pixelated}</style><h1>Latent diffusion in a frozen two-coordinate VAE</h1><p>Real MNIST "
          "1000 fit/200 validation/200 test, statically binarized at 128. Frozen 101652-parameter VAE first stage; deterministic posterior "
          "MEANS standardized with fit-only mean/std. Separate 1474-parameter 10->32 SiLU->32 SiLU->2 epsilon MLP, 8 sin/cos time "
          "features, 3 epochs of 96 Adam .003 updates, 32 cosine noise levels. Sampling starts from N(0,I) with independent innovations; "
          "UNCLIPPED continuous latent x0; undo the scaling and decode sigmoid means with the frozen decoder. Datamunge port of tflow's "
          "latent_diffusion (<a href='https://arxiv.org/abs/2112.10752'>Latent Diffusion Models</a>, tiny 2-D teaching adaptation).</p>"
          "<h2>Latent DDPM decoded means</h2><img src='samples.png'><h2>Same initial Gaussian: original VAE prior</h2>"
          "<img src='vae_prior.png'><h2>Fit-only full-covariance Gaussian mean-code prior</h2><img src='aggregate_prior.png'>"
          "<h2>Heldout known-noise latent denoising, level 16</h2><p>Rows digits 0..9; columns truth, frozen mean reconstruction, "
          "learned latent x0 decode, fit Gaussian latent x0 decode.</p><img src='denoising.png'><p>See metrics.json. No likelihood "
          "or image quality benchmark claim.</p>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "completeLatentDiffusionArtifacts");
  std::cout << "latentTestEpsilonMSE=" << Scalar(output, "loss") << " fitGaussian=" << base_mse << " frozenReconstructionMSE="
            << reconstruction_mse << ";frozenFirstStage/restore/lastNoise/repeat/immutable PASS (training " << train_seconds << " s)\n";

  // ---- TF parity ----
  if (!fs::exists(tf / "model_arrays.npz")) {
    std::cout << "no TF record at " << tf.string() << "; parity checks skipped\nArtifacts: " << root.string() << "\n";
    return;
  }
  std::cout << "TF parity (" << tf.string() << "):\n";
  if (replaying && tf_source) {
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
  // TF's final latent prior + TF's own standardized codes (and TF's first stage when it is the source).
  const auto           theirs = Load(tf / "data.npz");
  LatentDiffusionModel tf_model(false);
  replay::LoadInto(tf_model.store, tf / "model_arrays.npz");
  const auto   tf_out  = tf_model.Evaluate(theirs.at("test_latents"), tn, tt);
  const double out_diff = bpix::Compare(tf, "outputs.npz", tf_out);
  auto         tf_sample = Sample(tf_model, 32, 17201);
  if (tf_source) {
    tf_sample["codes"]  = Transform(tf_sample.at("latents"), Doubles(theirs.at("fit_code_mean")), Doubles(theirs.at("fit_code_scale")), true);
    tf_sample["images"] = vae.Decode(tf_sample.at("codes"));
  }
  const double smp_diff = bpix::Compare(tf, "samples.npz", tf_sample, false, true);
  check::Require(out_diff < 1e-4 && smp_diff < 1e-3, "TF's final weights reproduce TF's test outputs and samples");
  std::cout << "TF final weights: test epsilon MSE " << Scalar(tf_out, "loss") << " (TF record " << Scalar(Load(tf / "outputs.npz"), "loss")
            << ")\nArtifacts: " << root.string() << "\nAll latent_diffusion checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto o = diffusion::ParseOptions(argc, argv, "latent_diffusion", {"vae"},
                                           "epochs=N vae=auto|tf|ours|DIR out=PATH init=auto|tf|nn|PATH tf=DIR threads=N");
    Run(o);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
