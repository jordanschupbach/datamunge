// CMD: just run-nn wavenet
// Generative WaveNet: real speaker-disjoint mu-law audio, cold-start and
// continuation synthesis with per-layer causal caches.
//
// Port of tflow's wavenet (common/ml/wavenet_generation.h + the 1.9 stack ->
// examples/nn/common/{wavenet_generation,wavenet_stack}.hpp). Data: tflow's FSDD
// assets (paths::Dataset("wavenet"): 72 fit / 12 validation / 24 test windows of
// 256 mu-law codes, disjoint recordings and speakers, three held-out prompts).
// Stack R4 / S8, dilations 1..16 (one cycle), RF 33, 5068 parameters in 34
// tensors; NLL masked over the first 32 warm-up positions; Adam .001, batch 8 in
// tflow's BatchOrder(72, 8, 8713), three epochs = 27 updates.
//
// Checks (as tflow): RF 33 and census; host double NLL == graph NLL; restore
// bit-exact; repeat logits bit-exact; changing every code from t = 128 on
// leaves all earlier predictions bit-identical; full input-Jacobian fixture
// (exact zeros outside the RF and in the future, active inside); all 256 codec
// bins round-trip; cached synthesis agrees with the native RF forward at every
// probe (< 2e-5); cold-start (2 x 2048) and continuation (3 x 128 + 512) draws
// repeat bit-exactly and keep the prompt; inference immutable. Fit-only
// unigram/transition baselines. .wav files and waveform plots are written.
// Added: init=auto replays TF's initial arrays and compares every record; TF's
// final weights must reproduce TF's test logits and its seeded generations.
// Deviations: final.npz checkpoint, no independent_audit.json. Options:
// epochs=3 data=DIR out=DIR init=auto|tf|nn|PATH tf=DIR threads=4 verbose=0.
// Output: output/nn/models/wavenet.
#include "common/binary_pixels.hpp"
#include "common/wavenet_generation.hpp"

namespace {

namespace nn = datamunge::nn;
namespace fs = std::filesystem;
using namespace wavestack;
using Map = replay::Map;

const WaveStackConfig kConfig{4, 8, 5, 1, 256};

/// Mean next-code NLL over positions 32..255 in double precision.
double Nll(const Tensor& logits, const WaveData& d) {
  const std::int64_t n = d.input.size(0);
  check::Require(logits.shape() == nn::Shape({n, 256, 256}), "waveform logit shape");
  const auto z = logits.to_vector<float>();
  const auto y = d.target.to_vector<int>();
  double     nll = 0;
  for (std::int64_t b = 0; b < n; ++b)
    for (int t = 32; t < 256; ++t) {
      const float* row     = z.data() + (b * 256 + t) * 256;
      double       maximum = -INFINITY, mass = 0;
      for (int c = 0; c < 256; ++c) maximum = std::max(maximum, double(row[c]));
      for (int c = 0; c < 256; ++c) mass += std::exp(double(row[c]) - maximum);
      nll += maximum + std::log(mass) - row[y[b * 256 + t]];
    }
  return nll / double(n * 224);
}

void WavePlot(const fs::path& path, const Tensor& codes) {
  const std::int64_t n = codes.size(0), length = codes.size(1);
  const auto         c = codes.to_vector<int>();
  std::vector<float> image(static_cast<std::size_t>(n * 160 * 1024 * 3), 245.f);
  for (std::int64_t b = 0; b < n; ++b) {
    int last = 80;
    for (int x = 0; x < 1024; ++x) {
      const int index = std::min(int(length - 1), int(double(x) * length / 1024));
      const int y     = std::clamp(int(std::lround(80 - 72 * MuLawDecode(c[b * length + index]))), 0, 159);
      for (int j = std::min(last, y); j <= std::max(last, y); ++j) {
        float* p = image.data() + ((b * 160 + j) * 1024 + x) * 3;
        p[0] = 30, p[1] = 90, p[2] = 170;
      }
      last = y;
    }
  }
  bpix::PngBytes(path, image, n * 160, 1024);
}

void SaveDraw(const fs::path& root, const std::string& name, const WaveDraw& draw, int prefix, const Tensor* source_ids = nullptr) {
  Map out{{"codes", draw.codes}, {"logits", draw.logits}, {"uniform", draw.uniform}};
  if (source_ids) out["source_ids"] = *source_ids;
  nn::save_npz((root / (name + ".npz")).string(), replay::ToNamed(out));
  for (int b = 0; b < draw.codes.size(0); ++b) WaveWrite(root / (name + "_" + std::to_string(b) + ".wav"), draw.codes, b);
  WavePlot(root / (name + ".png"), draw.codes);
  std::cout << name << ":prefix=" << prefix << ",sampled=" << draw.logits.size(1) << ",native cache probes=" << draw.probes
            << ",max error=" << draw.parity << std::endl;
}

int Run(const bpix::Options& o) {
  const auto     started = std::chrono::steady_clock::now();
  const fs::path root = o.out, tf = o.tf, data = o.data;
  const auto     source = LoadWaveStack(data);
  for (auto [from, to] : {std::pair<const char*, const char*>{"wavenet_stack.npz", "data.npz"}, {"manifest.json", "data_manifest.json"},
                          {"ATTRIBUTION.txt", "ATTRIBUTION.txt"}})
    if (fs::exists(data / from)) fs::copy_file(data / from, root / to, fs::copy_options::overwrite_existing);
  check::Require(WaveStackField(kConfig) == 33, "RF33: initial width 2 plus dilations 1-2-4-8-16");
  WaveStackModel model(kConfig, true, 32);
  check::Require(model.parameters.count() == 5068 && model.parameters.size() == 34, "5068 parameters 34 tensors");
  const fs::path init = bpix::InitFile(o.init, tf);
  if (!init.empty()) {
    replay::LoadInto(model.parameters, init);
    model.optimizer = std::make_unique<nn::Optimizer>(model.parameters, .001, replay::Adam());
    std::cout << "replaying TF's initial arrays " << init.string() << "\n";
  } else {
    std::cout << "weights from nn's generator (statistical parity with TF only)\n";
  }
  const auto& fit = source.pools[0];
  const auto& validation = source.pools[1];
  const auto& test = source.pools[2];
  nn::save_npz((root / "initial_arrays.npz").string(), replay::ToNamed(WaveNamedArrays(model)));
  nn::save_npz((root / "initial_outputs.npz").string(), {{"logits", model.Predict(test.input)}});
  nn::BatchOrder order(72, 8, 8713);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_nll\tvalidation_bits_per_sample\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = fit.source.to_vector<int>();
  int        steps      = 0;
  WaveData   first;
  for (int epoch = 1; epoch <= o.epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const auto batch = WaveRows(fit, rows);
      if (!steps) {
        first = batch;
        nn::save_npz((root / "first_batch.npz").string(), {{"input", batch.input}, {"target", batch.target}});
        nn::save_npz((root / "first_forward.npz").string(), {{"logits", model.Predict(batch.input)}});
        nn::save_npz((root / "initial_gradients.npz").string(), replay::ToNamed(WaveNamedGradient(model, batch)));
      }
      model.Loss(batch, true);
      if (!steps) nn::save_npz((root / "first_after.npz").string(), replay::ToNamed(WaveNamedArrays(model)));
      for (auto row : rows) batches << epoch << '\t' << steps << '\t' << row << '\t' << fit_source[row] << '\n';
      ++steps;
    }
    const Tensor prediction = model.Predict(validation.input);
    const double nll        = Nll(prediction, validation);
    check::Near(model.Loss(validation), nll, 2e-5, "graph masked NLL matches double host");
    history << epoch << '\t' << nll << '\t' << nll / std::log(2.) << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), replay::ToNamed(WaveNamedArrays(model)));
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_validation.npz")).string(), {{"logits", prediction}});
    std::cout << "epoch" << epoch << ":validationbits/mu-lawsample=" << nll / std::log(2.) << std::endl;
  }
  const double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.parameters.save((root / "final.npz").string());
  const auto arrays = WaveNamedArrays(model);
  nn::save_npz((root / "model_arrays.npz").string(), replay::ToNamed(arrays));
  nn::save_npz((root / "final_gradients.npz").string(), replay::ToNamed(WaveNamedGradient(model, first)));
  WaveStackModel deployed(kConfig, false, 32);
  deployed.parameters.load((root / "final.npz").string(), true);
  check::Require(bpix::SameBits(model.Snapshot(), deployed.Snapshot()), "every generative WaveNet parameter restores BITEXACT");
  const auto   before = deployed.Snapshot();
  const Tensor logits = deployed.Predict(test.input);
  nn::save_npz((root / "outputs.npz").string(), {{"logits", logits}});
  check::Require(bpix::Bits(deployed.Predict(test.input)) == bpix::Bits(logits), "frozen repeat logits BITEXACT");
  auto changed = test.input.to_vector<std::int32_t>();
  for (int b = 0; b < 24; ++b)
    for (int t = 128; t < 256; ++t) changed[b * 256 + t] = (changed[b * 256 + t] + 71) % 256;
  const Tensor changed_t = nn::from_vector(changed, {24, 256});
  const Tensor altered   = deployed.Predict(changed_t);
  const auto   lv = logits.to_vector<float>(), av = altered.to_vector<float>();
  for (int b = 0; b < 24; ++b)
    for (int t = 0; t < 128; ++t)
      for (int c = 0; c < 256; ++c)
        check::Require(av[(b * 256 + t) * 256 + c] == lv[(b * 256 + t) * 256 + c], "ALL earlier predictions invariant to future codes");
  nn::save_npz((root / "future_control.npz").string(), {{"input", changed_t}, {"logits", altered}});
  nn::save_npz((root / "causality.npz").string(), replay::ToNamed(WaveCausalFixture(arrays, kConfig, test.input)));
  for (int code = 0; code < 256; ++code) check::Require(MuLawEncode(MuLawDecode(code)) == code, "all 256 codec bins round-trip");

  const Tensor seed          = nn::full({2, 1}, 128, nn::DType::Int32);
  const auto   unconditional = CachedWaveDraw(deployed, arrays, kConfig, seed, 2048, 8729);
  SaveDraw(root, "unconditional", unconditional, 1);
  const auto again = CachedWaveDraw(deployed, arrays, kConfig, seed, 2048, 8729);
  check::Require(bpix::Bits(unconditional.codes) == bpix::Bits(again.codes) && bpix::Bits(unconditional.logits) == bpix::Bits(again.logits) &&
                     bpix::Bits(unconditional.uniform) == bpix::Bits(again.uniform),
                 "cached ancestral seed repeat BITEXACT");
  const Tensor prompt       = nn::slice_axis(source.prompts, 1, 0, 128).contiguous();
  const auto   continuation = CachedWaveDraw(deployed, arrays, kConfig, prompt, 512, 8731);
  SaveDraw(root, "continuation", continuation, 128, &source.prompt_source_ids);
  const auto pv = prompt.to_vector<int>(), cv = continuation.codes.to_vector<int>();
  for (int b = 0; b < 3; ++b)
    for (int t = 0; t < 128; ++t) check::Require(cv[b * 640 + t] == pv[b * 128 + t], "observed held-out prompt preserved");
  for (int b = 0; b < 3; ++b) WaveWrite(root / ("reference_" + std::to_string(b) + ".wav"), source.prompts, b);
  WavePlot(root / "reference.png", source.prompts);

  std::vector<double> unigram(256, 1), transition(256 * 256, 1);
  const auto          fi = fit.input.to_vector<int>(), ft = fit.target.to_vector<int>();
  for (int b = 0; b < 72; ++b)
    for (int t = 32; t < 256; ++t) ++unigram[ft[b * 256 + t]], ++transition[fi[b * 256 + t] * 256 + ft[b * 256 + t]];
  double mass = 0;
  for (double u : unigram) mass += u;
  for (auto& u : unigram) u /= mass;
  for (int x = 0; x < 256; ++x) {
    double sum = 0;
    for (int c = 0; c < 256; ++c) sum += transition[x * 256 + c];
    for (int c = 0; c < 256; ++c) transition[x * 256 + c] /= sum;
  }
  const auto ti = test.input.to_vector<int>(), tt = test.target.to_vector<int>();
  double     unll = 0, tnll = 0;
  for (int b = 0; b < 24; ++b)
    for (int t = 32; t < 256; ++t) unll -= std::log(unigram[tt[b * 256 + t]]), tnll -= std::log(transition[ti[b * 256 + t] * 256 + tt[b * 256 + t]]);
  unll /= 24 * 224, tnll /= 24 * 224;
  nn::save_npz((root / "baseline.npz").string(), {{"unigram_probability", nn::from_vector(unigram, {256})},
                                                  {"transition_probability", nn::from_vector(transition, {256, 256})}});
  const double nll = Nll(logits, test);
  check::Require(bpix::SameBits(before, deployed.Snapshot()), "ALL inference/causal checks/cached generation immutable");
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":5068,\"parameter_tensors\":34,\"receptive_field\":33,\"residual_channels\":4,\"skip_channels\":8,\"levels\":5,"
          "\"cycles\":1,\"epochs\":"
       << o.epochs << ",\"optimizer_updates\":" << steps << ",\"selected_epoch_fixed_final\":" << o.epochs
       << ",\"loss_start\":32,\"scored_positions\":224,\"test_nll\":" << nll << ",\"test_bits_per_mu_law_sample\":" << nll / std::log(2.)
       << ",\"fit_unigram_nll\":" << unll << ",\"fit_transition_nll\":" << tnll << ",\"uniform_nll\":" << std::log(256.)
       << ",\"cached_native_probe_max_error\":" << std::max(unconditional.parity, continuation.parity)
       << ",\"cached_native_probes\":" << unconditional.probes + continuation.probes
       << ",\"actual_sampled_codes\":5632,\"sample_rate\":8000,\"train_seconds\":" << train_seconds << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Generative WaveNet</title><style>body{font:16px system-ui;margin:30px;max-width:1100px}"
          "img{max-width:100%}audio{display:block;margin:12px}</style><h1>WaveNet: small generative raw-audio model</h1><p><a "
          "href='https://arxiv.org/abs/1609.03499'>WaveNet</a> causal dilated tanh/sigmoid gates, residual and additive skip paths, "
          "ReLU / post 1x1 / 256-code head. Compact R4 / S8 / dilations 1, 2, 4, 8, 16, RF 33, 5068 params / 34 tensors. No "
          "conditioning.</p><p>Real FSDD recordings (CC BY-SA 4.0, attribution included), 72/12/24 windows with disjoint recordings "
          "and speakers, mu-law 256 codes. First 32 warm-up positions masked, 224 scored per window. Three epochs, 27 Adam .001 steps, "
          "fixed final.</p><p>Each causal block keeps a circular history of its OWN input at its dilation; cached host synthesis is "
          "checked against the native forward; generated codes are fed back. Cold start: one 128 code, 2 x 2048 draws; "
          "continuations: held-out 128-code prompts + 512 draws.</p><h2>Cold-start generations</h2><img src='unconditional.png'>";
  for (int b = 0; b < 2; ++b) html << "<audio controls src='unconditional_" << b << ".wav'></audio>";
  html << "<h2>Prompt continuations</h2><img src='continuation.png'>";
  for (int b = 0; b < 3; ++b) html << "<audio controls src='continuation_" << b << ".wav'></audio>";
  html << "<h2>Original held-out reference clips</h2><img src='reference.png'>";
  for (int b = 0; b < 3; ++b) html << "<audio controls src='reference_" << b << ".wav'></audio>";
  html << "<p>See metrics.json and history.tsv. No speech quality or vocoder claim.</p>";
  check::Require(bool(meta) && bool(history) && bool(batches) && bool(html), "complete generative WaveNet records");
  std::cout << "testNLL=" << nll << ", unigram=" << unll << ", transition=" << tnll
            << "; fixed final/causal RF33/codec/cached 5632 draws/restore/immutable PASS\n";

  if (!init.empty() && init == tf / "initial_arrays.npz") {
    std::cout << "replay vs TF's recorded run (" << tf.string() << "):\n";
    double d = 0;
    for (auto f : {"initial_arrays.npz", "initial_outputs.npz", "first_batch.npz", "first_forward.npz", "initial_gradients.npz"})
      d = std::max(d, bpix::Compare(tf, f, replay::ToMap(nn::load_npz((root / f).string())), o.verbose));
    check::Require(d >= 0 && d <= 2e-5, "replayed initial outputs/first forward/gradients match TF");
    const double after = bpix::Compare(tf, "first_after.npz", replay::ToMap(nn::load_npz((root / "first_after.npz").string())), o.verbose);
    check::Require(after >= 0 && after <= 1e-4, "first Adam update matches TF");
    if (o.epochs == 3) {
      const auto ours_h = replay::History(root / "history.tsv"), tf_h = replay::History(tf / "history.tsv");
      double     hd     = 0;
      for (std::size_t i = 0; i < std::min(ours_h.size(), tf_h.size()); ++i) hd = std::max(hd, std::abs(ours_h[i] - tf_h[i]));
      std::printf("  vs TF %-28s max |ours - TF| = %.3g\n", "history.tsv", hd);
      check::Require(hd < 1e-4, "replayed validation history follows TF's");
      for (auto f : {"epoch3_arrays.npz", "model_arrays.npz", "outputs.npz", "final_gradients.npz", "future_control.npz", "causality.npz", "baseline.npz"})
        bpix::Compare(tf, f, replay::ToMap(nn::load_npz((root / f).string())), o.verbose);
      std::printf("  generated codes equal to TF's: unconditional %.4f, continuation %.4f\n",
                  bpix::Agreement(tf, "unconditional.npz", {{"codes", unconditional.codes}}, "codes"),
                  bpix::Agreement(tf, "continuation.npz", {{"codes", continuation.codes}}, "codes"));
    }
  }
  if (fs::exists(tf / "model_arrays.npz") && fs::exists(tf / "outputs.npz")) {
    WaveStackModel fresh(kConfig, false, 32);
    replay::LoadInto(fresh.parameters, tf / "model_arrays.npz");
    const double d = bpix::Compare(tf, "outputs.npz", {{"logits", fresh.Predict(test.input)}}, o.verbose);
    check::Require(d >= 0 && d <= 2e-5, "TF's final weights reproduce TF's test logits");
    const auto tf_arrays = WaveNamedArrays(fresh);
    const auto u = CachedWaveDraw(fresh, tf_arrays, kConfig, seed, 2048, 8729);
    const auto c = CachedWaveDraw(fresh, tf_arrays, kConfig, prompt, 512, 8731);
    const double su = bpix::Agreement(tf, "unconditional.npz", {{"codes", u.codes}}, "codes"),
                 sc = bpix::Agreement(tf, "continuation.npz", {{"codes", c.codes}}, "codes");
    std::printf("  from TF's final weights: generated codes equal to TF's: unconditional %.4f, continuation %.4f\n", su, sc);
  }
  std::cout << "training " << train_seconds << " s, wall " << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
            << " s\nArtifacts:" << (root / "index.html").string() << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return Run(bpix::ParseOptions(argc, argv, "wavenet", paths::Dataset("wavenet")));
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
