// CMD: just run-nn image_gpt
// Compact ImageGPT: fit-only RGB palette, BOS-shifted pixel Transformer with
// RMSNorm and QuickGELU, ancestral colors.
//
// Port of tflow's image_gpt (common/ml/transformer_lm.h with an iGPT config ->
// examples/nn/common/{transformer_lm,decoder_transformer}.hpp). Data: tflow's
// image_gpt assets (paths::Dataset("image_gpt"): CIFAR 128 fit / 32 validation
// / 32 test images at 8x8, a frozen fit-only 16-color palette, raster palette
// codes with input [BOS 16, pixel 0..62] and target [pixel 0..63]). One
// decoder block (D 16, 2 heads, FF 32) with RMSNorm (gamma only) and QuickGELU,
// 17-token input / 16-class output: 3776 parameters in 15 tensors; Adam .003,
// batch 8 in tflow's BatchOrder(128, 8, 9127), three epochs = 48 updates.
//
// Checks (as tflow): palette/data shift; census; restore bit-exact; encoded
// input future Jacobian zero; all 64 current/future TARGET flips leave prefix
// logits bit-exact; future attention weights 0 and rows normalized; 32
// BOS-only ancestral images and 8 completions from 32 observed palette pixels
// (seed repeat bit-exact, prefix preserved, BOS never predicted); palette and
// weights immutable. Fit-only unigram / previous-code baselines and the
// palette quantization MSE. Added: init=auto replays TF's initial arrays and
// compares every record; TF's final weights must reproduce TF's test outputs
// and sampled codes. Deviations: final.npz checkpoint, no independent_audit.json.
// Options: epochs=3 data=DIR out=DIR tf=DIR init=auto|tf|nn threads=4 verbose=0.
// Output: output/nn/models/image_gpt.
#include "common/binary_pixels.hpp"
#include "common/transformer_lm.hpp"

namespace {

using lm::Tensor;
namespace nn = datamunge::nn;
namespace fs = std::filesystem;
using Map = replay::Map;

const lm::DecoderConfig kConfig{17, 16, 64, 16, 2, 32, true, true};

/// Palette codes [n, 64] -> RGB [n, 8, 8, 3].
Tensor Decode(const Tensor& codes, const Tensor& palette) {
  check::Require(codes.dtype() == nn::DType::Int32 && codes.dim() == 2 && codes.size(1) == 64, "64-pixel palette code images");
  const auto         c = codes.to_vector<int>();
  const auto         p = palette.to_vector<float>();
  std::vector<float> rgb(c.size() * 3);
  for (std::size_t i = 0; i < c.size(); ++i) {
    check::Require(c[i] >= 0 && c[i] < 16, "palette codes exclude BOS");
    for (int k = 0; k < 3; ++k) rgb[i * 3 + k] = p[c[i] * 3 + k];
  }
  return bpix::Floats(rgb, {codes.size(0), 8, 8, 3});
}

/// tflow's Draws: per image a BOS (+ observed truth prefix) prompt sampled to 64 pixels with seed + b.
Map Draws(lm::TransformerLM& model, int n, std::uint32_t seed, const Tensor* truth = nullptr) {
  const int                 known = truth ? 32 : 0, count = 64 - known;
  std::vector<std::int32_t> codes(n * 64), contexts, lengths;
  std::vector<double>       uniform;
  std::vector<float>        logits;
  const auto                tv = truth ? truth->to_vector<int>() : std::vector<int>();
  for (int b = 0; b < n; ++b) {
    std::vector<int> prefix{16};
    for (int j = 0; j < known; ++j) prefix.push_back(tv[b * 64 + j]);
    const auto draw = model.Sample(prefix, count, seed + b, 1);
    const auto dc   = draw.at("codes").to_vector<int>();
    check::Require(dc[0] == 16, "BOS prefix only, never predicted");
    for (int j = 0; j < 64; ++j) {
      check::Require(dc[j + 1] >= 0 && dc[j + 1] < 16, "every ancestral pixel a valid color");
      codes[b * 64 + j] = dc[j + 1];
      if (truth && j < known) check::Require(dc[j + 1] == tv[b * 64 + j], "observed palette prefix BITEXACT");
    }
    for (double u : draw.at("uniform").to_vector<double>()) uniform.push_back(u);
    for (float z : draw.at("logits").to_vector<float>()) logits.push_back(z);
    for (int c : draw.at("contexts").to_vector<int>()) contexts.push_back(c);
    for (int l : draw.at("context_lengths").to_vector<int>()) lengths.push_back(l);
    if (!b) check::Require(bpix::SameBits(draw, model.Sample(prefix, count, seed, 1)), "seed repeat all pixels/logits/uniforms/contexts BITEXACT");
  }
  Map out{{"codes", nn::from_vector(codes, {n, 64})},
          {"uniform", nn::from_vector(uniform, {n, count})},
          {"logits", bpix::Floats(logits, {n, count, 16})},
          {"contexts", nn::from_vector(contexts, {n, count, 64})},
          {"context_lengths", nn::from_vector(lengths, {n, count})}};
  if (truth) out["truth_codes"] = *truth;
  return out;
}

int Run(const lm::Options& o, const fs::path& data) {
  const auto     started = std::chrono::steady_clock::now();
  const fs::path root    = o.out;
  check::Require(fs::exists(data / "image_gpt.npz"), "image_gpt dataset missing: " + data.string() + " (tflow's image_gpt assets)");
  const auto   corpus  = replay::ToMap(nn::load_npz((data / "image_gpt.npz").string()));
  const Tensor palette = corpus.at("palette");
  check::Require(palette.dtype() == nn::DType::Float32 && palette.shape() == nn::Shape({16, 3}), "frozen fit-only 16-color RGB palette");
  const std::string palette_bytes = bpix::Bits(palette);
  fs::copy_file(data / "image_gpt.npz", root / "data.npz", fs::copy_options::overwrite_existing);
  if (fs::exists(data / "manifest.json")) fs::copy_file(data / "manifest.json", root / "data_manifest.json", fs::copy_options::overwrite_existing);
  for (std::string split : {"fit", "validation", "test"}) {
    const auto& x = corpus.at(split + "_input");
    const auto& y = corpus.at(split + "_target");
    const int   n = split == "fit" ? 128 : 32;
    check::Require(x.dtype() == nn::DType::Int32 && x.shape() == nn::Shape({n, 64}) && y.shape() == x.shape(), "64 WHOLE palette pixel codes");
    const auto xv = x.to_vector<int>(), yv = y.to_vector<int>();
    for (int i = 0; i < n * 64; ++i) {
      check::Require(yv[i] >= 0 && yv[i] < 16, "target palette code, NO labels/BOS");
      if (i % 64 == 0) check::Require(xv[i] == 16, "initial BOS");
      else check::Require(xv[i] == yv[i - 1], "STRICT previous-pixel shift");
    }
  }
  auto at = [&](const std::string& k) { return corpus.at(k); };

  lm::TransformerLM m(true, kConfig);
  check::Require(m.store.count() == 3776 && m.store.size() == 15, "3776 params 15 tensors / RMS gamma ONLY / QuickGELU");
  const fs::path init = lm::InitArrays(o);
  if (!init.empty()) {
    lm::LoadArrays(m.store, init);
    m.Train();
    std::cout << "replaying TF's initial arrays " << init.string() << "\n";
  } else {
    std::cout << "weights from nn's generator (statistical parity with TF only)\n";
  }
  nn::save_npz((root / "initial_arrays.npz").string(), m.Arrays());
  nn::save_npz((root / "initial_outputs.npz").string(), lm::Named(m.Evaluate(at("test_input"), at("test_target"))));
  nn::BatchOrder order(128, 8, 9127);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_nll\tvalidation_bits_per_palette_pixel\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto source = at("fit_source_ids").to_vector<int>();
  int        steps  = 0;
  Tensor     firstx, firsty;
  for (int e = 1; e <= o.epochs; ++e) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const Tensor x = lm::Rows(at("fit_input"), rows), y = lm::Rows(at("fit_target"), rows);
      if (!steps) {
        firstx = x, firsty = y;
        nn::save_npz((root / "first_batch.npz").string(), {{"input", x}, {"target", y}});
        nn::save_npz((root / "first_forward.npz").string(), lm::Named(m.Evaluate(x, y)));
        nn::save_npz((root / "initial_gradients.npz").string(), m.Gradient(x, y));
      }
      m.Update(x, y);
      if (!steps) nn::save_npz((root / "first_after.npz").string(), m.Arrays());
      for (auto r : rows) batches << e << '\t' << steps << '\t' << r << '\t' << source[r] << '\n';
      ++steps;
    }
    const auto   v   = m.Evaluate(at("validation_input"), at("validation_target"));
    const double nll = v.at("loss").item<float>();
    history << e << '\t' << nll << '\t' << nll / std::log(2.) << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(e) + "_arrays.npz")).string(), m.Arrays());
    nn::save_npz((root / ("epoch" + std::to_string(e) + "_validation.npz")).string(), lm::Named(v));
    std::cout << "epoch" << e << ":validationbits/PALETTEpixel=" << nll / std::log(2.) << std::endl;
  }
  const double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  m.store.save((root / "final.npz").string());
  nn::save_npz((root / "final_gradients.npz").string(), m.Gradient(firstx, firsty));
  lm::TransformerLM deployed(false, kConfig);
  deployed.store.load((root / "final.npz").string(), true);
  check::Require(lm::SameBits(m.Arrays(), deployed.Arrays()), "every ImageGPT weight restores BITEXACT");
  const auto before  = deployed.Arrays();
  const auto outputs = deployed.Evaluate(at("test_input"), at("test_target"));
  nn::save_npz((root / "outputs.npz").string(), lm::Named(outputs));
  nn::save_npz((root / "model_arrays.npz").string(), before);

  const Tensor single = lm::Rows(at("test_input"), {0}), target = lm::Rows(at("test_target"), {0});
  const Tensor original = deployed.Predict(single), jac = deployed.InputGradient(single);
  const auto   jv = jac.to_vector<float>(), ov = original.to_vector<float>();
  for (int t = 33; t < 64; ++t)
    for (int c = 0; c < 17; ++c) check::Require(jv[t * 17 + c] == 0, "current/future TARGET pixels cannot leak through shifted inputs");
  nn::save_npz((root / "causality.npz").string(), {{"input", single}, {"target", target}, {"logits", original}, {"input_gradient", jac}});
  const auto tv = target.to_vector<int>();
  for (int site = 0; site < 64; ++site) {
    std::vector<std::int32_t> changed(64);
    changed[0] = 16;
    for (int t = 1; t < 64; ++t) changed[t] = t - 1 >= site ? (tv[t - 1] + 7) % 16 : tv[t - 1];
    const auto control = deployed.Predict(nn::from_vector(changed, {1, 64})).to_vector<float>();
    for (int t = 0; t <= site; ++t)
      for (int c = 0; c < 16; ++c)
        check::Require(control[t * 16 + c] == ov[t * 16 + c], "ALL target-current/future palette flips leave prefix logits BITEXACT");
  }
  const auto weights = outputs.at("attention").to_vector<float>();
  for (int b = 0; b < 32; ++b)
    for (int h = 0; h < 2; ++h)
      for (int q = 0; q < 64; ++q) {
        double sum = 0;
        for (int k = 0; k < 64; ++k) {
          const float w = weights[((b * 2 + h) * 64 + q) * 64 + k];
          if (k > q) check::Require(w == 0, "future attention weight 0");
          sum += w;
        }
        check::Near(sum, 1, 2e-6, "all attention rows normalized");
      }
  auto samples      = Draws(deployed, 32, 9137);
  samples["images"] = Decode(samples.at("codes"), palette);
  nn::save_npz((root / "samples.npz").string(), lm::Named(samples));
  bpix::Png(root / "samples.png", samples.at("images"), 8, 2);
  const std::vector<std::int64_t> rows{0, 1, 2, 3, 4, 5, 6, 7};
  const Tensor                    truth = lm::Rows(at("test_target"), rows);
  auto                            completion = Draws(deployed, 8, 9187, &truth);
  completion["images"]                       = Decode(completion.at("codes"), palette);
  nn::save_npz((root / "completions.npz").string(), lm::Named(completion));
  const Tensor quantized = Decode(at("test_target"), palette);
  const auto   qv = quantized.to_vector<float>(), rgb = at("test_rgb").to_vector<float>(), cv = completion.at("images").to_vector<float>();
  std::vector<float> gallery(32 * 192);
  for (int b = 0; b < 8; ++b)
    for (int j = 0; j < 192; ++j) {
      gallery[(b * 4) * 192 + j]     = rgb[b * 192 + j];
      gallery[(b * 4 + 1) * 192 + j] = qv[b * 192 + j];
      gallery[(b * 4 + 2) * 192 + j] = j < 96 ? qv[b * 192 + j] : .5f;
      gallery[(b * 4 + 3) * 192 + j] = cv[b * 192 + j];
    }
  const Tensor gallery_t = bpix::Floats(gallery, {32, 8, 8, 3});
  nn::save_npz((root / "completion_gallery.npz").string(), {{"images", gallery_t}});
  bpix::Png(root / "completions.png", gallery_t, 4, 2);
  const auto         pv = palette.to_vector<float>();
  std::vector<float> palette_image(32 * 512 * 3);
  for (int y = 0; y < 32; ++y)
    for (int x = 0; x < 512; ++x)
      for (int c = 0; c < 3; ++c) palette_image[(y * 512 + x) * 3 + c] = float(std::floor(255 * double(pv[(x / 32) * 3 + c]) + .5));
  bpix::PngBytes(root / "palette.png", palette_image, 32, 512);
  double mse = 0;
  for (std::size_t i = 0; i < qv.size(); ++i) mse += (double(qv[i]) - rgb[i]) * (double(qv[i]) - rgb[i]);
  mse /= double(qv.size());
  const auto          fx = at("fit_input").to_vector<int>(), fy = at("fit_target").to_vector<int>();
  const auto          tx = at("test_input").to_vector<int>(), ty = at("test_target").to_vector<int>();
  std::vector<double> unigram(16, 1), transition(17 * 16, 1);
  for (int i = 0; i < 8192; ++i) ++unigram[fy[i]], ++transition[fx[i] * 16 + fy[i]];
  for (auto& u : unigram) u /= 8192 + 16;
  for (int c = 0; c < 17; ++c) {
    double sum = 0;
    for (int j = 0; j < 16; ++j) sum += transition[c * 16 + j];
    for (int j = 0; j < 16; ++j) transition[c * 16 + j] /= sum;
  }
  double unll = 0, tnll = 0;
  for (int i = 0; i < 2048; ++i) unll -= std::log(unigram[ty[i]]), tnll -= std::log(transition[tx[i] * 16 + ty[i]]);
  unll /= 2048, tnll /= 2048;
  nn::save_npz((root / "baseline.npz").string(),
               {{"unigram_probability", nn::from_vector(unigram, {16})}, {"transition_probability", nn::from_vector(transition, {17, 16})}});
  check::Require(lm::SameBits(before, deployed.Arrays()) && palette_bytes == bpix::Bits(palette),
                 "all inference and frozen palette immutable / excluded from optimizers");
  const double  nll = outputs.at("loss").item<float>();
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":3776,\"parameter_tensors\":15,\"palette_colors\":16,\"input_vocabulary\":17,\"bos_id\":16,\"dimensions\":16,"
          "\"heads\":2,\"feed_forward\":32,\"blocks\":1,\"rms_norm\":true,\"quick_gelu\":true,\"epochs\":"
       << o.epochs << ",\"optimizer_updates\":" << steps << ",\"selected_epoch_fixed_final\":" << o.epochs
       << ",\"test_nll_per_palette_pixel\":" << nll << ",\"test_bits_per_palette_pixel\":" << nll / std::log(2.)
       << ",\"test_code_perplexity\":" << std::exp(nll) << ",\"test_palette_quantization_rgb_mse\":" << mse << ",\"fit_unigram_nll\":" << unll
       << ",\"fit_transition_nll\":" << tnll << ",\"uniform_bits_per_palette_pixel\":4,\"actual_sampled_pixels\":2304,\"train_seconds\":"
       << train_seconds << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Tiny ImageGPT</title><style>body{font:16px system-ui;margin:30px;max-width:1000px}"
          "img{max-width:100%;image-rendering:pixelated}</style><h1>ImageGPT: pixel-sequence Transformer</h1><p><a "
          "href='https://proceedings.mlr.press/v119/chen20s.html'>Generative Pretraining From Pixels</a>, compact generative study: "
          "8x8 RGB images, K16 fit-only color palette, ONE d16 / two-head / FF32 causal decoder block, 3776 params in 15 tensors, "
          "RMS normalization (gamma only), QuickGELU x sigmoid(1.702x), learned token + 1-D position embeddings. Not a full "
          "512-color / ImageNet-scale reproduction.</p><p>Frozen palette (excluded from optimizers):</p><img src='palette.png'><p>"
          "Input [BOS 16, pixel 0..62], target [pixel 0..63]; BOS is input-only. Three epochs, 48 Adam .003 steps, fixed final. "
          "Categorical NLL / bits per QUANTIZED color pixel.</p><h2>First 32 ancestral samples</h2><img src='samples.png'><h2>First "
          "8 held-out completions</h2><p>Columns ORIGINAL 8-bit RGB, TRUE QUANTIZED palette RGB, OBSERVED top half (gray unknown), "
          "ancestral COMPLETION.</p><img src='completions.png'><p>See metrics.json.</p>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete ImageGPT records");
  std::cout << "test bits/PALETTEpixel=" << nll / std::log(2.) << ", unigram/transition NLL=" << unll << '/' << tnll
            << ", palette RGB MSE=" << mse << "; RMS-QuickGELU/BOS shift/causality/frozen palette/2304 ancestral/restore/immutable PASS\n";

  if (!init.empty() && init == o.tf / "initial_arrays.npz") {
    std::cout << "replay vs TF's recorded run (" << o.tf.string() << "):\n";
    double first = 0;
    for (auto f : {"initial_arrays.npz", "initial_outputs.npz", "first_batch.npz", "first_forward.npz", "initial_gradients.npz", "first_after.npz"})
      first = std::max(first, lm::CompareRecord(o, f, nn::load_npz((root / f).string())));
    check::Require(first >= 0 && first <= 2e-5, "replayed initial forward/gradients/first update match TF");
    if (o.epochs == 3) {
      const auto ours_h = replay::History(root / "history.tsv"), tf_h = replay::History(o.tf / "history.tsv");
      double     hd     = 0;
      for (std::size_t i = 0; i < std::min(ours_h.size(), tf_h.size()); ++i) hd = std::max(hd, std::abs(ours_h[i] - tf_h[i]));
      std::printf("  vs TF %-26s max |ours - TF| = %.3g\n", "history.tsv", hd);
      check::Require(hd < 1e-4, "replayed validation history follows TF's");
      for (auto f : {"epoch3_validation.npz", "model_arrays.npz", "outputs.npz", "final_gradients.npz", "causality.npz", "samples.npz",
                     "completions.npz", "baseline.npz"})
        lm::CompareRecord(o, f, nn::load_npz((root / f).string()));
    }
  }
  if (fs::exists(o.tf / "model_arrays.npz") && fs::exists(o.tf / "outputs.npz")) {
    lm::TransformerLM fresh(false, kConfig);
    lm::LoadArrays(fresh.store, o.tf / "model_arrays.npz");
    const double d = lm::CompareRecord(o, "outputs.npz", lm::Named(fresh.Evaluate(at("test_input"), at("test_target"))));
    check::Require(d >= 0 && d <= 2e-5, "TF's final weights reproduce TF's test outputs");
    const auto   s    = Draws(fresh, 32, 9137);
    const double same = bpix::Agreement(o.tf, "samples.npz", s, "codes");
    std::printf("  from TF's final weights: %.4f of 2048 sampled palette codes identical to TF's\n", same);
    check::Require(same == 1, "TF's final weights reproduce TF's sampled codes");
  }
  std::cout << "training " << train_seconds << " s, wall " << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
            << " s\nArtifacts:" << (root / "index.html").string() << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    fs::path           data = paths::Dataset("image_gpt");
    std::vector<char*> rest{argv[0]};
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]).rfind("data=", 0) == 0) data = std::string(argv[i]).substr(5);
      else rest.push_back(argv[i]);
    }
    return Run(lm::ParseOptions(int(rest.size()), rest.data(), "image_gpt", true), data);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
