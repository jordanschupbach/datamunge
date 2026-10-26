// CMD: just run-nn pixel_cnn
// Small genuine masked-convolution PixelCNN on binary 8x8 MNIST, strict prefix
// likelihood and ancestral sampling.
//
// Port of tflow's pixel_cnn (common/ml/pixel_cnn.h, common/nn/pixel_cnn.h ->
// examples/nn/common/{binary_pixels,pixel_cnn}.hpp). Data: tflow's saliency
// MNIST pools (paths::Dataset("saliency"), `just autoencoder-assets` in
// tflow_examples): 1000 fit / 200 validation / 200 test, half-pixel bilinear
// 28 -> 8, threshold >= .5. Four 3x3 masked convolutions (A then 3x B, width 8,
// ReLU) + 1x1 Bernoulli head, 1841 stored / 1033 active parameters; three
// epochs of 32-image Adam(.003) batches in tflow's BatchOrder(1000, 32, 8419).
//
// Checks (as tflow): parameter census; fresh restore bit-exact; single/batch
// parity; all 4096 logit/input Jacobian entries with every current/future one
// exactly zero; at all 64 sites flipping current+future pixels leaves every
// prefix probability bit-identical; seeded ancestral sampling repeats bit-exactly;
// 32-pixel prefix completions keep the prefix; inference is immutable.
// Added: data bit-identical to TF's; replay of TF's initial arrays (init=auto)
// matching TF's first forward/gradients/update; TF's final weights reproduce
// TF's test outputs. Deviations: checkpoint final.npz instead of a SaveV2
// bundle; no independent_audit.json. Options: epochs=3 data=DIR out=DIR
// init=auto|tf|nn|PATH tf=DIR threads=4 verbose=0. Output: output/nn/models/pixel_cnn.
#include "common/pixel_cnn.hpp"

// Experiment workflow from examples/nn/common/binary_pixels.hpp.
namespace bpix {
using namespace ::bpix;
inline double Run(const Spec& spec, const Options& o, const Factory& make, const Extra& extra = {}) {
  const auto  started = std::chrono::steady_clock::now();
  const fs::path root = spec.variant.empty() ? o.out : o.out / spec.variant;
  const fs::path tf   = spec.variant.empty() ? o.tf : o.tf / spec.variant;
  fs::create_directories(root);
  auto data = LoadBinary(o.data);
  nn::save_npz((root / "data.npz").string(), replay::ToNamed(data.record));
  if (fs::exists(o.data / "manifest.json"))
    fs::copy_file(o.data / "manifest.json", root / "data_manifest.json", fs::copy_options::overwrite_existing);
  if (fs::exists(tf / "data.npz")) {
    const auto theirs = replay::ToMap(nn::load_npz((tf / "data.npz").string()));
    for (const std::string s : {"fit", "validation", "test"})
      check::Require(Bits(theirs.at(s + "_images")) == Bits(data.binary[s == "fit" ? 0 : s == "validation" ? 1 : 2]) &&
                         Bits(theirs.at(s + "_gray")) == Bits(data.gray[s == "fit" ? 0 : s == "validation" ? 1 : 2]),
                     "resized/binarized pixels bit-identical to TF's data.npz");
    std::cout << "data: gray and binary 8x8 pools bit-identical to TF's data.npz\n";
  }
  const bool classes_used = make(false)->uses_classes;
  auto       C = [&](int split, const std::vector<std::int64_t>& rows) { return classes_used ? ClassesOf(data, split, rows) : Tensor(); };
  auto       All = [&](int split) { return classes_used ? data.pools[split].labels : Tensor(); };

  auto model = make(true);
  check::Require(model->store.count() == spec.parameters && std::int64_t(model->store.size()) == spec.tensors,
                 std::to_string(spec.parameters) + " parameters/" + std::to_string(spec.tensors) + " variable tensors/" +
                     std::to_string(spec.masked) + " masked coordinates");
  const fs::path init = InitFile(o.init, tf);
  if (!init.empty()) {
    replay::LoadInto(model->store, init);
    model->Train();
    std::cout << "replaying TF's initial arrays " << init.string() << "\n";
  } else {
    std::cout << "weights from nn's generator (statistical parity with TF only)\n";
  }
  nn::save_npz((root / "initial_arrays.npz").string(), replay::ToNamed(model->Arrays()));
  nn::save_npz((root / "initial_outputs.npz").string(), replay::ToNamed(model->Evaluate(data.binary[2], All(2))));
  nn::BatchOrder order(1000, 32, spec.batch_seed);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_nll_per_binary_pixel\tvalidation_bits_per_binary_pixel\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = data.pools[0].source.to_vector<int>();
  int        updates    = 0;
  Tensor     first_x, first_c;
  for (int epoch = 1; epoch <= o.epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const Tensor x = Rows(data.binary[0], rows), c = C(0, rows);
      if (!updates) {
        first_x = x;
        first_c = c;
        Map fb{{"images", x}};
        if (classes_used) fb["labels"] = c;
        nn::save_npz((root / "first_batch.npz").string(), replay::ToNamed(fb));
        nn::save_npz((root / "first_forward.npz").string(), replay::ToNamed(model->Evaluate(x, c)));
        nn::save_npz((root / "initial_gradients.npz").string(), replay::ToNamed(model->Gradient(x, c)));
      }
      model->Update(x, c);
      if (!updates) nn::save_npz((root / "first_after.npz").string(), replay::ToNamed(model->Arrays()));
      for (auto row : rows) batches << epoch << '\t' << updates << '\t' << row << '\t' << fit_source[row] << '\n';
      ++updates;
    }
    const double nll = model->Evaluate(data.binary[1], All(1)).at("loss").item<float>();
    history << epoch << '\t' << nll << '\t' << nll / std::log(2.) << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), replay::ToNamed(model->Arrays()));
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_samples.npz")).string(),
                 replay::ToNamed(model->Sample(16, spec.epoch_sample_seed, classes_used ? CycleClasses(16) : Tensor())));
    std::cout << "epoch" << epoch << ":validationbits/BINARYpixel=" << nll / std::log(2.) << std::endl;
  }
  const double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model->store.save((root / "final.npz").string());
  nn::save_npz((root / "final_gradients.npz").string(), replay::ToNamed(model->Gradient(first_x, first_c)));
  auto deployed = make(false);
  deployed->store.load((root / "final.npz").string(), true);
  check::Require(SameBits(model->Arrays(), deployed->Arrays()), "all weights fresh restore BITEXACT");
  const auto before  = deployed->Arrays();
  auto       outputs = deployed->Evaluate(data.binary[2], All(2));
  nn::save_npz((root / "outputs.npz").string(), replay::ToNamed(outputs));
  nn::save_npz((root / "model_arrays.npz").string(), replay::ToNamed(deployed->Arrays()));

  // Single/batch parity, causal Jacobian and future-flip guards.
  const Tensor single = Rows(data.binary[2], {0}), single_class = C(2, {0});
  const auto   probability = deployed->Probability(single, single_class);
  const auto   pv = probability.to_vector<float>(), ov = outputs.at("probability").to_vector<float>();
  double       parity = 0;
  for (int j = 0; j < 64; ++j) parity = std::max(parity, std::abs(double(pv[j]) - ov[j]));
  check::Require(parity < 1e-6, "single/batch parity");
  std::vector<float> jacobian(64 * 64);
  Map                causal{{"images", single}, {"probability", probability}};
  if (classes_used) causal["labels"] = single_class;
  const auto sv = single.to_vector<float>();
  for (int ordinal = 0; ordinal < 64; ++ordinal) {
    const int  site     = deployed->Order(ordinal);
    const auto gradient = deployed->CausalGradient(single, single_class, site).to_vector<float>();
    for (int j = 0; j < 64; ++j) {
      jacobian[site * 64 + j] = gradient[j];
      if (deployed->Order(j) >= ordinal) check::Require(gradient[j] == 0, "EVERY current and future pixel Jacobian exact zero");
    }
    auto changed = sv;
    for (int j = 0; j < 64; ++j)
      if (deployed->Order(j) >= ordinal) changed[j] = 1 - changed[j];
    const Tensor changed_t = Floats(changed, single.shape());
    const auto   control   = deployed->Probability(changed_t, single_class);
    const auto   cv        = control.to_vector<float>();
    for (int j = 0; j < 64; ++j)
      if (deployed->Order(j) <= ordinal)
        check::Require(cv[j] == pv[j], "all future flips leave ALL prefix conditional probabilities BITEXACT");
    if (ordinal % 8 == 0) {
      causal["changed_input" + std::to_string(site)]       = changed_t;
      causal["changed_probability" + std::to_string(site)] = control;
    }
  }
  causal["jacobian"] = Floats(jacobian, {64, 64});
  nn::save_npz((root / "causality.npz").string(), replay::ToNamed(causal));

  // Ancestral samples, repeat, prefix completions.
  const Tensor sample_classes = classes_used ? CycleClasses(32) : Tensor();
  auto         samples        = deployed->Sample(32, spec.sample_seed, sample_classes);
  nn::save_npz((root / "samples.npz").string(), replay::ToNamed(samples));
  check::Require(SameBits(samples, deployed->Sample(32, spec.sample_seed, sample_classes)),
                 "fixed seed ancestral samples AND conditionals BITEXACT");
  const std::vector<std::int64_t> first8{0, 1, 2, 3, 4, 5, 6, 7};
  const Tensor                    prefix      = Rows(data.binary[2], first8);
  auto                            completions = deployed->Sample(8, spec.completion_seed, C(2, first8), &prefix, 32);
  completions["truth"]                        = prefix;
  if (classes_used) completions["labels"] = C(2, first8);
  const auto px = prefix.to_vector<float>(), cx = completions.at("images").to_vector<float>();
  for (int b = 0; b < 8; ++b)
    for (int j = 0; j < 64; ++j)
      if (deployed->Order(j) < 32) check::Require(px[b * 64 + j] == cx[b * 64 + j], "completion observed prefix preserved exact");
  nn::save_npz((root / "completions.npz").string(), replay::ToNamed(completions));
  std::vector<float> gallery(24 * 64);
  for (int b = 0; b < 8; ++b)
    for (int j = 0; j < 64; ++j) {
      gallery[(b * 3) * 64 + j]     = px[b * 64 + j];
      gallery[(b * 3 + 1) * 64 + j] = deployed->Order(j) < 32 ? px[b * 64 + j] : .5f;
      gallery[(b * 3 + 2) * 64 + j] = cx[b * 64 + j];
    }
  const Tensor gallery_t = Floats(gallery, {24, 8, 8, 1});
  nn::save_npz((root / "completion_gallery.npz").string(), {{"images", gallery_t}});
  Png(root / "samples.png", samples.at("images"), 8);
  Png(root / "completions.png", gallery_t, 3);

  // Fit-only Jeffreys independent Bernoulli baseline.
  const auto         fit = data.binary[0].to_vector<float>(), test = data.binary[2].to_vector<float>();
  std::vector<float> means(64);
  double             baseline = 0;
  for (int j = 0; j < 64; ++j) {
    double sum = 0;
    for (int b = 0; b < 1000; ++b) sum += fit[b * 64 + j];
    const double p = (sum + .5) / 1001.;
    means[j]       = float(p);
    for (int b = 0; b < 200; ++b) baseline -= test[b * 64 + j] ? std::log(p) : std::log1p(-p);
  }
  baseline /= 12800;
  nn::save_npz((root / "baseline.npz").string(), {{"probability", Floats(means, {64})}});
  check::Require(SameBits(before, deployed->Arrays()), "all likelihood/causal Jacobian/sampling/completion immutable");
  std::string extra_json;
  if (extra) {
    extra_json = extra(*deployed, data, root, outputs);
    check::Require(SameBits(before, deployed->Arrays()), "example-specific controls also immutable");
  }

  const double  nll = outputs.at("loss").item<float>();
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{" << spec.metrics_prefix << "\"parameters\":" << spec.parameters << ",\"active_parameters\":" << spec.parameters - spec.masked
       << ",\"masked_unused_coordinates\":" << spec.masked << ",\"epochs\":" << o.epochs << ",\"optimizer_updates\":" << updates
       << ",\"selected_epoch_fixed_final\":" << o.epochs << ",\"test_nll_per_binary_pixel\":" << nll
       << ",\"test_bits_per_binary_pixel\":" << nll / std::log(2.) << ",\"fit_bernoulli_baseline_nll\":" << baseline
       << ",\"fit_bernoulli_baseline_bits\":" << baseline / std::log(2.) << extra_json << ",\"single_batch_max_error\":" << parity
       << ",\"verified_causal_jacobian_entries\":4096,\"ancestral_pixels\":2048,\"prefix_completion_pixels\":256,\"train_seconds\":"
       << train_seconds << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>" << spec.title
       << "</title><style>body{font:16px system-ui;margin:30px;max-width:1000px}img{max-width:100%;image-rendering:pixelated}</style>"
       << spec.html
       << "<h2>First 32 fixed-seed ancestral draws</h2><img src='samples.png'><h2>First 8 held-out completion cases</h2><p>Columns "
          "TRUE binary digit, OBSERVED first 32 pixels in model order (gray unknown), ancestral COMPLETION. Gray is a display mask "
          "only, not fed to the model.</p><img src='completions.png'><p>See metrics.json and history.tsv. No claim of readable "
          "digits or state-of-the-art likelihood.</p>";
  check::Require(bool(meta) && bool(history) && bool(batches) && bool(html), "complete " + spec.name + " records");
  std::cout << "test bits/BINARYpixel=" << nll / std::log(2.) << ", independent Bernoulli baseline=" << baseline / std::log(2.) << ";"
            << spec.tag << "/4096 causal zeros/64 future flip guards/restore/ancestral/prefix/immutable PASS\n";

  // ---- TF parity ----
  const auto close = [](double d, double tol) { return d >= 0 && d <= tol; };
  if (!init.empty() && init == tf / "initial_arrays.npz") {
    std::cout << "replay vs TF's recorded run (" << tf.string() << "):\n";
    double first = 0;
    for (auto f : {"initial_arrays.npz", "initial_outputs.npz", "first_batch.npz", "first_forward.npz", "initial_gradients.npz"})
      first = std::max(first, Compare(tf, f, replay::ToMap(nn::load_npz((root / f).string())), o.verbose));
    check::Require(close(first, 2e-5), "replayed initial outputs/first forward/gradients match TF");
    const double after = Compare(tf, "first_after.npz", replay::ToMap(nn::load_npz((root / "first_after.npz").string())), o.verbose);
    check::Require(close(after, 1e-4), "first Adam update matches TF");
    if (o.epochs == 3) {
      const auto ours_h = replay::History(root / "history.tsv"), tf_h = replay::History(tf / "history.tsv");
      double     hd     = 0;
      for (std::size_t i = 0; i < std::min(ours_h.size(), tf_h.size()); ++i) hd = std::max(hd, std::abs(ours_h[i] - tf_h[i]));
      std::printf("  vs TF %-28s max |ours - TF| = %.3g\n", "history.tsv", hd);
      for (auto f : {"epoch3_arrays.npz", "model_arrays.npz", "outputs.npz", "final_gradients.npz"})
        Compare(tf, f, replay::ToMap(nn::load_npz((root / f).string())), o.verbose);
      std::printf("  samples: %.4f of pixels equal to TF's; completions %.4f\n",
                  Agreement(tf, "samples.npz", samples, "images"), Agreement(tf, "completions.npz", completions, "images"));
      check::Require(hd < 2e-3, "replayed validation history follows TF's");
    }
  }
  if (fs::exists(tf / "model_arrays.npz") && fs::exists(tf / "outputs.npz")) {
    auto fresh = make(false);
    replay::LoadInto(fresh->store, tf / "model_arrays.npz");
    const auto   out = fresh->Evaluate(data.binary[2], All(2));
    const double d   = Compare(tf, "outputs.npz", out, o.verbose);
    check::Require(close(d, 2e-5), "TF's final weights reproduce TF's test outputs");
    const auto s = fresh->Sample(32, spec.sample_seed, sample_classes);
    std::printf("  from TF's final weights: test loss %.8f (TF %.8f); samples %.4f of pixels equal to TF's\n",
                out.at("loss").item<float>(), replay::Get(nn::load_npz((tf / "outputs.npz").string()), "loss").item<float>(),
                Agreement(tf, "samples.npz", s, "images"));
  }
  std::cout << "training " << train_seconds << " s, wall "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << " s\nArtifacts:" << (root / "index.html").string()
            << std::endl;
  return nll;
}
}  // namespace bpix



int main(int argc, char** argv) {
  try {
    const auto o = bpix::ParseOptions(argc, argv, "pixel_cnn", paths::Dataset("saliency"));
    bpix::Spec s;
    s.name = "pixel_cnn";
    s.parameters = 1841, s.tensors = 10, s.masked = 808;
    s.batch_seed = 8419, s.epoch_sample_seed = 8423, s.sample_seed = 8423, s.completion_seed = 8443;
    s.title = "PixelCNN masked convolutions";
    s.tag   = "maskedCNN";
    s.html  = "<h1>PixelCNN: masked convolutions</h1><p><a href='https://arxiv.org/abs/1601.06759'>Pixel Recurrent Neural "
              "Networks / PixelCNN</a>, compact CPU variant: four 3x3 masked convolutions, width 8 / ReLU, first mask A (raw "
              "centre excluded), three hidden mask B (centre allowed), 1x1 Bernoulli output. 1841 stored parameters / 1033 "
              "active / 808 masked unused coordinates. Nominal 9x9 receptive field; the standard masked-convolution blind spot "
              "is acknowledged.</p><p>Real MNIST 1000 fit / 200 validation / 200 test, half-pixel bilinear 28->8 then threshold "
              ">= .5, BINARY targets. NLL = mean(softplus(logit) - target logit), bits per BINARY pixel. Three epochs, 96 Adam "
              ".003 updates, fixed final; fit-only Jeffreys independent Bernoulli baseline.</p>";
    bpix::Run(s, o, [](bool training) { return std::make_unique<bpix::PixelCnnModel>(training); });
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
