// CMD: just run-nn pixel_cnn_pp
// Compact PixelCNN++ likelihood core: actual 8-bit RGB discretized logistic
// mixtures, whole-pixel causality and channel-coupled sampling.
//
// Port of tflow's pixel_cnn_pp (common/ml/pixel_cnn_pp.h,
// common/nn/discretized_logistic.h, common/nn/gated_pixel_cnn.h ->
// examples/nn/common/{pixel_cnn_pp,gated_pixel_cnn}.hpp). Data: CIFAR-10
// (paths::Dataset("cifar10"): data_batch_1.bin rows 0..127 fit / 128..159
// validation, test_batch.bin rows 64..95 test), exact integer 4x4 area mean to
// 8x8, round-half-up 8-bit codes, code / 127.5 - 1. Gated whole-pixel context
// (3 width-4 blocks) with 30 mixture parameters per pixel (3 components),
// 1410 stored / 962 active parameters; three epochs of Adam .003, batch 8 in
// tflow's BatchOrder(128, 8, 8623).
//
// Checks (as tflow): native likelihood fixtures (finite extreme tails/edge
// gradients; all 256 bins of 3 components x 3 channels sum to 1); census;
// restore bit-exact; single/batch parameter parity; all 12288 whole-pixel
// parameter/input Jacobian entries with every current/future RGB entry
// exactly zero; 64 future-flip guards keep all prefix parameters bit-exact;
// seeded sampling repeat; 32-pixel prefix completions; immutable inference.
// Added: the likelihood fixtures (deterministic inputs) and, with init=auto,
// the whole training run are compared with TF's records; TF's final weights
// must reproduce TF's test outputs. Deviations: final.npz checkpoint; the
// uint8 records (original/codes) are stored as int32 (nn has no uint8); no
// independent_audit.json. Options: epochs=3 data=DIR out=DIR
// init=auto|tf|nn|PATH tf=DIR threads=4 verbose=0. Output: output/nn/models/pixel_cnn_pp.
#include "common/cifar.hpp"
#include "common/pixel_cnn_pp.hpp"

namespace {

using bpix::Floats;
using pcpp::Map;
using pcpp::Tensor;
namespace nn = datamunge::nn;
namespace fs = std::filesystem;

/// tflow's Algebra: likelihood fixtures with fixed inputs; returns max |ours - TF| (or -1).
double Algebra(const fs::path& root, const fs::path& tf, bool verbose) {
  double worst = -1;
  for (bool normalization : {true, false}) {
    const int          n = normalization ? 256 : 8;
    std::vector<float> images(static_cast<std::size_t>(n * 3)), parameters(static_cast<std::size_t>(n * 30));
    for (int i = 0; i < n; ++i) {
      for (int c = 0; c < 3; ++c) {
        int code = normalization ? i : (i * 37 + c * 113) % 256;
        if (!normalization && i < 2) code = i ? 255 : 0;
        images[i * 3 + c] = float(double(code) / 127.5 - 1);
      }
      for (int k = 0; k < 3; ++k)
        for (int j = 0; j < 10; ++j) {
          float value = normalization ? 0 : float(.3 * std::sin((i * 30 + k * 10 + j) * .37));
          if (j == 0 && normalization) value = k == 0 ? -80 : k == 1 ? 0 : 80;
          if (j >= 1 && j <= 3 && normalization) value = k == 0 ? -2 : k == 1 ? 0 : 2;
          if (j >= 4 && j <= 6) value = normalization ? (k == 0 ? -7 : k == 1 ? -.3f : 3.f) : (k == 0 ? -8 : k == 1 ? -.4f : 2.f);
          if (!normalization && j >= 4 && j <= 6) value += .13f * (j - 4);
          parameters[i * 30 + k * 10 + j] = value;
        }
    }
    const Tensor x = Floats(images, {n, 1, 1, 3});
    Tensor       p = Floats(parameters, {n, 1, 1, 30});
    p.requires_grad_(true);
    const auto   f        = pcpp::DiscretizedLogisticRGB(x, p);
    const Tensor loss     = nn::div(nn::neg(nn::mean(f.pixel_log_probability)), 3.0);
    const Tensor gradient = nn::grad(loss, {p})[0];
    Map          out{{"images", x},
                     {"parameters", p.detach()},
                     {"loss", loss.detach()},
                     {"pixel_log_probability", f.pixel_log_probability.detach()},
                     {"component_log_probability", f.component_log_probability.detach()},
                     {"mixture_log_probability", f.mixture_log_probability.detach()},
                     {"channel_log_mass", f.channel_log_mass.detach()},
                     {"gradient", gradient.detach()}};
    bpix::RequireFinite(out, "finite extreme tail/edge gradients");
    if (normalization) {
      const auto m = out.at("channel_log_mass").to_vector<float>();  // [256, 3 components, 3 channels]
      for (int k = 0; k < 3; ++k)
        for (int c = 0; c < 3; ++c) {
          double mass = 0;
          for (int i = 0; i < 256; ++i) mass += std::exp(double(m[(i * 3 + k) * 3 + c]));
          check::Near(mass, 1, 2e-5, "sum ALL 256 bins including tails is 1");
        }
    }
    const std::string file = normalization ? "normalization.npz" : "likelihood_fixture.npz";
    nn::save_npz((root / file).string(), replay::ToNamed(out));
    worst = std::max(worst, bpix::Compare(tf, file, out, verbose, true));
  }
  std::cout << "discretized logistic fixtures: finite tails/edges, all 256 bins x 3 components x 3 channels normalized PASS\n";
  return worst;
}

int Run(const bpix::Options& o) {
  const auto     started = std::chrono::steady_clock::now();
  const fs::path root = o.out, tf = o.tf;
  const auto     training = data::LoadCifar({o.data / "data_batch_1.bin"}, data::Cifar::k10);
  const auto     testing  = data::LoadCifar({o.data / "test_batch.bin"}, data::Cifar::k10);
  std::vector<Tensor>                    pools, codes;
  std::vector<std::vector<std::int32_t>> code_values;
  Map                                    records;
  for (int k = 0; k < 3; ++k) {
    const int  n = k == 0 ? 128 : 32, start = k == 0 ? 0 : k == 1 ? 128 : 64;
    const auto& source = k == 2 ? testing : training;
    const auto src = source.images.to_vector<std::int32_t>(), labels_all = source.labels.to_vector<std::int32_t>();
    std::vector<std::int32_t> raw(static_cast<std::size_t>(n * 3072)), quantized(static_cast<std::size_t>(n * 192)), ids(n), labels(n);
    std::vector<float>        images(static_cast<std::size_t>(n * 192));
    for (int b = 0; b < n; ++b) {
      std::copy_n(src.data() + (start + b) * 3072, 3072, raw.data() + b * 3072);
      ids[b]    = start + b + (k == 2 ? 50000 : 0);
      labels[b] = labels_all[start + b];
      for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x)
          for (int c = 0; c < 3; ++c) {
            int sum = 0;
            for (int dy = 0; dy < 4; ++dy)
              for (int dx = 0; dx < 4; ++dx) sum += raw[b * 3072 + ((y * 4 + dy) * 32 + x * 4 + dx) * 3 + c];
            const int code                      = (sum + 8) / 16;
            quantized[b * 192 + (y * 8 + x) * 3 + c] = code;
            images[b * 192 + (y * 8 + x) * 3 + c]    = float(double(code) / 127.5 - 1);
          }
    }
    pools.push_back(Floats(images, {n, 8, 8, 3}));
    codes.push_back(nn::from_vector(quantized, {n, 8, 8, 3}));
    code_values.push_back(quantized);
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    records[split + "_original"]   = nn::from_vector(raw, {n, 32, 32, 3});
    records[split + "_codes"]      = codes.back();
    records[split + "_images"]     = pools.back();
    records[split + "_source_ids"] = nn::from_vector(ids, {n});
    records[split + "_labels"]     = nn::from_vector(labels, {n});
  }
  nn::save_npz((root / "data.npz").string(), replay::ToNamed(records));
  if (fs::exists(tf / "data.npz")) {
    const auto theirs = replay::ToMap(nn::load_npz((tf / "data.npz").string()));
    for (const std::string s : {"fit", "validation", "test"})
      check::Require(bpix::Bits(theirs.at(s + "_images")) == bpix::Bits(records.at(s + "_images")) &&
                         bpix::Bits(nn::cast(theirs.at(s + "_codes"), nn::DType::Int32)) == bpix::Bits(records.at(s + "_codes")),
                     "8x8 RGB codes/images bit-identical to TF's data.npz");
    std::cout << "data: 8x8 RGB codes and images bit-identical to TF's data.npz\n";
  }
  std::ofstream protocol(root / "source_protocol.json");
  protocol << "{\"fit_rows\":[0,127],\"validation_rows\":[128,159],\"test_rows\":[64,95],\"test_source_id_offset\":50000,"
              "\"resize\":\"integer4x4area/round-half-up-8bit\",\"mixtures\":3,\"parameter_layout\":\"component-major:logit,"
              "meanRGB,logscaleRGB,coeffRG-BR-BG\"}\n";
  const double fixtures = Algebra(root, tf, o.verbose);
  if (fixtures >= 0) check::Require(fixtures <= 1e-6, "likelihood fixtures match TF's (relative)");

  pcpp::PixelCnnPPModel model;
  check::Require(model.store.count() == 1410 && model.store.size() == 23, "1410 params/962 active/448 masked/23 vars");
  const fs::path init = bpix::InitFile(o.init, tf);
  if (!init.empty()) {
    replay::LoadInto(model.store, init);
    model.Train();
    std::cout << "replaying TF's initial arrays " << init.string() << "\n";
  } else {
    std::cout << "weights from nn's generator (statistical parity with TF only)\n";
  }
  nn::save_npz((root / "initial_arrays.npz").string(), replay::ToNamed(model.Arrays()));
  nn::save_npz((root / "initial_outputs.npz").string(), replay::ToNamed(model.Evaluate(pools[2])));
  nn::BatchOrder order(128, 8, 8623);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_nll_per_RGB_channel\tvalidation_bits_per_RGB_channel\n";
  batches << "epoch\tstep\trow\n";
  int    steps = 0;
  Tensor first;
  for (int epoch = 1; epoch <= o.epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const Tensor batch = nn::take_rows(pools[0], rows);
      if (!steps) {
        first = batch;
        nn::save_npz((root / "first_batch.npz").string(), {{"images", batch}});
        nn::save_npz((root / "first_forward.npz").string(), replay::ToNamed(model.Evaluate(batch)));
        nn::save_npz((root / "initial_gradients.npz").string(), replay::ToNamed(model.Gradient(batch)));
      }
      model.Update(batch);
      if (!steps) nn::save_npz((root / "first_after.npz").string(), replay::ToNamed(model.Arrays()));
      for (auto row : rows) batches << epoch << '\t' << steps << '\t' << row << '\n';
      ++steps;
    }
    const double nll = model.Evaluate(pools[1]).at("loss").item<float>();
    history << epoch << '\t' << nll << '\t' << nll / std::log(2.) << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), replay::ToNamed(model.Arrays()));
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_samples.npz")).string(), replay::ToNamed(model.Sample(8, 8639)));
    std::cout << "epoch" << epoch << ":validationbits/8bitRGBchannel=" << nll / std::log(2.) << std::endl;
  }
  const double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.store.save((root / "final.npz").string());
  nn::save_npz((root / "final_gradients.npz").string(), replay::ToNamed(model.Gradient(first)));
  pcpp::PixelCnnPPModel deployed(false);
  deployed.store.load((root / "final.npz").string(), true);
  check::Require(bpix::SameBits(model.Arrays(), deployed.Arrays()), "all RGB mixture weights restore BITEXACT");
  const auto before = deployed.Arrays();
  const auto out    = deployed.Evaluate(pools[2]);
  nn::save_npz((root / "outputs.npz").string(), replay::ToNamed(out));
  nn::save_npz((root / "model_arrays.npz").string(), replay::ToNamed(deployed.Arrays()));
  const Tensor single = nn::take_rows(pools[2], {0}), head = deployed.Head(single);
  const auto   hv = head.to_vector<float>(), ov = out.at("parameters").to_vector<float>();
  double       parity = 0;
  for (int j = 0; j < 1920; ++j) parity = std::max(parity, std::abs(double(hv[j]) - ov[j]));
  check::Require(parity < 2e-6, "single/batch RGB parameter parity");
  std::vector<float> jacobian(64 * 192);
  const auto         sv = single.to_vector<float>();
  for (int site = 0; site < 64; ++site) {
    const auto grad = deployed.CausalGradient(single, site).to_vector<float>();
    for (int j = 0; j < 192; ++j) {
      jacobian[site * 192 + j] = grad[j];
      if (j / 3 >= site) check::Require(grad[j] == 0, "all WHOLE current/future RGB Jacobian exact ZERO");
    }
    auto changed = sv;
    for (int j = site * 3; j < 192; ++j) changed[j] = -changed[j];
    const auto control = deployed.Head(Floats(changed, single.shape())).to_vector<float>();
    for (int j = 0; j < (site + 1) * 30; ++j)
      check::Require(control[j] == hv[j], "WHOLE PIXEL prefix: all 30 parameters invariant to future flips");
  }
  nn::save_npz((root / "causality.npz").string(), {{"images", single}, {"parameters", head}, {"jacobian", Floats(jacobian, {64, 64, 3})}});
  const auto samples = deployed.Sample(24, 8639);
  nn::save_npz((root / "samples.npz").string(), replay::ToNamed(samples));
  check::Require(bpix::SameBits(samples, deployed.Sample(24, 8639)), "repeat all RGB draws/components/parameters/uniforms BITEXACT");
  const Tensor prefix     = nn::take_rows(pools[2], {0, 1, 2, 3, 4, 5});
  auto         completion = deployed.Sample(6, 8641, &prefix, 32);
  completion["truth"]     = prefix;
  const auto px = prefix.to_vector<float>(), cx = completion.at("images").to_vector<float>();
  for (int b = 0; b < 6; ++b)
    for (int j = 0; j < 96; ++j) check::Require(px[b * 192 + j] == cx[b * 192 + j], "RGB completion prefix preserved");
  nn::save_npz((root / "completions.npz").string(), replay::ToNamed(completion));
  std::vector<float> gallery(18 * 192);
  for (int b = 0; b < 6; ++b)
    for (int j = 0; j < 192; ++j) {
      gallery[(b * 3) * 192 + j]     = px[b * 192 + j];
      gallery[(b * 3 + 1) * 192 + j] = j < 96 ? px[b * 192 + j] : 0;
      gallery[(b * 3 + 2) * 192 + j] = cx[b * 192 + j];
    }
  const Tensor gallery_t = Floats(gallery, {18, 8, 8, 3});
  nn::save_npz((root / "completion_gallery.npz").string(), {{"images", gallery_t}});
  bpix::Png(root / "samples.png", samples.at("images"), 6, 1);
  bpix::Png(root / "completions.png", gallery_t, 3, 1);
  // Fit-only per-channel 256-bin Jeffreys histogram baseline.
  std::vector<double> baseline(768, .5);
  for (std::size_t j = 0; j < code_values[0].size(); ++j) baseline[(j % 3) * 256 + code_values[0][j]] += 1;
  for (auto& v : baseline) v /= 128 * 64 + 128;
  double base_nll = 0;
  for (std::size_t j = 0; j < code_values[2].size(); ++j) base_nll -= std::log(baseline[(j % 3) * 256 + code_values[2][j]]);
  base_nll /= double(code_values[2].size());
  nn::save_npz((root / "baseline.npz").string(), {{"probability", nn::from_vector(baseline, {3, 256})}});
  const double nll = out.at("loss").item<float>();
  check::Require(bpix::SameBits(before, deployed.Arrays()), "likelihood-Jacobian-sampling-completion ALL immutable");
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":1410,\"active_parameters\":962,\"masked_unused_coordinates\":448,\"mixtures\":3,\"epochs\":" << o.epochs
       << ",\"optimizer_updates\":" << steps << ",\"selected_epoch_fixed_final\":" << o.epochs << ",\"test_nll_per_RGB_channel\":" << nll
       << ",\"test_bits_per_RGB_channel\":" << nll / std::log(2.) << ",\"fit_channel_histogram_bits\":" << base_nll / std::log(2.)
       << ",\"uniform_bits_per_RGB_channel\":8,\"single_batch_max_error\":" << parity
       << ",\"causal_jacobian_entries\":12288,\"train_seconds\":" << train_seconds << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>PixelCNN++ likelihood core</title><style>body{font:16px system-ui;margin:30px;"
          "max-width:1000px}img{max-width:100%;image-rendering:pixelated}</style><h1>PixelCNN++: exact RGB logistic mixture core</h1>"
          "<p><a href='https://arxiv.org/abs/1701.05517'>PixelCNN++</a> compact teaching variant: discretized 3-component logistic "
          "mixture with RGB channel coupling and a whole-pixel causal dual-stream gated context, 1410 params / 962 active / 448 "
          "masked, three width-4 blocks; no multiscale architecture or dropout. Class labels unused.</p><p>Real CIFAR 32 -> 8x8 "
          "via exact integer 4x4 area mean, half-up rounding to 8-bit codes, code / 127.5 - 1; 128 fit (train rows 0..127), 32 "
          "validation (128..159), 32 test (test rows 64..95). Three epochs, 48 Adam .003 updates, fixed final. Bits per 8-bit RGB "
          "CHANNEL; fit-only per-channel 256-bin Jeffreys histogram and 8-bit uniform baselines.</p><h2>First 24 unfiltered "
          "fixed-seed samples</h2><img src='samples.png'><h2>First 6 held-out completions</h2><p>TRUE 8-bit RGB / observed top "
          "half (gray unknown) / ancestral completion.</p><img src='completions.png'><p>See metrics.json. Short training may "
          "produce noisy colors; no natural-image quality or benchmark claim.</p>";
  check::Require(bool(meta) && bool(protocol) && bool(history) && bool(batches) && bool(html), "complete PixelCNN++ records");
  std::cout << "test bits/8bitRGBchannel=" << nll / std::log(2.) << ", fitting histogram baseline=" << base_nll / std::log(2.)
            << ", uniform 8; mixture/edges/normalization/coupling/whole-pixel causality/restore/ancestral/immutable PASS\n";

  if (!init.empty() && init == tf / "initial_arrays.npz") {
    std::cout << "replay vs TF's recorded run (" << tf.string() << "):\n";
    double d = 0;
    for (auto f : {"initial_arrays.npz", "initial_outputs.npz", "first_batch.npz", "first_forward.npz", "initial_gradients.npz"})
      d = std::max(d, bpix::Compare(tf, f, replay::ToMap(nn::load_npz((root / f).string())), o.verbose, true));
    check::Require(d >= 0 && d <= 2e-6, "replayed initial outputs/first forward/gradients match TF (relative)");
    const double after = bpix::Compare(tf, "first_after.npz", replay::ToMap(nn::load_npz((root / "first_after.npz").string())), o.verbose, true);
    check::Require(after >= 0 && after <= 1e-4, "first Adam update matches TF");
    if (o.epochs == 3) {
      const auto ours_h = replay::History(root / "history.tsv"), tf_h = replay::History(tf / "history.tsv");
      double     hd     = 0;
      for (std::size_t i = 0; i < std::min(ours_h.size(), tf_h.size()); ++i) hd = std::max(hd, std::abs(ours_h[i] - tf_h[i]));
      std::printf("  vs TF %-28s max |ours - TF| = %.3g\n", "history.tsv", hd);
      for (auto f : {"epoch3_arrays.npz", "model_arrays.npz", "outputs.npz", "final_gradients.npz"})
        bpix::Compare(tf, f, replay::ToMap(nn::load_npz((root / f).string())), o.verbose, true);
      std::printf("  samples: %.4f of channel codes equal to TF's; completions %.4f\n", bpix::Agreement(tf, "samples.npz", samples, "images"),
                  bpix::Agreement(tf, "completions.npz", completion, "images"));
      check::Require(hd < 2e-3, "replayed validation history follows TF's");
    }
  }
  if (fs::exists(tf / "model_arrays.npz") && fs::exists(tf / "outputs.npz")) {
    pcpp::PixelCnnPPModel fresh(false);
    replay::LoadInto(fresh.store, tf / "model_arrays.npz");
    const auto   tf_out = fresh.Evaluate(pools[2]);
    const double d      = bpix::Compare(tf, "outputs.npz", tf_out, o.verbose, true);
    check::Require(d >= 0 && d <= 2e-6, "TF's final weights reproduce TF's test outputs (relative)");
    std::printf("  from TF's final weights: test loss %.8f (TF %.8f); samples %.4f of channel codes equal to TF's\n",
                tf_out.at("loss").item<float>(), replay::Get(nn::load_npz((tf / "outputs.npz").string()), "loss").item<float>(),
                bpix::Agreement(tf, "samples.npz", fresh.Sample(24, 8639), "images"));
  }
  std::cout << "training " << train_seconds << " s, wall "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << " s\nArtifacts:" << (root / "index.html").string()
            << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return Run(bpix::ParseOptions(argc, argv, "pixel_cnn_pp", paths::Dataset("cifar10")));
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
