// CMD: just run-nn pix2pix
// Pix2Pix (Isola et al. 2017): paired MNIST Sobel edges -> grayscale digits
// with a tiny U-Net generator and a conditional 18x18 PatchGAN.
//
// Port of tflow's pix2pix. Preparation (in ~/projects/tflow_examples): just
// autoencoder-assets; the data are read from $DATAMUNGE_DATA_DIR/saliency
// (saliency_data.npz: 1000 fit / 200 validation / 200 test balanced digits).
// Fixed source transform: integer pixels recovered from float/255, zero-padded
// 3x3 Sobel, edge = 1 if |gx| + |gy| >= 256. Targets are the grayscale pixels.
// Two-level U-Net G with real skip concatenations and learned stride-2
// transposed convolutions (28 -> 14x14x4 -> 7x7x8 -> 14x14x4 ++ skip -> 28x28x2
// ++ source -> 3x3 sigmoid), 1390 parameters. Conditional PatchGAN D on
// concat(source, target): 4k4s2 -> 8k4s2 -> 1k3s1 raw 7x7 logits, 725
// parameters. D = mean patch logistic CE (real/fake), G = non-saturating patch
// CE + 100 mean L1 to the paired target. Separate Adam (.0002, beta1 .5); per
// batch a D step THEN the G step through the UPDATED D. Three default epochs
// of 32-row batches (96 steps each), fixed final checkpoint.
//
// Data and batch order are tflow's exactly; weights use datamunge's
// initializer unless init=PATH names a tflow initial_arrays.npz (then the run
// follows TF's trajectory up to float rounding). tflow's per-step .npz dumps
// (first_*, epoch*_arrays/preview, initial/final gradients) and index.html are
// not written. Outputs (data.npz, history, batches, initial/final checkpoint
// .npz, outputs.npz, condition_controls.npz, gallery.npz, paired_examples.png,
// metrics.json) go to output/nn/models/pix2pix.
// Options: epochs=3 data=PATH out=PATH init=PATH.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include "common/paths.hpp"
#include "common/pix2pix.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

double Value(const gan::Arrays& a, const std::string& key) { return a.at(key).item(); }

/// tflow Edges: binary zero-padded Sobel magnitude (|gx| + |gy| >= 256) of 0..255 pixels.
Tensor Edges(const Tensor& target) {
  const auto         in = target.contiguous().to_vector<float>();
  std::vector<float> out(in.size(), 0.f);
  const int          sx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}, sy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
  for (std::int64_t b = 0; b < target.size(0); ++b)
    for (int y = 0; y < 28; ++y)
      for (int x = 0; x < 28; ++x) {
        int gx = 0, gy = 0;
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            const int yy = y + dy, xx = x + dx;
            const int v  = yy < 0 || yy >= 28 || xx < 0 || xx >= 28
                               ? 0
                               : int(std::floor(double(in[b * 784 + yy * 28 + xx]) * 255 + .5));
            gx += sx[dy + 1][dx + 1] * v;
            gy += sy[dy + 1][dx + 1] * v;
          }
        out[b * 784 + y * 28 + x] = std::abs(gx) + std::abs(gy) >= 256 ? 1.f : 0.f;
      }
  return nn::from_vector(out, target.shape());
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  const auto pools = gan::LoadSaliency(data);
  std::filesystem::create_directories(root);
  std::vector<Tensor> edges;
  gan::Arrays         source;
  for (int k = 0; k < 3; ++k) {
    edges.push_back(Edges(pools[k].images));
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    source.emplace(split + "_images", pools[k].images);
    source.emplace(split + "_edges", edges.back());
    source.emplace(split + "_labels", pools[k].labels);
    source.emplace(split + "_source_ids", pools[k].source);
  }
  gan::SaveArrays((root / "data.npz").string(), source);
  gan::Pix2PixModel model;
  if (!init.empty()) model.Load(init);
  check::Require(model.Count(0) == 1390 && model.Count(1) == 725 && model.Group(0).size() + model.Group(1).size() == 16,
                 "2115 trainable parameters, 16 tensors");
  model.Save((root / "initial_arrays.npz").string());
  gan::SaveArrays((root / "initial_outputs.npz").string(), model.Evaluate(pools[2].images, edges[2]));
  nn::BatchOrder order(1000, 32, 6121);
  int            steps = 0;
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tdiscriminator_loss\tgenerator_loss\tg_adversarial\tl1\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto source_ids = nn::cast(pools[0].source, nn::DType::Int64).to_vector<std::int64_t>();
  Tensor     first_x, first_c;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const Tensor x = gan::Rows(pools[0].images, rows), c = gan::Rows(edges[0], rows);
      if (!steps) {
        first_x = x, first_c = c;
        gan::SaveArrays((root / "first_batch.npz").string(), {{"target", x}, {"source", c}});
      }
      for (int group : {1, 0}) {
        const auto before = model.AllArrays();
        model.Update(group, x, c);
        gan::RequireOnlyOwnersChanged(before, model.AllArrays(), {std::to_string(group) + "/"},
                                      "Pix2Pix step changes ONLY own group", !steps);
      }
      for (auto row : rows) batches << epoch << '\t' << steps << '\t' << row << '\t' << source_ids[row] << '\n';
      ++steps;
    }
    const auto v = model.Evaluate(pools[1].images, edges[1]);
    history << epoch;
    for (auto key : {"discriminator_loss", "generator_loss", "g_adversarial", "l1"}) history << '\t' << Value(v, key);
    history << '\n';
    std::cout << "epoch " << epoch << ": D=" << Value(v, "discriminator_loss") << ", G=" << Value(v, "generator_loss")
              << ", paired L1=" << Value(v, "l1") << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

  const auto checkpoint = (root / "final.npz").string();
  model.Save(checkpoint);
  gan::Pix2PixModel deployed(false, {1, 2});
  deployed.Load(checkpoint);
  check::Require(gan::BitEqual(model.AllArrays(), deployed.AllArrays()), "all G/D vars restore exact");
  for (int group = 0; group < 2; ++group)
    gan::SaveArrays((root / ("final_gradients_group" + std::to_string(group) + ".npz")).string(),
                    model.Gradient(group, first_x, first_c));
  const auto   before = deployed.AllArrays();
  const auto   output = deployed.Evaluate(pools[2].images, edges[2]);
  const Tensor repeat = deployed.Generate(edges[2]);
  check::Require(gan::BitEqual(repeat, output.at("fake")), "SOURCE ONLY translation without TARGET exact");
  const double parity = gan::MaxAbsDiff(deployed.Generate(gan::Rows(edges[2], {0})), gan::Rows(repeat, {0}));
  check::Require(parity < 2e-6, "single/batch parity");
  const Tensor zeros = nn::zeros(edges[2].shape());
  const Tensor blank = deployed.Generate(zeros), wrong_pair = deployed.ScorePair(zeros, pools[2].images),
               pair = deployed.ScorePair(edges[2], pools[2].images);
  check::Require(gan::BitEqual(pair, output.at("real_logit")), "source+target ONLY PatchD parity");
  const double g_effect = gan::MaxAbsDiff(repeat, blank), d_effect = gan::MaxAbsDiff(pair, wrong_pair);
  check::Require(g_effect > 1e-6 && d_effect > 1e-6, "BOTH G and D depend on SOURCE");
  gan::SaveArrays((root / "condition_controls.npz").string(),
                  {{"zero_source", zeros}, {"zero_source_prediction", blank}, {"zero_source_real_logits", wrong_pair}});
  check::Require(gan::BitEqual(before, deployed.AllArrays()), "immutable Pix2Pix inference");
  gan::SaveArrays((root / "outputs.npz").string(), output);
  const Tensor baseline = nn::reshape(gan::FitMean(pools[0].images, 200), {200, 28, 28, 1});
  gan::SaveArrays((root / "fit_mean_baseline.npz").string(), {{"image", baseline}});
  const auto pred = nn::cast(repeat, nn::DType::Float64), base = nn::cast(baseline, nn::DType::Float64),
             truth = nn::cast(pools[2].images, nn::DType::Float64);
  const double mse = nn::mean(nn::square(pred - truth)).item(), l1 = nn::mean(nn::abs(pred - truth)).item();
  const double base_mse = nn::mean(nn::square(base - truth)).item(), base_l1 = nn::mean(nn::abs(base - truth)).item();
  std::vector<Tensor> gallery_rows;
  for (std::int64_t b = 0; b < 10; ++b)
    for (const Tensor* t : std::vector<const Tensor*>{&edges[2], &pools[2].images, &repeat, &baseline})
      gallery_rows.push_back(gan::Flat784(gan::Rows(*t, {b})));
  const Tensor gallery = nn::concat(gallery_rows, 0);
  gan::SaveArrays((root / "gallery.npz").string(), {{"images", gallery}});
  gan::Mosaic(root / "paired_examples.png", gallery, 4);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":2115,\"epochs\":" << epochs << ",\"steps_per_group\":" << steps
       << ",\"optimizer_updates\":" << 2 * steps << ",\"selected_epoch_fixed_final\":" << epochs
       << ",\"test_l1\":" << l1 << ",\"test_mse\":" << mse << ",\"fit_mean_l1\":" << base_l1
       << ",\"fit_mean_mse\":" << base_mse << ",\"source_generator_max_effect\":" << g_effect
       << ",\"source_discriminator_max_effect\":" << d_effect << ",\"single_batch_max_error\":" << parity;
  for (auto key : {"discriminator_loss", "generator_loss", "g_adversarial"})
    meta << ",\"test_" << key << "\":" << Value(output, key);
  meta << ",\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "complete Pix2Pix records");
  std::cout << "fixed final=" << epochs << ", test L1=" << l1 << ", MSE=" << mse << ", mean baseline L1/MSE=" << base_l1
            << "/" << base_mse << ", test D/G=" << Value(output, "discriminator_loss") << "/"
            << Value(output, "generator_loss") << "; source only/conditional PatchGAN/restore/immutable PASS (training "
            << seconds << " s)\nArtifacts: " << root.string() << "\nAll pix2pix checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset("saliency"), out = paths::OutputRoot() / "models" / "pix2pix";
    std::string           init;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0)
        epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0)
        data = a.substr(5);
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else if (a.rfind("init=", 0) == 0)
        init = a.substr(5);
      else
        throw std::invalid_argument("expected epochs=N data=PATH out=PATH init=PATH");
    }
    check::Require(epochs > 0 && epochs <= 100, "three default or explicit 1..100 epochs");
    Run(data, out, epochs, init);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
