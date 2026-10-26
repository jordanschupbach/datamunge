// CMD: just run-nn infogan
// InfoGAN (Chen et al. 2016): unsupervised categorical/continuous latent codes on
// 1000 MNIST digits, with separate D -> Q -> G updates.
//
// Port of tflow's infogan. Preparation (in ~/projects/tflow_examples): just
// autoencoder-assets; the data are read from $DATAMUNGE_DATA_DIR/saliency
// (saliency_data.npz). Compact MLP teaching variant with a SEPARATE image-only
// Q network instead of shared D features; no real digit label enters training.
// G: Gaussian 16 + one-hot uniform category 10 + two uniform[-1, 1] codes ->
// 32 tanh -> 784 sigmoid (26800 parameters); D: 784 -> 32 tanh -> 1 logit
// (25153); Q: 784 -> 32 tanh -> 10 category logits + 2 Gaussian means with
// unit variance (25516). D: logistic source loss; Q: categorical CE + Gaussian
// NLL of the generated-prior codes; G: non-saturating source loss + the same
// recognition loss (lambda 1). MI lower bound = log 10 + 2 log 2 - recognition
// loss. Separate Adam (.0002, beta1 .5) per group; per batch D, then Q, then G,
// each recomputed after the previous write. Three default epochs of 32-row
// batches (96 steps per group), fixed final epoch.
//
// Noise, prior codes and batch order are tflow's exactly; weights use
// datamunge's initializer unless init=PATH names a tflow initial_arrays.npz.
// Traversals (category and both continuous codes) are structural pixel-effect
// checks, not semantic disentanglement claims. tflow's per-step .npz dumps and
// index.html are not written. Outputs go to output/nn/models/infogan.
// Options: epochs=3 data=PATH out=PATH init=PATH.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include "common/infogan.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

double Value(const gan::Arrays& a, const std::string& key) { return a.at(key).item(); }

void SavePrior(const std::filesystem::path& path, const gan::InfoCodes& codes, const Tensor& z) {
  gan::SaveArrays(path.string(), {{"category", codes.category}, {"continuous", codes.continuous}, {"raw", codes.raw}, {"z", z}});
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  std::filesystem::create_directories(root);
  const auto   pools = gan::LoadSaliency(data);
  const Tensor vn = gan::Gaussian(200, 16, 5323), tn = gan::Gaussian(200, 16, 5333);
  const auto   vc = gan::InfoPrior(200, 5347), tc = gan::InfoPrior(200, 5351);
  SavePrior(root / "validation_prior.npz", vc, vn);
  SavePrior(root / "test_prior.npz", tc, tn);
  gan::InfoGanModel model;
  if (!init.empty()) model.Load(init);
  check::Require(model.Count(0) == 26800 && model.Count(1) == 25153 && model.Count(2) == 25516 &&
                     model.Group(0).size() + model.Group(1).size() + model.Group(2).size() == 14,
                 "77469 parameters 14 tensors");
  model.Save((root / "initial_arrays.npz").string());
  gan::SaveArrays((root / "initial_outputs.npz").string(),
                  model.Evaluate(pools[2].images, tn, tc.category, tc.continuous));
  nn::BatchOrder order(1000, 32, 5369);
  int            steps = 0;
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tdiscriminator_loss\tgenerator_loss\trecognition_loss\tmi_lower_bound\tpixel_mmd\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto source  = nn::cast(pools[0].source, nn::DType::Int64).to_vector<std::int64_t>();
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    const Tensor noise = gan::Gaussian(1000, 16, 5363 + epoch);
    const auto   codes = gan::InfoPrior(1000, 5381 + epoch);
    order.begin_epoch();
    int seen = 0;
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const Tensor x = gan::Rows(pools[0].images, rows), z = gan::Rows(noise, rows),
                   c = gan::Rows(codes.category, rows), u = gan::Rows(codes.continuous, rows);
      if (!steps) gan::SaveArrays((root / "first_forward.npz").string(), model.Evaluate(x, z, c, u));
      for (int group : {1, 2, 0}) {
        const auto before = steps ? gan::Arrays{} : model.AllArrays();
        model.Update(group, x, z, c, u);
        if (!steps)
          gan::RequireOnlyOwnersChanged(before, model.AllArrays(), {std::to_string(group) + "/"},
                                        "only intended group updated");
      }
      for (auto row : rows) batches << epoch << '\t' << steps << '\t' << row << '\t' << source[row] << '\n';
      ++steps;
      seen += int(rows.size());
    }
    check::Require(seen == 1000, "all 1000 fit cases once per epoch");
    const auto v = model.Evaluate(pools[1].images, vn, vc.category, vc.continuous);
    history << epoch;
    for (auto key : {"discriminator_loss", "generator_loss", "recognition_loss", "mi_lower_bound", "pixel_mmd"})
      history << '\t' << Value(v, key);
    history << '\n';
    std::cout << "epoch " << epoch << ": D=" << Value(v, "discriminator_loss") << ", G=" << Value(v, "generator_loss")
              << ", recognition=" << Value(v, "recognition_loss") << ", MI bound=" << Value(v, "mi_lower_bound")
              << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

  const auto checkpoint = (root / "final.npz").string();
  model.Save(checkpoint);
  gan::InfoGanModel deployed(false, {1, 2, 3});
  deployed.Load(checkpoint);
  check::Require(gan::BitEqual(model.AllArrays(), deployed.AllArrays()), "all groups restore exact");
  const auto   before = deployed.AllArrays();
  const auto   output = deployed.Evaluate(pools[2].images, tn, tc.category, tc.continuous);
  const Tensor repeat = deployed.Sample(tn, tc.category, tc.continuous);
  check::Require(gan::BitEqual(repeat, output.at("fake")), "noise+codes ONLY generation exact");
  const auto recognition = deployed.Recognize(repeat);
  for (auto key : {"q_probability", "q_mean"})
    check::Require(gan::BitEqual(recognition.at(key), output.at(key)), "Q image ONLY without targets/noise/code exact");
  const double parity = gan::MaxAbsDiff(
      deployed.Sample(gan::Rows(tn, {0}), gan::Rows(tc.category, {0}), gan::Rows(tc.continuous, {0})), gan::Rows(repeat, {0}));
  check::Require(parity < 2e-6, "single batch parity");
  gan::SaveArrays((root / "outputs.npz").string(), output);
  gan::SaveArrays((root / "image_only_recognition.npz").string(), recognition);

  // Codes are sampled prior categories, not digit labels. Traversals include all outputs.
  std::vector<std::int64_t> grid_rows;
  std::vector<std::int32_t> grid_category;
  for (int c = 0; c < 10; ++c)
    for (int b = 0; b < 10; ++b) grid_rows.push_back(b), grid_category.push_back(c);
  const Tensor gz = gan::Rows(tn, grid_rows), gc = nn::from_vector(grid_category, {100}), gu = nn::zeros({100, 2});
  const Tensor grid = deployed.Sample(gz, gc, gu);
  const double cat_effect =
      gan::MaxAbsDiff(nn::slice_axis(grid, 0, 10, 100), nn::tile(nn::slice_axis(grid, 0, 0, 10), {9, 1}));
  gan::SaveArrays((root / "category_traversal.npz").string(), {{"z", gz}, {"category", gc}, {"continuous", gu}, {"fake", grid}});
  gan::Mosaic(root / "category_traversal.png", grid, 10);
  double cont_effect = 0;
  for (int axis = 0; axis < 2; ++axis) {
    std::vector<std::int64_t> rows;
    std::vector<float>        u(55 * 2, 0.f);
    for (int b = 0; b < 5; ++b)
      for (int k = 0; k < 11; ++k) rows.push_back(b), u[(b * 11 + k) * 2 + axis] = -1.f + .2f * k;
    const Tensor z = gan::Rows(tn, rows), c = nn::zeros({55}, nn::DType::Int32), uu = nn::from_vector(u, {55, 2});
    const Tensor images = deployed.Sample(z, c, uu);
    for (int b = 0; b < 5; ++b)
      cont_effect = std::max(cont_effect, gan::MaxAbsDiff(gan::Rows(images, {b * 11 + 10}), gan::Rows(images, {b * 11})));
    const auto name = "continuous" + std::to_string(axis);
    gan::SaveArrays((root / (name + "_traversal.npz")).string(), {{"z", z}, {"category", c}, {"continuous", uu}, {"fake", images}});
    gan::Mosaic(root / (name + "_traversal.png"), images, 11);
  }
  check::Require(cat_effect > 1e-5 && cont_effect > 1e-5, "both generated code paths affect pixels, not semantic certification");
  int  invalid = 0;
  auto bad_c   = grid_category;
  for (int k : {-1, 10}) {
    bad_c[0] = k;
    try {
      deployed.Sample(gz, nn::from_vector(bad_c, {100}), gu);
    } catch (const std::exception&) {
      ++invalid;
    }
  }
  std::vector<float> bad_u(200, 0.f);
  bad_u[0] = 2;
  try {
    deployed.Sample(gz, gc, nn::from_vector(bad_u, {100, 2}));
  } catch (const std::exception&) {
    ++invalid;
  }
  check::Require(invalid == 3, "invalid codes rejected");
  check::Require(gan::BitEqual(before, deployed.AllArrays()), "all inference immutable");
  gan::Mosaic(root / "generated.png", nn::slice_axis(repeat, 0, 0, 40), 8);

  const auto best  = nn::cast(nn::argmax(output.at("q_probability"), 1), nn::DType::Int64).to_vector<std::int64_t>();
  const auto truth = nn::cast(tc.category, nn::DType::Int64).to_vector<std::int64_t>();
  double     accuracy = 0;
  for (int b = 0; b < 200; ++b) accuracy += best[b] == truth[b];
  const double mse = nn::mean(nn::square(nn::cast(tc.continuous, nn::DType::Float64) -
                                         nn::cast(output.at("q_mean"), nn::DType::Float64)))
                         .item();
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":77469,\"epochs\":" << epochs << ",\"steps_per_group\":" << steps
       << ",\"optimizer_updates\":" << 3 * steps << ",\"selected_epoch_fixed_final\":" << epochs
       << ",\"generated_category_recognition_accuracy\":" << accuracy / 200
       << ",\"generated_continuous_recognition_mse\":" << mse << ",\"category_pixel_max_effect\":" << cat_effect
       << ",\"continuous_pixel_max_effect\":" << cont_effect << ",\"single_batch_max_error\":" << parity;
  for (auto key : {"discriminator_loss", "generator_loss", "recognition_loss", "categorical_loss", "continuous_loss",
                   "mi_lower_bound", "pixel_mmd"})
    meta << ",\"test_" << key << "\":" << Value(output, key);
  meta << ",\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "complete artifacts");
  std::cout << "fixed final=" << epochs << ", category accuracy=" << accuracy / 200 << ", continuous MSE=" << mse
            << ", MI bound=" << Value(output, "mi_lower_bound")
            << "; D then Q then G/restore/imageOnlyQ/codeOnlyG/immutable PASS (training " << seconds
            << " s)\nArtifacts: " << root.string() << "\nAll infogan checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset("saliency"), out = paths::OutputRoot() / "models" / "infogan";
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
