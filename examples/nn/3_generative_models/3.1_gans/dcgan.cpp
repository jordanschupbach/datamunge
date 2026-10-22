// CMD: just run-nn dcgan
// A compact DCGAN (Radford et al. 2016) on 1000 MNIST digits.
//
// Port of tflow's dcgan. Preparation (in ~/projects/tflow_examples): just
// autoencoder-assets; the data are read from $DATAMUNGE_DATA_DIR/saliency
// (saliency_data.npz: 1000 fit / 200 validation / 200 test balanced digits).
// G: Gaussian z (16) -> 7x7x8 projection/BN/ReLU -> 4x4 stride-2 TRANSPOSED
// conv 14x14x4/BN/ReLU -> 4x4 stride-2 TRANSPOSED conv 28x28x1, tanh.
// D: 4x4 stride-2 conv 4/Leaky .2 -> 4x4 stride-2 conv 8/BN/Leaky .2 -> 392
// -> 1 raw logit. G 6873, D 989 parameters; kernels N(0, .02), BN momentum
// .9, eps .001. Real pixels map 0..1 to -1..1. D minimizes mean
// softplus(-real logit) + mean softplus(fake logit); the non-saturating G
// minimizes mean softplus(-fake logit). Separate Adam (.0002, beta1 .5) for G
// and D. Per batch: D step on concat(real, fake) (joint BN statistics, only
// D's moving statistics updated), THEN the G step through the UPDATED D
// (G batch statistics with G's update; D inference BN). Three default
// epochs of 32-row batches (96 steps each), fixed final epoch, no labels.
//
// Validation/test D and G losses and a coarse 4x4-pixel IMQ MMD are game
// diagnostics, not image-quality measures. Noise banks and batch order are
// tflow's exactly; weights use datamunge's initializer unless init=PATH names
// a tflow initial_arrays.npz. Outputs (history, checkpoint .npz, sample
// sheets generated.png / fit_mean.png, metrics.json) go to
// output/nn/models/dcgan. Options: epochs=3 data=PATH out=PATH init=PATH.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include "common/gan.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

/// Logistic GAN algebra at extreme logits: finite losses and gradients, and
/// a strong non-saturating G gradient when D is confident a sample is fake.
void Algebra(const std::filesystem::path& root) {
  Tensor     real = nn::tensor({-80, -2, 2, 80}, {4, 1}), fake = nn::tensor({80, 2, -2, -80}, {4, 1});
  real.requires_grad_();
  fake.requires_grad_();
  const auto loss = gan::LogisticGanObjectives(real, fake);
  const auto dg   = nn::grad(loss.discriminator, {real, fake});
  const auto gg   = nn::grad(loss.generator, {fake});
  const gan::Arrays out{{"discriminator_loss", loss.discriminator.detach()},
                        {"generator_loss", loss.generator.detach()},
                        {"d_real_gradient", dg[0]},
                        {"d_fake_gradient", dg[1]},
                        {"g_fake_gradient", gg[0]}};
  gan::RequireFinite(out, "extreme logits finite GAN loss/gradients");
  check::Require(gg[0].at({3, 0}) < -.24, "non-saturating G has strong gradient when D confident fake");
  std::cout << "logistic algebra: D loss=" << loss.discriminator.item() << ", G loss=" << loss.generator.item()
            << ", dD/dreal=" << dg[0].values_str() << ", dG/dfake=" << gg[0].values_str() << '\n';
  auto saved = out;
  saved.emplace("real", real.detach());
  saved.emplace("fake", fake.detach());
  gan::SaveArrays((root / "logistic_algebra.npz").string(), saved);
}

double Value(const gan::Arrays& a, const std::string& key) { return a.at(key).item(); }

/// The fit-pool mean image repeated n times, [n, 784].
Tensor FitMean(const Tensor& fit, std::int64_t n) {
  return nn::tile(nn::mean(nn::reshape(fit, {-1, 784}), {0}, true), {n, 1});
}

void SaveSheet(const std::filesystem::path& path, const Tensor& images, std::int64_t cols) {
  // tflow's Mosaic: 3x pixels, 4-pixel gaps of gray 220.
  nn::save_image_grid(nn::reshape(images, {-1, 28, 28, 1}), path.string(), cols, 0, 1, 4, 220 / 255., 3);
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  const auto pools = gan::LoadSaliency(data);
  std::filesystem::create_directories(root);
  Algebra(root);
  const Tensor validation_z = gan::Gaussian(200, 16, 4823), test_z = gan::Gaussian(200, 16, 4831);
  gan::DcganModel model;
  if (!init.empty()) model.Load(init);
  check::Require(model.generator.count() == 6873 && model.discriminator.count() == 989 &&
                     model.generator.size() + model.discriminator.size() == 15 && model.g_state.size() + model.d_state.size() == 6,
                 "G 6873 + D 989 = 7862 parameters, 15 trainable tensors + 6 BN state variables");
  model.Save((root / "initial_arrays.npz").string());
  nn::BatchOrder order(1000, 32, 4847);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_discriminator_loss\tvalidation_generator_loss\tvalidation_coarse_pixel_mmd\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto source  = nn::cast(pools[0].source, nn::DType::Int64).to_vector<std::int64_t>();
  int        steps   = 0;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    const Tensor noise = gan::Gaussian(1000, 16, 4861 + epoch);
    order.begin_epoch();
    int seen = 0;
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const Tensor x = gan::Rows(pools[0].images, rows), z = gan::Rows(noise, rows);
      if (!steps) {
        const auto f = model.TrainingEvaluate(x, z);
        std::cout << "first batch (training BN): D loss=" << Value(f, "discriminator_loss")
                  << ", G loss=" << Value(f, "generator_loss") << '\n';
      }
      for (int group : {1, 0}) {
        const auto before = steps ? gan::Arrays{} : model.AllArrays();
        model.Update(group, x, z);
        if (!steps) {
          const auto after   = model.AllArrays();
          bool       changed = false;
          for (const auto& [key, value] : before) {
            const bool own = key.rfind(std::to_string(group) + "/", 0) == 0 ||
                             key.rfind(group == 0 ? "g_state/" : "d_state/", 0) == 0;
            if (own)
              changed |= !gan::BitEqual(value, after.at(key));
            else
              check::Require(gan::BitEqual(value, after.at(key)), "GAN step changes ONLY intended G or D group");
          }
          check::Require(changed, "intended GAN group updates");
        }
      }
      for (auto row : rows) batches << epoch << '\t' << steps << '\t' << row << '\t' << source[row] << '\n';
      ++steps;
      seen += int(rows.size());
    }
    check::Require(seen == 1000, "all fitting MNIST examples ONCE per epoch");
    const auto v = model.Evaluate(pools[1].images, validation_z);
    history << epoch << '\t' << Value(v, "discriminator_loss") << '\t' << Value(v, "generator_loss") << '\t'
            << Value(v, "pixel_mmd") << '\n';
    std::cout << "epoch " << epoch << ": diagnostic D loss=" << Value(v, "discriminator_loss")
              << ", G loss=" << Value(v, "generator_loss") << ", coarse MMD=" << Value(v, "pixel_mmd") << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

  // Final probes (training-mode forward and both gradients) must not touch weights or BN states.
  const auto probe_before = model.AllArrays();
  const auto probe_rows   = gan::Range(32);
  const Tensor probe_x = gan::Rows(pools[2].images, probe_rows), probe_z = gan::Rows(test_z, probe_rows);
  model.TrainingEvaluate(probe_x, probe_z);
  for (int group = 0; group < 2; ++group) model.Gradient(group, probe_x, probe_z);
  check::Require(gan::BitEqual(probe_before, model.AllArrays()), "final gradient probes DO not update BN states/weights");

  const auto checkpoint = (root / "final.npz").string();
  model.Save(checkpoint);
  gan::DcganModel deployed(false);
  deployed.Load(checkpoint);
  check::Require(gan::BitEqual(model.AllArrays(), deployed.AllArrays()), "all final G/D weights restore BIT EXACT");
  const auto   before = deployed.AllArrays();
  const auto   output = deployed.Evaluate(pools[2].images, test_z);
  const Tensor repeat = deployed.Generate(test_z);
  check::Require(gan::BitEqual(output.at("fake"), repeat), "noise-only generation BIT EXACT matches GAN evaluation");
  const auto single = deployed.Generate(gan::Rows(test_z, {0})).to_vector<float>();
  const auto all    = repeat.to_vector<float>();
  double     parity = 0;
  for (int i = 0; i < 784; ++i) parity = std::max(parity, std::abs(double(single[i]) - all[i]));
  check::Require(parity < 2e-6, "G single/batch parity");
  const Tensor changed_images = deployed.Generate(gan::Gaussian(200, 16, 4877));
  check::Require(!gan::BitEqual(changed_images, repeat), "different noise affects G images");
  const Tensor zero_images = deployed.Generate(nn::zeros({200, 16}));
  check::Require(!gan::BitEqual(zero_images, repeat), "G not constant zero-latent mapping");
  check::Require(gan::BitEqual(before, deployed.AllArrays()), "all G/D diagnostic generation immutable");

  const Tensor fit_mean = FitMean(pools[0].images, 200);
  const double mean_mmd =
      gan::ImqMmd(gan::GanCoarsePixels(nn::reshape(pools[2].images, {-1, 784})), gan::GanCoarsePixels(fit_mean)).item();
  double     real_accuracy = 0, fake_accuracy = 0;
  const auto real_logit = output.at("real_logit").to_vector<float>(), fake_logit = output.at("fake_logit").to_vector<float>();
  for (int b = 0; b < 200; ++b) real_accuracy += real_logit[b] >= 0, fake_accuracy += fake_logit[b] < 0;
  const double mean_pixel     = nn::mean(repeat).item();
  const double pixel_variance = nn::mean(nn::square(repeat - nn::mean(repeat, {0}, true))).item();
  SaveSheet(root / "generated.png", nn::slice_axis(repeat, 0, 0, 40), 8);
  SaveSheet(root / "fit_mean.png", nn::slice_axis(fit_mean, 0, 0, 8), 8);
  gan::SaveArrays((root / "outputs.npz").string(), output);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"generator_parameters\":6873,\"discriminator_parameters\":989,\"epochs\":" << epochs
       << ",\"steps_per_group\":" << steps << ",\"optimizer_updates\":" << 2 * steps
       << ",\"test_discriminator_loss\":" << Value(output, "discriminator_loss")
       << ",\"test_generator_loss\":" << Value(output, "generator_loss")
       << ",\"test_coarse_pixel_mmd\":" << Value(output, "pixel_mmd") << ",\"fit_mean_coarse_pixel_mmd\":" << mean_mmd
       << ",\"test_discriminator_real_accuracy\":" << real_accuracy / 200
       << ",\"test_discriminator_fake_accuracy\":" << fake_accuracy / 200 << ",\"generated_mean_pixel\":" << mean_pixel
       << ",\"generated_mean_per_pixel_variance\":" << pixel_variance << ",\"single_batch_max_error\":" << parity
       << ",\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "complete GAN artifacts");
  std::cout << "fixed FINAL epoch " << epochs << ", test D/G=" << Value(output, "discriminator_loss") << "/"
            << Value(output, "generator_loss") << ", coarse MMD=" << Value(output, "pixel_mmd")
            << ", mean baseline=" << mean_mmd << ", D real/fake accuracy=" << real_accuracy / 200 << "/"
            << fake_accuracy / 200 << ", mean pixel=" << mean_pixel << ", pixel variance=" << pixel_variance
            << "; D then G/restore/noiseOnly/immutable PASS (training " << seconds << " s)\nArtifacts: "
            << root.string() << "\nAll dcgan checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset("saliency"), out = paths::OutputRoot() / "models" / "dcgan";
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
    check::Require(epochs > 0 && epochs <= 100, "THREE default or explicit 1..100 epochs");
    Run(data, out, epochs, init);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
