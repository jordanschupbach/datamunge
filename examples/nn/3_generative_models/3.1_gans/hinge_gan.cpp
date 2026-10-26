// CMD: just run-nn hinge_gan
// Hinge-loss GAN (the SNGAN/Miyato et al. hinge formulation, without spectral
// normalization) on 1000 MNIST digits.
//
// Port of tflow's hinge_gan. Preparation (in ~/projects/tflow_examples): just
// autoencoder-assets; the data are read from $DATAMUNGE_DATA_DIR/saliency
// (saliency_data.npz: 1000 fit / 200 validation / 200 test balanced digits).
// Compact MLP G 16 -> 32 tanh -> 784 sigmoid (26416 parameters), D 784 -> 32
// tanh -> 1 RAW UNBOUNDED score (25153 parameters). D = mean relu(1 - real
// score) + mean relu(1 + fake score); G = -mean(fake score) with NO hinge on G
// (constant score derivative -1/N). At exact margins the ReLU subgradient is
// 0, as in TensorFlow. No sigmoid, cross entropy or probability calibration.
// Separate Adam (.0002, beta1 .5) for G and D; per batch a D step, THEN the G
// step recomputed through the UPDATED D. Three default epochs of 32-row
// batches (96 steps each), fixed final epoch, digit labels unused.
//
// Validation/test D and G losses (negative G loss is permitted),
// threshold-0 source accuracies and a coarse
// 4x4-pixel IMQ MMD are game diagnostics, not image-quality measures. Noise
// banks and batch order are tflow's exactly; weights use datamunge's
// initializer unless init=PATH names a tflow initial_arrays.npz (then the run
// follows TF's trajectory up to float rounding). tflow's per-step .npz dumps
// and index.html are not written. Outputs (history, batches, initial/final
// checkpoint .npz, outputs.npz, controls.npz, generated.png / fit_mean.png,
// metrics.json) go to output/nn/models/hinge_gan.
// Options: epochs=3 data=PATH out=PATH init=PATH.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include "common/gan.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

double Value(const gan::Arrays& a, const std::string& key) { return a.at(key).item(); }

/// Hinge algebra at extreme, boundary (+-1) and interior raw scores: finite
/// losses and gradients and a constant nonzero G score gradient.
void Algebra(const std::filesystem::path& root) {
  Tensor real = nn::tensor({-80, -2, 1, 2, 80, .5}, {6, 1}), fake = nn::tensor({80, 2, -1, -2, -80, -.5}, {6, 1});
  real.requires_grad_();
  fake.requires_grad_();
  const auto loss = gan::HingeGanObjectives(real, fake);
  const auto dg   = nn::grad(loss.discriminator, {real, fake});
  const auto gg   = nn::grad(loss.generator, {fake});
  gan::Arrays out{{"real", real.detach()},
                  {"fake", fake.detach()},
                  {"discriminator_loss", loss.discriminator.detach()},
                  {"generator_loss", loss.generator.detach()},
                  {"d_real_gradient", dg[0]},
                  {"d_fake_gradient", dg[1]},
                  {"g_fake_gradient", gg[0]}};
  gan::RequireFinite(out, "extreme scores finite hinge loss gradients");
  check::Require(gg[0].at({3, 0}) < -.15, "hinge G retains constant nonzero raw score gradient");
  std::cout << "hinge algebra: D loss=" << loss.discriminator.item() << ", G loss=" << loss.generator.item()
            << ", dG/dfake=" << gg[0].values_str() << '\n';
  gan::SaveArrays((root / "hinge_algebra.npz").string(), out);
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  const auto pools = gan::LoadSaliency(data);
  std::filesystem::create_directories(root);
  Algebra(root);
  const Tensor validation_z = gan::Gaussian(200, 16, 4823), test_z = gan::Gaussian(200, 16, 4831);
  gan::MlpGanModel model(gan::GanObjective::Hinge, false, 5701, 5711);
  if (!init.empty()) model.Load(init);
  check::Require(model.Count(0) == 26416 && model.Count(1) == 25153 && model.Group(0).size() + model.Group(1).size() == 8,
                 "G 26416 + D 25153 = 51569 parameters, 8 tensors");
  model.Save((root / "initial_arrays.npz").string());
  gan::SaveArrays((root / "initial_outputs.npz").string(), model.Evaluate(pools[2].images, test_z));
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
      if (!steps) gan::SaveArrays((root / "first_forward.npz").string(), model.Evaluate(x, z));
      for (int group : {1, 0}) {
        const auto before = steps ? gan::Arrays{} : model.AllArrays();
        model.Update(group, x, z);
        if (!steps)
          gan::RequireOnlyOwnersChanged(before, model.AllArrays(), {std::to_string(group) + "/"},
                                        "GAN step changes ONLY intended G or D group");
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

  const auto checkpoint = (root / "final.npz").string();
  model.Save(checkpoint);
  gan::MlpGanModel deployed(gan::GanObjective::Hinge, false, 1, 2, false);
  deployed.Load(checkpoint);
  check::Require(gan::BitEqual(model.AllArrays(), deployed.AllArrays()), "all final G/D weights restore BIT EXACT");
  const auto   before = deployed.AllArrays();
  const auto   output = deployed.Evaluate(pools[2].images, test_z);
  const Tensor repeat = deployed.Sample(test_z);
  check::Require(gan::BitEqual(output.at("fake"), repeat), "noise-only generation BIT EXACT matches GAN evaluation");
  const double parity = gan::MaxAbsDiff(deployed.Sample(gan::Rows(test_z, {0})), gan::Rows(repeat, {0}));
  check::Require(parity < 2e-6, "G single/batch parity");
  const Tensor changed_z = gan::Gaussian(200, 16, 4877), changed_images = deployed.Sample(changed_z);
  check::Require(!gan::BitEqual(changed_images, repeat), "different noise affects G images");
  const Tensor zero_images = deployed.Sample(nn::zeros({200, 16}));
  check::Require(!gan::BitEqual(zero_images, repeat), "G not constant zero-latent mapping");
  gan::SaveArrays((root / "controls.npz").string(), {{"changed_z", changed_z},
                                                     {"changed_images", changed_images},
                                                     {"zero_z", nn::zeros({200, 16})},
                                                     {"zero_images", zero_images}});
  check::Require(gan::BitEqual(before, deployed.AllArrays()), "all G/D diagnostic generation immutable");
  gan::SaveArrays((root / "outputs.npz").string(), output);

  const Tensor fit_mean = gan::FitMean(pools[0].images, 200);
  const double mean_mmd = gan::CoarseMmd(pools[2].images, fit_mean);
  double       real_accuracy = 0, fake_accuracy = 0;
  const auto   real_score = output.at("real_score").to_vector<float>(), fake_score = output.at("fake_score").to_vector<float>();
  for (int b = 0; b < 200; ++b) real_accuracy += real_score[b] >= 0, fake_accuracy += fake_score[b] < 0;
  const auto [mean_pixel, pixel_variance] = gan::PixelMoments(repeat);
  gan::Mosaic(root / "generated.png", nn::slice_axis(repeat, 0, 0, 40), 8);
  gan::Mosaic(root / "fit_mean.png", nn::slice_axis(fit_mean, 0, 0, 8), 8);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"generator_parameters\":26416,\"discriminator_parameters\":25153,\"epochs\":" << epochs
       << ",\"steps_per_group\":" << steps << ",\"optimizer_updates\":" << 2 * steps
       << ",\"selected_epoch_fixed_final\":" << epochs
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
            << ", mean baseline=" << mean_mmd << ", D real/fake accuracy (threshold 0)=" << real_accuracy / 200 << "/"
            << fake_accuracy / 200 << ", mean pixel=" << mean_pixel << ", pixel variance=" << pixel_variance
            << "; D then G/restore/noiseOnly/immutable PASS (training " << seconds << " s)\nArtifacts: "
            << root.string() << "\nAll hinge_gan checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset("saliency"), out = paths::OutputRoot() / "models" / "hinge_gan";
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
