// CMD: just run-nn auxiliary_classifier_gan
// Auxiliary-classifier GAN (Odena et al. 2017) on 1000 MNIST digits.
//
// Port of tflow's auxiliary_classifier_gan. Preparation (in ~/projects/tflow_examples): just
// autoencoder-assets; the data are read from $DATAMUNGE_DATA_DIR/saliency
// (saliency_data.npz: 1000 fit / 200 validation / 200 test balanced digits).
// G: Gaussian 16 + one-hot 10 -> 32 tanh -> 784 sigmoid (26736 parameters).
// D is IMAGE-ONLY: 784 -> 32 tanh shared features -> a source logit and a
// 10-class auxiliary head (25483 parameters). D minimizes the logistic source
// loss + CE(real class) + CE(fake requested class); the non-saturating G
// minimizes softplus(-fake logit) + CE(fake requested class). The first batch
// checks G's gradient equals source PLUS auxiliary components. Class IDs are
// validated (0..9, one per noise row) before the one-hot.
// Separate Adam (.0002, beta1 .5) for G and D; per batch a D step, THEN the G
// step recomputed through the UPDATED D. Three default epochs of 32-row
// batches (96 steps each), fixed final epoch.
//
// Validation/test D and G losses, threshold-0 source accuracies, auxiliary
// class accuracies and a coarse
// 4x4-pixel IMQ MMD are game diagnostics, not image-quality measures. Noise
// banks and batch order are tflow's exactly; weights use datamunge's
// initializer unless init=PATH names a tflow initial_arrays.npz (then the run
// follows TF's trajectory up to float rounding). tflow's per-step .npz dumps
// and index.html are not written. Outputs (history, batches, initial/final
// checkpoint .npz, outputs.npz, controls.npz, conditioning_controls.npz,
// same_z_class_grid.png, generated.png / fit_mean.png, metrics.json) go to
// output/nn/models/auxiliary_classifier_gan.
// Options: epochs=3 data=PATH out=PATH init=PATH.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include "common/auxiliary_classifier_gan.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

double Value(const gan::Arrays& a, const std::string& key) { return a.at(key).item(); }

/// Logistic GAN algebra at extreme logits: finite losses and gradients, and
/// a strong non-saturating G gradient when D is confident a sample is fake.
void Algebra(const std::filesystem::path& root) {
  Tensor real = nn::tensor({-80, -2, 2, 80}, {4, 1}), fake = nn::tensor({80, 2, -2, -80}, {4, 1});
  real.requires_grad_();
  fake.requires_grad_();
  const auto loss = gan::LogisticGanObjectives(real, fake);
  const auto dg   = nn::grad(loss.discriminator, {real, fake});
  const auto gg   = nn::grad(loss.generator, {fake});
  gan::Arrays out{{"real", real.detach()},
                  {"fake", fake.detach()},
                  {"discriminator_loss", loss.discriminator.detach()},
                  {"generator_loss", loss.generator.detach()},
                  {"d_real_gradient", dg[0]},
                  {"d_fake_gradient", dg[1]},
                  {"g_fake_gradient", gg[0]}};
  gan::RequireFinite(out, "extreme logits finite GAN loss gradients");
  check::Require(gg[0].at({3, 0}) < -.24, "non-saturating G has strong gradient when D confident fake");
  std::cout << "logistic algebra: D loss=" << loss.discriminator.item() << ", G loss=" << loss.generator.item()
            << ", dG/dfake=" << gg[0].values_str() << '\n';
  gan::SaveArrays((root / "logistic_algebra.npz").string(), out);
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  const auto pools = gan::LoadSaliency(data);
  std::filesystem::create_directories(root);
  Algebra(root);
  const Tensor validation_z = gan::Gaussian(200, 16, 4823), test_z = gan::Gaussian(200, 16, 4831);
  gan::AuxiliaryClassifierGanModel model;
  if (!init.empty()) model.Load(init);
  check::Require(model.Count(0) == 26736 && model.Count(1) == 25483 && model.Group(0).size() + model.Group(1).size() == 10,
                 "G 26736 + D 25483 = 52219 parameters, 10 tensors");
  model.Save((root / "initial_arrays.npz").string());
  gan::SaveArrays((root / "initial_outputs.npz").string(), model.Evaluate(pools[2].images, test_z, pools[2].labels));
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
      const Tensor x = gan::Rows(pools[0].images, rows), z = gan::Rows(noise, rows),
                   condition = gan::Rows(pools[0].labels, rows);
      if (!steps) {
        gan::SaveArrays((root / "first_forward.npz").string(), model.Evaluate(x, z, condition));
        const auto g_source = model.ComponentGradient(false, x, z, condition);
        const auto g_aux    = model.ComponentGradient(true, x, z, condition);
        for (const auto& [key, value] : model.Gradient(0, x, z, condition)) {
          const auto full = value.to_vector<float>(), a = g_source.at(key).to_vector<float>(),
                     b = g_aux.at(key).to_vector<float>();
          for (std::size_t i = 0; i < full.size(); ++i)
            check::Near(full[i], double(a[i]) + b[i], 2e-6, "G gradient = source PLUS auxiliary class components");
        }
        gan::SaveArrays((root / "first_g_source_gradients.npz").string(), g_source);
        gan::SaveArrays((root / "first_g_aux_gradients.npz").string(), g_aux);
      }
      for (int group : {1, 0}) {
        const auto before = steps ? gan::Arrays{} : model.AllArrays();
        model.Update(group, x, z, condition);
        if (!steps)
          gan::RequireOnlyOwnersChanged(before, model.AllArrays(), {std::to_string(group) + "/"},
                                        "GAN step changes ONLY intended G or D group");
      }
      for (auto row : rows) batches << epoch << '\t' << steps << '\t' << row << '\t' << source[row] << '\n';
      ++steps;
      seen += int(rows.size());
    }
    check::Require(seen == 1000, "all fitting MNIST examples ONCE per epoch");
    const auto v = model.Evaluate(pools[1].images, validation_z, pools[1].labels);
    history << epoch << '\t' << Value(v, "discriminator_loss") << '\t' << Value(v, "generator_loss") << '\t'
            << Value(v, "pixel_mmd") << '\n';
    std::cout << "epoch " << epoch << ": diagnostic D loss=" << Value(v, "discriminator_loss")
              << ", G loss=" << Value(v, "generator_loss") << ", coarse MMD=" << Value(v, "pixel_mmd") << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

  const auto checkpoint = (root / "final.npz").string();
  model.Save(checkpoint);
  gan::AuxiliaryClassifierGanModel deployed(false, 1, 2);
  deployed.Load(checkpoint);
  check::Require(gan::BitEqual(model.AllArrays(), deployed.AllArrays()), "all final G/D weights restore BIT EXACT");
  const auto   before = deployed.AllArrays();
  const auto   output = deployed.Evaluate(pools[2].images, test_z, pools[2].labels);
  const Tensor repeat = deployed.Sample(test_z, pools[2].labels);
  check::Require(gan::BitEqual(output.at("fake"), repeat), "noise+class generation BIT EXACT matches GAN evaluation");
  const double parity = gan::MaxAbsDiff(deployed.Sample(gan::Rows(test_z, {0}), gan::Rows(pools[2].labels, {0})), gan::Rows(repeat, {0}));
  check::Require(parity < 2e-6, "G single/batch parity");
  const Tensor changed_z = gan::Gaussian(200, 16, 4877), changed_images = deployed.Sample(changed_z, pools[2].labels);
  check::Require(!gan::BitEqual(changed_images, repeat), "different noise affects G images");
  const Tensor zero_images = deployed.Sample(nn::zeros({200, 16}), pools[2].labels);
  check::Require(!gan::BitEqual(zero_images, repeat), "G not constant zero-latent mapping");
  gan::SaveArrays((root / "controls.npz").string(), {{"changed_z", changed_z},
                                                     {"changed_images", changed_images},
                                                     {"zero_z", nn::zeros({200, 16})},
                                                     {"zero_images", zero_images}});
  // Same ten test noise vectors for every class (row = class), and D's real
  // logits under shifted labels: BOTH networks must depend on the class.
  std::vector<std::int64_t> grid_rows;
  std::vector<std::int32_t> grid_classes;
  for (int c = 0; c < 10; ++c)
    for (int k = 0; k < 10; ++k) grid_rows.push_back(k), grid_classes.push_back(c);
  const Tensor grid_z = gan::Rows(test_z, grid_rows), grid_labels = nn::from_vector(grid_classes, {100});
  const Tensor grid   = deployed.Sample(grid_z, grid_labels);
  const double g_condition_effect =
      gan::MaxAbsDiff(nn::slice_axis(grid, 0, 10, 100), nn::tile(nn::slice_axis(grid, 0, 0, 10), {9, 1}));
  auto wrong_classes = pools[2].labels.contiguous().to_vector<std::int32_t>();
  for (auto& c : wrong_classes) c = (c + 1) % 10;
  Tensor     wrong_labels = nn::from_vector(wrong_classes, {200});
  const auto wrong        = deployed.Evaluate(pools[2].images, test_z, wrong_labels);
  const double d_condition_effect = gan::MaxAbsDiff(output.at("real_logit"), wrong.at("real_logit"));
  check::Require(g_condition_effect > 1e-5 && d_condition_effect == 0,
                 "G conditional but D real IMAGE ONLY independent of G condition");
  check::Require(gan::BitEqual(output.at("real_class_probability"), wrong.at("real_class_probability")),
                 "changing conditions/CE targets leaves real class predictions BIT EXACT");
  const auto image_only = deployed.Classify(pools[2].images);
  check::Require(gan::BitEqual(image_only.at("source_logit"), output.at("real_logit")) &&
                     gan::BitEqual(image_only.at("class_probability"), output.at("real_class_probability")),
                 "D classifier requires IMAGE ONLY, no class/noise/targets");
  int invalid_rejected = 0;
  for (int invalid : {-1, 10}) {
    auto bad = wrong_classes;
    bad[0]   = invalid;
    try {
      deployed.Sample(test_z, nn::from_vector(bad, {200}));
    } catch (const std::exception&) {
      ++invalid_rejected;
    }
  }
  try {
    deployed.Sample(test_z, gan::Rows(pools[2].labels, {0}));
  } catch (const std::exception&) {
    ++invalid_rejected;
  }
  check::Require(invalid_rejected == 3, "negative/out of range/mismatched class labels rejected before OneHot");
  gan::SaveArrays((root / "conditioning_controls.npz").string(), {{"grid_z", grid_z},
                                                                  {"grid_labels", grid_labels},
                                                                  {"grid_images", grid},
                                                                  {"wrong_real_logits", wrong.at("real_logit")},
                                                                  {"wrong_real_class_probability", wrong.at("real_class_probability")},
                                                                  {"image_only_real_class_probability", image_only.at("class_probability")}});
  gan::Mosaic(root / "same_z_class_grid.png", grid, 10);
  check::Require(gan::BitEqual(before, deployed.AllArrays()), "all G/D diagnostic generation immutable");
  gan::SaveArrays((root / "outputs.npz").string(), output);

  const Tensor fit_mean = gan::FitMean(pools[0].images, 200);
  const double mean_mmd = gan::CoarseMmd(pools[2].images, fit_mean);
  double       real_accuracy = 0, fake_accuracy = 0;
  const auto   real_score = output.at("real_logit").to_vector<float>(), fake_score = output.at("fake_logit").to_vector<float>();
  for (int b = 0; b < 200; ++b) real_accuracy += real_score[b] >= 0, fake_accuracy += fake_score[b] < 0;
  const auto labels = nn::cast(pools[2].labels, nn::DType::Int64).to_vector<std::int64_t>();
  const auto real_best = nn::argmax(output.at("real_class_probability"), 1).to_vector<std::int64_t>();
  const auto fake_best = nn::argmax(output.at("fake_class_probability"), 1).to_vector<std::int64_t>();
  double     real_aux_accuracy = 0, fake_aux_accuracy = 0;
  for (int b = 0; b < 200; ++b) real_aux_accuracy += real_best[b] == labels[b], fake_aux_accuracy += fake_best[b] == labels[b];
  const auto [mean_pixel, pixel_variance] = gan::PixelMoments(repeat);
  gan::Mosaic(root / "generated.png", nn::slice_axis(repeat, 0, 0, 40), 8);
  gan::Mosaic(root / "fit_mean.png", nn::slice_axis(fit_mean, 0, 0, 8), 8);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"generator_parameters\":26736,\"discriminator_parameters\":25483,\"epochs\":" << epochs
       << ",\"steps_per_group\":" << steps << ",\"optimizer_updates\":" << 2 * steps
       << ",\"selected_epoch_fixed_final\":" << epochs << ",\"test_discriminator_loss\":" << Value(output, "discriminator_loss")
       << ",\"test_generator_loss\":" << Value(output, "generator_loss")
       << ",\"test_coarse_pixel_mmd\":" << Value(output, "pixel_mmd") << ",\"fit_mean_coarse_pixel_mmd\":" << mean_mmd
       << ",\"test_discriminator_real_accuracy\":" << real_accuracy / 200
       << ",\"test_discriminator_fake_accuracy\":" << fake_accuracy / 200 << ",\"generated_mean_pixel\":" << mean_pixel
       << ",\"generated_mean_per_pixel_variance\":" << pixel_variance
       << ",\"generator_class_probability_max_effect\":" << g_condition_effect
       << ",\"discriminator_real_source_condition_max_effect\":" << d_condition_effect
       << ",\"invalid_condition_controls_rejected\":" << invalid_rejected
       << ",\"test_real_aux_class_accuracy\":" << real_aux_accuracy / 200
       << ",\"test_fake_requested_aux_class_accuracy\":" << fake_aux_accuracy / 200
       << ",\"single_batch_max_error\":" << parity
       << ",\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "complete GAN artifacts");
  std::cout << "fixed FINAL epoch " << epochs << ", test D/G=" << Value(output, "discriminator_loss") << "/"
            << Value(output, "generator_loss") << ", coarse MMD=" << Value(output, "pixel_mmd")
            << ", mean baseline=" << mean_mmd << ", D real/fake accuracy=" << real_accuracy / 200 << "/"
            << fake_accuracy / 200 << ", aux class accuracy real/fake=" << real_aux_accuracy / 200 << "/"
            << fake_aux_accuracy / 200 << ", mean pixel=" << mean_pixel << ", pixel variance=" << pixel_variance
            << "; D then G/restore/noise+classOnly/immutable PASS (training " << seconds << " s)\nArtifacts: "
            << root.string() << "\nAll auxiliary_classifier_gan checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset("saliency"), out = paths::OutputRoot() / "models" / "auxiliary_classifier_gan";
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
