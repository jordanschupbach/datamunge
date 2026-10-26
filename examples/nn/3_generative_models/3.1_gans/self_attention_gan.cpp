// CMD: just run-nn self_attention_gan
// Self-attention GAN (Zhang et al. 2018), compact spatial-patch variant, on
// 1000 MNIST digits.
//
// Port of tflow's self_attention_gan. Preparation (in ~/projects/tflow_examples):
// just autoencoder-assets; the data are read from $DATAMUNGE_DATA_DIR/saliency
// (saliency_data.npz: 1000 fit / 200 validation / 200 test balanced digits).
// Real nonlocal attention in BOTH G and D over 16 spatial tokens (4x4 grid of
// 7x7 patches, 8 features each): softmax(q k^T) over all 16 positions with NO
// sqrt scaling (SAGAN's unscaled dot product), q/k 2-D, v 4-D, projection 8,
// residual output = input + gamma * projection, gamma initialised 0. G:
// Gaussian 16 -> 128 tanh -> 16x8 tokens -> attention -> shared 8 -> 49
// sigmoid patch decoder -> 28x28 (2730 parameters). D: patch 49 -> 8 tanh +
// learned 16x8 positions -> attention -> mean -> raw score (650 parameters).
// Spectral normalization on ALL 12 kernels (355 nontrainable u/v values); 16
// warm-up power iterations, then one update of D's vectors before each D Adam
// step and of G's vectors before each G Adam step. Hinge D / linear G, Adam
// .0002 beta1 .5, D then G recomputed through the updated D. Three default
// epochs of 32-row batches (96 steps each), fixed final epoch, labels unused.
//
// Noise banks and batch order are tflow's exactly; weights use datamunge's
// initializer unless init=PATH names a tflow initial_arrays.npz (G, D and the
// warmed-up u/v; then the run follows TF's trajectory up to float rounding).
// tflow's per-step .npz dumps, independent NumPy audit and index.html are not
// written. Outputs (history, batches, initial/final checkpoint .npz,
// final_gradients_group*.npz, outputs.npz, controls.npz,
// attention_controls.npz, generated.png / fit_mean.png, metrics.json) go to
// output/nn/models/self_attention_gan.
// Options: epochs=3 data=PATH out=PATH init=PATH.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include "common/self_attention_gan.hpp"
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
  gan::SelfAttentionGanModel model;
  if (!init.empty()) model.Load(init);
  check::Require(model.Count(0) == 2730 && model.Count(1) == 650 && model.Count(2) == 245 && model.Count(3) == 110,
                 "G 2730 + D 650 + 355 state");
  model.Save((root / "initial_arrays.npz").string());
  gan::SaveArrays((root / "initial_outputs.npz").string(), model.Evaluate(pools[2].images, test_z));
  nn::BatchOrder order(1000, 32, 4847);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_discriminator_loss\tvalidation_generator_loss\tvalidation_coarse_pixel_mmd\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto source  = nn::cast(pools[0].source, nn::DType::Int64).to_vector<std::int64_t>();
  int        steps   = 0;
  Tensor     first_x, first_z;
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
      if (!steps) first_x = x, first_z = z;
      for (int group : {1, 0}) {
        {
          const auto before = steps ? gan::Arrays{} : model.AllArrays();
          model.Advance(group);
          if (!steps)
            gan::RequireOnlyOwnersChanged(before, model.AllArrays(), {std::to_string(group + 2) + "/"},
                                          "power iteration changes ONLY own u/v state");
        }
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
  gan::SelfAttentionGanModel deployed(false, {1, 2, 3, 4});
  deployed.Load(checkpoint);
  check::Require(gan::BitEqual(model.AllArrays(), deployed.AllArrays()), "all 51 G/D/state tensors restore BIT EXACT");
  for (int group = 0; group < 2; ++group)
    gan::SaveArrays((root / ("final_gradients_group" + std::to_string(group) + ".npz")).string(),
                    model.Gradient(group, first_x, first_z));
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
  gan::Arrays attention_controls;
  for (int group = 0; group < 2; ++group) {
    const Tensor tokens = gan::Rows(output.at(group == 0 ? "g_attention_input" : "real_attention_input"), {0, 1});
    auto         values = tokens.contiguous().to_vector<float>();
    values[0] += 1.f, values[128] += 1.f;  // token 0, feature 0 of both rows
    const Tensor modified = nn::from_vector(values, tokens.shape());
    const auto   baseline = deployed.Attention(group == 0, tokens), changed = deployed.Attention(group == 0, modified);
    const auto   bo = baseline.at("output").contiguous().to_vector<float>(),
               co = changed.at("output").contiguous().to_vector<float>();
    double effect = 0;
    for (int b = 0; b < 2; ++b)
      for (int j = 8; j < 128; ++j) effect = std::max(effect, std::abs(double(bo[b * 128 + j]) - co[b * 128 + j]));
    check::Require(effect > 1e-9, "changing ONE spatial token affects OTHER positions via attention");
    const auto probability = baseline.at("probability").contiguous().to_vector<float>();
    for (int b = 0; b < 2; ++b)
      for (int q = 0; q < 16; ++q) {
        double sum = 0;
        for (int k = 0; k < 16; ++k) sum += probability[(b * 16 + q) * 16 + k];
        check::Near(sum, 1, 2e-6, "attention row stochastic");
      }
    const std::string prefix = std::to_string(group) + "/";
    attention_controls.emplace(prefix + "input", tokens);
    attention_controls.emplace(prefix + "modified_input", modified);
    attention_controls.emplace(prefix + "output", baseline.at("output"));
    attention_controls.emplace(prefix + "modified_output", changed.at("output"));
    attention_controls.emplace(prefix + "probability", baseline.at("probability"));
    std::cout << (group ? "D" : "G") << " attention: max effect of one token on OTHER positions=" << effect << '\n';
  }
  gan::SaveArrays((root / "attention_controls.npz").string(), attention_controls);
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
  meta << "{\"generator_parameters\":2730,\"discriminator_parameters\":650,\"epochs\":" << epochs
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
            << root.string() << "\nAll self_attention_gan checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset("saliency"), out = paths::OutputRoot() / "models" / "self_attention_gan";
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
