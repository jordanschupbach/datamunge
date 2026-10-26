// CMD: just run-nn bidirectional_gan
// BiGAN / ALI (Donahue et al. 2016; Dumoulin et al. 2016): joint-pair
// adversarial inference on eight 2-D Gaussians.
//
// Port of tflow's bidirectional_gan (no dataset: the ring of eight Gaussians,
// radius 2, sd .08, is drawn from tflow's seeded streams; 400 fit / 128
// validation / 128 test points). G: prior z 2 -> 16 tanh -> 2; E: data 2 -> 16
// tanh -> 2 (82 parameters each); joint D: concat(x, code) 4 -> 16 tanh -> 1
// logit (97). D separates (x, E(x)) from (G(z), z) with the logistic loss; E
// and G share the non-saturating objective mean softplus(real) + mean
// softplus(-fake) as SEPARATE groups. Separate Adam (.0002, beta1 .5); per
// batch D, then E, then G, each recomputed after the previous write (with D
// fixed, the E step leaves G's gradient EXACTLY unchanged). Three default
// epochs of 32-row batches (13 steps each), fixed final epoch. Reconstruction
// G(E(x)) and latent cycle E(G(z)) are untrained diagnostics, compared with the
// fit-mean predictor.
//
// Data, noise and batch order are tflow's exactly; weights use datamunge's
// initializer unless init=PATH names a tflow initial_arrays.npz. tflow's
// per-step .npz dumps and index.html are not written. Outputs (history,
// batches, checkpoints, outputs.npz, pair_controls.npz, generated.png /
// encoded.png / reconstruction.png scatter plots, metrics.json) go to
// output/nn/models/bidirectional_gan. Options: epochs=3 out=PATH init=PATH seed=N
// (seed=N > 0 replaces the default weight-initializer seeds; for seed-spread runs).

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include "common/bidirectional_gan.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

double Value(const gan::Arrays& a, const std::string& key) { return a.at(key).item(); }

Tensor Gaussian2(std::int64_t n, std::uint32_t seed) { return gan::Gaussian(n, 2, seed); }

/// tflow Points: ring of eight Gaussians (class i % 8), radius 2, sd .08.
Tensor Points(std::int64_t n, std::uint32_t seed) {
  const auto         noise = gan::GaussianNoise(n * 2, seed);
  std::vector<float> points(n * 2);
  for (std::int64_t i = 0; i < n; ++i)
    for (int j = 0; j < 2; ++j) {
      const double theta = 2 * std::acos(-1.) * double(i % 8) / 8;
      points[i * 2 + j]  = float(2 * (j ? std::sin(theta) : std::cos(theta)) + .08 * noise[i * 2 + j]);
    }
  return nn::from_vector(points, {n, 2});
}

/// tflow Plot: 512x512 scatter over [-3, 3]^2, real blue, fake red; returns points outside.
int Plot(const std::filesystem::path& path, const Tensor& real, const Tensor& fake) {
  std::vector<float> rgb(512 * 512 * 3, 248 / 255.f);
  int                clipped = 0;
  for (int group = 0; group < 2; ++group) {
    const auto p = (group ? fake : real).contiguous().to_vector<float>();
    for (std::size_t i = 0; i < p.size() / 2; ++i) {
      const double px = p[i * 2], py = p[i * 2 + 1];
      if (std::abs(px) > 3 || std::abs(py) > 3) {
        ++clipped;
        continue;
      }
      const int x = int(std::floor((px + 3) / 6 * 511 + .5)), y = int(std::floor((3 - py) / 6 * 511 + .5));
      for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx)
          if (dx * dx + dy * dy <= 4 && x + dx >= 0 && x + dx < 512 && y + dy >= 0 && y + dy < 512)
            for (int c = 0; c < 3; ++c)
              rgb[((y + dy) * 512 + x + dx) * 3 + c] =
                  (group ? (c == 0 ? 210 : 65) : (c == 2 ? 210 : 65)) / 255.f;
    }
  }
  nn::save_png(nn::from_vector(rgb, {512, 512, 3}), path.string());
  return clipped;
}

void Run(const std::filesystem::path& root, int epochs, const std::string& init, std::uint64_t seed) {
  std::filesystem::create_directories(root);
  const Tensor fit = Points(400, 5507), validation = Points(128, 5519), test = Points(128, 5521);
  const Tensor vz = Gaussian2(128, 6043), tz = Gaussian2(128, 6047);
  gan::SaveArrays((root / "data.npz").string(), {{"fit", fit}, {"validation", validation}, {"test", test}});
  gan::SaveArrays((root / "fixed_noise.npz").string(), {{"validation_z", vz}, {"test_z", tz}});
  gan::BidirectionalGanModel model(true, seed ? std::vector<std::uint64_t>{seed, seed + 1, seed + 2}
                                                : std::vector<std::uint64_t>{6007, 6011, 6029});
  if (!init.empty()) model.Load(init);
  check::Require(model.Count(0) == 82 && model.Count(1) == 82 && model.Count(2) == 97 &&
                     model.Group(0).size() + model.Group(1).size() + model.Group(2).size() == 12,
                 "G 82 / E 82 / joint D 97 = 261 params, 12 tensors");
  model.Save((root / "initial_arrays.npz").string());
  gan::SaveArrays((root / "initial_outputs.npz").string(), model.Evaluate(test, tz));
  nn::BatchOrder order(400, 32, 6089);
  int            steps = 0;
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tdiscriminator_loss\teg_loss\n";
  batches << "epoch\tstep\trow\n";
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    const Tensor noise = Gaussian2(400, 6067 + epoch);
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const Tensor x = gan::Rows(fit, rows), z = gan::Rows(noise, rows);
      if (!steps) gan::SaveArrays((root / "first_forward.npz").string(), model.Evaluate(x, z));
      gan::Arrays g_gradient_before_e;
      for (int group : {2, 1, 0}) {
        const auto before = model.AllArrays();
        if (!steps && group == 1) g_gradient_before_e = model.Gradient(0, x, z);
        if (!steps && group == 0) {
          check::Require(gan::BitEqual(model.Gradient(0, x, z), g_gradient_before_e),
                         "E update leaves G gradient EXACT unchanged: EG blocks independent with D fixed");
          gan::SaveArrays((root / "g_gradient_before_e.npz").string(), g_gradient_before_e);
        }
        model.Update(group, x, z);
        gan::RequireOnlyOwnersChanged(before, model.AllArrays(), {std::to_string(group) + "/"},
                                      "each BiGAN phase changes ONLY its own group", false);
      }
      for (auto row : rows) batches << epoch << '\t' << steps << '\t' << row << '\n';
      ++steps;
    }
    const auto v = model.Evaluate(validation, vz);
    history << epoch << '\t' << Value(v, "discriminator_loss") << '\t' << Value(v, "eg_loss") << '\n';
    std::cout << "epoch " << epoch << ": D=" << Value(v, "discriminator_loss") << ", EG=" << Value(v, "eg_loss")
              << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

  const auto checkpoint = (root / "final.npz").string();
  model.Save(checkpoint);
  gan::BidirectionalGanModel deployed(false, {1, 2, 3});
  deployed.Load(checkpoint);
  check::Require(gan::BitEqual(model.AllArrays(), deployed.AllArrays()), "all 12 vars restore exact");
  const auto   before = deployed.AllArrays();
  const auto   output = deployed.Evaluate(test, tz);
  const Tensor fake = deployed.Generate(tz), encoded = deployed.Encode(test);
  check::Require(gan::BitEqual(fake, output.at("fake")) && gan::BitEqual(encoded, output.at("encoded")),
                 "noise ONLY generation / data ONLY encoding");
  const double parity = std::max(gan::MaxAbsDiff(deployed.Generate(gan::Rows(tz, {0})), gan::Rows(fake, {0})),
                                 gan::MaxAbsDiff(deployed.Encode(gan::Rows(test, {0})), gan::Rows(encoded, {0})));
  check::Require(parity < 2e-6, "both G/E batch parity");
  const Tensor changed_z = Gaussian2(128, 6073), changed = deployed.Generate(changed_z);
  check::Require(!gan::BitEqual(changed, fake), "G responds to noise");
  const Tensor pair = deployed.ScorePair(test, encoded);
  check::Require(gan::BitEqual(pair, output.at("real_score")), "joint D pair only parity");
  const Tensor zeros = nn::zeros({128, 2});
  const Tensor zero_code = deployed.ScorePair(test, zeros), zero_image = deployed.ScorePair(zeros, encoded);
  const double code_effect = gan::MaxAbsDiff(pair, zero_code), image_effect = gan::MaxAbsDiff(pair, zero_image);
  check::Require(code_effect > 1e-5 && image_effect > 1e-5, "D uses BOTH joint pair parts");
  gan::SaveArrays((root / "pair_controls.npz").string(), {{"pair", pair},
                                                          {"zero_code", zero_code},
                                                          {"zero_image", zero_image},
                                                          {"changed_z", changed_z},
                                                          {"changed_fake", changed}});
  check::Require(gan::BitEqual(before, deployed.AllArrays()), "immutable all G/E/D inference");
  gan::SaveArrays((root / "outputs.npz").string(), output);
  const int outside_gen            = Plot(root / "generated.png", test, fake);
  const int outside_latent         = Plot(root / "encoded.png", tz, encoded);
  const int outside_reconstruction = Plot(root / "reconstruction.png", test, output.at("reconstruction"));
  auto mse = [](const Tensor& a, const Tensor& b) {
    return nn::mean(nn::square(nn::cast(a, nn::DType::Float64) - nn::cast(b, nn::DType::Float64))).item();
  };
  const double reconstruction_mse = mse(output.at("reconstruction"), test);
  const double latent_mse         = mse(output.at("latent_cycle"), tz);
  const double mean_mse = mse(nn::tile(nn::mean(nn::cast(fit, nn::DType::Float64), {0}, true), {128, 1}), test);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":261,\"epochs\":" << epochs << ",\"steps_per_group\":" << steps
       << ",\"optimizer_updates\":" << 3 * steps << ",\"selected_epoch_fixed_final\":" << epochs
       << ",\"test_discriminator_loss\":" << Value(output, "discriminator_loss")
       << ",\"test_eg_loss\":" << Value(output, "eg_loss") << ",\"test_reconstruction_mse\":" << reconstruction_mse
       << ",\"test_latent_cycle_mse\":" << latent_mse << ",\"fit_mean_reconstruction_mse\":" << mean_mse
       << ",\"joint_code_score_max_effect\":" << code_effect << ",\"joint_image_score_max_effect\":" << image_effect
       << ",\"outside_generated\":" << outside_gen << ",\"outside_encoded\":" << outside_latent
       << ",\"outside_reconstruction\":" << outside_reconstruction << ",\"single_batch_max_error\":" << parity
       << ",\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "complete BiGAN artifacts");
  std::cout << "fixed final=" << epochs << ", D/EG=" << Value(output, "discriminator_loss") << "/"
            << Value(output, "eg_loss") << ", untrained reconstruction MSE=" << reconstruction_mse
            << " vs mean=" << mean_mse << ", latent cycle MSE=" << latent_mse
            << "; joint paths/D then E then G/restore/immutable PASS (training " << seconds << " s)\nArtifacts: "
            << root.string() << "\nAll bidirectional_gan checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::filesystem::path out    = paths::OutputRoot() / "models" / "bidirectional_gan";
    std::string           init;
    std::uint64_t         seed = 0;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0)
        epochs = std::stoi(a.substr(7));
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else if (a.rfind("init=", 0) == 0)
        init = a.substr(5);
      else if (a.rfind("seed=", 0) == 0)
        seed = std::stoull(a.substr(5));
      else
        throw std::invalid_argument("expected epochs=N out=PATH init=PATH seed=N");
    }
    check::Require(epochs > 0 && epochs <= 100, "three default or explicit 1..100 epochs");
    Run(out, epochs, init, seed);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
