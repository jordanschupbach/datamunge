// CMD: just run-nn wasserstein_gan
// WGAN clipping versus WGAN-GP on eight 2D Gaussians; inexpensive three-epoch comparison.
//
// Port of tflow's wasserstein_gan (Arjovsky et al. 2017; Gulrajani et al.
// 2017). 400 fitting samples from eight equally weighted isotropic Gaussians
// at radius 2 (noise sigma .08), 128 validation and 128 test points. G 2 ->
// 16 tanh -> 2 linear (82 parameters), critic 2 -> 16 tanh -> 1 raw score
// (65). Critic minimizes mean(fake) - mean(real); G minimizes -mean(fake).
// Clipped WGAN: RMSProp 5e-5 (rho .9, epsilon inside the root), every critic
// parameter clipped to [-.01, .01] before training and after EVERY critic
// step. WGAN-GP: Adam 1e-4 (beta1 0, beta2 .9) plus 10 mean(||grad_x
// critic(xhat)||_2 - 1)^2 at xhat = alpha real + (1 - alpha)
// stop_gradient(fake), norm sqrt(sum g^2 + 1e-12). Three critic steps, then
// one G step through the updated critic, per batch of 32; fresh noise for
// every step. Same real batch reused in the three critic steps.
//
// tflow had to write the critic input Jacobian by hand (TF C++ has no second
// derivative of TanhGrad). Here the penalty differentiates the symbolic
// input gradient (nn::grad with create_graph) through the critic parameters,
// and the hand-derived Jacobian is kept as a check against it. Data, noise,
// alpha draws and batch order are tflow's exactly; weights use datamunge's
// initializer unless init=DIR points at a tflow output directory (its
// {clipped,gradient_penalty}/initial_arrays.npz). Outputs (history, fixed
// final checkpoint as .npz, scatter plots) go to output/nn/models/
// wasserstein_gan. The critic gap is an unconverged game diagnostic, not a
// certified Wasserstein distance. Options: epochs=3 out=PATH init=DIR.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include "common/gan.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

Tensor Alpha(std::int64_t n, std::uint32_t seed) {
  std::mt19937       rng(seed);
  std::vector<float> a(static_cast<std::size_t>(n));
  for (auto& v : a) v = float((double(rng()) + .5) / 4294967296.);
  return nn::from_vector(a, {n, 1});
}

Tensor Points(std::int64_t n, std::uint32_t seed) {
  const auto         noise = gan::GaussianNoise(n * 2, seed);
  std::vector<float> p(static_cast<std::size_t>(n * 2));
  for (std::int64_t i = 0; i < n; ++i)
    for (int j = 0; j < 2; ++j) {
      const double theta = 2 * std::acos(-1.) * double(i % 8) / 8;
      p[i * 2 + j]       = float(2 * (j ? std::sin(theta) : std::cos(theta)) + .08 * noise[i * 2 + j]);
    }
  return nn::from_vector(p, {n, 2});
}

/// Real (blue) and generated (red) points on fixed [-3, 3] axes; returns the count outside.
int Plot(const std::filesystem::path& path, const Tensor& real, const Tensor& fake) {
  std::vector<float> rgb(512 * 512 * 3, 248.f);
  int                outside = 0;
  for (int group = 0; group < 2; ++group) {
    const auto p = (group ? fake : real).contiguous().to_vector<float>();
    for (std::size_t i = 0; i < p.size() / 2; ++i) {
      const double px = p[2 * i], py = p[2 * i + 1];
      if (std::abs(px) > 3 || std::abs(py) > 3) {
        ++outside;
        continue;
      }
      const int x = int(std::floor((px + 3) / 6 * 511 + .5)), y = int(std::floor((3 - py) / 6 * 511 + .5));
      for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx)
          if (dx * dx + dy * dy <= 4 && x + dx >= 0 && x + dx < 512 && y + dy >= 0 && y + dy < 512)
            for (int c = 0; c < 3; ++c)
              rgb[((y + dy) * 512 + x + dx) * 3 + c] = group ? (c == 0 ? 210.f : 65.f) : (c == 2 ? 210.f : 65.f);
    }
  }
  nn::save_png(nn::from_vector(rgb, {512, 512, 3}), path.string(), 0, 255);
  return outside;
}

double Scalar(const gan::Arrays& a, const std::string& key) { return a.at(key).item(); }

void Train(const std::filesystem::path& root, const Tensor& fit, const Tensor& validation, const Tensor& test,
           int epochs, bool gp, const std::filesystem::path& init) {
  const auto started = std::chrono::steady_clock::now();
  const auto dir     = root / (gp ? "gradient_penalty" : "clipped");
  std::filesystem::create_directories(dir);
  const Tensor vz = gan::Gaussian(128, 2, 5527), tz = gan::Gaussian(128, 2, 5531), va = Alpha(128, 5537),
               ta = Alpha(128, 5549);
  gan::WassersteinGanModel model(gp);
  if (!init.empty()) model.Load((init / (gp ? "gradient_penalty" : "clipped") / "initial_arrays.npz").string());
  check::Require(model.generator.count() == 82 && model.critic.count() == 65, "147 parameters 8 tensors");
  model.Save((dir / "initial_arrays.npz").string());
  nn::BatchOrder batches_order(400, 32, 5557);
  int            gsteps = 0, dsteps = 0;
  double         clipmax = 0;
  std::ofstream  history(dir / "history.tsv"), batches(dir / "batches.tsv");
  history.precision(17);
  history << "epoch\tcritic_loss\tgenerator_loss\tcritic_gap\tpenalty\n";
  batches << "epoch\tstep\trow\n";
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    batches_order.begin_epoch();
    for (;;) {
      const auto rows = batches_order.next();
      if (rows.empty()) break;
      const auto   n = std::int64_t(rows.size());
      const Tensor x = gan::Rows(fit, rows);
      for (int phase = 0; phase < 3; ++phase) {
        const Tensor z = gan::Gaussian(n, 2, 5601 + dsteps), a = Alpha(n, 6601 + dsteps);
        if (!gsteps && phase == 0) {
          const auto f = model.Evaluate(x, z, a);
          std::cout << (gp ? "GP" : "clip") << " first critic batch: loss=" << Scalar(f, "critic_loss")
                    << ", gap=" << Scalar(f, "critic_gap") << ", penalty=" << Scalar(f, "penalty") << '\n';
        }
        const auto before = model.AllArrays();
        model.Update(1, x, z, a);
        const auto after = model.AllArrays();
        for (const auto& [key, value] : before)
          if (key[0] == '0')
            check::Require(gan::BitEqual(value, after.at(key)), "critic step never changes G");
          else if (!gp)
            for (float v : after.at(key).to_vector<float>()) {
              clipmax = std::max(clipmax, std::abs(double(v)));
              check::Require(std::abs(v) <= .01f, "ALL critic weights clipped AFTER each step");
            }
        ++dsteps;
      }
      const Tensor z = gan::Gaussian(n, 2, 7601 + gsteps), a = Alpha(n, 8601 + gsteps);
      const auto   before = model.AllArrays();
      model.Update(0, x, z, a);
      const auto after = model.AllArrays();
      for (const auto& [key, value] : before)
        if (key[0] == '1') check::Require(gan::BitEqual(value, after.at(key)), "G step never changes critic");
      for (auto row : rows) batches << epoch << '\t' << gsteps << '\t' << row << '\n';
      ++gsteps;
    }
    const auto v = model.Evaluate(validation, vz, va);
    history << epoch;
    for (auto key : {"critic_loss", "generator_loss", "critic_gap", "penalty"}) history << '\t' << Scalar(v, key);
    history << '\n';
    std::cout << (gp ? "GP" : "clip") << " epoch " << epoch << ": critic gap=" << Scalar(v, "critic_gap")
              << ", penalty=" << Scalar(v, "penalty") << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  const auto   final_path = (dir / "final.npz").string();
  model.Save(final_path);
  gan::WassersteinGanModel deployed(gp, false);
  deployed.Load(final_path);
  check::Require(gan::BitEqual(model.AllArrays(), deployed.AllArrays()), "all 8 vars restore exact");
  const auto before = deployed.AllArrays();
  const auto output = deployed.Evaluate(test, tz, ta);
  const auto hand = output.at("input_gradient").to_vector<float>(),
             symbolic = output.at("symbolic_input_gradient").to_vector<float>();
  double jacobian = 0;
  for (std::size_t i = 0; i < hand.size(); ++i) {
    jacobian = std::max(jacobian, std::abs(double(hand[i]) - symbolic[i]));
    check::Near(hand[i], symbolic[i], 2e-6, "explicit Jacobian matches actual input gradient");
  }
  const Tensor repeat = deployed.Generate(tz);
  check::Require(gan::BitEqual(repeat, output.at("fake")), "noise ONLY generation exact");
  const Tensor changed = deployed.Generate(gan::Gaussian(128, 2, 5573));
  check::Require(!gan::BitEqual(changed, repeat), "changed noise affects G");
  const auto single = deployed.Generate(gan::Rows(tz, {0})).to_vector<float>(), all = repeat.to_vector<float>();
  double     parity = 0;
  for (int j = 0; j < 2; ++j) parity = std::max(parity, std::abs(double(single[j]) - all[j]));
  check::Require(parity < 2e-6, "single batch parity");
  check::Require(gan::BitEqual(before, deployed.AllArrays()), "immutable inference");
  const int     outside = Plot(dir / "samples.png", test, repeat);
  std::ofstream meta(dir / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":147,\"gradient_penalty\":" << (gp ? "true" : "false") << ",\"epochs\":" << epochs
       << ",\"generator_steps\":" << gsteps << ",\"critic_steps\":" << dsteps
       << ",\"maximum_clipped_critic_parameter\":" << clipmax << ",\"plot_outside_count\":" << outside
       << ",\"single_batch_max_error\":" << parity << ",\"jacobian_max_error\":" << jacobian;
  for (auto key : {"critic_loss", "generator_loss", "critic_gap", "penalty"})
    meta << ",\"test_" << key << "\":" << Scalar(output, key);
  meta << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "complete WGAN records");
  std::cout << (gp ? "GP" : "clip") << ": D steps=" << dsteps << ", G steps=" << gsteps
            << ", test gap=" << Scalar(output, "critic_gap") << ", GP=" << Scalar(output, "penalty")
            << ", critic loss=" << Scalar(output, "critic_loss") << ", Jacobian max error=" << jacobian
            << "; isolation/clip/restore/noiseOnly/batch/immutable PASS (training " << seconds << " s)" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::filesystem::path out    = paths::OutputRoot() / "models" / "wasserstein_gan", init;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0)
        epochs = std::stoi(a.substr(7));
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else if (a.rfind("init=", 0) == 0)
        init = a.substr(5);
      else
        throw std::invalid_argument("expected epochs=N out=PATH init=DIR");
    }
    check::Require(epochs > 0 && epochs <= 100, "three default or explicit 1..100 epochs");
    std::filesystem::create_directories(out);
    const Tensor fit = Points(400, 5507), validation = Points(128, 5519), test = Points(128, 5521);
    gan::SaveArrays((out / "data.npz").string(), {{"fit", fit}, {"validation", validation}, {"test", test}});
    Train(out, fit, validation, test, epochs, false, init);
    Train(out, fit, validation, test, epochs, true, init);
    std::cout << "Artifacts: " << out.string() << "\nAll wasserstein_gan checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
