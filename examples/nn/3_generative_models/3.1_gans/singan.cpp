// CMD: just run-nn singan
// SinGAN (Shaham et al. 2019) core: one real CIFAR-10 image, three frozen
// coarse-to-fine patch GANs (16, 24, 32 pixels) and fixed-noise reconstruction.
//
// Port of tflow's singan. Data: data_batch_1.bin row 0 from
// $DATAMUNGE_DATA_DIR/cifar10, scaled to [-1, 1]; the pyramid uses the
// antialiased Keys bicubic of sr_data.hpp. Each scale: residual fully
// convolutional G (previous + amplitude noise -> 3 convs -> previous + tanh,
// 371 parameters) and a 3-layer local patch D (297), LSGAN objectives (not the
// paper's WGAN-GP) plus 10 x MSE of the fixed-noise reconstruction (noise only
// at the coarsest scale, zeros above). Amplitudes: 1 at scale 0, then .1 x RMS
// error of the upsampled reconstruction from the frozen lower scales. Scales
// k > 0 warm-start G and D from scale k - 1 with a cold Adam (.0005, beta1 .5);
// lower scales stay frozen. Per scale three default epochs of 16 D-then-G
// steps, each with a freshly drawn sample through the frozen lower scales
// (seed 7501 + 10000 k + 307 step). Then 12 samples, a 40x48 rectangular
// sample, the reconstruction and zero-noise-per-scale controls are drawn.
//
// The image, pyramid and every noise draw are tflow's exactly; weights use
// datamunge's initializer unless init=PATH names tflow's
// output/models/singan/scale0/initial_arrays.npz (then all three scales follow
// TF's trajectory up to float rounding: later scales warm-start from our own
// previous scale, as tflow). tflow's per-step .npz dumps, independent audit and
// index.html are not written. Outputs (data.npz, history, steps, per scale
// scale<k>/ initial_arrays / final.npz / outputs.npz /
// final_gradients_group*.npz, samples.npz, rectangular.npz,
// reconstruction.npz, noise_controls.npz, amplitudes.npz, the three PNGs,
// metrics.json) go to
// output/nn/models/singan.
// Options: epochs=3 data=PATH out=PATH init=PATH.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include "common/paths.hpp"
#include "common/single_image_gan.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using gan::Arrays;

namespace {

double Value(const Arrays& a, const std::string& key) { return a.at(key).item(); }
Tensor Zeros(int n, int h, int w) { return nn::zeros({n, h, w, 3}); }
/// tflow Noise: GaussianNoise of [n, h, w, 3] (scale > 0) or [n, h, w, 1] broadcast to RGB (scale 0).
Tensor Noise(int n, int h, int w, int scale, int seed) {
  const int  c   = scale ? 3 : 1;
  const auto raw = gan::GaussianNoise(std::int64_t(n) * h * w * c, std::uint32_t(seed));
  std::vector<float> out(std::size_t(n) * h * w * 3);
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = float(raw[(i / 3) * c + (scale ? i % 3 : 0)]);
  return nn::from_vector(out, {n, h, w, 3});
}
const int sizes[3] = {16, 24, 32};

using Models = std::vector<std::unique_ptr<gan::SingleImageGAN>>;

/// Coarse-to-fine draw through the first `count` scales (fixed reconstruction
/// noise when `reconstruction`; scale `zero_scale` gets zero noise).
Tensor Draw(Models& models, const std::vector<float>& amplitudes, const std::vector<Tensor>& fixed, int count, int n,
            int seed, bool reconstruction = false, int height = 32, int width = 32, Arrays* capture = nullptr,
            int zero_scale = -1) {
  Tensor result = Zeros(n, height / 2, width / 2);
  for (int k = 0; k < count; ++k) {
    const int h = k == 0 ? height / 2 : k == 1 ? height * 3 / 4 : height, w = k == 0 ? width / 2 : k == 1 ? width * 3 / 4 : width;
    if (k) result = sr::SRBicubic(result, h, w);
    Tensor z = reconstruction ? fixed[k] : Noise(n, h, w, k, seed + 97 * k);
    if (k == zero_scale) z = Zeros(n, h, w);
    const Tensor previous = result;
    result                = models[k]->Generate(previous, z, amplitudes[k]);
    if (capture) {
      const auto prefix = "scale" + std::to_string(k) + "_";
      capture->emplace(prefix + "noise", z);
      capture->emplace(prefix + "previous", previous);
      capture->emplace(prefix + "fake", result);
    }
  }
  return result;
}

/// [-1, 1] RGB images upscaled `scale` x, `columns` per row, no gaps.
void Png(const std::filesystem::path& path, const Tensor& images, std::int64_t columns, std::int64_t scale) {
  nn::save_image_grid(images, path.string(), columns, -1, 1, 0, 220 / 255., scale);
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  std::filesystem::create_directories(root);
  const auto   source = data::LoadCifar({data / "data_batch_1.bin"}, data::Cifar::k10);
  const Tensor image  = 2.0 * sr::SRPairs(source, {0}).hr - 1.0;
  Arrays       pyramid{{"source", image}};
  std::vector<Tensor> targets, fixed;
  for (int k = 0; k < 3; ++k) {
    targets.push_back(k == 2 ? image : sr::SRBicubic(image, sizes[k], sizes[k]));
    fixed.push_back(k ? Zeros(1, sizes[k], sizes[k]) : Noise(1, 16, 16, 0, 7411));
    pyramid.emplace("target" + std::to_string(k), targets.back());
    pyramid.emplace("fixed_noise" + std::to_string(k), fixed.back());
  }
  gan::SaveArrays((root / "data.npz").string(), pyramid);
  Models             models;
  std::vector<float> amplitudes;
  std::ofstream      history(root / "history.tsv"), steps(root / "steps.tsv");
  history.precision(17);
  history << "scale\tepoch\td_loss\tg_loss\tadversarial_loss\treconstruction_loss\n";
  steps << "scale\tepoch\tstep\tseed\n";
  const auto started = std::chrono::steady_clock::now();
  for (int k = 0; k < 3; ++k) {
    const auto dir = root / ("scale" + std::to_string(k));
    std::filesystem::create_directories(dir);
    std::vector<Arrays> lower;
    for (auto& m : models) lower.push_back(m->AllArrays());
    const Tensor prev_rec = k ? sr::SRBicubic(Draw(models, amplitudes, fixed, k, 1, 0, true), sizes[k], sizes[k]) : Zeros(1, 16, 16);
    const double error =
        nn::sum(nn::square(nn::cast(prev_rec, nn::DType::Float64) - nn::cast(targets[k], nn::DType::Float64))).item();
    amplitudes.push_back(k ? float(.1 * std::sqrt(error / double(prev_rec.numel()))) : 1.f);
    models.push_back(std::make_unique<gan::SingleImageGAN>());
    auto& model = *models.back();
    check::Require(model.Count(0) == 371 && model.Count(1) == 297, "668 parameters / scale, 12 tensors");
    if (k)
      model.Load((root / ("scale" + std::to_string(k - 1)) / "final.npz").string());  // warm start G and D, cold Adam
    else if (!init.empty())
      model.Load(init);
    model.Save((dir / "initial_arrays.npz").string());
    Arrays diagnostic;
    for (int epoch = 1; epoch <= epochs; ++epoch) {
      for (int step = 0; step < 16; ++step) {
        const int    global = (epoch - 1) * 16 + step, seed = 7501 + 10000 * k + global * 307;
        const Tensor prev   = k ? sr::SRBicubic(Draw(models, amplitudes, fixed, k, 1, seed), sizes[k], sizes[k]) : Zeros(1, 16, 16);
        Arrays       batch{{"image", targets[k]},
                           {"previous", prev},
                           {"noise", Noise(1, sizes[k], sizes[k], k, seed + 211)},
                           {"rec_previous", prev_rec},
                           {"rec_noise", fixed[k]},
                           {"amplitude", gan::SingleImageGAN::Scalar(amplitudes[k])}};
        if (!global) diagnostic = batch;
        for (int group : {1, 0}) {
          const auto before = model.AllArrays();
          model.Update(group, batch);
          gan::RequireOnlyOwnersChanged(before, model.AllArrays(), {std::to_string(group) + "/"},
                                        "only current G or D group updated", !global);
        }
        steps << k << '\t' << epoch << '\t' << global << '\t' << seed << '\n';
      }
      const auto output = model.Evaluate(diagnostic);
      history << k << '\t' << epoch;
      for (auto key : {"d_loss", "g_loss", "adversarial_loss", "reconstruction_loss"}) history << '\t' << Value(output, key);
      history << '\n';
      std::cout << "scale " << k << " epoch " << epoch << ": amp=" << amplitudes[k]
                << ", D/G=" << Value(output, "d_loss") << '/' << Value(output, "g_loss")
                << ", reconstruction MSE=" << Value(output, "reconstruction_loss") << std::endl;
    }
    const auto checkpoint = (dir / "final.npz").string();
    model.Save(checkpoint);
    for (int group = 0; group < 2; ++group)
      gan::SaveArrays((dir / ("final_gradients_group" + std::to_string(group) + ".npz")).string(),
                      model.Gradient(group, diagnostic));
    gan::SaveArrays((dir / "outputs.npz").string(), model.Evaluate(diagnostic));
    for (int i = 0; i < k; ++i)
      check::Require(gan::BitEqual(lower[i], models[i]->AllArrays()), "ALL lower G/D weights frozen for entire subsequent scale");
    auto restored = std::make_unique<gan::SingleImageGAN>(false, std::vector<std::uint64_t>{1, 2});
    restored->Load(checkpoint);
    check::Require(gan::BitEqual(model.AllArrays(), restored->AllArrays()), "all 12 scale weights BIT EXACT restore");
    const Tensor expected = model.Generate(diagnostic.at("previous"), diagnostic.at("noise"), amplitudes[k]);
    const Tensor actual   = restored->Generate(diagnostic.at("previous"), diagnostic.at("noise"), amplitudes[k]);
    check::Require(gan::BitEqual(expected, actual), "noise AND previous ONLY inference/restore exact");
    models.back() = std::move(restored);
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  std::vector<Arrays> before;
  for (auto& m : models) before.push_back(m->AllArrays());
  Arrays       outputs;
  const Tensor samples = Draw(models, amplitudes, fixed, 3, 12, 7901, false, 32, 32, &outputs);
  gan::SaveArrays((root / "samples.npz").string(), outputs);
  outputs.clear();
  const Tensor rectangular = Draw(models, amplitudes, fixed, 3, 4, 7919, false, 40, 48, &outputs);
  gan::SaveArrays((root / "rectangular.npz").string(), outputs);
  outputs.clear();
  const Tensor reconstruction = Draw(models, amplitudes, fixed, 3, 1, 0, true, 32, 32, &outputs);
  gan::SaveArrays((root / "reconstruction.npz").string(), outputs);
  outputs.clear();
  for (int k = 0; k < 3; ++k) {
    const Tensor controlled = Draw(models, amplitudes, fixed, 3, 12, 7901, false, 32, 32, nullptr, k);
    outputs.emplace("zero_noise_scale" + std::to_string(k), controlled);
    check::Require(gan::MaxAbsDiff(controlled, samples) > 1e-7, "ALL three noise levels affect final RGB");
  }
  gan::SaveArrays((root / "noise_controls.npz").string(), outputs);
  for (int k = 0; k < 3; ++k)
    check::Require(gan::BitEqual(before[k], models[k]->AllArrays()), "ALL inference including rectangular immutable");
  Png(root / "samples.png", samples, 4, 4);
  Png(root / "rectangular.png", rectangular, 2, 3);
  Png(root / "reconstruction.png", nn::concat({image, reconstruction}, 0), 2, 6);
  const Tensor s64 = nn::cast(samples, nn::DType::Float64);
  const double mse = nn::mean(nn::square(nn::cast(reconstruction, nn::DType::Float64) - nn::cast(image, nn::DType::Float64))).item();
  const double diversity = nn::mean(nn::square(s64 - nn::mean(s64, {0}, true))).item();
  // Frozen generator chain inputs for gan_diagnostics-style reuse (scale<k>/final.npz + these amplitudes).
  gan::SaveArrays((root / "amplitudes.npz").string(), {{"amplitudes", nn::from_vector(amplitudes, {3})}});
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"epochs_per_scale\":" << epochs << ",\"scales\":3,\"parameters_per_scale\":668,\"steps_per_group_per_scale\":"
       << epochs * 16 << ",\"total_optimizer_updates\":" << epochs * 16 * 6 << ",\"amplitudes\":[" << amplitudes[0] << ','
       << amplitudes[1] << ',' << amplitudes[2] << "],\"training_image_reconstruction_mse\":" << mse
       << ",\"sample_pixel_variance\":" << diversity << ",\"heldout_accuracy\":false,\"selected_epoch_fixed_final\":"
       << epochs << ",\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(meta) && bool(history) && bool(steps), "complete SinGAN records");
  std::cout << "single-image 3 scale reconstruction MSE=" << mse << ", sample variance=" << diversity
            << "; frozen lower/phase isolation/restore/noise levels/rectangular/immutable PASS (training " << seconds
            << " s)\nArtifacts: " << root.string() << "\nAll singan checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset("cifar10"), out = paths::OutputRoot() / "models" / "singan";
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
    check::Require(epochs > 0 && epochs <= 100, "three cheap epochs PER scale or deliberate 1..100 override");
    Run(data, out, epochs, init);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
