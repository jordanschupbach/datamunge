// CMD: just run-nn style_gan
// StyleGAN (Karras et al. 2019, AdaIN) and StyleGAN2 (Karras et al. 2020,
// modulated/demodulated convolution): compact 8x8 MNIST CPU runs of three
// epochs each.
//
// Port of tflow's style_gan. Preparation (in ~/projects/tflow_examples): just
// autoencoder-assets; the data are read from $DATAMUNGE_DATA_DIR/saliency
// (saliency_data.npz), resampled 28 -> 8 bilinear (half-pixel). Both
// variants: mapping z (8) -> pixel norm -> 8 leaky -> 8 leaky = w, a learned
// 4x4x4 constant, a styled 3x3 layer at 4x4 and (after nearest x2) at 8x8 with
// explicit spatial noise times learned per-channel strengths (initialised 0),
// then a sigmoid 1-channel output. StyleGAN1: plain conv + noise + bias +
// leaky, then instance norm with an affine (1 + scale, shift) style from w
// (661 G parameters). StyleGAN2: per-example modulated conv with weight
// demodulation, no AdaIN, a modulated non-demodulated 1x1 RGB (625). Shared
// critic: 4x4/2 conv 4 -> 3x3 conv 8 -> 128 -> score (493), LSGAN objectives,
// separate Adam (.0002, beta1 .5), D then G, 96 steps per epoch, fixed final
// checkpoint. Controls: Z-only mapping, W-space synthesis, style mixing per
// layer, noise effects, truncation (psi 1 / .5 / 0 around the mean of 2048
// mapped prior draws), and for StyleGAN2 invariance of the demodulated output
// to a doubled first-layer style.
//
// Data, noise/latent banks and batch order are tflow's exactly; weights use
// datamunge's initializer unless init=DIR names tflow's output/models/style_gan
// (then adain/ and modulated/initial_arrays.npz are replayed and both runs
// follow TF's trajectories up to float rounding). tflow's per-step .npz dumps
// and index.html are not written. Outputs (data.npz and per variant adain/,
// modulated/: inputs, history, batches, initial/final checkpoint .npz,
// final_gradients_group*.npz, outputs.npz, style_controls.npz, generated.png,
// style_mixing.png, truncation_noise.png, metrics.json) go to
// output/nn/models/style_gan.
// Options: epochs=3 data=PATH out=PATH init=DIR.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include "common/paths.hpp"
#include "common/style_gan.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

double Value(const gan::Arrays& a, const std::string& key) { return a.at(key).item(); }

/// 28x28 -> 8x8 bilinear with half-pixel centres (scale 3.5).
Tensor Resize8(const Tensor& images) {
  const auto         in = images.contiguous().to_vector<float>();
  const std::int64_t n  = images.size(0);
  std::vector<float> out(n * 64);
  for (std::int64_t b = 0; b < n; ++b)
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) {
        const double py = (y + .5) * 3.5 - .5, px = (x + .5) * 3.5 - .5;
        const int    iy = int(std::floor(py)), ix = int(std::floor(px));
        double       v = 0;
        for (int dy = 0; dy < 2; ++dy)
          for (int dx = 0; dx < 2; ++dx)
            v += (dy ? py - iy : 1 - (py - iy)) * (dx ? px - ix : 1 - (px - ix)) * in[b * 784 + (iy + dy) * 28 + ix + dx];
        out[b * 64 + y * 8 + x] = float(v);
      }
  return nn::from_vector(out, {n, 8, 8, 1});
}

void Mosaic(const std::filesystem::path& path, const Tensor& images, std::int64_t columns) {
  nn::save_image_grid(nn::reshape(images, {-1, 8, 8, 1}), path.string(), columns, 0, 1, 4, 220 / 255., 8);
}

struct Inputs {
  Tensor z, noise4, noise8;
};
/// z [n, 8] from GaussianNoise(zseed); per row 80 noise values from nseed: 16 for 4x4, 64 for 8x8.
Inputs Draw(int n, std::uint32_t zseed, std::uint32_t nseed) {
  const Tensor z     = gan::Gaussian(n, 8, zseed);
  const Tensor noise = gan::Gaussian(n, 80, nseed);
  return {z, nn::reshape(nn::slice(noise, {0, 0}, {n, 16}), {n, 4, 4, 1}),
          nn::reshape(nn::slice(noise, {0, 16}, {n, 64}), {n, 8, 8, 1})};
}
void SaveInputs(const std::filesystem::path& path, const Inputs& x) {
  gan::SaveArrays(path.string(), {{"z", x.z}, {"noise4", x.noise4}, {"noise8", x.noise8}});
}
/// Every row replaced by row 0.
Tensor CommonNoise(const Tensor& x) {
  std::vector<std::int64_t> rows(std::size_t(x.size(0)), 0);
  return gan::Rows(x, rows);
}

void Train(const std::filesystem::path& root, const std::vector<Tensor>& images, int epochs, bool version2,
           const std::string& init) {
  const auto dir = root / (version2 ? "modulated" : "adain");
  std::filesystem::create_directories(dir);
  const auto validation = Draw(200, 6637, 6653), test = Draw(200, 6659, 6661);
  SaveInputs(dir / "validation_inputs.npz", validation);
  SaveInputs(dir / "test_inputs.npz", test);
  gan::StyleGanModel model(version2);
  if (!init.empty()) model.Load((std::filesystem::path(init) / (version2 ? "modulated" : "adain") / "initial_arrays.npz").string());
  check::Require(model.Count(0) == (version2 ? 625 : 661) && model.Count(1) == 493, "styled G/D parameter counts");
  model.Save((dir / "initial_arrays.npz").string());
  gan::SaveArrays((dir / "initial_outputs.npz").string(), model.Evaluate(images[2], test.z, test.noise4, test.noise8));
  const auto changed_initial = Draw(200, 6659, 6673);
  check::Require(gan::BitEqual(model.GenerateZ(test.z, test.noise4, test.noise8),
                               model.GenerateZ(test.z, changed_initial.noise4, changed_initial.noise8)),
                 "initial noise strength 0 means EXACT no effect");
  nn::BatchOrder order(1000, 32, 6679);
  int            steps = 0;
  std::ofstream  history(dir / "history.tsv"), batches(dir / "batches.tsv");
  history.precision(17);
  history << "epoch\td_loss\tg_loss\n";
  batches << "epoch\tstep\trow\n";
  Tensor     fx, fz, f4, f8;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    const auto input = Draw(1000, 6689 + epoch, 6701 + epoch);
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const Tensor x = gan::Rows(images[0], rows), z = gan::Rows(input.z, rows), n4 = gan::Rows(input.noise4, rows),
                   n8 = gan::Rows(input.noise8, rows);
      if (!steps) fx = x, fz = z, f4 = n4, f8 = n8;
      for (int group : {1, 0}) {
        const auto before = model.AllArrays();
        model.Update(group, x, z, n4, n8);
        gan::RequireOnlyOwnersChanged(before, model.AllArrays(), {std::to_string(group) + "/"},
                                      "styled GAN step updates ONLY own group", !steps);
      }
      for (auto row : rows) batches << epoch << '\t' << steps << '\t' << row << '\n';
      ++steps;
    }
    const auto v = model.Evaluate(images[1], validation.z, validation.noise4, validation.noise8);
    history << epoch << '\t' << Value(v, "d_loss") << '\t' << Value(v, "g_loss") << '\n';
    std::cout << (version2 ? "modulated" : "adain") << " epoch " << epoch << ": D/G=" << Value(v, "d_loss") << '/'
              << Value(v, "g_loss") << std::endl;
  }
  const double seconds    = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  const auto   checkpoint = (dir / "final.npz").string();
  model.Save(checkpoint);
  for (int group = 0; group < 2; ++group)
    gan::SaveArrays((dir / ("final_gradients_group" + std::to_string(group) + ".npz")).string(),
                    model.Gradient(group, fx, fz, f4, f8));
  gan::StyleGanModel deployed(version2, false, {1, 2});
  deployed.Load(checkpoint);
  check::Require(gan::BitEqual(model.AllArrays(), deployed.AllArrays()), "all styled GAN weights restore exact");
  const auto   before = deployed.AllArrays();
  const auto   output = deployed.Evaluate(images[2], test.z, test.noise4, test.noise8);
  const Tensor w      = deployed.Map(test.z);
  check::Require(gan::BitEqual(w, output.at("mapped")), "Z ONLY mapping exact");
  const Tensor plain  = deployed.GenerateZ(test.z, test.noise4, test.noise8),
               from_w = deployed.GenerateW(w, w, test.noise4, test.noise8);
  check::Require(gan::BitEqual(plain, output.at("fake")) && gan::BitEqual(plain, from_w),
                 "Z + noise ONLY matches W styles + noise ONLY synthesis");
  const double parity = gan::MaxAbsDiff(
      deployed.GenerateZ(gan::Rows(test.z, {0}), gan::Rows(test.noise4, {0}), gan::Rows(test.noise8, {0})),
      gan::Rows(plain, {0}));
  check::Require(parity < 3e-6, "styled G batch parity");
  const auto   other   = Draw(200, 6719, 6673);
  const Tensor other_w = deployed.Map(other.z);
  const Tensor mixed = deployed.GenerateW(w, other_w, test.noise4, test.noise8),
               mixed_first = deployed.GenerateW(other_w, w, test.noise4, test.noise8),
               changed_noise = deployed.GenerateZ(test.z, other.noise4, other.noise8);
  const double style_effect = std::max(gan::MaxAbsDiff(plain, mixed), gan::MaxAbsDiff(plain, mixed_first));
  const double noise_effect = gan::MaxAbsDiff(plain, changed_noise);
  check::Require(style_effect > 1e-5 && noise_effect > 1e-7, "actual per layer styles AND learned noise affect pixels");
  const auto   center_inputs = Draw(2048, 6733, 6737);
  const Tensor center_w      = deployed.Map(center_inputs.z);
  const Tensor center = nn::cast(nn::mean(nn::cast(center_w, nn::DType::Float64), {0}), nn::DType::Float32);
  const Tensor n4 = CommonNoise(test.noise4), n8 = CommonNoise(test.noise8);
  gan::Arrays  controls{{"w", w},
                        {"other_z", other.z},
                        {"other_w", other_w},
                        {"mixed_second", mixed},
                        {"mixed_first", mixed_first},
                        {"changed_noise4", other.noise4},
                        {"changed_noise8", other.noise8},
                        {"changed_noise_fake", changed_noise},
                        {"center_z", center_inputs.z},
                        {"center_w", center_w},
                        {"center", center},
                        {"common_noise4", n4},
                        {"common_noise8", n8}};
  controls.emplace("changed_only_noise4", deployed.GenerateZ(test.z, other.noise4, test.noise8));
  controls.emplace("changed_only_noise8", deployed.GenerateZ(test.z, test.noise4, other.noise8));
  if (version2) {
    const Tensor doubled = output.at("first_style") * 2.0;
    const auto   scaled  = deployed.FirstStyleControl(test.z, test.noise4, test.noise8, doubled);
    check::Require(gan::MaxAbsDiff(plain, scaled.at("fake")) <= 3e-6,
                   "demodulated kernel invariant to common positive input style factor up to epsilon");
    controls.emplace("doubled_first_style", doubled);
    controls.emplace("doubled_first_style_fake", scaled.at("fake"));
    controls.emplace("doubled_first_style_demodulator", scaled.at("demodulator"));
  }
  std::vector<Tensor> gallery(40), mix_gallery(40);
  for (int level = 0; level < 3; ++level) {
    const float  psi       = level == 0 ? 1 : level == 1 ? .5f : 0;
    const Tensor truncated = nn::expand_dims(center, 0) + double(psi) * (w - nn::expand_dims(center, 0));
    const Tensor image     = deployed.GenerateW(truncated, truncated, n4, n8);
    controls.emplace("truncated_w" + std::to_string(level), truncated);
    controls.emplace("truncated_fake" + std::to_string(level), image);
    if (level == 2)
      check::Require(gan::MaxAbsDiff(image, CommonNoise(image)) <= 3e-6, "psi 0 SAME w + SAME noise gives SAME image");
    for (int b = 0; b < 8; ++b) gallery[b * 5 + level] = gan::Rows(image, {b});
  }
  const Tensor other_plain = deployed.GenerateW(other_w, other_w, test.noise4, test.noise8);
  for (int b = 0; b < 8; ++b) {
    gallery[b * 5 + 3] = gan::Rows(plain, {b});
    gallery[b * 5 + 4] = gan::Rows(changed_noise, {b});
    mix_gallery[b * 5] = gan::Rows(plain, {b});
    mix_gallery[b * 5 + 1] = gan::Rows(mixed, {b});
    mix_gallery[b * 5 + 2] = gan::Rows(mixed_first, {b});
    mix_gallery[b * 5 + 3] = gan::Rows(other_plain, {b});
    mix_gallery[b * 5 + 4] = gan::Rows(changed_noise, {b});
  }
  controls.emplace("other_plain", other_plain);
  controls.emplace("truncation_gallery", nn::concat(gallery, 0));
  controls.emplace("mix_gallery", nn::concat(mix_gallery, 0));
  gan::SaveArrays((dir / "style_controls.npz").string(), controls);
  Mosaic(dir / "truncation_noise.png", controls.at("truncation_gallery"), 5);
  Mosaic(dir / "style_mixing.png", controls.at("mix_gallery"), 5);
  Mosaic(dir / "generated.png", nn::slice_axis(plain, 0, 0, 40), 8);
  check::Require(gan::BitEqual(before, deployed.AllArrays()), "all mapping/style mix/truncation/noise inference immutable");
  gan::SaveArrays((dir / "outputs.npz").string(), output);
  std::ofstream meta(dir / "metrics.json");
  meta.precision(17);
  meta << "{\"style2\":" << (version2 ? "true" : "false") << ",\"generator_parameters\":" << model.Count(0)
       << ",\"discriminator_parameters\":493,\"epochs\":" << epochs << ",\"steps_per_group\":" << steps
       << ",\"optimizer_updates\":" << 2 * steps << ",\"selected_epoch_fixed_final\":" << epochs
       << ",\"test_d_loss\":" << Value(output, "d_loss") << ",\"test_g_loss\":" << Value(output, "g_loss")
       << ",\"style_pixel_max_effect\":" << style_effect << ",\"noise_pixel_max_effect\":" << noise_effect
       << ",\"single_batch_max_error\":" << parity << ",\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "complete both styled GAN records");
  std::cout << (version2 ? "StyleGAN2" : "StyleGAN1") << ": test D/G=" << Value(output, "d_loss") << '/'
            << Value(output, "g_loss") << ", style effect=" << style_effect << ", noise effect=" << noise_effect
            << "; constant/mapping/W mix/truncation/noise/restore/immutable PASS (training " << seconds << " s)"
            << std::endl;
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  std::filesystem::create_directories(root);
  const auto          pools = gan::LoadSaliency(data);
  std::vector<Tensor> resized;
  gan::Arrays         records;
  for (int k = 0; k < 3; ++k) {
    resized.push_back(Resize8(pools[k].images));
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    records.emplace(split + "_original", pools[k].images);
    records.emplace(split + "_images", resized.back());
    records.emplace(split + "_labels", pools[k].labels);
    records.emplace(split + "_source_ids", pools[k].source);
  }
  gan::SaveArrays((root / "data.npz").string(), records);
  Train(root, resized, epochs, false, init);
  Train(root, resized, epochs, true, init);
  std::cout << "Artifacts: " << root.string() << "\nAll style_gan checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset("saliency"), out = paths::OutputRoot() / "models" / "style_gan";
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
        throw std::invalid_argument("expected epochs=N data=PATH out=PATH init=DIR");
    }
    check::Require(epochs > 0 && epochs <= 100, "three default or explicit 1..100 epochs");
    Run(data, out, epochs, init);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
