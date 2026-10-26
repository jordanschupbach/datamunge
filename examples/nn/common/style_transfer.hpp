// Gatys content/Gram objectives and projected-Adam pixel optimization: a port
// of tflow's common/ml/style_transfer.h (reused by johnson_fast_style).
// Images: JPEG decoded with datamunge::image::read_jpeg (libjpeg-turbo exact,
// TF DecodeJpeg defaults), centre square crop, TF ResizeBilinear (legacy
// corners) via nn::resize_bilinear, / 255.
// TF -> datamunge: the generated image is a leaf tensor updated by
// nn::Optimizer (Adam .03) and projected to [0, 1] in place; the image
// checkpoint is best_image.npz (TF SaveV2 bundle best_image).
#pragma once

#include <chrono>
#include <datamunge/image/jpeg.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

#include "common/perceptual.hpp"

namespace style {

/// Per-image Gram matrix divided by the number of positions: [B, C, C].
inline nn::Tensor StyleGram(const nn::Tensor& features) {
  const nn::Tensor flat = nn::reshape(features, {features.size(0), -1, features.size(3)});
  return nn::matmul(flat, flat, true) / double(flat.size(1));
}

/// .5 * sum((F - P)^2).
inline nn::Tensor GatysContentLoss(const nn::Tensor& features, const nn::Tensor& target) {
  return .5 * nn::sum(nn::squared_difference(features, target));
}

/// .25 * mean((G - A)^2) = the Gatys 1/(4 N^2 M^2) squared-Gram term with M-normalised Grams.
inline nn::Tensor GatysStyleLayer(const nn::Tensor& features, const nn::Tensor& target_gram) {
  return .25 * nn::mean(nn::squared_difference(StyleGram(features), target_gram));
}

/// Equal .2 weights over conv1_1..conv5_1.
inline nn::Tensor GatysStyleLoss(const std::array<nn::Tensor, 5>& features, const std::array<nn::Tensor, 5>& targets) {
  nn::Tensor total;
  for (std::size_t i = 0; i < 5; ++i) {
    const nn::Tensor layer = GatysStyleLayer(features[i], targets[i]);
    total                  = i ? total + layer : layer;
  }
  return .2 * total;
}

/// [1, side, side, 3] in [0, 1]: decode, centre square crop, ResizeBilinear, / 255.
inline nn::Tensor LoadStyleImage(const std::filesystem::path& path, int side) {
  datamunge::image::JpegDecodeOptions options;
  options.channels       = 3;
  const nn::Tensor decoded = nn::from_image(datamunge::image::read_jpeg(path.string(), options));  // [H, W, 3] 0..255
  const std::int64_t h = decoded.size(0), w = decoded.size(1), square = std::min(h, w);
  const nn::Tensor   crop    = nn::slice(decoded, {(h - square) / 2, (w - square) / 2, 0}, {square, square, 3});
  const nn::Tensor   resized = nn::resize_bilinear(nn::expand_dims(crop, 0), {side, side}) / 255.0;
  check::Require(resized.shape() == nn::Shape({1, side, side, 3}), "style-transfer RGB image shape");
  return resized.contiguous();
}

inline void WriteStylePng(const nn::Tensor& rgb, const std::filesystem::path& filename) {
  check::Require(rgb.dim() == 4 && rgb.size(0) == 1 && rgb.size(3) == 3, "single RGB image PNG");
  const auto         v = rgb.contiguous().to_vector<float>();
  std::vector<float> pixels(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    const double value = v[i];
    check::Require(std::isfinite(value) && value >= 0 && value <= 1, "finite projected image pixels");
    pixels[i] = float(std::lround(value * 255));
  }
  nn::save_png(nn::from_vector(pixels, {rgb.size(1), rgb.size(2), 3}), filename.string(), 0, 255);
}

inline void OptimizeGatys(const TensorMap& weights, const std::filesystem::path& assets, int side, int steps, float strength,
                          const std::filesystem::path& directory) {
  check::Require(side >= 32 && side % 16 == 0 && steps >= 2 && std::isfinite(strength) && strength > 0,
                 "positive style strength,>=2 steps and image side multiple of16>=32");
  const nn::Tensor content = LoadStyleImage(assets / "content.jpg", side), style = LoadStyleImage(assets / "style.jpg", side);
  std::vector<float>                    init(std::size_t(content.numel()));
  std::mt19937                          rng(17);
  std::uniform_real_distribution<float> noise(0, 1);
  const auto                            cv = content.to_vector<float>();
  for (std::size_t i = 0; i < init.size(); ++i) init[i] = .75f * cv[i] + .25f * noise(rng);
  const nn::Tensor initial = nn::from_vector(init, content.shape());
  const FrozenVgg19 vgg(weights);
  nn::Tensor                content_target;
  std::array<nn::Tensor, 5> style_targets;
  {
    nn::NoGradGuard guard;
    content_target   = vgg.Extract(content).content;
    const auto other = vgg.Extract(style);
    for (std::size_t i = 0; i < 5; ++i) style_targets[i] = StyleGram(other.style[i]);
  }
  nn::Tensor image = initial.clone().requires_grad_();
  auto evaluate = [&](double content_scale, double style_scale) {
    const auto       generated    = vgg.Extract(image);
    const nn::Tensor content_loss = GatysContentLoss(generated.content, content_target);
    const nn::Tensor style_loss   = GatysStyleLoss(generated.style, style_targets);
    const nn::Tensor objective    = content_loss / content_scale + double(strength) * (style_loss / style_scale);
    return std::array<nn::Tensor, 3>{objective, content_loss, style_loss};
  };
  double initial_content = 0, initial_style = 0;
  {
    nn::NoGradGuard guard;
    const auto      raw = evaluate(1, 1);
    initial_content = raw[1].item(), initial_style = raw[2].item();
  }
  check::Require(initial_content > 1e-12 && initial_style > 1e-12, "nondegenerate initial content/style targets");
  // TF feeds the scales as float32 scalars.
  const double  scale_c = double(float(initial_content)), scale_s = double(float(initial_style));
  nn::OptimizerConfig cfg;
  cfg.kind = nn::OptimizerKind::Adam;
  nn::Optimizer optimizer(std::vector<nn::Tensor>{image}, .03, cfg);
  std::filesystem::create_directories(directory);
  const auto    prefix = (directory / "best_image.npz").string();
  std::ofstream history(directory / "history.tsv");
  history << "step\tobjective\tcontent_loss\tstyle_loss\n";
  std::cout << "== Gatys: full frozen ImageNet VGG-19,average pools,content conv4_2/style conv1_1..conv5_1; " << side << "px," << steps
            << " steps,projected Adam.03" << std::endl;
  std::cout << " 20024384 verified imported parameters;12944960 used through conv5_1; only " << image.numel() << " RGB pixels optimized" << std::endl;
  std::cout << " initial=.75 content+.25 seeded noise; fixed raw loss denominators content=" << initial_content << ", style=" << initial_style
            << ", relative style strength=" << strength << " (constant coefficients for all steps)" << std::endl;
  double     best = INFINITY, initial_objective = 0;
  int        selected = -1;
  const auto started  = std::chrono::steady_clock::now();
  for (int step = 0; step <= steps; ++step) {
    if (step) {
      const auto losses = evaluate(scale_c, scale_s);
      optimizer.step(nn::grad(losses[0], {image}));
      nn::NoGradGuard guard;
      image.copy_(nn::clip(image, 0.0, 1.0));
    }
    std::array<double, 3> values;
    {
      nn::NoGradGuard guard;
      const auto      losses = evaluate(scale_c, scale_s);
      for (std::size_t i = 0; i < 3; ++i) values[i] = losses[i].item();
    }
    const double value = values[0];
    check::Require(std::isfinite(value), "finite pixel-optimization objective");
    history << step << '\t' << value << '\t' << values[1] << '\t' << values[2] << '\n';
    history.flush();
    if (step == 0) initial_objective = value;
    if (value < best) {
      best = value, selected = step;
      nn::save_npz(prefix, {{"image/pixels", image.detach()}});
    }
    if (step == 0 || step % 5 == 0 || step == steps)
      std::cout << "  step " << step << ": objective=" << value << ", relative content=" << values[1] / initial_content
                << ", relative style=" << values[2] / initial_style << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  {
    nn::NoGradGuard guard;
    image.copy_(nn::load_npz(prefix).at(0).second);
  }
  const auto   restored      = evaluate(scale_c, scale_s);
  const double gradient_norm = nn::norm(nn::grad(restored[0], {image})[0]).item();
  check::Near(restored[0].item(), best, 1e-6, "restored image objective matches selected step");
  check::Require(best < initial_objective - 1e-3 && selected > 0, "combined Gatys objective must improve");
  const double final_style = restored[2].item();
  check::Require(final_style < initial_style * .999 && std::isfinite(gradient_norm) && gradient_norm > 0,
                 "style statistics improve and image gradient remains finite nonzero");
  const nn::Tensor result = image.detach().clone();
  const double     change = nn::mean(nn::abs(result - initial)).item();
  check::Require(change > 1e-4, "actual generated pixels changed");
  vgg.CheckFrozen();
  WriteStylePng(content, directory / "content.png");
  WriteStylePng(style, directory / "style.png");
  WriteStylePng(initial, directory / "initial.png");
  WriteStylePng(result, directory / "stylized.png");
  WriteStylePng(nn::concat({content, style, initial, result}, 2), directory / "comparison.png");
  if (std::filesystem::exists(assets / "manifest.json"))
    std::filesystem::copy_file(assets / "manifest.json", directory / "assets_manifest.json", std::filesystem::copy_options::overwrite_existing);
  check::Require(bool(history), "write full pixel optimization history");
  std::cout << " selected step=" << selected << "; restored objective=" << best << " verified; relative content/style="
            << restored[1].item() / initial_content << '/' << final_style / initial_style << ", gradient norm=" << gradient_norm
            << ", mean pixel change=" << change << std::endl;
  std::cout << " all32 pretrained tensors /20024384 values unchanged bitwise; frozen feature targets, image-only checkpoint/history and "
               "content/style/initial/stylized PNGs: "
            << directory.string() << " (optimization " << seconds << " s)\n full short Gatys optimization demonstration passed" << std::endl;
}

}  // namespace style
