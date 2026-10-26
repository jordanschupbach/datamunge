// Alternating context-encoder GAN training and generator-only restoration: a
// port of tflow's common/ml/context_encoder.h.
// TF -> datamunge: the D step detaches the generator patch (nn::NoGradGuard,
// TF StopGradient) and updates D's moving BN only; the G step adds the
// non-saturating loss through D in inference mode and updates G's moving BN
// only, as tflow's control dependencies do. The checkpoint is best.npz with
// "generator/<name>" weights and "generator_state/<name>" moving BN; the
// deployment model has no discriminator or optimizer (TF inspects the
// GraphDef for the same property).
#pragma once

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

#include "common/checks.hpp"
#include "common/cifar.hpp"
#include "common/inpainting.hpp"
#include "common/inpainting_data.hpp"

namespace inpaint {

/// Valid mask [1, 32, 32, 1]: 0 inside the centre 16x16 hole.
inline nn::Tensor ContextMask() {
  std::vector<float> mask(1024);
  for (int y = 0; y < 32; ++y)
    for (int x = 0; x < 32; ++x) mask[std::size_t(y * 32 + x)] = !(y >= 8 && y < 24 && x >= 8 && x < 24);
  return nn::from_vector(mask, {1, 32, 32, 1});
}

using Snapshot = std::vector<std::vector<float>>;
inline Snapshot ContextSnapshot(const nn::ParameterStore& p, const nn::ParameterStore& state) {
  Snapshot out;
  for (const auto& v : p.parameters()) out.push_back(v.to_vector<float>());
  for (const auto& v : state.parameters()) out.push_back(v.to_vector<float>());
  return out;
}

class ContextModel {
 public:
  nn::ParameterStore             generator{71}, gstate{72}, discriminator{73}, dstate{74};
  std::unique_ptr<nn::Optimizer> g_optimizer, d_optimizer;
  ContextConfig                  config = ContextConfig::Compact();
  explicit ContextModel(bool train = true) {
    nn::NoGradGuard  guard;
    const nn::Tensor patch = ContextEncoder(generator, gstate, nn::zeros({1, 32, 32, 3}), config, false).patch;
    if (train) {
      ContextDiscriminator(discriminator, dstate, patch, config, false);
      nn::OptimizerConfig optimizer;
      optimizer.kind  = nn::OptimizerKind::Adam;
      optimizer.beta1 = .5;
      g_optimizer     = std::make_unique<nn::Optimizer>(generator, .002, optimizer);
      d_optimizer     = std::make_unique<nn::Optimizer>(discriminator, .0002, optimizer);
    }
  }
  /// {G total, reconstruction, G adversarial} (training-mode G, inference-mode D); `update` steps G and its BN.
  std::array<double, 3> Generator(const InpaintData& data, bool update = false) {
    std::unique_ptr<nn::NoGradGuard> guard;
    if (!update) guard = std::make_unique<nn::NoGradGuard>();
    const auto       training       = ContextEncoder(generator, gstate, data.masked, config, true, update);
    const auto       fool           = ContextDiscriminator(discriminator, dstate, training.patch, config, false);
    const nn::Tensor reconstruction = ContextReconstruction(training.patch, data.patch);
    const nn::Tensor adversarial    = ContextBCE(fool.logits, true);
    const nn::Tensor g_loss         = .999 * reconstruction + .001 * adversarial;
    const std::array<double, 3> values = {g_loss.item(), reconstruction.item(), adversarial.item()};
    for (double v : values) check::Require(std::isfinite(v) && v >= 0, "finite context G objectives");
    if (update) g_optimizer->step(nn::grad(g_loss, generator.parameters()));
    return values;
  }
  /// One D step on real targets and detached training-mode G patches (2 x mean logistic loss).
  double Discriminator(const InpaintData& data) {
    nn::Tensor fake;
    {
      nn::NoGradGuard guard;
      fake = ContextEncoder(generator, gstate, data.masked, config, true, false).patch;
    }
    const std::int64_t batch  = data.masked.size(0);
    const auto         d      = ContextDiscriminator(discriminator, dstate, nn::concat({data.patch, fake}, 0), config, true, true);
    const nn::Tensor   labels = nn::concat({nn::ones({batch}), nn::zeros({batch})}, 0);
    const nn::Tensor   d_loss = 2.0 * nn::mean(nn::softplus(d.logits) - labels * d.logits);
    const double       value  = d_loss.item();
    check::Require(std::isfinite(value) && value >= 0, "finite context discriminator objective");
    d_optimizer->step(nn::grad(d_loss, discriminator.parameters()));
    return value;
  }
  double Validation(const InpaintData& data) {
    nn::NoGradGuard guard;
    return ContextReconstruction(ContextEncoder(generator, gstate, data.masked, config, false).patch, data.patch).item();
  }
  /// Composited RGB in [0, 1] from the masked image alone.
  nn::Tensor Predict(const nn::Tensor& masked) {
    nn::NoGradGuard  guard;
    const nn::Tensor patch = ContextEncoder(generator, gstate, masked, config, false).patch;
    const nn::Tensor out   = .5 * (ContextComposite(masked, patch, ContextMask()) + 1.0);
    for (float v : out.to_vector<float>()) check::Require(std::isfinite(v), "finite composited inpainting pixels");
    return out;
  }
  void Save(const std::string& path) const {
    nn::NamedTensors arrays;
    for (std::size_t i = 0; i < generator.size(); ++i) arrays.push_back({"generator/" + generator.names()[i], generator.parameters()[i]});
    for (std::size_t i = 0; i < gstate.size(); ++i) arrays.push_back({"generator_state/" + gstate.names()[i], gstate.parameters()[i]});
    nn::save_npz(path, arrays);
  }
  void Restore(const std::string& path) {
    nn::NoGradGuard guard;
    const auto      arrays = nn::load_npz(path);
    check::Require(arrays.size() == generator.size() + gstate.size(), "complete context generator checkpoint");
    for (const auto& [name, value] : arrays) {
      if (name.rfind("generator_state/", 0) == 0)
        gstate.set(name.substr(16), value);
      else if (name.rfind("generator/", 0) == 0)
        generator.set(name.substr(10), value);
      else
        throw std::runtime_error("unexpected context checkpoint array " + name);
    }
  }
};

inline std::vector<std::size_t> Rows(std::int64_t start, std::int64_t end) {
  std::vector<std::size_t> rows;
  for (std::int64_t i = start; i < end; ++i) rows.push_back(std::size_t(i));
  return rows;
}

inline double ContextValidation(ContextModel& model, const InpaintData& data) {
  double sum = 0;
  for (std::int64_t row = 0; row < data.truth.size(0); row += 8) {
    const auto rows = Rows(row, std::min<std::int64_t>(row + 8, data.truth.size(0)));
    sum += double(rows.size()) * model.Validation(InpaintBatch(data, rows));
  }
  return sum / double(data.truth.size(0));
}

inline nn::Tensor ContextPredict(ContextModel& model, const InpaintData& data) {
  std::vector<nn::Tensor> parts;
  for (std::int64_t row = 0; row < data.truth.size(0); row += 8)
    parts.push_back(model.Predict(InpaintBatch(data.masked, Rows(row, std::min<std::int64_t>(row + 8, data.truth.size(0))))));
  return nn::concat(parts, 0).contiguous();
}

/// masked | context mean | learned | truth | 4x red error, 3x scaled, 8 rows.
inline void ContextPreview(const InpaintData& data, const nn::Tensor& learned, const nn::Tensor& mean, const std::filesystem::path& path) {
  const int  rows = int(std::min<std::int64_t>(8, data.truth.size(0))), scale = 3, W = 5 * 96;
  const auto t = data.truth.to_vector<float>(), l = learned.to_vector<float>(), m = data.masked.to_vector<float>(), f = mean.to_vector<float>();
  std::vector<float> image(std::size_t(rows) * 96 * W * 3);
  for (int n = 0; n < rows; ++n)
    for (int y = 0; y < 32; ++y)
      for (int x = 0; x < 32; ++x)
        for (int c = 0; c < 3; ++c) {
          const std::size_t i     = std::size_t(((n * 32 + y) * 32 + x) * 3 + c);
          const double      truth = t[i], value = l[i];
          const double values[] = {(m[i] + 1) / 2, f[i], value, truth, c == 0 ? 4 * std::abs(value - truth) : 0};
          for (int col = 0; col < 5; ++col)
            for (int dy = 0; dy < scale; ++dy)
              for (int dx = 0; dx < scale; ++dx)
                image[(std::size_t((n * 32 + y) * scale + dy) * W + std::size_t((col * 32 + x) * scale + dx)) * 3 + std::size_t(c)] =
                    float(std::lround(255 * std::clamp(values[col], 0., 1.)));
        }
  nn::save_png(nn::from_vector(image, {rows * 96, W, 3}), path.string(), 0, 255);
}

}  // namespace inpaint
