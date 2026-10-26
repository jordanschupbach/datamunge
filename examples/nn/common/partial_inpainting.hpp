// Short full-objective partial-convolution inpainting and target-free
// deployment: a port of tflow's common/ml/partial_inpainting.h.
// TF -> datamunge: training-mode U-Net calls update the moving BN only on
// optimizer steps; VGG targets for the ground truth are cached once. The
// checkpoint is best.npz ("partial/<name>" weights, "statistics/<name>" moving
// BN); the deployment model has no VGG or optimizer and takes only the masked
// image and mask (TF inspects the GraphDef placeholders for the same property).
#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>

#include "common/checks.hpp"
#include "common/cifar.hpp"
#include "common/partial_data.hpp"
#include "common/partial_losses.hpp"

namespace inpaint {

struct PartialPrepared {
  PartialData     data;
  PartialFeatures features;
};
inline PartialPrepared PartialBatch(const PartialPrepared& prepared, const std::vector<std::size_t>& rows) {
  return {PartialBatch(prepared.data, rows),
          {InpaintBatch(prepared.features[0], rows), InpaintBatch(prepared.features[1], rows), InpaintBatch(prepared.features[2], rows)}};
}

using PartialSnapshot = std::vector<std::vector<float>>;

class PartialModel {
 public:
  nn::ParameterStore                  parameters{83}, state{84};
  std::unique_ptr<style::FrozenVgg16> vgg;
  std::unique_ptr<nn::Optimizer>      optimizer;
  explicit PartialModel(const style::TensorMap* weights = nullptr) {
    {
      nn::NoGradGuard guard;
      PartialUNet(parameters, state, nn::zeros({1, 32, 32, 3}), nn::ones({1, 32, 32, 3}), PartialConfig::Compact(), false);
    }
    if (weights) {
      vgg = std::make_unique<style::FrozenVgg16>(*weights);
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      optimizer   = std::make_unique<nn::Optimizer>(parameters, .0002, config);
    }
  }
  /// {total, valid, hole, perceptual, style_raw, style_comp, TV}; `update` steps Adam and the moving BN.
  std::array<double, 7> Loss(const PartialPrepared& batch, bool train = false, bool update = false) {
    check::Require(bool(vgg) && (!update || train), "partial objective needs training model");
    std::unique_ptr<nn::NoGradGuard> guard;
    if (!update) guard = std::make_unique<nn::NoGradGuard>();
    const auto&      d    = batch.data;
    const nn::Tensor raw  = PartialUNet(parameters, state, d.input, d.mask, PartialConfig::Compact(), train, update).image;
    const nn::Tensor comp = PartialComposite(d.truth, raw, d.mask);
    const auto       all  = PartialVgg(*vgg, nn::concat({raw, comp}, 0));
    const std::int64_t n  = raw.size(0);
    PartialFeatures  rf, cf;
    for (std::size_t i = 0; i < 3; ++i) {
      const auto& f = all[i];
      rf[i]         = nn::slice(f, {0, 0, 0, 0}, {n, f.size(1), f.size(2), f.size(3)});
      cf[i]         = nn::slice(f, {n, 0, 0, 0}, {n, f.size(1), f.size(2), f.size(3)});
    }
    const auto            losses = PartialObjective(raw, d.truth, d.mask, rf, cf, batch.features);
    std::array<double, 7> result;
    for (std::size_t i = 0; i < 7; ++i) {
      result[i] = losses[i].item();
      check::Require(std::isfinite(result[i]) && result[i] >= 0, "finite every partial loss component");
    }
    if (update) optimizer->step(nn::grad(losses[0], parameters.parameters()));
    return result;
  }
  PartialPrepared Prepare(const PartialData& data) {
    check::Require(bool(vgg), "VGG required only while preparing training targets");
    nn::NoGradGuard                   guard;
    PartialPrepared                   result{data, {}};
    std::array<std::vector<nn::Tensor>, 3> parts;
    for (std::int64_t row = 0; row < data.truth.size(0); row += 2) {
      std::vector<std::size_t> rows;
      for (std::int64_t i = row; i < std::min<std::int64_t>(row + 2, data.truth.size(0)); ++i) rows.push_back(std::size_t(i));
      const auto f = PartialVgg(*vgg, InpaintBatch(data.truth, rows));
      for (std::size_t i = 0; i < 3; ++i) parts[i].push_back(f[i]);
    }
    for (std::size_t i = 0; i < 3; ++i) result.features[i] = nn::concat(parts[i], 0).contiguous();
    return result;
  }
  /// {composite, raw} from the masked image and mask only.
  std::pair<nn::Tensor, nn::Tensor> Predict(const PartialData& data) {
    nn::NoGradGuard  guard;
    const nn::Tensor raw       = PartialUNet(parameters, state, data.input, data.mask, PartialConfig::Compact(), false).image.contiguous();
    const nn::Tensor composite = PartialComposite(data.input, raw, data.mask).contiguous();
    for (const auto& t : {composite, raw})
      for (float v : t.to_vector<float>()) check::Require(std::isfinite(v), "finite raw partial prediction");
    return {composite, raw};
  }
  PartialSnapshot Snapshot() const {
    PartialSnapshot out;
    for (const auto& v : parameters.parameters()) out.push_back(v.to_vector<float>());
    for (const auto& v : state.parameters()) out.push_back(v.to_vector<float>());
    return out;
  }
  void Save(const std::string& path) const {
    nn::NamedTensors arrays;
    for (std::size_t i = 0; i < parameters.size(); ++i) arrays.push_back({"partial/" + parameters.names()[i], parameters.parameters()[i]});
    for (std::size_t i = 0; i < state.size(); ++i) arrays.push_back({"statistics/" + state.names()[i], state.parameters()[i]});
    nn::save_npz(path, arrays);
  }
  void Restore(const std::string& path) {
    nn::NoGradGuard guard;
    const auto      arrays = nn::load_npz(path);
    check::Require(arrays.size() == parameters.size() + state.size(), "complete partial UNet checkpoint");
    for (const auto& [name, value] : arrays) {
      if (name.rfind("statistics/", 0) == 0)
        state.set(name.substr(11), value);
      else if (name.rfind("partial/", 0) == 0)
        parameters.set(name.substr(8), value);
      else
        throw std::runtime_error("unexpected partial checkpoint array " + name);
    }
  }
};

inline std::vector<std::size_t> PartialRows(std::int64_t start, std::int64_t end) {
  std::vector<std::size_t> rows;
  for (std::int64_t i = start; i < end; ++i) rows.push_back(std::size_t(i));
  return rows;
}

inline double PartialValidation(PartialModel& model, const PartialPrepared& data) {
  double             sum = 0;
  const std::int64_t n   = data.data.truth.size(0);
  for (std::int64_t row = 0; row < n; row += 2) {
    const auto rows = PartialRows(row, std::min<std::int64_t>(row + 2, n));
    sum += double(rows.size()) * model.Loss(PartialBatch(data, rows))[0];
  }
  return sum / double(n);
}

inline std::pair<nn::Tensor, nn::Tensor> PartialPredict(PartialModel& model, const PartialData& data) {
  std::vector<nn::Tensor> composite, raw;
  const std::int64_t      n = data.truth.size(0);
  for (std::int64_t row = 0; row < n; row += 2) {
    const auto out = model.Predict(PartialBatch(data, PartialRows(row, std::min<std::int64_t>(row + 2, n))));
    composite.push_back(out.first);
    raw.push_back(out.second);
  }
  return {nn::concat(composite, 0).contiguous(), nn::concat(raw, 0).contiguous()};
}

/// observed | visible mean | prediction | truth | 4x red error, 3x scaled, 8 rows.
inline void PartialPreview(const PartialData& data, const nn::Tensor& predicted, const nn::Tensor& mean, const std::filesystem::path& path) {
  const int  rows = int(std::min<std::int64_t>(8, data.truth.size(0)));
  const auto in = data.input.to_vector<float>(), me = mean.to_vector<float>(), pr = predicted.to_vector<float>(), tr = data.truth.to_vector<float>();
  std::vector<float> image(std::size_t(rows) * 96 * 480 * 3);
  for (int n = 0; n < rows; ++n)
    for (int y = 0; y < 32; ++y)
      for (int x = 0; x < 32; ++x) {
        const std::size_t base  = std::size_t(((n * 32 + y) * 32 + x) * 3);
        double            error = 0;
        for (int c = 0; c < 3; ++c) error += std::abs(pr[base + std::size_t(c)] - tr[base + std::size_t(c)]) / 3;
        for (int c = 0; c < 3; ++c) {
          const std::size_t i        = base + std::size_t(c);
          const double      values[] = {in[i], me[i], pr[i], tr[i], c == 0 ? 4 * error : 0};
          for (int col = 0; col < 5; ++col)
            for (int dy = 0; dy < 3; ++dy)
              for (int dx = 0; dx < 3; ++dx)
                image[(std::size_t((n * 32 + y) * 3 + dy) * 480 + std::size_t((col * 32 + x) * 3 + dx)) * 3 + std::size_t(c)] =
                    float(std::lround(255 * std::clamp(values[col], 0., 1.)));
        }
      }
  nn::save_png(nn::from_vector(image, {rows * 96, 480, 3}), path.string(), 0, 255);
}

}  // namespace inpaint
