// Six-stage parallel confidence/PAF OpenPose (COCO18), short fully supervised
// training and image-only restored inference: a port of tflow's
// common/nn/openpose.h (the network) and common/ml/openpose.h (the harness).
// TF -> datamunge: eager forward/backward instead of a session graph; the
// checkpoint is best.npz ("openpose/<name>" weights) instead of a SaveV2
// bundle; the deployment model is built without an optimizer and Predict()
// takes only RGB (TF inspects the GraphDef placeholders for the same property).
#pragma once

#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>

#include "common/paf_decode.hpp"
#include <datamunge/nn/image_io.hpp>
#include <datamunge/nn/openpose_models.hpp>
#include <datamunge/nn/paf_evaluation.hpp>

namespace pose {

using datamunge::nn::models::OpenPose;
using datamunge::nn::models::OpenPoseResult;
using Snapshot = std::vector<std::vector<float>>;

class OpenPoseModel {
public:
  nn::ParameterStore parameters{197};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit OpenPoseModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      OpenPose(parameters, nn::zeros({1, 128, 128, 3}), 8);
    }
    if (training) {
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .001, config);
    }
  }
  /// {total, heat1, paf1, ..., heat6, paf6}; `update` applies one Adam step.
  std::array<double, 13> Loss(const PafData &data, bool update = false) {
    check::Require(bool(optimizer), "supervised PAF objectives exist only in training graph");
    const auto target = PreparePafTargets(data);
    std::unique_ptr<nn::NoGradGuard> guard;
    if (!update)
      guard = std::make_unique<nn::NoGradGuard>();
    const auto network = OpenPose(parameters, data.rgb, 8);
    std::vector<nn::Tensor> losses(13);
    nn::Tensor total;
    for (std::size_t stage = 0; stage < 6; ++stage) {
      losses[1 + 2 * stage] =
          PafMaskedMSE(network.heatmaps[stage], target.heatmap, target.heatmask);
      losses[2 + 2 * stage] = PafMaskedMSE(network.pafs[stage], target.paf, target.pafmask);
      total = stage ? total + losses[1 + 2 * stage] + losses[2 + 2 * stage]
                    : losses[1] + losses[2];
    }
    losses[0] = total;
    std::array<double, 13> result{};
    for (std::size_t i = 0; i < 13; ++i) {
      result[i] = losses[i].item();
      check::Require(std::isfinite(result[i]) && result[i] >= 0,
                     "every finite heatmap/PAF stage objective");
    }
    if (update)
      optimizer->step(nn::grad(total, parameters.parameters()));
    return result;
  }
  /// Final-stage (heatmap [N,16,16,19], PAF [N,16,16,38]) from RGB alone.
  std::pair<nn::Tensor, nn::Tensor> Predict(const nn::Tensor &rgb) {
    nn::NoGradGuard guard;
    const auto network = OpenPose(parameters, rgb, 8);
    const auto heat = network.heatmaps[5].contiguous(), paf = network.pafs[5].contiguous();
    check::Require(heat.shape() == nn::Shape{rgb.size(0), 16, 16, 19} &&
                       paf.shape() == nn::Shape{rgb.size(0), 16, 16, 38},
                   "final heatmap/PAF field dimensions");
    for (const auto &field : {heat, paf})
      for (float v : field.to_vector<float>())
        check::Require(std::isfinite(v), "finite unbounded signed fields");
    return {heat, paf};
  }
  [[nodiscard]] Snapshot Values() const {
    Snapshot out;
    for (const auto &v : parameters.parameters())
      out.push_back(v.to_vector<float>());
    return out;
  }
  void Save(const std::string &path) const {
    nn::NamedTensors arrays;
    for (std::size_t i = 0; i < parameters.size(); ++i)
      arrays.push_back({"openpose/" + parameters.names()[i], parameters.parameters()[i]});
    nn::save_npz(path, arrays);
  }
  void Restore(const std::string &path) {
    nn::NoGradGuard guard;
    const auto arrays = nn::load_npz(path);
    check::Require(arrays.size() == parameters.size(), "complete OpenPose checkpoint");
    for (const auto &[name, value] : arrays) {
      check::Require(name.rfind("openpose/", 0) == 0,
                     "unexpected OpenPose checkpoint array " + name);
      parameters.set(name.substr(9), value);
    }
  }
};

inline std::vector<std::size_t> PoseRows(std::int64_t start, std::int64_t end) {
  std::vector<std::size_t> rows;
  for (std::int64_t i = start; i < end; ++i)
    rows.push_back(std::size_t(i));
  return rows;
}

/// Final fields for every image, predicted two images at a time.
inline std::pair<nn::Tensor, nn::Tensor> OpenPosePredict(OpenPoseModel &model,
                                                         const PafData &data) {
  std::vector<nn::Tensor> heat, paf;
  for (std::int64_t n = 0; n < data.rgb.size(0); n += 2) {
    const auto value = model.Predict(
        PoseBatch(data.rgb, PoseRows(n, std::min<std::int64_t>(n + 2, data.rgb.size(0)))));
    heat.push_back(value.first);
    paf.push_back(value.second);
  }
  return {nn::concat(heat, 0).contiguous(), nn::concat(paf, 0).contiguous()};
}

inline double OpenPoseValidation(OpenPoseModel &model, const PafData &data) {
  double total = 0;
  for (std::int64_t n = 0; n < data.rgb.size(0); ++n)
    total += model.Loss(PafBatch(data, {std::size_t(n)}))[0];
  return total / double(data.rgb.size(0));
}

inline std::vector<PafDecoded> DecodePafBatch(const nn::Tensor &heat, const nn::Tensor &paf) {
  const PafField h(heat), f(paf);
  std::vector<PafDecoded> result;
  for (int n = 0; n < int(heat.size(0)); ++n)
    result.push_back(DecodePafs(h, f, n));
  return result;
}

inline double HostPafMSE(const nn::Tensor &p, const nn::Tensor &t, const nn::Tensor &m) {
  return datamunge::nn::models::PafFieldMSE(p, t, m).item();
}
using datamunge::nn::models::PafMeanField;

struct PafSummary {
  double heat_mse, paf_mse, precision, recall, f1;
  PafMetrics score;
};

inline PafSummary PrintPafScore(const std::string &label, const nn::Tensor &heat,
                                const nn::Tensor &paf, const PafTargets &truth,
                                const PafData &data, const std::vector<PafDecoded> &decoded) {
  const auto score = ScorePafs(decoded, data);
  const double precision = score.precision(), recall = score.recall(),
               f1 = precision + recall ? (2 * nn::full({}, precision, nn::DType::Float64) *
                                          recall / (precision + recall))
                                             .item()
                                       : 0;
  PafSummary s{HostPafMSE(heat, truth.heatmap, truth.heatmask),
               HostPafMSE(paf, truth.paf, truth.pafmask),
               precision,
               recall,
               f1,
               score};
  std::cout << " " << label << ":heatMSE=" << s.heat_mse << ",PAFMSE=" << s.paf_mse
            << ",grouped annotated-joint precision/recall/F1=" << 100 * precision << "/"
            << 100 * recall << "/" << 100 * f1 << "%,correct/pred/GT=" << score.correct << "/"
            << score.predicted << "/" << score.annotated << ",people=" << score.people << "/"
            << score.true_people << " (custom greedy .2box matching;not COCO AP)\n";
  return s;
}

inline void PafPreview(const PafData &data, const std::vector<PafDecoded> &decoded,
                       const nn::Tensor &heat, const nn::Tensor &paf,
                       const std::filesystem::path &file) {
  const auto tiles = datamunge::nn::models::PafPreviewTiles(data, decoded, heat, paf);
  nn::save_png(tiles.reshape({tiles.size(0) * 128, 640, 3}), file.string());
}
} // namespace pose
