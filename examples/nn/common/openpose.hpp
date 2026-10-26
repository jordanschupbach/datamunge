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

namespace pose {

struct OpenPoseResult {
  std::array<nn::Tensor, 6> heatmaps, pafs;
  std::vector<nn::Tensor>   stages;
};

/// VGG-10 backbone + CPM-2 features, then six stages of a PAF (38) and a
/// confidence (19) branch; stages > 0 see the previous predictions and the
/// features. `divisor` divides every hidden width (8 = compact model).
inline OpenPoseResult OpenPose(nn::ParameterStore& p, const nn::Tensor& rgb, int divisor = 1) {
  if (divisor < 1 || 64 % divisor || 128 % divisor) throw std::invalid_argument("OpenPose width divisor must divide64 and128");
  auto conv = [&](const std::string& name, const nn::Tensor& input, std::int64_t out, std::int64_t kernel, bool activate = true) {
    nn::ConvLayerOptions config;
    config.init       = nn::init::he_normal();
    config.activation = activate ? nn::Activation::Relu : nn::Activation::None;
    return nn::conv2d(p, name, input, out, kernel, config);
  };
  OpenPoseResult result;
  // Published input convention: BGR bytes/256-.5, starting from RGB[0,1].
  nn::Tensor x = nn::gather(rgb, nn::from_vector(std::vector<std::int64_t>{2, 1, 0}, {3}), 3) * (255. / 256) - .5;
  const int  channels[] = {64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 256, 128};
  for (int layer = 0; layer < 12; ++layer) {
    x = conv("backbone/" + std::to_string(layer), x, channels[layer] / divisor, 3);
    result.stages.push_back(x);
    if (layer == 1 || layer == 3 || layer == 7) {
      x = nn::max_pool2d(x, 2, 2, nn::Padding::Valid);
      result.stages.push_back(x);
    }
  }
  const nn::Tensor features = x;
  for (int stage = 0; stage < 6; ++stage) {
    nn::Tensor input = features;
    if (stage > 0) {
      input = nn::concat({result.pafs[std::size_t(stage - 1)], result.heatmaps[std::size_t(stage - 1)], features}, 3);
      result.stages.push_back(input);
    }
    for (int branch = 0; branch < 2; ++branch) {
      x                = input;
      const int layers = stage == 0 ? 5 : 7;
      for (int layer = 0; layer < layers; ++layer) {
        const bool final  = layer + 1 == layers;
        const int  out    = final ? (branch == 0 ? 38 : 19) : (stage == 0 && layer == 3 ? 512 : 128) / divisor;
        const int  kernel = stage == 0 ? (layer < 3 ? 3 : 1) : (layer < 5 ? 7 : 1);
        x = conv("stage" + std::to_string(stage) + "/" + (branch == 0 ? "paf/" : "heat/") + std::to_string(layer), x, out, kernel, !final);
        result.stages.push_back(x);
      }
      (branch == 0 ? result.pafs : result.heatmaps)[std::size_t(stage)] = x;
    }
  }
  return result;
}

using Snapshot = std::vector<std::vector<float>>;

class OpenPoseModel {
 public:
  nn::ParameterStore             parameters{197};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit OpenPoseModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      OpenPose(parameters, nn::zeros({1, 128, 128, 3}), 8);
    }
    if (training) {
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      optimizer   = std::make_unique<nn::Optimizer>(parameters, .001, config);
    }
  }
  /// {total, heat1, paf1, ..., heat6, paf6}; `update` applies one Adam step.
  std::array<double, 13> Loss(const PafData& data, bool update = false) {
    check::Require(bool(optimizer), "supervised PAF objectives exist only in training graph");
    const auto                       target = PreparePafTargets(data);
    std::unique_ptr<nn::NoGradGuard> guard;
    if (!update) guard = std::make_unique<nn::NoGradGuard>();
    const auto              network = OpenPose(parameters, data.rgb, 8);
    std::vector<nn::Tensor> losses(13);
    nn::Tensor              total;
    for (std::size_t stage = 0; stage < 6; ++stage) {
      losses[1 + 2 * stage] = PafMaskedMSE(network.heatmaps[stage], target.heatmap, target.heatmask);
      losses[2 + 2 * stage] = PafMaskedMSE(network.pafs[stage], target.paf, target.pafmask);
      total = stage ? total + losses[1 + 2 * stage] + losses[2 + 2 * stage] : losses[1] + losses[2];
    }
    losses[0] = total;
    std::array<double, 13> result{};
    for (std::size_t i = 0; i < 13; ++i) {
      result[i] = losses[i].item();
      check::Require(std::isfinite(result[i]) && result[i] >= 0, "every finite heatmap/PAF stage objective");
    }
    if (update) optimizer->step(nn::grad(total, parameters.parameters()));
    return result;
  }
  /// Final-stage (heatmap [N,16,16,19], PAF [N,16,16,38]) from RGB alone.
  std::pair<nn::Tensor, nn::Tensor> Predict(const nn::Tensor& rgb) {
    nn::NoGradGuard guard;
    const auto      network = OpenPose(parameters, rgb, 8);
    const auto      heat = network.heatmaps[5].contiguous(), paf = network.pafs[5].contiguous();
    check::Require(heat.shape() == nn::Shape{rgb.size(0), 16, 16, 19} && paf.shape() == nn::Shape{rgb.size(0), 16, 16, 38},
                   "final heatmap/PAF field dimensions");
    for (const auto& field : {heat, paf})
      for (float v : field.to_vector<float>()) check::Require(std::isfinite(v), "finite unbounded signed fields");
    return {heat, paf};
  }
  [[nodiscard]] Snapshot Values() const {
    Snapshot out;
    for (const auto& v : parameters.parameters()) out.push_back(v.to_vector<float>());
    return out;
  }
  void Save(const std::string& path) const {
    nn::NamedTensors arrays;
    for (std::size_t i = 0; i < parameters.size(); ++i) arrays.push_back({"openpose/" + parameters.names()[i], parameters.parameters()[i]});
    nn::save_npz(path, arrays);
  }
  void Restore(const std::string& path) {
    nn::NoGradGuard guard;
    const auto      arrays = nn::load_npz(path);
    check::Require(arrays.size() == parameters.size(), "complete OpenPose checkpoint");
    for (const auto& [name, value] : arrays) {
      check::Require(name.rfind("openpose/", 0) == 0, "unexpected OpenPose checkpoint array " + name);
      parameters.set(name.substr(9), value);
    }
  }
};

inline std::vector<std::size_t> PoseRows(std::int64_t start, std::int64_t end) {
  std::vector<std::size_t> rows;
  for (std::int64_t i = start; i < end; ++i) rows.push_back(std::size_t(i));
  return rows;
}

/// Final fields for every image, predicted two images at a time.
inline std::pair<nn::Tensor, nn::Tensor> OpenPosePredict(OpenPoseModel& model, const PafData& data) {
  std::vector<nn::Tensor> heat, paf;
  for (std::int64_t n = 0; n < data.rgb.size(0); n += 2) {
    const auto value = model.Predict(PoseBatch(data.rgb, PoseRows(n, std::min<std::int64_t>(n + 2, data.rgb.size(0)))));
    heat.push_back(value.first);
    paf.push_back(value.second);
  }
  return {nn::concat(heat, 0).contiguous(), nn::concat(paf, 0).contiguous()};
}

inline double OpenPoseValidation(OpenPoseModel& model, const PafData& data) {
  double total = 0;
  for (std::int64_t n = 0; n < data.rgb.size(0); ++n) total += model.Loss(PafBatch(data, {std::size_t(n)}))[0];
  return total / double(data.rgb.size(0));
}

inline std::vector<PafDecoded> DecodePafBatch(const nn::Tensor& heat, const nn::Tensor& paf) {
  const PafField          h(heat), f(paf);
  std::vector<PafDecoded> result;
  for (int n = 0; n < int(heat.size(0)); ++n) result.push_back(DecodePafs(h, f, n));
  return result;
}

/// Host masked MSE: per-image mean over mask > .5, averaged over images with any mask.
inline double HostPafMSE(const nn::Tensor& prediction, const nn::Tensor& truth, const nn::Tensor& mask) {
  check::Require(prediction.shape() == truth.shape() && truth.shape() == mask.shape(), "host masked field metric shapes");
  const auto          p = prediction.to_vector<float>(), t = truth.to_vector<float>(), m = mask.to_vector<float>();
  const std::int64_t  batch = truth.size(0), size = truth.numel() / batch;
  double              total = 0;
  int                 images = 0;
  for (std::int64_t n = 0; n < batch; ++n) {
    double sum   = 0;
    int    count = 0;
    for (std::int64_t i = 0; i < size; ++i)
      if (m[std::size_t(n * size + i)] > .5) {
        const double difference = double(p[std::size_t(n * size + i)]) - t[std::size_t(n * size + i)];
        sum += difference * difference;
        ++count;
      }
    if (count) total += sum / count, ++images;
  }
  return images ? total / images : 0;
}

/// Per-element masked mean of the fitting targets, repeated `count` times.
inline nn::Tensor PafMeanField(const nn::Tensor& fitting, const nn::Tensor& mask, std::int64_t count) {
  const auto          f = fitting.to_vector<float>(), m = mask.to_vector<float>();
  const std::int64_t  size = fitting.numel() / fitting.size(0);
  std::vector<float>  result(std::size_t(count * size));
  for (std::int64_t i = 0; i < size; ++i) {
    double sum     = 0;
    int    present = 0;
    for (std::int64_t n = 0; n < fitting.size(0); ++n)
      if (m[std::size_t(n * size + i)] > .5) sum += f[std::size_t(n * size + i)], ++present;
    const float value = present ? float(sum / present) : 0.f;
    for (std::int64_t n = 0; n < count; ++n) result[std::size_t(n * size + i)] = value;
  }
  auto shape = fitting.shape();
  shape[0]   = count;
  return nn::from_vector(result, shape);
}

struct PafSummary {
  double heat_mse, paf_mse, precision, recall, f1;
  PafMetrics score;
};

inline PafSummary PrintPafScore(const std::string& label, const nn::Tensor& heat, const nn::Tensor& paf, const PafTargets& truth,
                                const PafData& data, const std::vector<PafDecoded>& decoded) {
  const auto   score = ScorePafs(decoded, data);
  const double precision = score.precision(), recall = score.recall(), f1 = precision + recall ? 2 * precision * recall / (precision + recall) : 0;
  PafSummary   s{HostPafMSE(heat, truth.heatmap, truth.heatmask), HostPafMSE(paf, truth.paf, truth.pafmask), precision, recall, f1, score};
  std::cout << " " << label << ":heatMSE=" << s.heat_mse << ",PAFMSE=" << s.paf_mse << ",grouped annotated-joint precision/recall/F1=" << 100 * precision
            << "/" << 100 * recall << "/" << 100 * f1 << "%,correct/pred/GT=" << score.correct << "/" << score.predicted << "/" << score.annotated
            << ",people=" << score.people << "/" << score.true_people << " (custom greedy .2box matching;not COCO AP)\n";
  return s;
}

/// Per image: RGB | truth skeletons (green) | decoded skeletons (red) | max joint confidence | max PAF magnitude.
inline void PafPreview(const PafData& data, const std::vector<PafDecoded>& decoded, const nn::Tensor& heat, const nn::Tensor& paf,
                       const std::filesystem::path& file) {
  const std::int64_t batch = data.rgb.size(0), P = data.joints.size(1);
  const auto         rgb = data.rgb.to_vector<float>(), jv = data.joints.to_vector<float>();
  const auto         counts = data.counts.to_vector<std::int32_t>();
  const PafField     h(heat), f(paf);
  std::vector<float> pixels(std::size_t(batch * 128 * 640 * 3), 0.f);
  auto view = [&](std::int64_t y, std::int64_t x, int c) -> float& { return pixels[std::size_t((y * 640 + x) * 3 + c)]; };
  for (std::int64_t n = 0; n < batch; ++n)
    for (int y = 0; y < 128; ++y)
      for (int x = 0; x < 128; ++x) {
        for (int column = 0; column < 3; ++column)
          for (int c = 0; c < 3; ++c)
            view(n * 128 + y, column * 128 + x, c) =
                float(std::clamp(int(std::lround(255 * rgb[std::size_t(((n * 128 + y) * 128 + x) * 3 + c)])), 0, 255));
        float peak = 0, magnitude = 0;
        for (int j = 0; j < 18; ++j) peak = std::max(peak, h(n, y / 8, x / 8, j));
        for (int e = 0; e < 19; ++e) magnitude = std::max(magnitude, std::hypot(f(n, y / 8, x / 8, 2 * e), f(n, y / 8, x / 8, 2 * e + 1)));
        view(n * 128 + y, 384 + x, 0) = float(std::lround(255 * std::clamp(peak, 0.f, 1.f)));
        view(n * 128 + y, 512 + x, 2) = float(std::lround(255 * std::clamp(magnitude, 0.f, 1.f)));
      }
  auto dot = [&](std::int64_t n, int column, long x, long y, int channel) {
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx)
        if (x + dx >= 0 && x + dx < 128 && y + dy >= 0 && y + dy < 128)
          for (int c = 0; c < 3; ++c) view(n * 128 + y + dy, column * 128 + x + dx, c) = c == channel ? 255.f : 0.f;
  };
  auto line = [&](std::int64_t n, int column, double ax, double ay, double bx, double by, int channel) {
    const int steps = std::max(1, int(std::ceil(std::hypot(bx - ax, by - ay))));
    for (int i = 0; i <= steps; ++i) dot(n, column, std::lround(ax + (bx - ax) * i / steps), std::lround(ay + (by - ay) * i / steps), channel);
  };
  auto joint = [&](std::int64_t n, int person, int j, int c) { return jv[std::size_t(((n * P + person) * 18 + j) * 3 + c)]; };
  for (std::int64_t n = 0; n < batch; ++n) {
    for (int person = 0; person < counts[std::size_t(n)]; ++person)
      for (const auto& edge : kPoseEdges)
        if (joint(n, person, edge[0], 2) > 0 && joint(n, person, edge[1], 2) > 0)
          line(n, 1, joint(n, person, edge[0], 0), joint(n, person, edge[0], 1), joint(n, person, edge[1], 0), joint(n, person, edge[1], 1), 1);
    for (const auto& person : decoded[std::size_t(n)].people)
      for (int index : person.links) {
        const auto& link = decoded[std::size_t(n)].links[std::size_t(index)];
        const auto& a    = decoded[std::size_t(n)].peaks[std::size_t(link.a)];
        const auto& b    = decoded[std::size_t(n)].peaks[std::size_t(link.b)];
        line(n, 2, (a.x + .5) * 8 - .5, (a.y + .5) * 8 - .5, (b.x + .5) * 8 - .5, (b.y + .5) * 8 - .5, 0);
      }
  }
  nn::save_png(nn::from_vector(pixels, {batch * 128, 640, 3}), file.string(), 0, 255);
}

}  // namespace pose
