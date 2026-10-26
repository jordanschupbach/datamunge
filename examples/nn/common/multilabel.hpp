// Short ResNet-18 sigmoid multi-label training, BN restoration and
// validation-only thresholds: a port of tflow's common/ml/multilabel.h.
// TF -> datamunge: training-mode passes update the moving BN statistics
// eagerly (tflow runs the collected update ops with the train op); the
// "training-mode objective without update" uses update_stats=false. The
// checkpoint is best.npz ("parameters/<name>" and "state/<name>"); the
// deployment check is structural (a second model without optimizer or class
// weights; TF inspects GraphDef placeholders). PNGs via nn::save_png.
#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>

#include "common/multilabel_metrics.hpp"
#include "common/resnet.hpp"

namespace multilabel {

/// mean(BCE(logits, target) * (1 + (w - 1) * target)): positive bits weighted by w per class.
inline nn::Tensor MultilabelObjective(const nn::Tensor& logits, const nn::Tensor& target, const nn::Tensor& positive_weights) {
  const nn::Tensor weights = 1.0 + (positive_weights - 1.0) * target;
  return nn::mean(nn::sigmoid_cross_entropy(logits, target, nn::Reduction::None) * weights);
}

inline resnet::ResNetConfig MultilabelConfig(int divisor = 8) {
  resnet::ResNetConfig config;
  config.depth         = 18;
  config.classes       = 80;
  config.width_divisor = divisor;
  config.small_stem    = true;
  return config;
}

using MultilabelSnapshot = std::vector<std::vector<float>>;

class MultilabelModel {
 public:
  nn::ParameterStore             parameters{251}, state{257};
  nn::Tensor                     weights;
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit MultilabelModel(bool training = true, const nn::Tensor& positive_weights = nn::Tensor(), std::uint64_t seed = 251)
      : parameters(seed) {
    {
      nn::NoGradGuard guard;
      resnet::ResNet(parameters, state, nn::zeros({1, 64, 64, 3}) - .5, MultilabelConfig(), false);
    }
    if (training) {
      check::Require(positive_weights.defined() && positive_weights.shape() == nn::Shape{80}, "fitting-only positive weights");
      weights = positive_weights;
      nn::OptimizerConfig opt;
      opt.kind  = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .001, opt);
    }
  }
  nn::Tensor Logits(const nn::Tensor& rgb, bool training, bool update) {
    auto config         = MultilabelConfig();
    config.update_stats = update;
    return resnet::ResNet(parameters, state, rgb - .5, config, training).logits;
  }
  /// Weighted BCE; `update` takes an Adam step (training-mode BN, moving stats updated);
  /// `training` alone evaluates the training-mode objective without touching anything.
  double Loss(const MultilabelData& data, bool update = false, bool training = false) {
    check::Require(bool(optimizer), "training model required for weighted loss");
    std::unique_ptr<nn::NoGradGuard> guard;
    if (!update) guard = std::make_unique<nn::NoGradGuard>();
    const nn::Tensor loss  = MultilabelObjective(Logits(data.rgb, update || training, update), data.labels, weights);
    const double     value = loss.item();
    check::Require(std::isfinite(value) && value >= 0, "finite weighted multi-label objective");
    if (update) optimizer->step(nn::grad(loss, parameters.parameters()));
    return value;
  }
  nn::Tensor Predict(const nn::Tensor& rgb) {
    nn::NoGradGuard guard;
    return nn::sigmoid(Logits(rgb, false, false));
  }
  MultilabelSnapshot Snapshot() const {
    MultilabelSnapshot out;
    for (const auto& v : parameters.parameters()) out.push_back(v.to_vector<float>());
    for (const auto& v : state.parameters()) out.push_back(v.to_vector<float>());
    return out;
  }
  void Save(const std::string& path) const {
    nn::NamedTensors arrays;
    for (std::size_t i = 0; i < parameters.size(); ++i) arrays.push_back({"parameters/" + parameters.names()[i], parameters.parameters()[i]});
    for (std::size_t i = 0; i < state.size(); ++i) arrays.push_back({"state/" + state.names()[i], state.parameters()[i]});
    nn::save_npz(path, arrays);
  }
  void Restore(const std::string& path) {
    nn::NoGradGuard guard;
    const auto      arrays = nn::load_npz(path);
    check::Require(arrays.size() == parameters.size() + state.size(), "complete multi-label checkpoint");
    for (const auto& [name, value] : arrays) {
      if (name.rfind("parameters/", 0) == 0)
        parameters.set(name.substr(11), value);
      else if (name.rfind("state/", 0) == 0)
        state.set(name.substr(6), value);
      else
        throw std::runtime_error("unexpected multi-label checkpoint array " + name);
    }
  }
};

inline nn::Tensor MultilabelPredict(MultilabelModel& model, const MultilabelData& data) {
  std::vector<nn::Tensor> parts;
  const auto              n = std::size_t(data.rgb.size(0));
  for (std::size_t start = 0; start < n; start += 8) {
    std::vector<std::size_t> rows;
    for (std::size_t i = start; i < std::min(start + 8, n); ++i) rows.push_back(i);
    parts.push_back(model.Predict(MultilabelRows(data.rgb, rows)));
  }
  return nn::concat(parts, 0).contiguous();
}

inline void PrintMultilabel(const std::string& name, const nn::Tensor& probability, const nn::Tensor& labels, double threshold) {
  const auto score = ScoreMultilabel(probability, labels, threshold);
  std::cout << " " << name << ":t=" << threshold << ",mAP=" << 100 * score.map << "% (" << score.ap_classes << " positive-support classes,"
            << score.absent_classes << " absent),microAP=" << 100 * score.micro_ap << "%,micro/macroF1=" << 100 * score.micro_f1 << "/"
            << 100 * score.macro_f1 << "%,Hamming=" << score.hamming << ",exact=" << 100 * score.exact << "%,NLL=" << score.nll
            << ",Brier=" << score.brier << ",TP/FP/FN/TN=" << score.tp << "/" << score.fp << "/" << score.fn << "/" << score.tn << "\n";
}

/// 6x4 grid: each image with an 8x10 truth bitmap (green) and a decision bitmap
/// (green TP, red FP, yellow FN, dark TN).
inline void MultilabelPreview(const MultilabelData& data, const nn::Tensor& probability, double threshold, const std::filesystem::path& path) {
  const int          H = 6 * 64, W = 4 * 160;
  std::vector<float> view(std::size_t(H) * W * 3, 0.f);
  const auto         rgb = data.rgb.to_vector<float>(), truth_v = data.labels.to_vector<float>(), prob = probability.to_vector<float>();
  auto               at  = [&](int y, int x, int c) -> float& { return view[(std::size_t(y) * W + std::size_t(x)) * 3 + std::size_t(c)]; };
  for (int n = 0; n < 24; ++n) {
    const int row = n / 4, col = n % 4;
    for (int y = 0; y < 64; ++y)
      for (int x = 0; x < 64; ++x)
        for (int c = 0; c < 3; ++c)
          at(row * 64 + y, col * 160 + x, c) = float(std::clamp(int(std::lround(255 * rgb[std::size_t(((n * 64 + y) * 64 + x) * 3 + c)])), 0, 255));
    for (int label = 0; label < 80; ++label) {
      const bool truth = truth_v[std::size_t(n * 80 + label)] > .5, predicted = prob[std::size_t(n * 80 + label)] >= threshold;
      const int  y = (label / 10) * 8, x = (label % 10) * 4;
      for (int dy = 1; dy < 7; ++dy)
        for (int dx = 0; dx < 3; ++dx) {
          const int yy = row * 64 + y + dy, xx = col * 160 + 116 + x + dx;
          at(yy, col * 160 + 68 + x + dx, 1) = truth ? 255.f : 35.f;
          if (truth && predicted)
            at(yy, xx, 1) = 255;
          else if (predicted)
            at(yy, xx, 0) = 255;
          else if (truth)
            at(yy, xx, 0) = 255, at(yy, xx, 1) = 255;
          else
            for (int c = 0; c < 3; ++c) at(yy, xx, c) = 20;
        }
    }
  }
  nn::save_png(nn::from_vector(view, {H, W, 3}), path.string(), 0, 255);
}

}  // namespace multilabel
