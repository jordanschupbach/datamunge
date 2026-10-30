// Bias-free normalized classifier with ArcFace angular / CosFace cosine
// margins, and the short CNN ArcFace/CosFace demos (closed-set digits 0-4
// plus unseen-digit verification/retrieval): a port of tflow's
// common/nn/angular_margin.h and common/ml/angular_margin.h.
// Training changes only the labelled target; inference has no label input.
#pragma once

#include <cmath>
#include <set>

#include "common/facenet.hpp"

namespace metric {

namespace nn = datamunge::nn;

using nn::models::AngularCrossEntropy;
using nn::models::AngularHead;
using nn::models::AngularMarginConfig;
using nn::models::AngularMarginKind;
using nn::models::ApplyAngularMargin;
using nn::models::NormalizedAngularHead;
/// Compact NN1 + normalized 128x5 head, scale 16, ArcFace m .3 / CosFace m .2.
class AngularModel {
public:
  nn::ParameterStore parameters{17};
  std::unique_ptr<nn::Optimizer> optimizer;
  AngularMarginConfig cfg;
  explicit AngularModel(bool cosface) {
    cfg.kind = cosface ? AngularMarginKind::kCosFace : AngularMarginKind::kArcFace;
    cfg.margin = cosface ? .2f : .3f;
    cfg.scale = 16;
    {
      nn::NoGradGuard guard;
      Head(nn::zeros({1, 28, 28, 1}), nn::Tensor());
    }
    nn::OptimizerConfig oc;
    oc.kind = nn::OptimizerKind::Adam;
    optimizer = std::make_unique<nn::Optimizer>(parameters, .001, oc);
  }
  AngularHead Head(const nn::Tensor &x, const nn::Tensor &labels) {
    const auto net = FaceNetNN1(parameters, x, FaceNetConfig::Compact());
    return NormalizedAngularHead(parameters, net.raw, labels, 128, 5, cfg);
  }
  nn::Tensor Encode(const nn::Tensor &x) {
    nn::NoGradGuard guard;
    const nn::Tensor e = Head(x, nn::Tensor()).embedding;
    check::Require(e.shape() == nn::Shape({x.size(0), 128}), "angular embedding shape");
    check::Require(
        nn::all(nn::less_equal(
                    nn::abs(nn::sum(nn::square(e.to(nn::DType::Float64)), {1}) - 1.), 2e-6))
            .item<bool>(),
        "angular unit embeddings");
    return e;
  }
  nn::Tensor Predict(const nn::Tensor &x) {
    nn::NoGradGuard guard;
    return Head(x, nn::Tensor()).inference;
  }
  double Loss(const nn::Tensor &x, const nn::Tensor &y, bool update = false) {
    for (auto v : y.to_vector<std::int64_t>())
      check::Require(v >= 0 && v < 5, "training class range");
    double value = 0;
    if (!update) {
      nn::NoGradGuard guard;
      value = AngularCrossEntropy(Head(x, y).training, y, 5).item();
    } else {
      const nn::Tensor loss = AngularCrossEntropy(Head(x, y).training, y, 5);
      value = loss.item();
      optimizer->step(
          nn::grad(loss + .00005 * WeightSquares(parameters), parameters.parameters()));
    }
    check::Require(std::isfinite(value), "finite angular margin loss");
    return value;
  }
  void Save(const std::string &path) const {
    parameters.save(path);
  }
  void Restore(const std::string &path) {
    parameters.load(path);
  }
};

using nn::models::AngularLabels;
struct AngularClassScore {
  double loss = 0, accuracy = 0;
  int confusion[5][5] = {};
};
inline AngularClassScore ScoreAngularClasses(AngularModel &model, const nn::ImageDataset &d,
                                             const std::vector<std::size_t> &rows) {
  std::vector<nn::Tensor> parts;
  for (std::size_t start = 0; start < rows.size(); start += 64) {
    const auto end = std::min(rows.size(), start + 64);
    parts.push_back(
        model.Predict(SiameseImages(d, {rows.begin() + start, rows.begin() + end})));
  }
  const auto logits = nn::concat(parts, 0);
  const auto y = AngularLabels(d, rows);
  const auto counts = nn::models::AngularClassificationConfusion(logits, y);
  const auto values = counts.to_vector<std::int64_t>();
  AngularClassScore score;
  score.loss = AngularCrossEntropy(logits.to(nn::DType::Float64), y, 5).item();
  score.accuracy = nn::mean(nn::equal(nn::argmax(logits, 1), y).to(nn::DType::Float64)).item();
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 5; ++j)
      score.confusion[i][j] = int(values[std::size_t(i * 5 + j)]);
  return score;
}

} // namespace metric
