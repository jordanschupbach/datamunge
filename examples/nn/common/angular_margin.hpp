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

enum class AngularMarginKind { kArcFace, kCosFace };

struct AngularMarginConfig {
  AngularMarginKind kind = AngularMarginKind::kArcFace;
  float             margin = .5f, scale = 64.f, epsilon = 1e-6f;
  bool              monotonic = true;
  void Validate() const {
    if (!std::isfinite(margin) || margin < 0 || margin >= 1.57079632679f || !std::isfinite(scale) || scale <= 0 ||
        !std::isfinite(epsilon) || epsilon <= 0 || epsilon >= .01f)
      throw std::invalid_argument("invalid angular margin/scale/clipping epsilon");
  }
};

/// scale * (cosine with the target column replaced by phi(cosine)).
inline nn::Tensor ApplyAngularMargin(const nn::Tensor& cosine, const nn::Tensor& labels, int classes,
                                     const AngularMarginConfig& cfg) {
  cfg.Validate();
  if (classes < 2) throw std::invalid_argument("at least two angular classes");
  if (cfg.margin == 0) return cosine * double(cfg.scale);
  nn::Tensor phi;
  if (cfg.kind == AngularMarginKind::kCosFace) {
    phi = cosine - double(cfg.margin);
  } else {
    const nn::Tensor c    = nn::clip(cosine, double(-1.f + cfg.epsilon), double(1.f - cfg.epsilon));
    const nn::Tensor sine = nn::sqrt(nn::maximum(1.0 - nn::square(c), nn::scalar_like(double(cfg.epsilon), c)));
    phi = c * double(std::cos(cfg.margin)) - sine * double(std::sin(cfg.margin));
    if (cfg.monotonic)
      phi = nn::where(nn::greater(c, double(std::cos(3.14159265358979323846f - cfg.margin))), phi,
                      c - double(std::sin(cfg.margin) * cfg.margin));
  }
  const nn::Tensor target = nn::one_hot(labels, classes, 1.0, 0.0, cosine.dtype());
  return (cosine + target * (phi - cosine)) * double(cfg.scale);
}

struct AngularHead {
  nn::Tensor embedding, weights, cosine, training, inference;
};

/// Unit rows of `raw` against unit columns of "angular/W" [dimensions, classes].
/// `labels` may be undefined (inference only: `training` stays undefined).
inline AngularHead NormalizedAngularHead(nn::ParameterStore& p, const nn::Tensor& raw, const nn::Tensor& labels,
                                         int dimensions, int classes, const AngularMarginConfig& cfg) {
  cfg.Validate();
  if (dimensions < 1 || classes < 2) throw std::invalid_argument("angular head dimensions/classes");
  AngularHead      out;
  const nn::Tensor w = p.get("angular/W", {dimensions, classes}, nn::init::glorot_uniform());
  out.embedding      = UnitEmbedding(raw);
  out.weights = w * nn::rsqrt(nn::maximum(nn::sum(nn::square(w), {0}, true), nn::scalar_like(double(1e-12f), w)));
  out.cosine    = nn::clip(nn::matmul(out.embedding, out.weights), -1.0, 1.0);
  out.inference = out.cosine * double(cfg.scale);
  if (labels.defined()) out.training = ApplyAngularMargin(out.cosine, labels, classes, cfg);
  return out;
}

/// -mean(sum(log_softmax(logits) * one_hot(labels))).
inline nn::Tensor AngularCrossEntropy(const nn::Tensor& logits, const nn::Tensor& labels, int classes) {
  return -nn::mean(nn::sum(nn::log_softmax(logits) * nn::one_hot(labels, classes, 1.0, 0.0, logits.dtype()), {1}));
}

/// Compact NN1 + normalized 128x5 head, scale 16, ArcFace m .3 / CosFace m .2.
class AngularModel {
 public:
  nn::ParameterStore             parameters{17};
  std::unique_ptr<nn::Optimizer> optimizer;
  AngularMarginConfig            cfg;
  explicit AngularModel(bool cosface) {
    cfg.kind   = cosface ? AngularMarginKind::kCosFace : AngularMarginKind::kArcFace;
    cfg.margin = cosface ? .2f : .3f;
    cfg.scale  = 16;
    {
      nn::NoGradGuard guard;
      Head(nn::zeros({1, 28, 28, 1}), nn::Tensor());
    }
    nn::OptimizerConfig oc;
    oc.kind   = nn::OptimizerKind::Adam;
    optimizer = std::make_unique<nn::Optimizer>(parameters, .001, oc);
  }
  AngularHead Head(const nn::Tensor& x, const nn::Tensor& labels) {
    const auto net = FaceNetNN1(parameters, x, FaceNetConfig::Compact());
    return NormalizedAngularHead(parameters, net.raw, labels, 128, 5, cfg);
  }
  nn::Tensor Encode(const nn::Tensor& x) {
    nn::NoGradGuard  guard;
    const nn::Tensor e = Head(x, nn::Tensor()).embedding;
    check::Require(e.shape() == nn::Shape({x.size(0), 128}), "angular embedding shape");
    const auto v = e.contiguous().to_vector<float>();
    for (std::int64_t i = 0; i < x.size(0); ++i) {
      double norm = 0;
      for (int j = 0; j < 128; ++j) {
        const double value = v[std::size_t(i * 128 + j)];
        check::Require(std::isfinite(value), "finite angular embedding");
        norm += value * value;
      }
      check::Near(norm, 1, 2e-6, "angular unit embedding");
    }
    return e;
  }
  nn::Tensor Predict(const nn::Tensor& x) {
    nn::NoGradGuard guard;
    return Head(x, nn::Tensor()).inference;
  }
  double Loss(const nn::Tensor& x, const nn::Tensor& y, bool update = false) {
    for (auto v : y.to_vector<std::int64_t>()) check::Require(v >= 0 && v < 5, "training class range");
    double value = 0;
    if (!update) {
      nn::NoGradGuard guard;
      value = AngularCrossEntropy(Head(x, y).training, y, 5).item();
    } else {
      const nn::Tensor loss = AngularCrossEntropy(Head(x, y).training, y, 5);
      value                 = loss.item();
      optimizer->step(nn::grad(loss + .00005 * WeightSquares(parameters), parameters.parameters()));
    }
    check::Require(std::isfinite(value), "finite angular margin loss");
    return value;
  }
  void Save(const std::string& path) const { parameters.save(path); }
  void Restore(const std::string& path) { parameters.load(path); }
};

inline nn::Tensor AngularLabels(const nn::ImageDataset& d, const std::vector<std::size_t>& rows) {
  std::vector<std::int64_t> ids;
  for (auto r : rows) ids.push_back(std::int64_t(r));
  return nn::take_rows(d.labels, ids);
}

struct AngularClassScore {
  double loss = 0, accuracy = 0;
  int    confusion[5][5] = {};
};
inline AngularClassScore ScoreAngularClasses(AngularModel& model, const nn::ImageDataset& d, const std::vector<std::size_t>& rows) {
  AngularClassScore score;
  const auto        labels = Labels(d);
  for (std::size_t start = 0; start < rows.size(); start += 64) {
    const auto                     end = std::min(rows.size(), start + 64);
    const std::vector<std::size_t> part(rows.begin() + std::ptrdiff_t(start), rows.begin() + std::ptrdiff_t(end));
    const nn::Tensor               logits = model.Predict(SiameseImages(d, part));
    check::Require(logits.shape() == nn::Shape({std::int64_t(part.size()), 5}), "normalized classifier shape");
    const auto l = logits.to_vector<float>();
    for (std::size_t i = 0; i < part.size(); ++i) {
      int    predicted = 0;
      double maximum   = -INFINITY;
      for (int c = 0; c < 5; ++c) {
        const double value = l[i * 5 + c];
        check::Require(std::isfinite(value) && std::abs(value) <= 16.00001, "finite bounded cosine inference logit");
        if (value > maximum) maximum = value, predicted = c;
      }
      double sum = 0;
      for (int c = 0; c < 5; ++c) sum += std::exp(l[i * 5 + c] - maximum);
      const int truth = labels[part[i]];
      check::Require(truth >= 0 && truth < 5, "closed-set truth range");
      score.loss += maximum + std::log(sum) - l[i * 5 + std::size_t(truth)];
      ++score.confusion[truth][predicted];
    }
  }
  int correct = 0, total = 0;
  for (int y = 0; y < 5; ++y)
    for (int p = 0; p < 5; ++p) {
      total += score.confusion[y][p];
      if (y == p) correct += score.confusion[y][p];
    }
  check::Require(total == int(rows.size()), "every held-out classifier row counted");
  score.loss /= total;
  score.accuracy = double(correct) / total;
  return score;
}

}  // namespace metric
