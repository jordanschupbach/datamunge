/// @file
/// @brief Decision-boundary probes for differentiable classifiers.
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/projected_robustness.hpp"

namespace datamunge::nn {

/// @brief Scores for a batch of independent examples, shaped [examples,classes].
using RobustnessClassifier = std::function<Tensor(const Tensor&)>;

/// @brief Probe a classifier with the minimum-distance successful Carlini–Wagner iterate.
///
/// For each coefficient, optimizes squared displacement plus a positive-part
/// confidence margin in unconstrained coordinates. Only initially correct
/// rows are eligible. Rows without a successful iterate retain their input.
/// @param x Float32 CPU inputs [examples,features], in [0,1].
/// @param labels True class indices [examples].
/// @param predict Differentiable classifier with at least two classes.
/// @param coefficients Positive penalty coefficients tested independently.
/// @param steps Number of Adam iterations for each coefficient.
/// @param confidence Required competitor score advantage.
/// @return Selected inputs with the same shape as x.
/// @throws std::invalid_argument Invalid optimization settings.
inline Tensor carlini_wagner_l2(const Tensor& x, const Tensor& labels,
    const RobustnessClassifier& predict, std::span<const float> coefficients,
    int steps, double confidence = .05) {
  if (!predict || steps < 1 || coefficients.empty() || !std::isfinite(confidence) || confidence < 0 ||
      !x.defined() || x.dim() != 2 || x.numel() == 0 || x.dtype() != DType::Float32 ||
      x.device().backend != compute::Backend::CPU || !labels.defined() ||
      labels.shape() != Shape{x.size(0)}) {
    throw std::invalid_argument("carlini_wagner_l2: invalid inputs or settings");
  }
  const auto initial = predict(x).detach();
  if (initial.dim() != 2 || initial.size(0) != x.size(0) || initial.size(1) < 2) {
    throw std::invalid_argument("carlini_wagner_l2: invalid logits");
  }
  const auto truth = one_hot(labels, initial.size(1));
  const auto eligible = equal(argmax(initial, 1), labels);
  const auto centered = clip(2*x.to(DType::Float64)-1, -.999999, .999999);
  const auto start = .5 * log((1+centered)/(1-centered));
  auto best = full({x.size(0)}, std::numeric_limits<float>::infinity());
  auto selected = x.detach();
  for (float coefficient : coefficients) {
    if (!std::isfinite(coefficient) || coefficient <= 0) {
      throw std::invalid_argument("carlini_wagner_l2: invalid penalty coefficient");
    }
    auto u = start, m = zeros_like(start), v = zeros_like(start);
    for (int step = 1; step <= steps; ++step) {
      const auto t = tanh(u);
      auto point = (.5*(1+t)).to(DType::Float32).detach();
      point.requires_grad_(true);
      const auto logits = predict(point);
      const auto margin = sum(truth*logits, {1}) -
          max(where(equal(truth,0), logits, -std::numeric_limits<float>::infinity()), {1});
      const auto distance = sum(square(point-x), {1});
      const auto improve = logical_and(eligible, logical_and(greater_equal(-margin,confidence), less(distance,best)));
      selected = where(improve.reshape({-1,1}),point,selected).detach();
      best = where(improve,distance,best).detach();
      const auto g = .5*(1-square(t)) * grad(sum(distance+coefficient*relu(margin+confidence)),{point}).at(0)
          .to(DType::Float64);
      m = .9*m+.1*g;
      v = .999*v+.001*square(g);
      const double rate = .035*std::sqrt(1-std::pow(.999,step))/(1-std::pow(.9,step));
      u = u-rate*m/(sqrt(v)+1e-8);
    }
  }
  return selected;
}

/// @brief Move each row toward its closest locally linear decision boundary.
/// @param x Float32 CPU inputs [examples,features], in [0,1].
/// @param predict Differentiable classifier whose rows are independent.
/// @param steps Maximum number of iterations.
/// @return Input states [steps+1,examples,features]. Changed rows stop moving.
/// @throws std::invalid_argument Invalid input, classifier geometry or iteration count.
inline Tensor deepfool_path(const Tensor& x, const RobustnessClassifier& predict, int steps) {
  if (!predict || !x.defined() || x.dim()!=2 || x.numel()==0 || x.dtype()!=DType::Float32 ||
      x.device().backend != compute::Backend::CPU || steps<1) {
    throw std::invalid_argument("deepfool_path: invalid inputs or iteration count");
  }
  const auto initial = predict(x).detach();
  if (initial.dim()!=2 || initial.size(0)!=x.size(0) || initial.size(1)<2) {
    throw std::invalid_argument("deepfool_path: invalid logits");
  }
  const auto rows=x.size(0), features=x.size(1), classes=initial.size(1);
  const auto original=argmax(initial,1).to(DType::Int64).to_vector<std::int64_t>();
  auto state=x.detach();
  std::vector<Tensor> path{state};
  for (int step=0;step<steps;++step) {
    auto leaf=state.detach();
    leaf.requires_grad_(true);
    const auto logits=predict(leaf);
    const auto scores=logits.to_vector<float>();
    const auto winner=argmax(logits,1).to(DType::Int64).to_vector<std::int64_t>();
    std::vector<std::vector<float>> jacobian;
    for (std::int64_t cls=0;cls<classes;++cls) {
      const auto score=sum(slice_axis(logits,1,cls,cls+1));
      jacobian.push_back(grad(score,{leaf},{.retain_graph=cls+1<classes}).at(0).to_vector<float>());
    }
    auto values=state.to_vector<float>();
    for (std::int64_t row=0;row<rows;++row) {
      const auto base=original[row];
      if (winner[row]!=base) continue;
      double closest=std::numeric_limits<double>::infinity(), scale=0;
      auto best=base;
      for (std::int64_t cls=0;cls<classes;++cls) {
        if (cls==base) continue;
        double norm=0;
        for (std::int64_t k=row*features;k<(row+1)*features;++k) {
          const double delta=double(jacobian[cls][k])-jacobian[base][k];
          norm+=delta*delta;
        }
        const double gap=std::abs(double(scores[row*classes+cls])-scores[row*classes+base]);
        if (norm>1e-16 && gap/std::sqrt(norm)<closest) {
          closest=gap/std::sqrt(norm);
          best=cls;
          scale=1.02*(gap+1e-7)/norm;
        }
      }
      for (std::int64_t k=row*features;k<(row+1)*features;++k) {
        values[k]=std::clamp(values[k]+scale*(double(jacobian[best][k])-jacobian[base][k]),0.,1.);
      }
    }
    state=from_vector(values,x.shape());
    path.push_back(state);
  }
  return stack(path);
}

/// @brief Fit a single bounded displacement shared by a calibration batch.
/// @param calibration Unit-box calibration matrix [examples,features].
/// @param objective Differentiable scalar score to increase on that batch.
/// @param radius Maximum absolute coordinate displacement, at most 0.5.
/// @param rate Positive projected step size.
/// @param steps Positive update count.
/// @return Universal displacement trajectory [steps+1,features], beginning at zero.
inline Tensor universal_sign_path(const Tensor& calibration,
    const std::function<Tensor(const Tensor&)>& objective, double radius, double rate, int steps) {
  if (!calibration.defined() || calibration.dim()!=2 || calibration.numel()==0 ||
      !objective || radius>.5) {
    throw std::invalid_argument("universal_sign_path: invalid calibration data or radius");
  }
  const auto center=full_like(sum(calibration,{0}),.5).detach();
  return projected_sign_ascent(center,center,[&](const Tensor& state) {
    return objective(clip(calibration+(state-.5),0.,1.));
  },radius,rate,steps).states-.5;
}

} // namespace datamunge::nn
