// Straight-line integrated gradients with midpoint quadrature and batched calls:
// a port of tflow's common/nn/integrated_gradients.h.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <functional>
#include <vector>

#include "common/checks.hpp"

namespace ig_reference {

namespace nn = datamunge::nn;
using nn::Tensor;

struct IntegratedResult {
  Tensor attribution, mean_gradient, path_gradient, nodes;  // float32 x3, float64 [steps]
  int    steps = 0;
};
/// Ordinary input derivative at a batch of points [count, ...] for fixed int32 classes [count].
using InputGradientFunction = std::function<Tensor(const Tensor& points, const Tensor& classes)>;

/// (x - x0) * mean_k dF/dx(x0 + alpha_k (x - x0)), alpha_k = (k + .5) / steps; the
/// steps * n path points are evaluated `batch` at a time (point j: case j % n, node j / n).
inline IntegratedResult IntegratedGradients(const Tensor& input, const Tensor& baseline, const Tensor& targets, int steps,
                                            int batch, const InputGradientFunction& gradient) {
  check::Require(input.dtype() == nn::DType::Float32 && baseline.dtype() == nn::DType::Float32 &&
                     input.shape() == baseline.shape() && input.dim() >= 2 && input.size(0) > 0 && input.numel() > 0,
                 "matching nonempty float32 input/baseline batches");
  const int n = int(input.size(0));
  check::Require(targets.dtype() == nn::DType::Int32 && targets.shape() == nn::Shape{n} && steps > 0 && batch > 0,
                 "fixed targets/positive quadrature steps/batch");
  const std::int64_t width = input.numel() / n;
  const auto         x = input.contiguous().to_vector<float>(), x0 = baseline.contiguous().to_vector<float>();
  const auto         t = targets.contiguous().to_vector<int>();
  std::vector<double> nodes(steps), sums(x.size());
  for (int k = 0; k < steps; ++k) nodes[k] = (k + .5) / steps;
  std::vector<float> trace(std::size_t(steps) * x.size());
  for (int start = 0; start < steps * n; start += batch) {
    const int          count = std::min(batch, steps * n - start);
    std::vector<float> points(std::size_t(count) * width);
    std::vector<int>   classes(count);
    for (int j = 0; j < count; ++j) {
      const int flat = start + j, b = flat % n, k = flat / n;
      classes[j]         = t[b];
      const double alpha = nodes[k];
      for (std::int64_t i = 0; i < width; ++i) {
        const std::int64_t source     = b * width + i;
        const double       difference = double(x[source]) - x0[source];
        const double       product    = alpha * difference;
        points[j * width + i]         = float(double(x0[source]) + product);
      }
    }
    nn::Shape shape = input.shape();
    shape[0]        = count;
    const Tensor g  = gradient(nn::from_vector(points, shape), nn::from_vector(classes, {count}));
    check::Require(g.dtype() == nn::DType::Float32 && g.shape() == shape, "ordinary input derivative shape matches every path point");
    const auto values = g.contiguous().to_vector<float>();
    for (int j = 0; j < count; ++j) {
      const int flat = start + j, b = flat % n;
      for (std::int64_t i = 0; i < width; ++i) {
        const double v = values[j * width + i];
        check::Require(std::isfinite(v), "finite every sampled path derivative");
        trace[flat * width + i] = float(v);
        sums[b * width + i] += v;
      }
    }
  }
  std::vector<float> mean(x.size()), attribution(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double average = sums[i] / steps;
    mean[i]              = float(average);
    attribution[i]       = float((double(x[i]) - x0[i]) * average);
  }
  nn::Shape trace_shape = input.shape();
  trace_shape.insert(trace_shape.begin(), steps);
  return {nn::from_vector(attribution, input.shape()), nn::from_vector(mean, input.shape()), nn::from_vector(trace, trace_shape),
          nn::from_vector(nodes, {steps}), steps};
}

}  // namespace ig
