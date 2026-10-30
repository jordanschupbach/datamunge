// SmoothGrad: the signed mean of ordinary gradients under an explicit Gaussian noise
// bank. A port of tflow's common/nn/smoothgrad.h (GaussianNoise itself lives in
// common/autoencoder.hpp as ae::GaussianNoise, bit-identical to tflow's).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <vector>

#include "common/autoencoder.hpp"
#include "common/checks.hpp"
#include "common/integrated_gradients.hpp"

namespace smooth_reference {

namespace nn = datamunge::nn;
using nn::Tensor;
using ig::InputGradientFunction;

/// tflow nn::GaussianNoise: float64 [samples, input_shape...] Box-Muller bank from mt19937(seed).
inline Tensor GaussianNoise(const nn::Shape& input_shape, int samples, std::uint32_t seed) {
  check::Require(samples > 0 && input_shape.size() >= 2 && nn::shape_numel(input_shape) > 0, "nonempty noise shape/samples");
  return ae::GaussianNoise(input_shape, samples, seed);
}

struct SmoothGradResult {
  Tensor mean, mean_absolute, mean_square, sample_variance, standard_error, standard_noise, noisy_input, sample_gradient;
};

/// Point (k, b) = [clip](float(x_b + sigma z_kb)); statistics of dF/dx over the k samples.
inline SmoothGradResult SmoothGrad(const Tensor& input, const Tensor& targets, const Tensor& standard_noise, float sigma,
                                   bool clipped, int batch, const InputGradientFunction& gradient) {
  check::Require(input.dtype() == nn::DType::Float32 && input.dim() >= 2 && input.numel() > 0 &&
                     standard_noise.dtype() == nn::DType::Float64 && standard_noise.dim() == input.dim() + 1 &&
                     std::isfinite(sigma) && sigma >= 0 && batch > 0,
                 "finite SmoothGrad configuration");
  const int n = int(input.size(0)), samples = int(standard_noise.size(0));
  nn::Shape expected = input.shape();
  expected.insert(expected.begin(), samples);
  check::Require(samples > 0 && standard_noise.shape() == expected && targets.dtype() == nn::DType::Int32 &&
                     targets.shape() == nn::Shape{n},
                 "noise bank shape and fixed targets");
  const std::int64_t width = input.numel() / n;
  const auto         x = input.contiguous().to_vector<float>();
  const auto         z = standard_noise.contiguous().to_vector<double>();
  const auto         t = targets.contiguous().to_vector<int>();
  std::vector<double> sums(x.size()), absolute(x.size()), squares(x.size());
  std::vector<float>  noisy(z.size()), samples_g(z.size());
  for (int start = 0; start < samples * n; start += batch) {
    const int          count = std::min(batch, samples * n - start);
    std::vector<float> points(std::size_t(count) * width);
    std::vector<int>   classes(count);
    for (int j = 0; j < count; ++j) {
      const int flat = start + j, b = flat % n;
      classes[j] = t[b];
      for (std::int64_t i = 0; i < width; ++i) {
        const std::int64_t index = flat * width + i;
        check::Require(std::isfinite(z[index]), "finite stored Gaussian samples");
        float value = float(double(x[b * width + i]) + double(sigma) * z[index]);
        if (clipped) value = std::clamp(value, 0.f, 1.f);
        points[j * width + i] = value;
        noisy[index]          = value;
      }
    }
    nn::Shape shape = input.shape();
    shape[0]        = count;
    const Tensor g  = gradient(nn::from_vector(points, shape), nn::from_vector(classes, {count}));
    check::Require(g.dtype() == nn::DType::Float32 && g.shape() == shape, "ordinary derivative shape at every noisy point");
    const auto values = g.contiguous().to_vector<float>();
    for (int j = 0; j < count; ++j) {
      const int flat = start + j, b = flat % n;
      for (std::int64_t i = 0; i < width; ++i) {
        const double value = values[j * width + i];
        check::Require(std::isfinite(value), "finite noisy input derivative");
        samples_g[flat * width + i] = float(value);
        const std::int64_t index    = b * width + i;
        sums[index] += value;
        absolute[index] += std::abs(value);
        squares[index] += value * value;
      }
    }
  }
  std::vector<float> mean(x.size()), mabs(x.size()), msq(x.size()), var(x.size()), se(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double m = sums[i] / samples,
                 v = samples > 1 ? std::max(0., (squares[i] - sums[i] * sums[i] / samples) / (samples - 1)) : 0;
    mean[i] = float(m);
    mabs[i] = float(absolute[i] / samples);
    msq[i]  = float(squares[i] / samples);
    var[i]  = float(v);
    se[i]   = float(std::sqrt(v / samples));
  }
  const auto& s = input.shape();
  return {nn::from_vector(mean, s),     nn::from_vector(mabs, s),         nn::from_vector(msq, s),
          nn::from_vector(var, s),      nn::from_vector(se, s),           standard_noise,
          nn::from_vector(noisy, expected), nn::from_vector(samples_g, expected)};
}

}  // namespace smooth
