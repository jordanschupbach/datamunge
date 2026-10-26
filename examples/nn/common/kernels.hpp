// datamunge::nn port of tflow_examples src/common/ml/kernels.h.
// Pairwise squared distances, RBF kernels and random Fourier features.
// Distances sum direct coordinate differences to avoid cancellation near exact
// matches. This creates a [queries, references, dimensions] temporary; chunk
// large inputs. These educational examples use small, low-dimensional data.
// For k(x,y)=exp(-gamma*||x-y||^2), draw W~N(0,2*gamma), b~U(0,2*pi),
// then z(x)=sqrt(2/D)*cos(xW+b). W and b are fixed, seeded host tensors drawn
// from the same std::mt19937_64 stream as the TF version (bit-identical).
// Rahimi & Recht: https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
#pragma once

#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <random>
#include <stdexcept>
#include <vector>

namespace ml {
namespace nn = datamunge::nn;

inline nn::Tensor SquaredDistances(const nn::Tensor& x, const nn::Tensor& y) {
  const auto delta = nn::expand_dims(x, 1) - nn::expand_dims(y, 0);
  return nn::sum(nn::square(delta), {2});
}

inline nn::Tensor RbfKernel(const nn::Tensor& x, const nn::Tensor& y, float gamma) {
  if (!std::isfinite(gamma) || gamma <= 0) throw std::invalid_argument("RBF gamma must be positive");
  return nn::exp(double(-gamma) * SquaredDistances(x, y));
}

struct RandomFourierFeatures {
  RandomFourierFeatures(int input_width, int output_width, float gamma, std::uint64_t seed) {
    if (input_width <= 0 || output_width <= 0 || !std::isfinite(gamma) || gamma <= 0)
      throw std::invalid_argument("positive RFF widths/gamma required");
    std::mt19937_64                       rng(seed);
    std::normal_distribution<float>       normal(0, std::sqrt(2 * gamma));
    std::uniform_real_distribution<float> uniform(0, 2 * std::acos(-1.f));
    std::vector<float>                    w(std::size_t(input_width) * output_width), b(output_width);
    for (auto& v : w) v = normal(rng);
    for (auto& v : b) v = uniform(rng);
    weights = nn::from_vector(w, {input_width, output_width});
    phase   = nn::from_vector(b, {output_width});
  }
  nn::Tensor Transform(const nn::Tensor& x) const {
    return double(std::sqrt(2.f / float(phase.numel()))) * nn::cos(nn::matmul(x, weights) + phase);
  }
  nn::Tensor weights, phase;
};

}  // namespace ml
