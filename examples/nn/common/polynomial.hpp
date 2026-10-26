// datamunge::nn port of tflow_examples src/common/ml/polynomial.h.
// Polynomial design matrices from a float64 [N,1] tensor.
// Legendre recurrence spans the same degree-d polynomial space as powers,
// but is better conditioned for x in [-1,1]. Column zero is the intercept.
#pragma once

#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <vector>

namespace ml {
namespace nn = datamunge::nn;

inline nn::Tensor PolynomialFeatures(const nn::Tensor& x, int degree) {
  if (degree < 0) throw std::invalid_argument("polynomial degree must be nonnegative");
  std::vector<nn::Tensor> columns{nn::ones_like(x)};
  if (degree) columns.push_back(x);
  for (int n = 2; n <= degree; ++n)
    columns.push_back((double(2 * n - 1) * (x * columns[n - 1]) - double(n - 1) * columns[n - 2]) / double(n));
  return nn::concat(columns, 1);
}

}  // namespace ml
