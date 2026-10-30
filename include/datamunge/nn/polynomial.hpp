// datamunge::nn port of tflow_examples src/common/ml/polynomial.h.
// Polynomial design matrices from a float64 [N,1] tensor.
// Legendre recurrence spans the same degree-d polynomial space as powers,
// but is better conditioned for x in [-1,1]. Column zero is the intercept.
#pragma once

/// @file polynomial.hpp
/// @brief Legendre design matrices for scalar predictors.

#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/ops.hpp>
#include <stdexcept>
#include <vector>

namespace datamunge::nn::ml {
namespace nn = datamunge::nn;

/// @brief Build degree-ordered Legendre columns from one scalar predictor.
///
/// Caller supplies rank two and one column; values outside [-1,1] are
/// accepted but may be poorly conditioned. The operation is differentiable.
/// @param x Floating [N,1] tensor; preferably Float64 and values in [-1,1].
/// @param degree Nonnegative maximum degree; zero emits only the intercept.
/// @return Design [N,degree+1] with P0=1, P1=x and the Legendre recurrence.
/// @throws std::invalid_argument On negative degree or invalid concat shapes.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::ml;
/// const auto design=m::PolynomialFeatures(n::tensor({-1,0,1},{3,1}),2);
/// @endcode
///
/// @par References
/// - NIST DLMF, https://dlmf.nist.gov/18.9
/// @see LeastSquaresQr()
inline nn::Tensor PolynomialFeatures(const nn::Tensor& x, int degree) {
  if (degree < 0)
    throw std::invalid_argument("polynomial degree must be nonnegative");
  std::vector<nn::Tensor> columns{nn::ones_like(x)};
  if (degree)
    columns.push_back(x);
  for (int n = 2; n <= degree; ++n)
    columns.push_back(
        (double(2 * n - 1) * (x * columns[n - 1]) - double(n - 1) * columns[n - 2]) / double(n));
  return nn::concat(columns, 1);
}

} // namespace datamunge::nn::ml
