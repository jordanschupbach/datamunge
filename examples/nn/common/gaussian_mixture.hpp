// datamunge::nn port of tflow_examples src/common/ml/gaussian_mixture.h.
// Full-covariance Gaussian-mixture EM with float64 nn operations.
// E step: batched Cholesky solves, log determinants and log-sum-exp. M step:
// responsibility-weighted means/covariances and normalized mixture masses.
// Clamp covariance eigenvalues to a positive floor (constrained covariance
// MLE), rather than adding a ridge to every eigenvalue. Reject a collapsed
// component with negligible effective count; callers can restart the fit.
// Small datasets only: the M step materializes an [N,K,D,D] temporary.
#pragma once

#include <cmath>
#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>

namespace ml {
namespace nn = datamunge::nn;

struct GaussianMixtureState {
  nn::Tensor weights, means, covariances;
};
struct GaussianMixtureStep {
  GaussianMixtureState next;
  nn::Tensor           responsibilities, log_density;
  double               log_likelihood = 0;
};

class GaussianMixture {
 public:
  GaussianMixture(int components, int dimensions, double covariance_floor = 1e-4)
      : k_(components), d_(dimensions), floor_(covariance_floor) {
    if (k_ <= 0 || d_ <= 0 || !std::isfinite(covariance_floor) || covariance_floor <= 0)
      throw std::invalid_argument("positive GMM dimensions and covariance floor required");
  }

  GaussianMixtureStep Step(const nn::Tensor& x, const GaussianMixtureState& state) const {
    using nn::DType;
    if (x.dtype() != DType::Float64 || x.dim() != 2 || x.size(0) < 1 || x.size(1) != d_ ||
        state.weights.dtype() != DType::Float64 || state.weights.shape() != nn::Shape{k_} ||
        state.means.dtype() != DType::Float64 || state.means.shape() != nn::Shape{k_, d_} ||
        state.covariances.dtype() != DType::Float64 || state.covariances.shape() != nn::Shape{k_, d_, d_})
      throw std::invalid_argument("GMM requires matching float64 data/state shapes");
    double sum = 0;
    for (double weight : state.weights.to_vector<double>()) {
      if (!std::isfinite(weight) || weight <= 0) throw std::invalid_argument("mixture weights must be positive");
      sum += weight;
    }
    if (std::abs(sum - 1) > 1e-10) throw std::invalid_argument("mixture weights must sum to one");
    nn::NoGradGuard guard;
    const auto delta     = nn::expand_dims(x, 1) - nn::expand_dims(state.means, 0);           // [N, K, D]
    const auto chol      = nn::cholesky(state.covariances);                                  // [K, D, D]
    const auto whitened  = nn::triangular_solve(chol, nn::transpose(delta, {1, 2, 0}));      // [K, D, N]
    const auto quadratic = nn::transpose(nn::sum(nn::square(whitened), {1}), {1, 0});        // [N, K]
    const auto logdet    = 2.0 * nn::sum(nn::log(nn::matrix_diag_part(chol)), {1});         // [K]
    const auto log_joint =
        nn::log(state.weights) + -.5 * (quadratic + (logdet + d_ * std::log(2 * std::acos(-1.))));
    const auto maximum        = nn::max(log_joint, {1}, true);
    const auto log_normalizer = maximum + nn::log(nn::sum(nn::exp(log_joint - maximum), {1}, true));
    GaussianMixtureStep out;
    out.responsibilities  = nn::exp(log_joint - log_normalizer);
    out.log_density       = nn::reshape(log_normalizer, {-1});
    out.log_likelihood    = nn::sum(out.log_density, {0}).item();
    const auto counts     = nn::sum(out.responsibilities, {0});
    const auto safe       = nn::maximum(counts, nn::scalar_like(1e-100, counts));
    out.next.weights      = counts / nn::sum(counts, {0});
    out.next.means        = nn::matmul(out.responsibilities, x, true, false) / nn::expand_dims(safe, 1);
    const auto centered   = nn::expand_dims(x, 1) - nn::expand_dims(out.next.means, 0);
    const auto outer      = nn::expand_dims(centered, 3) * nn::expand_dims(centered, 2);
    const auto weighted   = outer * nn::expand_dims(nn::expand_dims(out.responsibilities, 2), 3);
    const auto covariance = nn::sum(weighted, {0}) / nn::reshape(safe, {k_, 1, 1});
    const auto eigen      = nn::self_adjoint_eig(covariance);
    const auto clipped    = nn::maximum(eigen.e, nn::scalar_like(floor_, eigen.e));
    out.next.covariances  = nn::matmul(eigen.v * nn::expand_dims(clipped, 1), eigen.v, false, true);
    for (double c : counts.to_vector<double>())
      if (!std::isfinite(c) || c < 1e-8) throw std::runtime_error("GMM component collapsed; restart with different initialization");
    return out;
  }

 private:
  std::int64_t k_, d_;
  double       floor_;
};

}  // namespace ml
