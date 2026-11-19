#pragma once

/// @file weight_gaussian.hpp
/// @brief A stochastic weight-averaging Gaussian from ordered snapshots.

#include <datamunge/nn/ops.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Diagonal plus low-rank Gaussian approximation to an SGD trajectory.
struct SwagFit {
  Tensor mean;       ///< Float64 running mean, shape [weights].
  Tensor diagonal;   ///< Float64 population variance along each weight coordinate.
  Tensor deviations; ///< Float64 circular buffer [rank, weights] of deviations from running means.
};

/// @brief Fit a SWAG approximation to chronological weight snapshots.
/// The diagonal uses all snapshots. The low-rank buffer keeps the most recent
/// deviations, each centered at the running mean immediately after that snapshot.
/// @param snapshots Finite CPU floating matrix [snapshots, weights].
/// @param rank Number of retained deviations, between two and the snapshot count.
/// @return Detached Float64 approximation.
/// @throws std::invalid_argument Invalid geometry, dtype or snapshots.
/// @code
/// auto fit = dm::nn::FitSwag(dm::nn::tensor({1., 2., 3., 4.}, {2, 2}), 2);
/// @endcode
inline SwagFit FitSwag(const Tensor& snapshots, std::int64_t rank) {
  if (!snapshots.defined() || snapshots.device().backend != compute::Backend::CPU ||
      snapshots.dim() != 2 || snapshots.size(1) <= 0 ||
      (snapshots.dtype() != DType::Float32 && snapshots.dtype() != DType::Float64) ||
      rank < 2 || rank > snapshots.size(0)) {
    throw std::invalid_argument("invalid SWAG snapshot geometry, rank or dtype");
  }
  const auto rows = snapshots.size(0), weights = snapshots.size(1);
  const auto values = snapshots.to_vector<double>();
  std::vector<double> mean(weights), second(weights), deviations(rank * weights);
  for (std::int64_t i = 0; i < rows; ++i) {
    for (std::int64_t j = 0; j < weights; ++j) {
      const double value = values[i * weights + j];
      if (!std::isfinite(value))
        throw std::invalid_argument("SWAG snapshots must be finite");
      mean[j] += (value - mean[j]) / (i + 1);
      second[j] += (value * value - second[j]) / (i + 1);
      deviations[(i % rank) * weights + j] = value - mean[j];
    }
  }
  for (std::int64_t j = 0; j < weights; ++j) {
    const double v = second[j] - mean[j] * mean[j];
    if (!std::isfinite(v))
      throw std::invalid_argument("SWAG snapshot variance overflowed");
    second[j] = std::max(0., v);
  }
  return {from_vector(mean, {weights}), from_vector(second, {weights}),
          from_vector(deviations, {rank, weights})};
}

/// @brief Transform standard Gaussian innovations into SWAG weight draws.
/// @param fit Result from FitSwag().
/// @param diagonal_noise Finite CPU innovations [draws, weights].
/// @param low_rank_noise Finite CPU innovations [draws, rank].
/// @return Float64 Gaussian draws with covariance half diagonal, half low-rank.
/// @throws std::invalid_argument Invalid fit or innovation geometry.
/// @code
/// namespace n = dm::nn;
/// auto f = n::FitSwag(n::tensor({1., 2., 3., 4.}, {2, 2}), 2);
/// auto draws = n::DrawSwag(f, n::zeros({4, 2}), n::zeros({4, 2}));
/// @endcode
inline Tensor DrawSwag(const SwagFit& fit, const Tensor& diagonal_noise, const Tensor& low_rank_noise) {
  if (!fit.mean.defined() || !fit.diagonal.defined() || !fit.deviations.defined() ||
      !diagonal_noise.defined() || !low_rank_noise.defined() ||
      fit.mean.dim() != 1 || fit.mean.numel() <= 0 || fit.diagonal.shape() != fit.mean.shape() ||
      fit.deviations.dim() != 2 || fit.deviations.size(1) != fit.mean.size(0) ||
      fit.deviations.size(0) < 2 || fit.mean.dtype() != DType::Float64 ||
      fit.diagonal.dtype() != DType::Float64 || fit.deviations.dtype() != DType::Float64 ||
      diagonal_noise.device().backend != compute::Backend::CPU ||
      low_rank_noise.device().backend != compute::Backend::CPU ||
      diagonal_noise.dim() != 2 || diagonal_noise.size(0) <= 0 ||
      diagonal_noise.size(1) != fit.mean.size(0) ||
      low_rank_noise.shape() != Shape{diagonal_noise.size(0), fit.deviations.size(0)}) {
    throw std::invalid_argument("invalid SWAG draw or innovation geometry");
  }
  if (!all(is_finite(fit.mean)).item<bool>() || !all(is_finite(fit.deviations)).item<bool>() ||
      !all(logical_and(is_finite(fit.diagonal), greater_equal(fit.diagonal, 0))).item<bool>() ||
      !all(is_finite(diagonal_noise)).item<bool>() || !all(is_finite(low_rank_noise)).item<bool>()) {
    throw std::invalid_argument("SWAG draws require finite inputs and nonnegative diagonal variance");
  }
  const auto rank = fit.deviations.size(0);
  return fit.mean + sqrt(.5 * fit.diagonal) * cast(diagonal_noise, DType::Float64)
         + matmul(cast(low_rank_noise, DType::Float64), fit.deviations) / std::sqrt(2. * (rank - 1));
}

/// @brief Axis on which a diagonal-Gaussian linear readout draws innovations.
enum class LinearReparameterization {
  Weights,     ///< Draw one shared coefficient vector for the rows.
  Activations ///< Draw each row independently from its marginal distribution.
};

/// @brief One sampled linear readout and the coefficient KL to a unit Gaussian prior.
struct VariationalLinearDraw {
  Tensor value; ///< Sampled latent activation, shape [rows, 1].
  Tensor kl;    ///< Scalar coefficient KL divergence.
};

/// @brief Reparameterize a linear readout with a diagonal Gaussian weight posterior.
/// @param design Floating design matrix [rows, weights].
/// @param mean Gaussian coefficient means [weights, 1].
/// @param log_sd Log standard deviations [weights, 1].
/// @param innovation Standard normal values [weights, 1] for weight sampling or
/// [rows, 1] for local activation sampling, in the same floating dtype.
/// @param method Weight sampling preserves the joint covariance across rows.
/// Local sampling preserves only the marginal distribution at each row.
/// @return Differentiable sampled readout and analytic Gaussian prior KL.
/// @throws std::invalid_argument Incompatible shapes, dtypes or method.
/// @code
/// namespace n = dm::nn;
/// auto draw = n::VariationalLinearReparameterize(n::ones({2, 3}), n::zeros({3, 1}),
///     n::zeros({3, 1}), n::ones({2, 1}), n::LinearReparameterization::Activations);
/// @endcode
inline VariationalLinearDraw VariationalLinearReparameterize(
    const Tensor& design, const Tensor& mean, const Tensor& log_sd,
    const Tensor& innovation, LinearReparameterization method) {
  if (!design.defined() || !mean.defined() || !log_sd.defined() || !innovation.defined() ||
      design.dim() != 2 || design.size(0) <= 0 || design.size(1) <= 0 ||
      mean.shape() != Shape{design.size(1), 1} || log_sd.shape() != mean.shape() ||
      (method != LinearReparameterization::Weights && method != LinearReparameterization::Activations) ||
      innovation.shape() != Shape{method == LinearReparameterization::Weights ? design.size(1)
                                                                             : design.size(0), 1} ||
      (design.dtype() != DType::Float32 && design.dtype() != DType::Float64) ||
      mean.dtype() != design.dtype() || log_sd.dtype() != design.dtype() ||
      innovation.dtype() != design.dtype()) {
    throw std::invalid_argument("invalid variational linear readout geometry, dtype or sampling method");
  }
  const auto sd = exp(log_sd);
  const auto variance = square(sd);
  const auto kl = .5 * sum(square(mean) + variance - (1. + 2. * log_sd));
  if (method == LinearReparameterization::Weights)
    return {matmul(design, mean + sd * innovation), kl};
  const auto center = matmul(design, mean);
  const auto activation_variance = matmul(square(design), variance);
  return {center + sqrt(activation_variance) * innovation, kl};
}

}  // namespace datamunge::nn
