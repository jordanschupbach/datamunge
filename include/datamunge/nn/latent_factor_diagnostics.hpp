#pragma once
/// @file latent_factor_diagnostics.hpp
/// @brief Fitting-only standardized linear probes and heldout Gaussian-latent diagnostics.
#include "datamunge/nn/gaussian_autoencoder.hpp"
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/io.hpp"
#include "datamunge/nn/linalg.hpp"

namespace datamunge::nn::models {
/// @brief Diagnose heldout factor recovery from Gaussian posterior means with fitting-only probes.
///
/// Fits population mean/std (floor1e-8), centered targets, all-axis ridge and a
/// single-axis ridge probe. The best axis maximizes absolute FIT Pearson correlation
/// (ties choose lowest index); test correlation never affects fitting. Ridge solves
/// (X'X+ridge I)C=X'Y in Float64 without penalizing the restored target mean.
/// Targets are numeric factor indices/continuous values, not class predictions.
/// Heldout R2 uses the heldout target mean in its denominator; negative values
/// remain visible. Constant fitting axes correlate zero; each test target must vary.
/// Mean KL per unit uses fitting mu/log variance only. This is neither a MIG/DCI
/// nor the beta-VAE paper's classifier metric and does not certify disentanglement.
///
/// Returns13 named arrays: latent_mean/std, mu_variance, mean_kl_per_unit [L];
/// factor_mean [K]; fit/test_correlation and ridge_coefficients [L,K];
/// best_axis_by_fit [K] Int32; ridge/single_prediction [Nt,K]; ridge/single_r2 [K].
/// All other arrays are Float64, detached owning diagnostics. Inputs are unchanged;
/// no shared state. Work includes O(Nf*L^2+L^3+Nt*L*K); normal equations may be
/// ill-conditioned despite positive ridge. Nonfinite intermediate/output values
/// raise rather than silently returning unusable diagnostics.
/// @param fit_mu_t Finite nonempty Float32/64 fitting posterior means [Nf,L].
/// @param test_mu_t Finite matching-dtype heldout means [Nt,L].
/// @param fit_factors_t Finite numeric fitting targets [Nf,K], integer or floating.
/// @param test_factors_t Finite numeric heldout targets [Nt,K], same target dtype.
/// @param fit_log_variance_t Finite matching-shape/dtype fitting log variances [Nf,L].
/// @param ridge Finite positive diagonal penalty, default0.001.
/// @return Owning NamedTensors with13 detached arrays in the documented order.
/// @throws std::invalid_argument For invalid inputs, constant test targets, nonfinite
///         arithmetic or an invalid/underflowed ridge; solve() may reject singular systems.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto fit=n::tensor({0.,1.,2.},{3,1});
/// const auto test=n::tensor({3.,4.},{2,1});
/// const auto diagnostic=n::models::LatentFactorDiagnostics(fit,test,fit,test,n::zeros({3,1}));
/// // diagnostic[8] selects axis0 using fitting labels only; all13 arrays own lifetime.
/// @endcode
/// @par References
/// - Hoerl and Kennard (1970), Ridge Regression: Biased Estimation for Nonorthogonal
///   Problems, https://doi.org/10.1080/00401706.1970.10488634 .
/// - Kingma and Welling (2013/2014), https://arxiv.org/abs/1312.6114 , Gaussian KL.
/// @see BetaVariationalAutoencoder(), solve(), NamedTensors
inline NamedTensors LatentFactorDiagnostics(
    const Tensor& fit_mu_t,
    const Tensor& test_mu_t,
    const Tensor& fit_factors_t,
    const Tensor& test_factors_t,
    const Tensor& fit_log_variance_t,
    double        ridge = .001) {
  NoGradGuard guard;
  gaussian_detail::Match(fit_mu_t, fit_log_variance_t);
  aae_detail::Matrix(test_mu_t, fit_mu_t.size(1), fit_mu_t.dtype(), false);
  if (!fit_factors_t.defined() || fit_factors_t.dim() != 2 ||
      fit_factors_t.size(0) != fit_mu_t.size(0) || fit_factors_t.size(1) <= 0 ||
      !test_factors_t.defined() || test_factors_t.dim() != 2 ||
      test_factors_t.size(0) != test_mu_t.size(0) ||
      test_factors_t.size(1) != fit_factors_t.size(1) ||
      test_factors_t.dtype() != fit_factors_t.dtype() || !std::isfinite(ridge) || ridge <= 0)
    throw std::invalid_argument("invalid latent probe shapes, target dtype or ridge");
  for (const auto& t : {fit_factors_t, test_factors_t})
    for (double v : t.to_vector<double>())
      if (!std::isfinite(v))
        throw std::invalid_argument("finite targets required");
  const auto fit_mu  = fit_mu_t.contiguous().to_vector<double>(),
             test_mu = test_mu_t.contiguous().to_vector<double>(),
             fit_lv  = fit_log_variance_t.contiguous().to_vector<double>();
  const auto fit_f   = fit_factors_t.contiguous().to_vector<double>(),
             test_f  = test_factors_t.contiguous().to_vector<double>();
  const auto nf = fit_mu_t.size(0), nt = test_mu_t.size(0), L = fit_mu_t.size(1),
             K = fit_factors_t.size(1);
  std::vector<double> center(L), scale(L), factor_mean(K), variance(L), unit_kl(L), x(nf * L),
      y(nf * K), xt(nt * L);
  for (std::int64_t j = 0; j < L; ++j) {
    double m = 0, v = 0, k = 0;
    for (std::int64_t b = 0; b < nf; ++b)
      m += fit_mu[b * L + j];
    m /= nf;
    for (std::int64_t b = 0; b < nf; ++b) {
      const double mu = fit_mu[b * L + j], lv = fit_lv[b * L + j];
      v += (mu - m) * (mu - m);
      k += .5 * (mu * mu + std::exp(lv) - 1 - lv);
    }
    v /= nf;
    const double sd = std::max(std::sqrt(v), 1e-8);
    center[j]       = m;
    scale[j]        = sd;
    variance[j]     = v;
    unit_kl[j]      = k / nf;
    for (std::int64_t b = 0; b < nf; ++b)
      x[b * L + j] = (fit_mu[b * L + j] - m) / sd;
    for (std::int64_t b = 0; b < nt; ++b)
      xt[b * L + j] = (test_mu[b * L + j] - m) / sd;
  }
  for (std::int64_t j = 0; j < K; ++j) {
    double f = 0;
    for (std::int64_t b = 0; b < nf; ++b)
      f += fit_f[b * K + j];
    f /= nf;
    factor_mean[j] = f;
    for (std::int64_t b = 0; b < nf; ++b)
      y[b * K + j] = fit_f[b * K + j] - f;
  }
  /// @brief Compute split-specific Pearson correlations; constant axes return zero.
  /// @param mu Split means, row-major.
  /// @param factors Split targets, row-major.
  /// @param n Split row count.
  /// @return L by K correlations in row-major order.
  auto correlation =
      [L, K](const std::vector<double>& mu, const std::vector<double>& factors, std::int64_t n) {
        std::vector<double> out(L * K);
        for (std::int64_t d = 0; d < L; ++d)
          for (std::int64_t j = 0; j < K; ++j) {
            double a = 0, b = 0, aa = 0, bb = 0, ab = 0;
            for (std::int64_t i = 0; i < n; ++i) {
              a += mu[i * L + d];
              b += factors[i * K + j];
            }
            a /= n;
            b /= n;
            for (std::int64_t i = 0; i < n; ++i) {
              const double u = mu[i * L + d] - a, v = factors[i * K + j] - b;
              aa += u * u;
              bb += v * v;
              ab += u * v;
            }
            out[d * K + j] = aa > 1e-20 && bb > 0 ? ab / std::sqrt(aa * bb) : 0;
          }
        return out;
      };
  const auto fit_corr  = correlation(fit_mu, fit_f, nf),
             test_corr = correlation(test_mu, test_f, nt);
  // Ridge normal equations (X^T X + .001 I) C = X^T Y, solved in float64.
  auto                X = from_vector(x, {nf, L}), Y = from_vector(y, {nf, K});
  auto                A    = matmul(transpose(X), X) + mul(eye(L, DType::Float64), ridge);
  auto                coef = solve(A, matmul(transpose(X), Y)).contiguous();
  const auto          c    = coef.to_vector<double>();
  std::vector<double> prediction(nt * K), single(nt * K), ridge_r2(K), single_r2(K);
  std::vector<int>    axis(K);
  for (std::int64_t j = 0; j < K; ++j) {
    std::int64_t best = 0;
    for (std::int64_t d = 1; d < L; ++d)
      if (std::abs(fit_corr[d * K + j]) > std::abs(fit_corr[best * K + j]))
        best = d;
    axis[j]          = best;
    double numerator = 0, denominator = ridge;
    for (std::int64_t i = 0; i < nf; ++i) {
      numerator += x[i * L + best] * y[i * K + j];
      denominator += std::pow(x[i * L + best], 2);
    }
    const double one       = numerator / denominator;
    double       truthmean = 0;
    for (std::int64_t i = 0; i < nt; ++i)
      truthmean += test_f[i * K + j];
    truthmean /= nt;
    double ss = 0, err = 0, oneerr = 0;
    for (std::int64_t i = 0; i < nt; ++i) {
      double p = factor_mean[j];
      for (std::int64_t d = 0; d < L; ++d)
        p += xt[i * L + d] * c[d * K + j];
      const double q        = factor_mean[j] + xt[i * L + best] * one;
      prediction[i * K + j] = p;
      single[i * K + j]     = q;
      const double target   = test_f[i * K + j];
      ss += std::pow(target - truthmean, 2);
      err += std::pow(p - target, 2);
      oneerr += std::pow(q - target, 2);
    }
    if (!(ss > 0))
      throw std::invalid_argument("each test factor must vary");
    ridge_r2[j]  = 1 - err / ss;
    single_r2[j] = 1 - oneerr / ss;
  }
  NamedTensors result{
      {"latent_mean", from_vector(center, {L})},
      {"latent_std", from_vector(scale, {L})},
      {"factor_mean", from_vector(factor_mean, {K})},
      {"mu_variance", from_vector(variance, {L})},
      {"mean_kl_per_unit", from_vector(unit_kl, {L})},
      {"fit_correlation", from_vector(fit_corr, {L, K})},
      {"test_correlation", from_vector(test_corr, {L, K})},
      {"ridge_coefficients", coef},
      {"best_axis_by_fit", from_vector(axis, {K})},
      {"ridge_prediction", from_vector(prediction, {nt, K})},
      {"single_prediction", from_vector(single, {nt, K})},
      {"ridge_r2", from_vector(ridge_r2, {K})},
      {"single_r2", from_vector(single_r2, {K})}};
  for (const auto& entry : result)
    for (double v : entry.second.to_vector<double>())
      if (!std::isfinite(v))
        throw std::invalid_argument("nonfinite latent diagnostic arithmetic");
  return result;
}
} // namespace datamunge::nn::models
