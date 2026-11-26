#pragma once

/// @file reliability.hpp
/// @brief Internal-consistency reliability: Cronbach's alpha with Feldt and asymptotic intervals, item
/// analysis, a one-factor maximum-likelihood model and McDonald's omega with bootstrap intervals.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::stats {

/// @brief Statistics for one item of a scale.
/// @par Example
/// @code
/// dm::stats::ItemStatistics s{.mean = 3};
/// std::println("{}", s.mean); // 3
/// @endcode
struct ItemStatistics {
    double mean{};              ///< Item mean.
    double sd{};                ///< Item standard deviation (n - 1 divisor).
    double item_rest{};         ///< Corrected item-total correlation: item versus the sum of the other items.
    double alpha_if_deleted{};  ///< Raw alpha of the remaining items.
};

/// @brief Result of cronbach_alpha().
/// @par Example
/// @code
/// dm::stats::AlphaEstimate a;
/// std::println("{}", a.items); // 0
/// @endcode
struct AlphaEstimate {
    std::size_t n{};                 ///< Respondents (complete rows).
    std::size_t items{};             ///< Number of items p.
    double alpha{};                  ///< Raw alpha \f$\frac{p}{p-1}(1-\sum s^2_j/s^2_X)\f$ from the covariance matrix.
    double standardized_alpha{};     ///< Alpha of standardized items, \f$p\bar r/(1+(p-1)\bar r)\f$.
    double average_r{};              ///< Mean inter-item correlation.
    double standard_error{};         ///< Asymptotic (distribution-free) standard error, Duhachek–Iacobucci (2004).
    double feldt_lower{};            ///< Feldt (1965) interval assuming compound symmetry and normality.
    double feldt_upper{};            ///< Upper Feldt limit.
    double asymptotic_lower{};       ///< alpha - z * standard_error.
    double asymptotic_upper{};       ///< alpha + z * standard_error.
    std::vector<ItemStatistics> item_statistics; ///< Per-item statistics.
};

/// @brief Cronbach's alpha with item analysis and two confidence intervals.
///
/// Alpha equals the reliability of the sum score when items are essentially tau-equivalent (equal loadings
/// on one factor, uncorrelated errors); for congeneric items it is a lower bound. The Feldt interval uses
/// \f$(1-\alpha)/(1-\hat\alpha)\sim F_{n-1,(n-1)(p-1)}\f$; the asymptotic one uses the distribution-free
/// variance of van Zyl, Neudecker and Nel (2000), as reported by `psych::alpha` ("ase").
/// @param data n-by-p matrix, rows are respondents and columns items.
/// @param confidence Confidence level in (0, 1).
/// @return Estimates and intervals.
/// @throws std::invalid_argument For fewer than two items or three rows, nonfinite data, a constant item, or
/// confidence outside (0, 1).
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4, 2, {1, 2, 2, 2, 3, 4, 4, 5});
/// std::println("{:.3f}", dm::stats::cronbach_alpha(x).alpha); // 0.967
/// @endcode
AlphaEstimate cronbach_alpha(const linalg::DenseMatrix<double>& data, double confidence = 0.95);

/// @brief One-factor model \f$\Sigma=\lambda\lambda^\top+\Psi\f$ fitted by maximum likelihood.
/// @par Example
/// @code
/// dm::stats::OneFactorFit f;
/// std::println("{}", f.loadings.size()); // 0
/// @endcode
struct OneFactorFit {
    std::vector<double> loadings;      ///< \f$\lambda\f$ (covariance units, sign chosen so the sum is positive).
    std::vector<double> uniquenesses;  ///< Diagonal of \f$\Psi\f$.
    double chi_square{};               ///< Likelihood-ratio test of the model, \f$nF_{ML}\f$ (lavaan's statistic).
    double df{};                       ///< p(p+1)/2 - 2p.
    double p_value{};                  ///< Chi-square p-value (NaN when df <= 0).
    std::size_t iterations{};          ///< EM iterations.
    bool converged{};                  ///< Whether the largest parameter change fell below the tolerance.
    bool heywood{};                    ///< Whether some uniqueness reached its floor (1e-8 of the item variance).
};

/// @brief Fits a one-factor model to a covariance matrix by maximum likelihood (EM algorithm).
///
/// The input S (divisor n - 1) is rescaled to the ML divisor n before fitting, so loadings and
/// uniquenesses match lavaan's and are smaller than unbiased-scale values by \f$\sqrt{(n-1)/n}\f$.
///
/// Maximizes the Wishart likelihood \f$-\tfrac n2[\log|\Sigma|+\operatorname{tr}(S\Sigma^{-1})]\f$ with the
/// factor variance fixed at 1, using the Rubin–Thayer EM algorithm started from principal-axis values. This
/// is the estimator of `lavaan::cfa(..., std.lv = TRUE)` and `factanal` with one factor.
/// @param covariance Sample covariance S (p >= 3, positive definite).
/// @param n Sample size used for the test statistic.
/// @param max_iterations EM iteration limit.
/// @param tolerance Convergence tolerance on parameters relative to item standard deviations.
/// @return The fit.
/// @throws std::invalid_argument If S is not square with p >= 3 or not positive definite, or n < 2.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> s(3, 3, {1, .5, .5, .5, 1, .5, .5, .5, 1});
/// std::println("{:.3f}", dm::stats::one_factor_fit(s, 100).loadings[0]); // 0.704
/// @endcode
OneFactorFit one_factor_fit(const linalg::DenseMatrix<double>& covariance, std::size_t n,
                            std::size_t max_iterations = 100000, double tolerance = 1e-12);

/// @brief Options for mcdonald_omega().
/// @par Example
/// @code
/// dm::stats::OmegaOptions o{.replicates = 0};
/// std::println("{}", o.replicates); // 0
/// @endcode
struct OmegaOptions {
    double confidence{0.95};     ///< Level of the bootstrap intervals.
    std::size_t replicates{1000}; ///< Nonparametric bootstrap replicates (0 skips the intervals).
    std::uint64_t seed{1};       ///< Bootstrap seed.
};

/// @brief Result of mcdonald_omega().
/// @par Example
/// @code
/// dm::stats::OmegaEstimate o;
/// std::println("{}", o.replicates); // 0
/// @endcode
struct OmegaEstimate {
    std::size_t n{};              ///< Respondents.
    double omega{};               ///< \f$(\sum\lambda)^2/((\sum\lambda)^2+\sum\psi)\f$: model-implied total variance.
    double omega_observed{};      ///< \f$(\sum\lambda)^2/\mathbf 1^\top S\mathbf 1\f$: observed total variance (Bentler).
    double alpha{};               ///< Cronbach's alpha of the same data, for comparison.
    OneFactorFit fit;             ///< The one-factor model.
    std::size_t replicates{};     ///< Bootstrap replicates used.
    double omega_lower{};         ///< Percentile bootstrap interval for omega (NaN without replicates).
    double omega_upper{};         ///< Upper limit.
    double alpha_lower{};         ///< Percentile bootstrap interval for alpha.
    double alpha_upper{};         ///< Upper limit.
};

/// @brief McDonald's omega (total) from a one-factor maximum-likelihood model, with bootstrap intervals.
///
/// Omega is the share of sum-score variance due to the common factor. It equals the sum-score reliability for
/// congeneric items with uncorrelated errors, where alpha is too low when loadings differ. Rows are resampled
/// for the percentile intervals of omega and alpha (Kelley and Pornprasertmanit 2016).
/// @param data n-by-p item matrix (p >= 3).
/// @param options Interval options.
/// @return Estimates.
/// @throws std::invalid_argument As for cronbach_alpha(), for p < 3 or a singular covariance.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(5, 3, {1, 2, 1, 2, 2, 3, 3, 4, 3, 4, 3, 5, 5, 5, 4});
/// std::println("{:.3f}", dm::stats::mcdonald_omega(x, {.replicates = 0}).omega); // 0.936
/// @endcode
OmegaEstimate mcdonald_omega(const linalg::DenseMatrix<double>& data, const OmegaOptions& options = {});

} // namespace datamunge::stats
