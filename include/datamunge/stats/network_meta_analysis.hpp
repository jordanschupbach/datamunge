#pragma once

/// @file network_meta_analysis.hpp
/// @brief Contrast-based network meta-analysis by generalised least squares, with multi-arm studies,
/// common and random effects, heterogeneity/inconsistency decomposition and P-score ranking.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief One study's relative effects: contrasts of each arm against the study's first (baseline) arm.
///
/// For a multi-arm study the contrasts share the baseline arm and are correlated; `covariance` holds their
/// full covariance matrix (for arm-level data, Var(baseline) off the diagonal; see
/// network_study_from_arms()).
/// @par Example
/// @code
/// dm::stats::NetworkStudy s{.treatments = {0, 1}, .effects = {0.4},
///                           .covariance = dm::linalg::DenseMatrix<double>(1, 1, 0.04)};
/// std::println("{}", s.effects.size()); // 1
/// @endcode
struct NetworkStudy {
    std::vector<std::size_t> treatments;    ///< Treatment indices of the arms; the first is the study baseline.
    std::vector<double> effects;            ///< Contrasts treatments[j] versus treatments[0], j = 1..arms-1.
    linalg::DenseMatrix<double> covariance; ///< (arms - 1) x (arms - 1) covariance of the contrasts.
};

/// @brief Builds a study from arm-level estimates (for example log odds or means) and their variances.
///
/// Contrasts are differences from the first arm; their covariance has \f$s_0^2+s_j^2\f$ on the diagonal and
/// \f$s_0^2\f$ off it, which is exactly the correlation induced by the shared baseline arm.
/// @param treatments Treatment index of each arm (distinct; at least two).
/// @param estimates Arm-level estimates.
/// @param variances Their variances (> 0).
/// @return The study.
/// @throws std::invalid_argument For fewer than two arms, mismatched lengths, repeated treatments or
/// invalid values.
/// @par Example
/// @code
/// const auto s = dm::stats::network_study_from_arms({2, 0, 1}, {0.5, 0.2, 0.1}, {0.04, 0.05, 0.06});
/// std::println("{:.2f} {:.2f}", s.effects[0], s.covariance(0, 1)); // -0.30 0.04
/// @endcode
NetworkStudy network_study_from_arms(const std::vector<std::size_t>& treatments, const std::vector<double>& estimates,
                                     const std::vector<double>& variances);

/// @brief Estimators of the common between-study variance in a network.
/// @par Example
/// @code
/// const auto e = dm::stats::NetworkTauEstimator::reml;
/// std::println("{}", e == dm::stats::NetworkTauEstimator::dersimonian_laird); // false
/// @endcode
enum class NetworkTauEstimator {
    dersimonian_laird, ///< Generalised method of moments from the network Q (netmeta's default).
    reml,              ///< Restricted maximum likelihood (Fisher scoring).
};

/// @brief Options for network_meta_analysis().
/// @par Example
/// @code
/// const dm::stats::NetworkMetaOptions o{.reference = 2};
/// std::println("{}", o.larger_is_better); // true
/// @endcode
struct NetworkMetaOptions {
    std::size_t reference{0};                                       ///< Reference treatment (row of the reported effects).
    NetworkTauEstimator estimator{NetworkTauEstimator::dersimonian_laird}; ///< tau2 estimator.
    double confidence{0.95};                                        ///< Interval coverage.
    bool larger_is_better{true};                                    ///< Direction for P-scores.
    std::size_t max_iterations{1000};                               ///< REML iteration cap.
    double tolerance{1e-12};                                        ///< REML convergence threshold.
};

/// @brief All pairwise relative effects from one model (common or random effects).
/// @par Example
/// @code
/// dm::stats::NetworkEstimates e;
/// std::println("{}", e.p_scores.size()); // 0
/// @endcode
struct NetworkEstimates {
    linalg::DenseMatrix<double> estimate;       ///< estimate(i, j): effect of treatment i versus j (d_i - d_j).
    linalg::DenseMatrix<double> standard_error; ///< Its standard error (0 on the diagonal).
    linalg::DenseMatrix<double> lower;          ///< Lower confidence limit.
    linalg::DenseMatrix<double> upper;          ///< Upper confidence limit.
    linalg::DenseMatrix<double> p_value;        ///< Two-sided z-test p-value (1 on the diagonal).
    std::vector<double> p_scores;               ///< P-score of each treatment in [0, 1] (mean probability of being better).
};

/// @brief Result of network_meta_analysis().
/// @par Example
/// @code
/// dm::stats::NetworkMetaAnalysis n;
/// std::println("{}", n.treatments); // 0
/// @endcode
struct NetworkMetaAnalysis {
    std::size_t treatments{};      ///< Number of treatments.
    std::size_t studies{};         ///< Number of studies.
    std::size_t contrasts{};       ///< Number of independent contrasts (sum of arms - 1).
    std::size_t designs{};         ///< Number of distinct designs (sets of treatments compared).
    NetworkEstimates common;       ///< Common-effect (fixed-effect) estimates.
    NetworkEstimates random;       ///< Random-effects estimates.
    double q{};                    ///< Network Q (generalised Cochran's Q of the common-effect model).
    double q_df{};                 ///< contrasts - (treatments - 1).
    double q_p_value{};            ///< Its chi-squared p-value.
    double q_within{};             ///< Heterogeneity within designs.
    double q_within_df{};          ///< Its degrees of freedom.
    double q_within_p_value{};     ///< Its p-value.
    double q_between{};            ///< Inconsistency between designs (q - q_within).
    double q_between_df{};         ///< Its degrees of freedom.
    double q_between_p_value{};    ///< Its p-value.
    double tau2{};                 ///< Common between-study variance of each contrast.
    double tau{};                  ///< sqrt(tau2).
    double i2{};                   ///< \f$I^2=\max(0,(Q-df)/Q)\f$ in percent.
    std::size_t iterations{};      ///< REML iterations (0 for DL).
    bool converged{true};          ///< Whether REML converged.
};

/// @brief Network meta-analysis combining direct and indirect evidence by generalised least squares.
///
/// Each contrast is modelled as \f$d_{t}-d_{b}\f$ with \f$d_{\text{reference}}=0\f$, so indirect comparisons
/// follow from consistency: an A-versus-B estimate borrows strength from A-C and B-C trials. The weight
/// matrix is the inverse of the block-diagonal within-study covariance, which handles multi-arm studies
/// exactly; random effects add \f$\tau^2\f$ to each contrast's variance and \f$\tau^2/2\f$ to the covariance
/// of contrasts sharing a baseline (homogeneous heterogeneity). The common-effect Q is split into
/// heterogeneity within designs and inconsistency between designs (design-by-treatment decomposition).
/// P-scores average, over the other treatments, the one-sided probability that a treatment is better.
/// Matches `netmeta::netmeta` (graph-theoretical method), `decomp.design()` and `netrank()`.
/// @param studies The studies.
/// @param options Reference, estimator, confidence and ranking direction.
/// @return Estimates of every pairwise comparison, heterogeneity and inconsistency statistics, P-scores.
/// @throws std::invalid_argument For invalid studies (bad sizes, repeated or out-of-range treatments,
/// non-positive-definite covariances), fewer than two treatments, a disconnected network or an invalid
/// reference.
/// @par Example
/// @code
/// std::vector<dm::stats::NetworkStudy> s{
///     {{0, 1}, {0.5}, dm::linalg::DenseMatrix<double>(1, 1, 0.04)},
///     {{1, 2}, {0.3}, dm::linalg::DenseMatrix<double>(1, 1, 0.04)}};
/// const auto n = dm::stats::network_meta_analysis(s);
/// std::println("{:.2f}", n.common.estimate(2, 0)); // 0.80
/// @endcode
NetworkMetaAnalysis network_meta_analysis(const std::vector<NetworkStudy>& studies,
                                          const NetworkMetaOptions& options = {});

} // namespace datamunge::stats
