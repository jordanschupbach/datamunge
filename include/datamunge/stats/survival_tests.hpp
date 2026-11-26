#pragma once

/// @file survival_tests.hpp
/// @brief Weighted log-rank (Fleming–Harrington G-rho-gamma) tests for two or more groups, optionally stratified.

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/survival.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Options for survival_group_test().
/// @par Example
/// @code
/// const dm::stats::SurvivalGroupTestOptions peto{.rho = 1};
/// std::println("{}", peto.gamma); // 0
/// @endcode
struct SurvivalGroupTestOptions {
    double rho{0};                ///< Weight exponent on \f$\hat S(t-)\f$ (0: log-rank, 1: Peto–Peto/Prentice).
    double gamma{0};              ///< Weight exponent on \f$1-\hat S(t-)\f$ (positive values emphasize late differences).
    std::vector<int> strata{};    ///< Optional stratum label per observation (empty: one stratum).
};

/// @brief Result of a weighted log-rank test.
/// @par Example
/// @code
/// const dm::stats::SurvivalGroupTest t;
/// std::println("{}", t.degrees_of_freedom); // 0
/// @endcode
struct SurvivalGroupTest {
    std::vector<int> groups;          ///< Group labels in increasing order.
    std::vector<double> observed;     ///< Weighted observed events per group, summed over strata.
    std::vector<double> expected;     ///< Weighted expected events per group under the null.
    linalg::DenseMatrix<double> variance; ///< Covariance of observed minus expected (hypergeometric).
    double chi_square{};              ///< \f$(O-E)^\top V^{-}(O-E)\f$ over groups with positive expectation.
    std::size_t degrees_of_freedom{}; ///< Number of such groups minus one.
    double p_value{};                 ///< Upper chi-square tail.
};

/// @brief Weighted log-rank test of equal survival across groups, as R's `survdiff` (which covers \f$\gamma=0\f$).
///
/// At each distinct event time within a stratum, the observed events of each group are compared
/// with their expectation \f$d\,n_g/n\f$ under a common hazard, weighted by
/// \f$w(t)=\hat S(t-)^\rho\{1-\hat S(t-)\}^\gamma\f$, where \f$\hat S\f$ is the pooled left-continuous
/// Kaplan–Meier estimate within the stratum. Sums over strata give O − E and its hypergeometric
/// covariance, and the quadratic form drops one group (a generalized inverse).
/// @param observations Right-censored records.
/// @param groups Group label per observation (at least two distinct labels).
/// @param options Weight exponents and optional strata.
/// @return Observed, expected, covariance, statistic, degrees of freedom and p-value.
/// @throws std::invalid_argument For mismatched lengths, invalid times, fewer than two groups,
/// negative exponents, or a singular covariance.
/// @par Example
/// @code
/// const std::vector<dm::stats::SurvivalObservation> obs{{1, true}, {3, true}, {4, false}, {6, true},
///                                                      {2, true}, {5, true}, {8, true}, {9, false}};
/// const auto t = dm::stats::survival_group_test(obs, {0, 0, 0, 0, 1, 1, 1, 1});
/// std::println("{:.3f} {}", t.chi_square, t.degrees_of_freedom); // 0.846 1
/// @endcode
/// @par References
/// - N. Mantel (1966). "Evaluation of survival data and two new rank order statistics arising in
///   its consideration." Cancer Chemotherapy Reports 50(3), 163-170.
/// - D. P. Harrington and T. R. Fleming (1982). "A class of rank test procedures for censored
///   survival data." Biometrika 69(3), 553-566. doi:10.1093/biomet/69.3.553
/// @see prentice_wilcoxon_test(), kaplan_meier()
[[nodiscard]] SurvivalGroupTest survival_group_test(const std::vector<SurvivalObservation>& observations,
                                                    const std::vector<int>& groups,
                                                    const SurvivalGroupTestOptions& options = {});

} // namespace datamunge::stats
