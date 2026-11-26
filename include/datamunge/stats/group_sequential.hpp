#pragma once

/// @file group_sequential.hpp
/// @brief One-sided group-sequential efficacy boundaries from alpha-spending functions and boundary-crossing probabilities.

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Lan–DeMets alpha-spending function families.
/// @par Example
/// @code
/// const auto f = dm::stats::SpendingFunction::pocock;
/// std::println("{}", f == dm::stats::SpendingFunction::obrien_fleming); // false
/// @endcode
enum class SpendingFunction {
    obrien_fleming,    ///< \f$\alpha(t)=2-2\Phi(z_{1-\alpha/2}/\sqrt t)\f$: spends almost nothing early.
    pocock,            ///< \f$\alpha(t)=\alpha\log\{1+(e-1)t\}\f$: nearly constant nominal levels.
    power,             ///< \f$\alpha(t)=\alpha t^{\rho}\f$ with \f$\rho>0\f$ (Kim and DeMets).

    /// \f$\alpha(t)=\alpha(1-e^{-\gamma t})/(1-e^{-\gamma})\f$ for \f$\gamma\ne0\f$, and \f$\alpha t\f$
    /// for \f$\gamma=0\f$ (Hwang, Shih and DeCani).
    hwang_shih_decani,
};

/// @brief Error spending specification.
/// @par Example
/// @code
/// const dm::stats::AlphaSpending spending{.function = dm::stats::SpendingFunction::power, .parameter = 3};
/// std::println("{}", spending.alpha); // 0.025
/// @endcode
struct AlphaSpending {
    SpendingFunction function{SpendingFunction::obrien_fleming}; ///< Spending family.
    double alpha{.025};      ///< Total one-sided type I error in (0, 0.5).
    double parameter{1};     ///< \f$\rho\f$ for the power family or \f$\gamma\f$ for Hwang–Shih–DeCani; unused otherwise.
};

/// @brief Cumulative type I error spent by information fraction t.
/// @param t Information fraction in [0, 1] (values above 1 are treated as 1).
/// @param spending Family, total alpha and parameter.
/// @return \f$\alpha(t)\f$, increasing from 0 at t = 0 to alpha at t = 1.
/// @throws std::invalid_argument For negative or nonfinite t, alpha outside (0, 0.5), or a nonpositive \f$\rho\f$.
/// @par Example
/// @code
/// std::println("{:.6f}", dm::stats::alpha_spent(.5, {})); // 0.001525
/// @endcode
/// @par References
/// K. K. G. Lan and D. L. DeMets (1983). "Discrete sequential boundaries for clinical trials".
/// Biometrika 70(3), 659-663. doi:10.1093/biomet/70.3.659
/// @see group_sequential_boundaries()
[[nodiscard]] double alpha_spent(double t, const AlphaSpending& spending);

/// @brief Efficacy boundaries and their error spending.
/// @par Example
/// @code
/// const dm::stats::GroupSequentialBoundary b;
/// std::println("{}", b.z.size()); // 0
/// @endcode
struct GroupSequentialBoundary {
    std::vector<double> fractions;       ///< Information fractions \f$t_k\f$ of the looks.
    std::vector<double> z;               ///< Upper boundaries \f$b_k\f$ for the standardized statistic \f$Z_k\f$.
    std::vector<double> nominal_alpha;   ///< \f$1-\Phi(b_k)\f$, the one-look p-value threshold.
    std::vector<double> cumulative_alpha; ///< \f$\alpha(t_k)\f$ spent by look k.
};

/// @brief Computes one-sided efficacy boundaries by alpha spending.
///
/// With \f$Z_k\f$ the standardized statistic at information fraction \f$t_k\f$, the
/// canonical joint distribution has \f$\operatorname{cov}(Z_j,Z_k)=\sqrt{t_j/t_k}\f$ for
/// \f$j\le k\f$. Each \f$b_k\f$ solves \f$P_0(Z_1<b_1,\dots,Z_{k-1}<b_{k-1},Z_k\ge b_k)=
/// \alpha(t_k)-\alpha(t_{k-1})\f$ given the earlier boundaries. Probabilities use the
/// Armitage–McPherson–Rowe recursion on the score scale with composite Simpson
/// integration (absolute error near \f$10^{-9}\f$). There is no futility boundary.
/// When \f$t_K<1\f$, only \f$\alpha(t_K)\f$ is spent.
/// @param fractions Strictly increasing information fractions in (0, 1].
/// @param spending Spending function, alpha and parameter.
/// @return Boundaries, nominal levels and cumulative alpha.
/// @throws std::invalid_argument For an empty, unsorted or out-of-range fraction vector, or invalid spending.
/// @par Example
/// @code
/// const auto b = dm::stats::group_sequential_boundaries({.25, .5, .75, 1}, {});
/// std::println("{:.4f} {:.4f}", b.z[0], b.z[3]); // 4.3326 2.0141
/// @endcode
/// @par References
/// - K. K. G. Lan and D. L. DeMets (1983), doi:10.1093/biomet/70.3.659
/// - C. Jennison and B. W. Turnbull (2000). Group Sequential Methods with Applications to
///   Clinical Trials. Chapman & Hall/CRC, ch. 7 and 19. ISBN 978-0-8493-0316-6
/// - P. Armitage, C. K. McPherson and B. C. Rowe (1969). "Repeated significance tests on
///   accumulating data". JRSS A 132(2), 235-244. doi:10.2307/2343787
/// @see boundary_crossing_probabilities(), alpha_spent()
[[nodiscard]] GroupSequentialBoundary group_sequential_boundaries(const std::vector<double>& fractions,
                                                                  const AlphaSpending& spending);

/// @brief Probabilities of first crossing each efficacy boundary under a drift.
/// @par Example
/// @code
/// const dm::stats::CrossingProbabilities p;
/// std::println("{}", p.total); // 0
/// @endcode
struct CrossingProbabilities {
    std::vector<double> stage;        ///< \f$P(\text{stop for efficacy at look }k)\f$.
    std::vector<double> cumulative;   ///< Running sums of `stage`.
    double total{};                   ///< Probability of crossing at any look (power, or size when the drift is 0).

    /// Expected information fraction at stopping, \f$\sum_k t_kP(\text{stop at }k)+t_K P(\text{no crossing})\f$.
    double expected_fraction{};
};

/// @brief Crossing probabilities for given boundaries when \f$E[Z_k]=\theta\sqrt{t_k}\f$.
///
/// The drift \f$\theta\f$ is the expected value of the statistic at full information,
/// e.g. \f$\delta/\sqrt{4\sigma^2/N}\f$ for a two-arm mean difference with N patients.
/// @param fractions Strictly increasing information fractions in (0, 1].
/// @param z Upper boundaries, one per look (may be infinite to skip a look).
/// @param drift Drift \f$\theta\f$, finite.
/// @return Stagewise and total crossing probabilities and the expected stopping fraction.
/// @throws std::invalid_argument For invalid fractions, mismatched lengths, NaN boundaries or a nonfinite drift.
/// @par Example
/// @code
/// const auto b = dm::stats::group_sequential_boundaries({.25, .5, .75, 1}, {});
/// const auto p = dm::stats::boundary_crossing_probabilities(b.fractions, b.z, 3);
/// std::println("{:.4f}", p.total); // 0.8442
/// @endcode
/// @see group_sequential_boundaries(), drift_for_power()
[[nodiscard]] CrossingProbabilities boundary_crossing_probabilities(const std::vector<double>& fractions,
                                                                    const std::vector<double>& z, double drift);

/// @brief Drift at which the boundaries give the requested power.
///
/// Dividing by the fixed-design drift \f$z_{1-\alpha}+z_{1-\beta}\f$ and squaring gives the
/// maximum-information inflation factor of the sequential design.
/// @param fractions Strictly increasing information fractions in (0, 1].
/// @param z Upper boundaries, one per look.
/// @param power Target probability of crossing, above the size of the boundaries and below 1.
/// @return Drift \f$\theta\f$ solving `boundary_crossing_probabilities(...).total == power`.
/// @throws std::invalid_argument For invalid inputs or a power outside (size, 1).
/// @par Example
/// @code
/// const auto b = dm::stats::group_sequential_boundaries({.25, .5, .75, 1}, {});
/// std::println("{:.4f}", dm::stats::drift_for_power(b.fractions, b.z, .9)); // 3.2710
/// @endcode
/// @see boundary_crossing_probabilities()
[[nodiscard]] double drift_for_power(const std::vector<double>& fractions, const std::vector<double>& z,
                                     double power);

} // namespace datamunge::stats
