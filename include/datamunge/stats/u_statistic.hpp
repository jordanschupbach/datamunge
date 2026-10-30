#pragma once

/// @file u_statistic.hpp
/// @brief Complete and randomly incomplete U-statistics, conditional projections, jackknife and exact two-point laws.

#include <datamunge/stats/resample_types.hpp>
#include <datamunge/dstruct/dataframe.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace datamunge::stats {

/// @brief Enumeration or random-subset settings for UStatistic.
///
/// @par Example
/// @code
/// const dm::stats::UStatistic u(2, {1000,42}); // exactly 1000 draws, possibly repeated
/// @endcode
/// @see UStatistic
struct UStatisticOptions {
    std::size_t max_terms{0}; ///< Zero enumerates every m-subset; positive B samples B subsets with replacement across draws.
    std::uint64_t seed{42}; ///< Seed for local mt19937_64 in each call; ignored for complete enumeration.
};

/// @brief U estimate and legacy first-projection variance diagnostics, not total uncertainty.
///
/// The variance field is m^2*zeta1/n. For complete fits this is an asymptotic projection
/// estimate; it is not an exact finite-sample variance and does not handle degeneracy.
/// For incomplete fits conditional means are noisy and some rows may be uncovered, so this
/// field is not a calibrated data-sampling or total variance. Use analyze() to separate
/// conditional Monte Carlo error and complete-fit delete-one jackknife estimates.
///
/// @par Example
/// @code
/// const auto result=dm::stats::UStatistic(1).run({1,2,3},[](const auto& x){return x[0];});
/// std::print("{:.3f}\n",result.estimate); // 2.000
/// @endcode
/// @see UStatisticAnalysis, UStatistic::analyze()
struct UStatisticResult {
    double estimate{0.0}; ///< Mean of evaluated kernel values.
    double zeta1{0.0}; ///< Sum of (conditional_mean-estimate)^2 over covered rows, divided by covered_rows-1; zero if fewer than two.
    double variance{0.0}; ///< Legacy leading-order estimate m^2*zeta1/n, including noisy projections for incomplete fits.
    double standard_error{0.0}; ///< Square root of non-negative variance; not an automatic confidence-interval guarantee.
    std::size_t degree{0}; ///< Kernel degree m.
    std::size_t num_terms{0}; ///< Number of evaluations; repeated subsets count separately.
    bool complete{false}; ///< True only for deliberate enumeration, never inferred from a positive draw budget.
};

/// @brief Owned U-statistic fit, per-observation diagnostics and separately identified uncertainties.
///
/// @par Example
/// @code
/// const auto fit=dm::stats::UStatistic(2).analyze({1,2,4},[](const auto& x){return 0.5*(x[0]-x[1])*(x[0]-x[1]);});
/// std::print("{}\n",fit.summary().to_md(4));
/// @endcode
/// @see UStatistic::analyze()
struct UStatisticAnalysis {
    UStatisticResult result; ///< Existing estimate and projection fields, with their original semantics.
    dstruct::DataFrame projections; ///< row, value, inclusions, conditional_mean; leave_one_out and pseudo_value are NaN unless complete with n>m.
    double kernel_variance{0.0}; ///< Conditional variance of a uniformly chosen subset value: complete denominator M, incomplete unbiased sample denominator B-1; NaN for B=1.
    double monte_carlo_variance{0.0}; ///< Zero for complete enumeration; kernel_variance/B for incomplete sampling, conditional on the fixed data.
    double jackknife_variance{std::numeric_limits<double>::quiet_NaN()}; ///< Delete-one sampling-variance estimate, only complete with n>m; no degeneracy correction.

    /// @brief A one-row table separating the legacy projection, jackknife and Monte Carlo errors.
    /// @return Owned columns n, degree, terms, complete, covered_rows, estimate,
    ///         projection_se, jackknife_se, conditional_mc_se and kernel_variance. No confidence interval is implied.
    /// @throws std::invalid_argument If projections lacks the numeric inclusions column produced by analyze().
    /// @par Example
    /// See the class example.
    /// @see UStatistic::analyze()
    [[nodiscard]] dstruct::DataFrame summary() const;
};

/// @brief Exact finite-sample law of a complete U-statistic under a two-point i.i.d. population.
///
/// @par Example
/// @code
/// const auto law=dm::stats::UStatistic(2).two_point_law(4,-1,1,0.5,[](const auto& x){return x[0]*x[1];});
/// std::print("{:.6f}\n",law.variance); // 0.166667
/// @endcode
/// @see UStatistic::two_point_law()
struct UStatisticLaw {
    dstruct::DataFrame distribution; ///< n+1 rows: high_count, probability and estimate conditional on that count.
    std::size_t sample_size{0}; ///< Number n of independent observations in the law.
    double target{0.0}; ///< Population E[h] computed by the degree-m binomial mixture.
    double mean{0.0}; ///< Weighted finite-sample expectation; equals target up to rounding.
    double variance{0.0}; ///< Exact finite-sample variance up to numerical summation, including all Hoeffding orders.
    double first_order_variance{0.0}; ///< m^2*zeta1/n from the known population conditional means.
};

/// @brief Average a symmetric degree-m kernel over all distinct-index subsets or random subset draws.
///
/// Complete enumeration averages C(n,m) terms. Random draws contain distinct indices within
/// each tuple, but can repeat subsets across draws. Each next subset is uniform conditional
/// on the preceding shuffle state. The kernel must be symmetric and deterministic; neither
/// property is tested. Complete tuples are in index order, random tuples in shuffle order.
/// For i.i.d. data and integrable h, both estimators are unbiased for E[h]. Asymptotic
/// normality additionally requires fixed degree, finite second moment and nonzero first
/// projection variance; growing-degree and degenerate cases need separate theory.
///
/// Fitting uses O(n+m) working storage and O(T*m) bookkeeping plus T callback costs.
/// Complete T=C(n,m) may be prohibitive; no automatic switch to incomplete sampling occurs.
/// Inputs are borrowed and unchanged; each call owns its RNG. Same seeds reproduce draws
/// within an STL implementation; uniform_int_distribution is not a cross-STL bitstream contract.
/// Concurrent calls are safe only if the caller's callback is safe to call concurrently.
///
/// @par Example
/// @code
/// const dm::stats::UStatistic u(2);
/// const auto fit=u.run({1,2,4},[](const auto& x){return 0.5*(x[0]-x[1])*(x[0]-x[1]);});
/// std::print("{:.6f}\n",fit.estimate); // 2.333333
/// @endcode
///
/// @par References
/// - W. Hoeffding (1948). A class of statistics with asymptotically normal distribution.
///   Annals of Mathematical Statistics 19(3), 293-325. https://doi.org/10.1214/aoms/1177730196
/// - G. Blom (1976). Some properties of incomplete U-statistics. Biometrika 63(3), 573-580.
///   https://doi.org/10.1093/biomet/63.3.573
/// @see analyze(), incomplete_path(), two_point_law()
class UStatistic {
public:
    /// @brief Stores the kernel degree and evaluation settings without evaluating a kernel.
    /// @param degree Kernel tuple size m, at least one.
    /// @param options Complete/incomplete settings, default {0,42}.
    /// @throws std::invalid_argument If degree is zero.
    /// @par Example
    /// See the class example.
    explicit UStatistic(std::size_t degree, UStatisticOptions options = {});

    /// @brief Computes a complete or incomplete U average and the legacy projection estimate.
    /// @param sample Finite observations, length at least degree(); no null filtering or mutation.
    /// @param kernel Finite-valued deterministic symmetric callback on vectors of length degree().
    /// @return Owned result; projection standard_error is not a total uncertainty estimate.
    /// @throws std::invalid_argument For too few or non-finite observations, or non-finite callback values.
    /// @throws std::bad_function_call For an empty callback; other callback exceptions propagate.
    /// @throws std::overflow_error If the complete term count cannot fit size_t.
    /// @par Example
    /// See the class example.
    /// @see analyze(), UStatisticResult
    [[nodiscard]] UStatisticResult run(const std::vector<double>& sample, const SymmetricKernel& kernel) const;

    /// @brief Adds conditional subset variance, coverage and a complete-fit delete-one jackknife.
    ///
    /// With complete enumeration and n>m, U_(-i)=(n*U-m*g_i)/(n-m); pseudo-values are
    /// n*U-(n-1)*U_(-i), and jackknife variance is (n-1)/n sum((U_(-i)-U)^2).
    /// This equals ((n-1)/(n-m))^2 times the legacy projection variance. It is an estimate,
    /// not the exact population variance, and need not be valid for a degenerate kernel.
    /// Uncovered conditional means and unavailable delete-one diagnostics are NaN. Incomplete
    /// Monte Carlo variance estimates only variability from random subset draws with the data fixed.
    /// @param sample Finite sample of size n>=degree(), unchanged.
    /// @param kernel Deterministic symmetric callback, finite on every evaluated tuple.
    /// @return Owned analysis; no kernel is evaluated more often than in run().
    /// @throws std::invalid_argument As run().
    /// @throws std::bad_function_call For an empty callback; other callback exceptions propagate.
    /// @throws std::overflow_error For an unrepresentable complete term count.
    /// @par Example
    /// See UStatisticAnalysis.
    /// @see run(), term_details()
    [[nodiscard]] UStatisticAnalysis analyze(const std::vector<double>& sample, const SymmetricKernel& kernel) const;

    /// @brief Records each evaluated tuple for small worked examples or sampling audits.
    /// @param sample Finite sample, size at least degree().
    /// @param kernel Deterministic symmetric finite-valued callback.
    /// @return Owned table: term (one-based), i1,...,im (zero-based sample indices),
    ///         x1,...,xm (observations), kernel. O(T*m) output storage; positive budgets may repeat tuples.
    /// @throws std::invalid_argument As run().
    /// @throws std::bad_function_call For an empty callback; other callback exceptions propagate.
    /// @throws std::overflow_error For an unrepresentable complete term count.
    ///
    /// @par Example
    /// @code
    /// const auto terms=dm::stats::UStatistic(2).term_details({1,2,4},[](const auto& x){return std::abs(x[0]-x[1]);});
    /// std::print("{}\n",terms.to_md(3));
    /// @endcode
    /// @see analyze()
    [[nodiscard]] dstruct::DataFrame term_details(const std::vector<double>& sample, const SymmetricKernel& kernel) const;

    /// @brief Calibrates conditional Monte Carlo error against a complete reference on a fixed sample.
    ///
    /// Deliberately computes the complete reference, so this diagnostic is for tractable samples.
    /// Every budget uses the same repeated seeds options().seed+r (unsigned wrap), pairing
    /// repetitions across budgets. options().max_terms is ignored. The expected RMSE is
    /// sqrt(complete kernel_variance/B), since draws are independent with replacement across subsets.
    /// @param sample Finite observations, size at least degree().
    /// @param kernel Deterministic symmetric finite-valued callback.
    /// @param budgets Nonempty positive term counts; order is retained.
    /// @param repetitions Positive independent seeded runs per budget, default 256.
    /// @return Owned rows: terms, repetitions, complete_estimate, mean_estimate, bias_to_complete,
    ///         rmse_to_complete, expected_rmse, mean_estimated_mc_variance. Work proportional
    ///         to C(n,m)+repetitions*sum(budgets) kernel calls.
    /// @throws std::invalid_argument For invalid samples, empty/zero budgets, zero repetitions or non-finite values.
    /// @throws std::bad_function_call For an empty callback; other callback exceptions propagate.
    /// @throws std::overflow_error For an unrepresentable complete term count.
    ///
    /// @par Example
    /// @code
    /// const auto path=dm::stats::UStatistic(2).incomplete_path({1,2,4},
    ///     [](const auto& x){return std::abs(x[0]-x[1]);},{10,100},64);
    /// std::print("{}\n",path.to_md(4));
    /// @endcode
    /// @see analyze()
    [[nodiscard]] dstruct::DataFrame incomplete_path(const std::vector<double>& sample, const SymmetricKernel& kernel,
        const std::vector<std::size_t>& budgets, std::size_t repetitions = 256) const;

    /// @brief Exact law of the complete U-statistic for independent draws from {low,high}.
    ///
    /// If K~Binomial(n,p), a tuple's number of high observations given K is hypergeometric.
    /// The method evaluates h on the m+1 possible tuple compositions and uses their weights
    /// to compute U conditional on each K, its mean/variance and known first projection.
    /// This is a finite sum, not simulation; settings max_terms/seed are ignored. O(n*m)
    /// arithmetic and O(n+m) storage plus m+1 kernel calls. Numerical log-combination
    /// arithmetic can lose precision for extremely large n. No growing-degree CLT is asserted.
    /// @param n Sample size, at least degree(), less than SIZE_MAX.
    /// @param low Finite first population value.
    /// @param high Finite second population value, distinct from low (ordering need not increase).
    /// @param p Probability of the second value, finite in [0,1].
    /// @param kernel Deterministic symmetric finite-valued callback.
    /// @return Owned count distribution, population target, sampling moments and first-order variance.
    /// @throws std::invalid_argument For invalid n, population values/probability or non-finite callback values.
    /// @throws std::bad_function_call For an empty callback; other callback exceptions propagate.
    ///
    /// @par Example
    /// See UStatisticLaw.
    /// @see analyze()
    [[nodiscard]] UStatisticLaw two_point_law(std::size_t n, double low, double high, double p,
                                             const SymmetricKernel& kernel) const;

    /// @brief Configured kernel degree.
    /// @return Tuple size supplied at construction.
    [[nodiscard]] std::size_t degree() const noexcept { return degree_; }
    /// @brief Configured enumeration or sampling settings.
    /// @return Borrowed options, valid for the object's lifetime.
    [[nodiscard]] const UStatisticOptions& options() const noexcept { return options_; }

private:
    /// @brief Shared enumeration/sampling engine, optionally recording tuples.
    /// @param sample Finite observations, validated inside the method.
    /// @param kernel Symmetric finite-valued callback.
    /// @param terms Optional output table; null skips tuple recording.
    /// @return Owned analysis with one projection row per observation.
    UStatisticAnalysis evaluate(const std::vector<double>& sample, const SymmetricKernel& kernel,
                                dstruct::DataFrame* terms) const;
    std::size_t degree_; ///< Kernel tuple size.
    UStatisticOptions options_; ///< Enumeration budget and local RNG seed.
};

} // namespace datamunge::stats
