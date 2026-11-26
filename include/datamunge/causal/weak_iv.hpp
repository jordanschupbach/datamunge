#pragma once

/// @file weak_iv.hpp
/// @brief Weak-instrument-robust inference for one endogenous regressor: the Anderson-Rubin test
/// and its inverted confidence set, alongside 2SLS and LIML.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::causal {

/// @brief Shape of a confidence set from an inverted test.
enum class ConfidenceSetShape {
    interval,   ///< A bounded interval [lower, upper].
    two_rays,   ///< The union (-inf, a] and [b, inf) with a < b.
    whole_line, ///< Every value of the coefficient.
    empty,      ///< No value (possible with more than one instrument).
};

/// @brief A confidence set as a union of closed pieces, with infinite ends where unbounded.
/// @par Example
/// @code
/// const dm::causal::ConfidenceSet s{dm::causal::ConfidenceSetShape::interval, {{0.1, 0.4}}};
/// std::println("{} {}", s.contains(0.2), s.contains(0.5)); // true false
/// @endcode
struct ConfidenceSet {
    ConfidenceSetShape shape{ConfidenceSetShape::interval}; ///< Shape.
    std::vector<std::pair<double, double>> pieces;          ///< Disjoint closed pieces in order.

    /// @brief Whether a coefficient value lies in the set.
    /// @param value Coefficient value.
    /// @return True when some piece contains it.
    [[nodiscard]] bool contains(double value) const {
        for (const auto& [lo, hi] : pieces)
            if (lo <= value && value <= hi)
                return true;
        return false;
    }
};

/// @brief A k-class estimate (2SLS has k = 1, LIML the smallest root) with homoskedastic inference.
/// @par Example
/// @code
/// const dm::causal::KClassEstimate e{.estimate = 0.13, .standard_error = 0.05};
/// std::println("{:.1f}", e.estimate / e.standard_error); // 2.6
/// @endcode
struct KClassEstimate {
    double estimate{};       ///< Coefficient of the endogenous regressor.
    double standard_error{}; ///< Conventional standard error, sigma^2 = RSS / (n - p - 1).
    double lower{};          ///< Lower t(n - p - 1) confidence limit.
    double upper{};          ///< Upper t(n - p - 1) confidence limit.
    double k{};              ///< The k-class parameter.
};

/// @brief Options for weak_iv_inference().
/// @par Example
/// @code
/// const dm::causal::WeakIvOptions o{.null_value = 0.1, .alpha = 0.1};
/// std::println("{}", o.intercept); // true
/// @endcode
struct WeakIvOptions {
    double null_value{0.0}; ///< Coefficient value tested by the Anderson-Rubin test.
    double alpha{0.05};     ///< Size of the test; sets have level 1 - alpha.
    bool intercept{true};   ///< Add a column of ones to the exogenous controls.
};

/// @brief Result of weak_iv_inference().
/// @par Example
/// @code
/// dm::causal::WeakIvInference r;
/// r.first_stage_f = 3.2;
/// std::println("{}", r.first_stage_f < 10); // true
/// @endcode
struct WeakIvInference {
    KClassEstimate tsls;          ///< Two-stage least squares.
    KClassEstimate liml;          ///< Limited-information maximum likelihood.
    double first_stage_f{};       ///< F statistic of the excluded instruments in the first stage.
    double ar_statistic{};        ///< Anderson-Rubin F statistic at the null value.
    double ar_p_value{};          ///< Its p-value from F(L, n - L - p).
    std::size_t df1{};            ///< L, the number of (linearly independent) instruments.
    std::size_t df2{};            ///< n - L - p.
    ConfidenceSet ar_set;         ///< Values not rejected by the AR test at level alpha.
};

/// @brief Anderson-Rubin test and confidence set, with 2SLS and LIML, for one endogenous regressor.
///
/// After the exogenous controls (and intercept) are partialled out, the Anderson-Rubin statistic
/// for \f$H_0:\beta=\beta_0\f$ is the F statistic of a regression of \f$y-\beta_0 d\f$ on the
/// instruments,
/// \f[AR(\beta_0)=\frac{(y-\beta_0d)^\top P_Z(y-\beta_0d)/L}{(y-\beta_0d)^\top M_Z(y-\beta_0d)/(n-L-p)},\f]
/// whose null distribution does not depend on instrument strength. Inverting it gives the set
/// of \f$\beta\f$ where a quadratic in \f$\beta\f$ is nonnegative: a bounded interval, two rays,
/// the whole line (instruments uninformative) or, when overidentifying restrictions are rejected,
/// the empty set. Results match R's `ivmodel` (`AR.test`, `KClass`, `LIML`).
/// @param outcome Outcome y.
/// @param endogenous Endogenous regressor d.
/// @param instruments n-by-L excluded instruments.
/// @param controls n-by-p exogenous controls, or an empty matrix.
/// @param options Null value, size and intercept.
/// @return Estimates, first-stage F, the AR test and the AR confidence set.
/// @throws std::invalid_argument For mismatched sizes, nonfinite values, alpha outside (0, 1), or
/// too few observations.
/// @throws std::runtime_error For collinear controls or instruments.
/// @par Example
/// @code
/// std::vector<double> y, d;
/// dm::linalg::DenseMatrix<double> z(200, 1);
/// for (int i = 0; i < 200; ++i) {
///     z(i, 0) = std::sin(1.3 * i);
///     const double u = std::cos(2.1 * i);
///     d.push_back(z(i, 0) + u);
///     y.push_back(2 * d.back() + u);
/// }
/// const auto r = dm::causal::weak_iv_inference(y, d, z, {}, {.null_value = 2});
/// std::println("{}", r.ar_set.contains(2.0)); // true
/// @endcode
WeakIvInference weak_iv_inference(const std::vector<double>& outcome, const std::vector<double>& endogenous,
                                  const linalg::DenseMatrix<double>& instruments,
                                  const linalg::DenseMatrix<double>& controls = {},
                                  const WeakIvOptions& options = {});

} // namespace datamunge::causal
