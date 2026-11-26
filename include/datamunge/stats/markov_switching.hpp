#pragma once

/// @file markov_switching.hpp
/// @brief Markov-switching (Hamilton) regression with regime-specific coefficients and variances.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Options for fit_markov_switching().
/// @par Example
/// @code
/// const dm::stats::MarkovSwitchingOptions options{.regimes = 3};
/// std::println("{}", options.switching_variance); // true
/// @endcode
struct MarkovSwitchingOptions {
    std::size_t regimes{2};          ///< Number of regimes k (at least 2).
    bool intercept{true};            ///< Prepend a switching constant to the regressors.
    bool switching_variance{true};   ///< Regime-specific \f$\sigma^2_s\f$ (otherwise one shared variance).
    std::size_t em_iterations{500};  ///< EM iterations before quasi-Newton refinement.
    std::size_t max_iterations{2000}; ///< L-BFGS iteration limit for the refinement.
};

/// @brief A fitted Markov-switching regression.
/// @par Example
/// @code
/// const dm::stats::MarkovSwitchingFit f;
/// std::println("{}", f.expected_durations.size()); // 0
/// @endcode
struct MarkovSwitchingFit {
    linalg::DenseMatrix<double> transition;   ///< k by k, entry (i, j) = \f$P(s_t=j\mid s_{t-1}=i)\f$; rows sum to 1.
    linalg::DenseMatrix<double> coefficients; ///< k by p regression coefficients (row s for regime s; constant first).
    std::vector<double> variances;            ///< \f$\sigma^2_s\f$ per regime (all equal without switching variance).
    std::vector<double> steady_state;         ///< Ergodic regime probabilities, also the initial distribution.
    std::vector<double> expected_durations;   ///< \f$1/(1-p_{ss})\f$ periods.

    /// Parameters in statsmodels' `MarkovRegression` order: transition probabilities
    /// \f$p_{i\to j}\f$ for destinations \f$j<k-1\f$ (destination-major, then source), coefficients
    /// (regressor-major, then regime), then variances.
    std::vector<double> parameters;
    std::vector<double> standard_errors;      ///< From the inverse numerical Hessian of the log likelihood in these parameters.
    double log_likelihood{};                  ///< Maximized log likelihood.
    double aic{};                             ///< \f$-2\ell+2q\f$.
    double bic{};                             ///< \f$-2\ell+q\ln n\f$.
    linalg::DenseMatrix<double> filtered_probabilities; ///< n by k, \f$P(s_t\mid y_{1:t})\f$.
    linalg::DenseMatrix<double> smoothed_probabilities; ///< n by k, \f$P(s_t\mid y_{1:n})\f$ (Kim smoother).
    bool converged{};                         ///< Whether the final gradient was small.
};

/// @brief Fits \f$y_t=x_t^\top\beta_{s_t}+\varepsilon_t\f$, \f$\varepsilon_t\sim N(0,\sigma^2_{s_t})\f$, with
/// \f$s_t\f$ a k-state Markov chain, by maximum likelihood, as statsmodels' `MarkovRegression`.
///
/// The likelihood comes from Hamilton's filter, started from the chain's steady-state
/// distribution. Estimation starts from statsmodels' default values (equal transition
/// probabilities, coefficients interpolated between 0 and OLS, and variances from 1/10 of the OLS
/// residual variance up to all of it). EM iterations follow, then L-BFGS over logit-transformed
/// transition probabilities and log variances. Regime labels are those reached from this start.
/// Mixture likelihoods can have several local maxima, so compare fits from different specifications
/// before trusting one.
/// @param y Response (finite, at least 10 values).
/// @param x n by p matrix of switching regressors (may have zero columns; must not include a constant when `intercept`).
/// @param options Regimes, constant, variance switching and iteration limits.
/// @return Transition matrix, coefficients, variances, standard errors, likelihood and regime probabilities.
/// @throws std::invalid_argument For fewer than two regimes, mismatched sizes, nonfinite data, too
/// few observations for the parameters, or no regressors at all.
/// @par Example
/// @code
/// std::mt19937 g(9); // raw mt19937 output is portable
/// std::vector<double> y;
/// int s = 0;
/// for (int t = 0; t < 400; ++t) {
///     if (g() % 20 == 0)
///         s = 1 - s;
///     y.push_back((s ? 3.0 : 0.0) + 2 * (g() / 4294967296.0 - .5));
/// }
/// const auto f = dm::stats::fit_markov_switching(y, dm::linalg::DenseMatrix<double>(400, 0));
/// std::println("{:.1f}", f.coefficients(1, 0) - f.coefficients(0, 0)); // 3.0
/// @endcode
/// @par References
/// - J. D. Hamilton (1989). "A new approach to the economic analysis of nonstationary time series
///   and the business cycle." Econometrica 57(2), 357-384. doi:10.2307/1912559
/// - C.-J. Kim (1994). "Dynamic linear models with Markov-switching." Journal of Econometrics
///   60(1-2), 1-22. doi:10.1016/0304-4076(94)90036-1
[[nodiscard]] MarkovSwitchingFit fit_markov_switching(const std::vector<double>& y, const linalg::DenseMatrix<double>& x,
                                                      const MarkovSwitchingOptions& options = {});

} // namespace datamunge::stats
