#pragma once

/// @file hawkes.hpp
/// @brief Self-exciting (Hawkes) point processes on the line with an exponential kernel.

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace datamunge::stats {

/// @brief Parameters of \f$\lambda(t)=\mu+\sum_{t_i<t}\alpha e^{-\beta(t-t_i)}\f$.
/// @par Example
/// @code
/// const dm::stats::HawkesParameters p{.mu = 0.5, .alpha = 0.8, .beta = 1.5};
/// std::println("{:.4f}", p.alpha / p.beta); // 0.5333
/// @endcode
struct HawkesParameters {
    double mu{1};    ///< Baseline (immigrant) rate.
    double alpha{0.5}; ///< Jump in intensity after each event.
    double beta{1};  ///< Exponential decay rate; alpha / beta is the branching ratio.
};

/// @brief Log-likelihood of event times on [0, T] for a process that starts empty at time 0.
///
/// \f$\ell=\sum_k\log\lambda(t_k)-\mu T-\frac\alpha\beta\sum_k(1-e^{-\beta(T-t_k)})\f$, computed in O(n) by
/// the recursion \f$A_k=e^{-\beta(t_k-t_{k-1})}(1+A_{k-1})\f$. Matches `emhawkes::logLik` with
/// `lambda_component0 = 0` (whose first inter-arrival time marks the origin).
/// @param times Strictly increasing event times in (0, T].
/// @param params Parameters (all positive).
/// @param end_time Observation end T (NaN uses the last event time).
/// @return The log-likelihood.
/// @throws std::invalid_argument For nonincreasing or nonpositive times, T before the last event, or
/// nonpositive parameters.
/// @par Example
/// @code
/// const double l = dm::stats::hawkes_log_likelihood({1.0}, {.mu = 1, .alpha = 0.5, .beta = 1}, 2.0);
/// std::println("{:.4f}", l); // -2.3161
/// @endcode
double hawkes_log_likelihood(const std::vector<double>& times, const HawkesParameters& params,
                             double end_time = std::numeric_limits<double>::quiet_NaN());

/// @brief Options for hawkes_fit().
/// @par Example
/// @code
/// const dm::stats::HawkesFitOptions o{.end_time = 100};
/// std::println("{}", o.max_iterations); // 200
/// @endcode
struct HawkesFitOptions {
    double end_time{std::numeric_limits<double>::quiet_NaN()}; ///< Observation end (NaN: last event).
    std::size_t max_iterations{200}; ///< Newton iterations per start.
};

/// @brief Maximum-likelihood fit of an exponential Hawkes process.
/// @par Example
/// @code
/// dm::stats::HawkesFit f;
/// std::println("{}", f.converged); // false
/// @endcode
struct HawkesFit {
    HawkesParameters parameters;            ///< MLE of (mu, alpha, beta).
    std::array<double, 3> standard_errors{}; ///< From the inverse observed information (mu, alpha, beta).
    double log_likelihood{};                ///< Maximised log-likelihood.
    double branching_ratio{};               ///< alpha / beta: expected offspring per event (< 1 for stationarity).
    double aic{};                           ///< -2 log_likelihood + 6.
    std::size_t iterations{};               ///< Newton iterations of the best start.
    bool converged{false};                  ///< Whether the gradient vanished at the reported optimum.
};

/// @brief Maximum-likelihood estimation of an exponential-kernel Hawkes process.
///
/// Maximises hawkes_log_likelihood() over log parameters by damped Newton steps with the analytic
/// gradient and a finite-difference Hessian, from several starting values (branching ratios 0.2, 0.5
/// and 0.8 and decay rates around n / T), keeping the best. Standard errors are from the observed
/// information on the natural scale. The branching ratio is not constrained; a value at or above 1
/// indicates a nonstationary (explosive) fit. Matches `emhawkes::hfit(lambda_component0 = 0)`.
/// @param times Strictly increasing event times in (0, T].
/// @param options End time and iteration limit.
/// @return The fit.
/// @throws std::invalid_argument For fewer than three events or invalid times.
/// @par Example
/// @code
/// const auto t = dm::stats::hawkes_simulate({.mu = 0.5, .alpha = 1, .beta = 2}, 2000, 7);
/// const auto f = dm::stats::hawkes_fit(t, {.end_time = 2000});
/// std::println("{}", std::abs(f.branching_ratio - 0.5) < 0.1); // true
/// @endcode
HawkesFit hawkes_fit(const std::vector<double>& times, const HawkesFitOptions& options = {});

/// @brief Simulates an exponential Hawkes process on [0, T] by Ogata's thinning, starting empty.
/// @param params Parameters (all positive).
/// @param end_time Horizon T.
/// @param seed Random seed.
/// @return Event times in increasing order.
/// @throws std::invalid_argument For nonpositive parameters or horizon.
/// @par Example
/// @code
/// const auto t = dm::stats::hawkes_simulate({.mu = 1, .alpha = 0.5, .beta = 1}, 100, 1);
/// std::println("{}", std::is_sorted(t.begin(), t.end())); // true
/// @endcode
std::vector<double> hawkes_simulate(const HawkesParameters& params, double end_time, std::uint64_t seed);

/// @brief Time-rescaling residuals \f$\Lambda(t_k)-\Lambda(t_{k-1})\f$ (with \f$t_0=0\f$).
///
/// Under the fitted model these are independent standard exponentials (Ogata 1988), so a QQ plot or
/// a Kolmogorov-Smirnov test of them checks goodness of fit.
/// @param times Strictly increasing event times.
/// @param params Parameters.
/// @return One compensator increment per event.
/// @throws std::invalid_argument As for hawkes_log_likelihood().
/// @par Example
/// @code
/// const auto r = dm::stats::hawkes_residuals({1.0, 2.0}, {.mu = 1, .alpha = 0.5, .beta = 1});
/// std::println("{:.4f} {:.4f}", r[0], r[1]); // 1.0000 1.3161
/// @endcode
std::vector<double> hawkes_residuals(const std::vector<double>& times, const HawkesParameters& params);

} // namespace datamunge::stats
