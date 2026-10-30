#pragma once

/// @file time_series.hpp
/// @brief Time-series building blocks: differencing, sample autocorrelation and partial
///        autocorrelation, and the Ljung-Box portmanteau test.

#include <datamunge/stats/hypothesis_test_result.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Lagged differences, like R's `diff(x, lag, differences)`.
///
/// One pass maps \f$x_t\f$ to \f$x_t-x_{t-\text{lag}}\f$, shortening the series by @p lag; the pass
/// is repeated @p differences times.
/// @param x The series.
/// @param lag The lag (1 for ordinary differences, the period for seasonal ones).
/// @param differences How many times to difference.
/// @return The differenced series, @c lag*differences values shorter.
/// @throws std::invalid_argument if @p lag is 0 or the series is too short.
std::vector<double> difference(const std::vector<double>& x, std::size_t lag = 1, std::size_t differences = 1);

/// @brief Sample autocorrelations \f$r_0,\dots,r_K\f$, as R's `acf()` computes them.
///
/// \f$r_k=\sum_{t=1}^{n-k}(x_t-\bar x)(x_{t+k}-\bar x)\big/\sum_{t=1}^{n}(x_t-\bar x)^2\f$, so
/// \f$r_0=1\f$. The common denominator keeps the sequence positive semi-definite.
/// @param x The series (at least 2 values, not constant).
/// @param max_lag Largest lag \f$K\f$ (less than the length of @p x).
/// @return @c max_lag+1 values, starting at lag 0.
/// @throws std::invalid_argument for a series that is too short or constant.
std::vector<double> acf(const std::vector<double>& x, std::size_t max_lag);

/// @brief Sample partial autocorrelations \f$\phi_{11},\dots,\phi_{KK}\f$ by the Durbin-Levinson
///        recursion on acf(), as R's `pacf()`.
///
/// \f$\phi_{kk}\f$ is the last coefficient of the best linear predictor of \f$x_t\f$ from
/// \f$x_{t-1},\dots,x_{t-k}\f$; for an AR(p) process it vanishes beyond lag p.
/// @param x The series.
/// @param max_lag Largest lag \f$K\f$ (at least 1).
/// @return @c max_lag values, starting at lag 1.
/// @throws std::invalid_argument as acf(), or if @p max_lag is 0.
std::vector<double> pacf(const std::vector<double>& x, std::size_t max_lag);

/// @brief Theoretical autocorrelations \f$\rho_0,\dots,\rho_K\f$ of a stationary ARMA process, like
///        R's `ARMAacf()`.
///
/// The process is \f$x_t=\sum_i a_i x_{t-i}+\varepsilon_t+\sum_j b_j\varepsilon_{t-j}\f$. Its
/// \f$\psi\f$-weights solve \f$\psi_j=b_j+\sum_i a_i\psi_{j-i}\f$, and
/// \f$\gamma_k=\sigma^2\sum_j\psi_j\psi_{j+k}\f$; the sum is truncated once the weights are
/// negligible, so the AR part must be stationary.
/// @param ar The \f$a_i\f$ (may be empty).
/// @param ma The \f$b_j\f$ (may be empty).
/// @param max_lag Largest lag \f$K\f$.
/// @return @c max_lag+1 values, starting at \f$\rho_0=1\f$.
/// @throws std::invalid_argument if the \f$\psi\f$-weights do not die out (non-stationary AR part).
std::vector<double> arma_acf(const std::vector<double>& ar, const std::vector<double>& ma, std::size_t max_lag);

/// @brief Ljung-Box test that the first @p lag autocorrelations of @p residuals are all zero.
///
/// \f$Q=n(n+2)\sum_{k=1}^{h}r_k^2/(n-k)\f$ is compared with a chi-squared distribution on
/// \f$h-\f$ @p fitdf degrees of freedom, matching R's `Box.test(type = "Ljung-Box")`.
/// @param residuals The series to test, usually model residuals.
/// @param lag Number of autocorrelations \f$h\f$.
/// @param fitdf Degrees of freedom used by the fitted model (p+q+P+Q for an ARIMA).
/// @return statistic = Q, parameter1 = degrees of freedom, p_value.
/// @throws std::invalid_argument if @p lag is not greater than @p fitdf or is too large.
HypothesisTestResult ljung_box_test(const std::vector<double>& residuals, std::size_t lag, std::size_t fitdf = 0);

}  // namespace datamunge::stats
