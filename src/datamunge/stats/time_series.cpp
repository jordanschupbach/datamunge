/// @file time_series.cpp
/// @brief Differencing, ACF, PACF (Durbin-Levinson) and the Ljung-Box test.

#include <datamunge/stats/time_series.hpp>

#include <datamunge/random/distributions.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::stats {

std::vector<double> difference(const std::vector<double>& x, std::size_t lag, std::size_t differences) {
    if (lag == 0) throw std::invalid_argument("difference: lag must be at least 1");
    if (x.size() < lag * differences + 1) throw std::invalid_argument("difference: series too short");
    std::vector<double> out = x;
    for (std::size_t d = 0; d < differences; ++d) {
        std::vector<double> next(out.size() - lag);
        for (std::size_t t = lag; t < out.size(); ++t) next[t - lag] = out[t] - out[t - lag];
        out = std::move(next);
    }
    return out;
}

std::vector<double> acf(const std::vector<double>& x, std::size_t max_lag) {
    const std::size_t n = x.size();
    if (n < 2 || max_lag >= n) throw std::invalid_argument("acf: need max_lag < length of series");
    double mean = 0.0;
    for (double v : x) mean += v;
    mean /= static_cast<double>(n);
    double c0 = 0.0;
    for (double v : x) c0 += (v - mean) * (v - mean);
    if (c0 == 0.0) throw std::invalid_argument("acf: series is constant");
    std::vector<double> r(max_lag + 1);
    for (std::size_t k = 0; k <= max_lag; ++k) {
        double s = 0.0;
        for (std::size_t t = 0; t + k < n; ++t) s += (x[t] - mean) * (x[t + k] - mean);
        r[k] = s / c0;
    }
    return r;
}

std::vector<double> pacf(const std::vector<double>& x, std::size_t max_lag) {
    if (max_lag == 0) throw std::invalid_argument("pacf: max_lag must be at least 1");
    const std::vector<double> r = acf(x, max_lag);
    std::vector<double> out(max_lag), phi(max_lag + 1, 0.0), prev(max_lag + 1, 0.0);
    double v = 1.0;  // prediction error variance, relative to r_0
    for (std::size_t k = 1; k <= max_lag; ++k) {
        double num = r[k];
        for (std::size_t j = 1; j < k; ++j) num -= prev[j] * r[k - j];
        const double kk = num / v;
        phi[k] = kk;
        for (std::size_t j = 1; j < k; ++j) phi[j] = prev[j] - kk * prev[k - j];
        v *= 1.0 - kk * kk;
        out[k - 1] = kk;
        prev = phi;
    }
    return out;
}

std::vector<double> arma_acf(const std::vector<double>& ar, const std::vector<double>& ma, std::size_t max_lag) {
    // psi-weights until they are negligible, then autocovariances from their products.
    std::vector<double> psi{1.0};
    double              tail = 1.0;
    for (std::size_t j = 1; j < 200000; ++j) {
        double v = j <= ma.size() ? ma[j - 1] : 0.0;
        for (std::size_t i = 1; i <= ar.size() && i <= j; ++i) v += ar[i - 1] * psi[j - i];
        psi.push_back(v);
        tail = 0.99 * tail + std::fabs(v);  // smoothed recent magnitude
        if (j > ma.size() + ar.size() + max_lag && j > 50 && tail < 1e-14) break;
        if (!std::isfinite(v) || std::fabs(v) > 1e12) break;
    }
    if (tail >= 1e-14) throw std::invalid_argument("arma_acf: psi-weights do not die out; AR part not stationary");
    std::vector<double> gamma(max_lag + 1, 0.0);
    for (std::size_t k = 0; k <= max_lag; ++k)
        for (std::size_t j = 0; j + k < psi.size(); ++j) gamma[k] += psi[j] * psi[j + k];
    for (std::size_t k = 1; k <= max_lag; ++k) gamma[k] /= gamma[0];
    gamma[0] = 1.0;
    return gamma;
}

HypothesisTestResult ljung_box_test(const std::vector<double>& residuals, std::size_t lag, std::size_t fitdf) {
    const std::size_t n = residuals.size();
    if (lag <= fitdf) throw std::invalid_argument("ljung_box_test: lag must exceed fitdf");
    const std::vector<double> r = acf(residuals, lag);
    double q = 0.0;
    for (std::size_t k = 1; k <= lag; ++k) q += r[k] * r[k] / static_cast<double>(n - k);
    q *= static_cast<double>(n) * static_cast<double>(n + 2);
    HypothesisTestResult out;
    out.statistic  = q;
    out.parameter1 = static_cast<double>(lag - fitdf);
    out.p_value    = 1.0 - random::chi_squared_cdf(q, out.parameter1);
    return out;
}

}  // namespace datamunge::stats
