#include <datamunge/stats/arima.hpp>

#include <datamunge/optim/differential_evolution.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

namespace {

std::vector<double> lag_difference(const std::vector<double>& x, std::size_t lag) {
    if (x.size() <= lag) return {};
    std::vector<double> out(x.size() - lag);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = x[i + lag] - x[i];
    return out;
}

std::vector<double> convolve(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> out(a.size() + b.size() - 1, 0.0);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) out[i + j] += a[i] * b[j];
    return out;
}

std::vector<double> build_ar_poly(const std::vector<double>& phi) {
    std::vector<double> poly(phi.size() + 1, 0.0);
    poly[0] = 1.0;
    for (std::size_t i = 0; i < phi.size(); ++i) poly[i + 1] = -phi[i];
    return poly;
}

std::vector<double> build_ma_poly(const std::vector<double>& theta) {
    std::vector<double> poly(theta.size() + 1, 0.0);
    poly[0] = 1.0;
    for (std::size_t i = 0; i < theta.size(); ++i) poly[i + 1] = theta[i];
    return poly;
}

std::vector<double> build_seasonal_ar_poly(const std::vector<double>& seasonal_phi, std::size_t s) {
    std::vector<double> poly(seasonal_phi.size() * s + 1, 0.0);
    poly[0] = 1.0;
    for (std::size_t i = 0; i < seasonal_phi.size(); ++i) poly[(i + 1) * s] = -seasonal_phi[i];
    return poly;
}

std::vector<double> build_seasonal_ma_poly(const std::vector<double>& seasonal_theta, std::size_t s) {
    std::vector<double> poly(seasonal_theta.size() * s + 1, 0.0);
    poly[0] = 1.0;
    for (std::size_t i = 0; i < seasonal_theta.size(); ++i) poly[(i + 1) * s] = seasonal_theta[i];
    return poly;
}

// (1 - B^lag)^order, expanded via the binomial theorem with alternating sign.
std::vector<double> build_diff_poly(std::size_t order, std::size_t lag) {
    std::vector<double> poly(order * lag + 1, 0.0);
    poly[0] = 1.0;
    double coeff = 1.0;
    for (std::size_t k = 1; k <= order; ++k) {
        coeff = coeff * static_cast<double>(order - k + 1) / static_cast<double>(k) * -1.0;
        poly[k * lag] = coeff;
    }
    return poly;
}

// Regression-style coefficients c_i such that, for a polynomial poly with poly[0] == 1,
// the equation poly(B) x_t = 0 rearranges to x_t = sum_i c_i * x_{t-i-1}. That is, c_i = -poly[i+1].
std::vector<double> ar_regression_coefficients(const std::vector<double>& poly) {
    std::vector<double> out(poly.empty() ? 0 : poly.size() - 1);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = -poly[i + 1];
    return out;
}

// Regression-style coefficients b_j such that an MA polynomial poly with poly[0] == 1
// contributes sum_j b_j * e_{t-j-1} to the prediction; b_j = poly[j+1].
std::vector<double> ma_regression_coefficients(const std::vector<double>& poly) {
    std::vector<double> out(poly.empty() ? 0 : poly.size() - 1);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = poly[i + 1];
    return out;
}

// Conditional sum-of-squares fit: conditions on the first ar.size() observations of w
// (skipping them from the sum), treats residuals before that point as exactly zero, and
// recursively computes one-step-ahead residuals for the rest. Always fills `residuals` to
// the same length as `w` (zero for the conditioned-on prefix).
double css_fit(const std::vector<double>& w, const std::vector<double>& ar, const std::vector<double>& ma,
               double mean, std::vector<double>& residuals) {
    const std::size_t m = w.size();
    const std::size_t P = ar.size();
    const std::size_t Q = ma.size();
    residuals.assign(m, 0.0);
    double sse = 0.0;
    for (std::size_t t = P; t < m; ++t) {
        double pred = 0.0;
        for (std::size_t i = 0; i < P; ++i) pred += ar[i] * (w[t - 1 - i] - mean);
        for (std::size_t j = 0; j < Q; ++j) {
            if (j + 1 > t) continue; // no residual that far back yet; treated as zero
            pred += ma[j] * residuals[t - 1 - j];
        }
        const double e = (w[t] - mean) - pred;
        residuals[t] = e;
        sse += e * e;
    }
    return sse;
}

class ARIMAObjective : public optim::ArbitraryFunction {
 public:
    ARIMAObjective(const std::vector<double>& w, std::size_t p, std::size_t seasonal_p, std::size_t q,
                    std::size_t seasonal_q, std::size_t s, bool mean_free)
        : w_(w), p_(p), seasonal_p_(seasonal_p), q_(q), seasonal_q_(seasonal_q), s_(s), mean_free_(mean_free) {}

    double evaluate(const std::vector<double>& params) override {
        std::size_t offset = 0;
        const std::vector<double> phi(params.begin() + static_cast<long>(offset), params.begin() + static_cast<long>(offset + p_));
        offset += p_;
        const std::vector<double> seasonal_phi(params.begin() + static_cast<long>(offset),
                                                params.begin() + static_cast<long>(offset + seasonal_p_));
        offset += seasonal_p_;
        const std::vector<double> theta(params.begin() + static_cast<long>(offset), params.begin() + static_cast<long>(offset + q_));
        offset += q_;
        const std::vector<double> seasonal_theta(params.begin() + static_cast<long>(offset),
                                                   params.begin() + static_cast<long>(offset + seasonal_q_));
        offset += seasonal_q_;
        const double mean = mean_free_ ? params[offset] : 0.0;

        const auto ar_poly = convolve(build_ar_poly(phi), build_seasonal_ar_poly(seasonal_phi, s_));
        const auto ma_poly = convolve(build_ma_poly(theta), build_seasonal_ma_poly(seasonal_theta, s_));
        const auto ar = ar_regression_coefficients(ar_poly);
        const auto ma = ma_regression_coefficients(ma_poly);

        std::vector<double> residuals;
        const double sse = css_fit(w_, ar, ma, mean, residuals);
        if (!std::isfinite(sse)) return std::numeric_limits<double>::max() / 4.0;
        return sse;
    }

 private:
    const std::vector<double>& w_;
    std::size_t p_, seasonal_p_, q_, seasonal_q_, s_;
    bool mean_free_;
};

std::vector<double> integrate_lag(const std::vector<double>& future_diffs, const std::vector<double>& tail,
                                   std::size_t lag) {
    std::vector<double> result(future_diffs.size());
    for (std::size_t i = 0; i < future_diffs.size(); ++i) {
        const double prev = (i < lag) ? tail[tail.size() - lag + i] : result[i - lag];
        result[i] = future_diffs[i] + prev;
    }
    return result;
}

} // namespace

ARIMA::ARIMA(const std::vector<double>& y, ARIMAOptions options) : options_(options) { fit(y); }

void ARIMA::fit(const std::vector<double>& y) {
    n_ = y.size();
    const auto& o = options_;
    if ((o.seasonal_p || o.seasonal_d || o.seasonal_q) && o.seasonal_period == 0)
        throw std::invalid_argument("ARIMA: seasonal_period must be nonzero when a seasonal order is given");

    std::vector<double> level = y;
    for (std::size_t k = 0; k < o.d; ++k) {
        if (level.empty()) throw std::invalid_argument("ARIMA: series too short for the requested differencing order");
        regular_diff_tail_.push_back({level.back()});
        level = lag_difference(level, 1);
    }
    for (std::size_t k = 0; k < o.seasonal_d; ++k) {
        if (level.size() < o.seasonal_period)
            throw std::invalid_argument("ARIMA: series too short for the requested seasonal differencing order");
        seasonal_diff_tail_.push_back(
            std::vector<double>(level.end() - static_cast<long>(o.seasonal_period), level.end()));
        level = lag_difference(level, o.seasonal_period);
    }
    w_ = level;

    const std::size_t ar_order = o.p + o.seasonal_p * o.seasonal_period;
    const std::size_t ma_order = o.q + o.seasonal_q * o.seasonal_period;
    if (w_.size() <= ar_order)
        throw std::invalid_argument("ARIMA: not enough observations (after differencing) for the requested AR order");

    const bool mean_free = o.include_mean && o.d == 0 && o.seasonal_d == 0;
    const std::size_t n_params = o.p + o.seasonal_p + o.q + o.seasonal_q + (mean_free ? 1 : 0);

    std::vector<double> best_params(n_params, 0.0);
    if (mean_free) {
        double w_mean = 0.0;
        for (const double v : w_) w_mean += v;
        w_mean /= static_cast<double>(w_.size());
        best_params.back() = w_mean;
    }

    if (n_params > 0) {
        std::vector<double> lower(n_params), upper(n_params);
        for (std::size_t i = 0; i < n_params - (mean_free ? 1 : 0); ++i) {
            lower[i] = -o.coefficient_bound;
            upper[i] = o.coefficient_bound;
        }
        if (mean_free) {
            double w_mean = 0.0, w_var = 0.0;
            for (const double v : w_) w_mean += v;
            w_mean /= static_cast<double>(w_.size());
            for (const double v : w_) w_var += (v - w_mean) * (v - w_mean);
            w_var /= static_cast<double>(w_.size());
            const double spread = 4.0 * std::max(std::sqrt(w_var), 1e-6);
            lower.back() = w_mean - spread;
            upper.back() = w_mean + spread;
        }

        ARIMAObjective objective(w_, o.p, o.seasonal_p, o.q, o.seasonal_q, o.seasonal_period, mean_free);
        optim::DEOptions de_options;
        de_options.population_size = o.de_population_size;
        de_options.max_generations = o.de_max_generations;
        de_options.seed = o.seed;
        optim::DifferentialEvolution de(de_options);
        de.optimize(objective, best_params, lower, upper);
    }

    std::size_t offset = 0;
    phi_.assign(best_params.begin() + static_cast<long>(offset), best_params.begin() + static_cast<long>(offset + o.p));
    offset += o.p;
    seasonal_phi_.assign(best_params.begin() + static_cast<long>(offset), best_params.begin() + static_cast<long>(offset + o.seasonal_p));
    offset += o.seasonal_p;
    theta_.assign(best_params.begin() + static_cast<long>(offset), best_params.begin() + static_cast<long>(offset + o.q));
    offset += o.q;
    seasonal_theta_.assign(best_params.begin() + static_cast<long>(offset), best_params.begin() + static_cast<long>(offset + o.seasonal_q));
    offset += o.seasonal_q;
    mean_ = mean_free ? best_params[offset] : 0.0;

    const auto ar_poly = convolve(build_ar_poly(phi_), build_seasonal_ar_poly(seasonal_phi_, o.seasonal_period));
    const auto ma_poly = convolve(build_ma_poly(theta_), build_seasonal_ma_poly(seasonal_theta_, o.seasonal_period));
    combined_ar_ = ar_regression_coefficients(ar_poly);
    combined_ma_ = ma_regression_coefficients(ma_poly);

    const double sse = css_fit(w_, combined_ar_, combined_ma_, mean_, w_residuals_);
    n_used_ = w_.size() - ar_order;
    sigma2_ = sse / static_cast<double>(n_used_);
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    log_likelihood_ = -0.5 * static_cast<double>(n_used_) * (std::log(kTwoPi) + std::log(sigma2_) + 1.0);
    const auto k_params = static_cast<double>(n_params + 1); // + sigma2
    aic_ = -2.0 * log_likelihood_ + 2.0 * k_params;
    bic_ = -2.0 * log_likelihood_ + std::log(static_cast<double>(n_used_)) * k_params;

    const auto diff_poly_d = build_diff_poly(o.d, 1);
    const auto seasonal_diff_poly_D = build_diff_poly(o.seasonal_d, o.seasonal_period == 0 ? 1 : o.seasonal_period);
    const auto full_ar_poly = convolve(convolve(ar_poly, diff_poly_d), seasonal_diff_poly_D);
    full_ar_regression_ = ar_regression_coefficients(full_ar_poly);
    full_ma_regression_ = combined_ma_;

    const std::size_t lead = o.d + o.seasonal_d * o.seasonal_period;
    fitted_.assign(n_, 0.0);
    residuals_.assign(n_, 0.0);
    for (std::size_t t = 0; t < n_; ++t) {
        if (t < lead + ar_order) {
            fitted_[t] = y[t];
            residuals_[t] = 0.0;
        } else {
            const double e = w_residuals_[t - lead];
            residuals_[t] = e;
            fitted_[t] = y[t] - e;
        }
    }
}

std::vector<double> ARIMA::forecast(std::size_t horizon) const {
    if (horizon == 0) return {};
    const std::size_t m = w_.size();
    const std::size_t P = combined_ar_.size();
    const std::size_t Q = combined_ma_.size();

    std::vector<double> w_forecast(horizon);
    for (std::size_t k = 0; k < horizon; ++k) {
        double pred = 0.0;
        for (std::size_t i = 0; i < P; ++i) {
            const std::size_t back = i + 1;
            double wv;
            if (back <= k) {
                wv = w_forecast[k - back] - mean_;
            } else if (back - k <= m) {
                wv = w_[m - (back - k)] - mean_;
            } else {
                wv = 0.0;
            }
            pred += combined_ar_[i] * wv;
        }
        for (std::size_t j = 0; j < Q; ++j) {
            const std::size_t back = j + 1;
            if (back <= k) continue; // future innovations are zero in expectation
            if (back - k <= m) pred += combined_ma_[j] * w_residuals_[m - (back - k)];
        }
        w_forecast[k] = pred + mean_;
    }

    std::vector<double> current = w_forecast;
    for (std::size_t step = options_.seasonal_d; step-- > 0;)
        current = integrate_lag(current, seasonal_diff_tail_[step], options_.seasonal_period);
    for (std::size_t step = options_.d; step-- > 0;) current = integrate_lag(current, regular_diff_tail_[step], 1);
    return current;
}

std::pair<std::vector<double>, std::vector<double>> ARIMA::forecast_with_intervals(std::size_t horizon) const {
    const auto point = forecast(horizon);

    std::vector<double> psi(horizon, 0.0);
    if (horizon > 0) psi[0] = 1.0;
    for (std::size_t j = 1; j < horizon; ++j) {
        double val = 0.0;
        for (std::size_t i = 0; i < full_ar_regression_.size() && i < j; ++i) val += full_ar_regression_[i] * psi[j - 1 - i];
        if (j - 1 < full_ma_regression_.size()) val += full_ma_regression_[j - 1];
        psi[j] = val;
    }

    std::vector<double> se(horizon);
    double cumulative = 0.0;
    for (std::size_t h = 0; h < horizon; ++h) {
        cumulative += psi[h] * psi[h];
        se[h] = std::sqrt(sigma2_ * cumulative);
    }
    return {point, se};
}

} // namespace datamunge::stats
