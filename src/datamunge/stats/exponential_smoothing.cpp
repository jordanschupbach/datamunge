#include <datamunge/stats/exponential_smoothing.hpp>

#include <datamunge/optim/differential_evolution.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::stats {

namespace {

struct State {
    double level{0.0};
    double trend{0.0};
    std::vector<double> season; // length seasonal_period; index by (t % seasonal_period)
};

State initial_state(const ExponentialSmoothingOptions& o, const std::vector<double>& y) {
    State s;
    if (o.seasonal != SeasonalType::None) {
        const std::size_t m = o.seasonal_period;
        double mean1 = 0.0;
        for (std::size_t i = 0; i < m; ++i) mean1 += y[i];
        mean1 /= static_cast<double>(m);
        s.level = mean1;
        if (o.trend != TrendType::None) {
            double mean2 = 0.0;
            for (std::size_t i = m; i < 2 * m; ++i) mean2 += y[i];
            mean2 /= static_cast<double>(m);
            s.trend = (mean2 - mean1) / static_cast<double>(m);
        }
        s.season.resize(m);
        double norm = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            s.season[i] = (o.seasonal == SeasonalType::Additive) ? (y[i] - mean1) : (y[i] / mean1);
            norm += s.season[i];
        }
        norm /= static_cast<double>(m);
        for (std::size_t i = 0; i < m; ++i) {
            if (o.seasonal == SeasonalType::Additive) s.season[i] -= norm;
            else s.season[i] /= norm;
        }
    } else {
        s.level = y[0];
        if (o.trend != TrendType::None) s.trend = (y.size() > 1) ? (y[1] - y[0]) : 0.0;
    }
    return s;
}

// Runs the smoothing recursion over y starting from `state`, scoring indices
// [warmup, y.size()); returns the sum of squared one-step-ahead errors. Any of the output
// pointers may be null when the caller only needs the SSE (e.g. during parameter search).
double run(const ExponentialSmoothingOptions& o, const std::vector<double>& y, std::size_t warmup, State state,
           double a, double b, double g, double p, std::vector<double>* out_fitted, std::vector<double>* out_residuals,
           State* out_final_state) {
    const std::size_t m = o.seasonal_period;
    double sse = 0.0;
    for (std::size_t t = warmup; t < y.size(); ++t) {
        const std::size_t phase = (m > 0) ? (t % m) : 0;
        const double trend_component =
            (o.trend == TrendType::None) ? 0.0 : (o.trend == TrendType::AdditiveDamped ? p * state.trend : state.trend);
        const double level_plus_trend = state.level + trend_component;

        double fitted_t;
        if (o.seasonal == SeasonalType::None) fitted_t = level_plus_trend;
        else if (o.seasonal == SeasonalType::Additive) fitted_t = level_plus_trend + state.season[phase];
        else fitted_t = level_plus_trend * state.season[phase];

        const double residual_t = y[t] - fitted_t;
        sse += residual_t * residual_t;
        if (out_fitted) (*out_fitted)[t] = fitted_t;
        if (out_residuals) (*out_residuals)[t] = residual_t;

        double new_level;
        if (o.seasonal == SeasonalType::None) new_level = a * y[t] + (1.0 - a) * level_plus_trend;
        else if (o.seasonal == SeasonalType::Additive)
            new_level = a * (y[t] - state.season[phase]) + (1.0 - a) * level_plus_trend;
        else new_level = a * (y[t] / state.season[phase]) + (1.0 - a) * level_plus_trend;

        double new_trend = state.trend;
        if (o.trend == TrendType::Additive) new_trend = b * (new_level - state.level) + (1.0 - b) * state.trend;
        else if (o.trend == TrendType::AdditiveDamped)
            new_trend = b * (new_level - state.level) + (1.0 - b) * p * state.trend;

        if (o.seasonal == SeasonalType::Additive) state.season[phase] = g * (y[t] - new_level) + (1.0 - g) * state.season[phase];
        else if (o.seasonal == SeasonalType::Multiplicative)
            state.season[phase] = g * (y[t] / new_level) + (1.0 - g) * state.season[phase];

        state.level = new_level;
        state.trend = new_trend;
    }
    if (out_final_state) *out_final_state = state;
    return sse;
}

class ExpSmoothingObjective : public optim::ArbitraryFunction {
 public:
    ExpSmoothingObjective(const ExponentialSmoothingOptions& options, const std::vector<double>& y,
                           std::size_t warmup, State init_state)
        : options_(options), y_(y), warmup_(warmup), init_state_(std::move(init_state)) {}

    double evaluate(const std::vector<double>& params) override {
        std::size_t idx = 0;
        const double a = options_.alpha >= 0.0 ? options_.alpha : params[idx++];
        const double b =
            (options_.trend != TrendType::None) ? (options_.beta >= 0.0 ? options_.beta : params[idx++]) : 0.0;
        const double g = (options_.seasonal != SeasonalType::None)
                              ? (options_.gamma >= 0.0 ? options_.gamma : params[idx++])
                              : 0.0;
        const double p = (options_.trend == TrendType::AdditiveDamped)
                              ? (options_.phi >= 0.0 ? options_.phi : params[idx++])
                              : 1.0;
        return run(options_, y_, warmup_, init_state_, a, b, g, p, nullptr, nullptr, nullptr);
    }

 private:
    const ExponentialSmoothingOptions& options_;
    const std::vector<double>& y_;
    std::size_t warmup_;
    State init_state_;
};

} // namespace

ExponentialSmoothing::ExponentialSmoothing(const std::vector<double>& y, ExponentialSmoothingOptions options)
    : options_(options) {
    fit(y);
}

void ExponentialSmoothing::fit(const std::vector<double>& y) {
    n_ = y.size();
    const auto& o = options_;
    if (o.seasonal != SeasonalType::None && o.seasonal_period < 2)
        throw std::invalid_argument("ExponentialSmoothing: seasonal_period must be at least 2 when seasonal smoothing is enabled");

    warmup_ = (o.seasonal != SeasonalType::None) ? o.seasonal_period : (o.trend != TrendType::None ? 2 : 1);
    const std::size_t min_required = (o.seasonal != SeasonalType::None) ? 2 * o.seasonal_period : warmup_;
    if (n_ < min_required || n_ <= warmup_)
        throw std::invalid_argument("ExponentialSmoothing: series too short for the requested trend/seasonal configuration");

    const State init = initial_state(o, y);

    std::size_t n_free = 0;
    if (o.alpha < 0.0) ++n_free;
    if (o.trend != TrendType::None && o.beta < 0.0) ++n_free;
    if (o.seasonal != SeasonalType::None && o.gamma < 0.0) ++n_free;
    if (o.trend == TrendType::AdditiveDamped && o.phi < 0.0) ++n_free;

    std::vector<double> best_params(n_free, 0.3);
    if (n_free > 0) {
        std::vector<double> lower(n_free, 1e-4), upper(n_free, 1.0 - 1e-4);
        ExpSmoothingObjective objective(o, y, warmup_, init);
        optim::DEOptions de_options;
        de_options.population_size = o.de_population_size;
        de_options.max_generations = o.de_max_generations;
        de_options.seed = o.seed;
        optim::DifferentialEvolution de(de_options);
        de.optimize(objective, best_params, lower, upper);
    }

    std::size_t idx = 0;
    alpha_ = o.alpha >= 0.0 ? o.alpha : best_params[idx++];
    beta_ = (o.trend != TrendType::None) ? (o.beta >= 0.0 ? o.beta : best_params[idx++]) : 0.0;
    gamma_ = (o.seasonal != SeasonalType::None) ? (o.gamma >= 0.0 ? o.gamma : best_params[idx++]) : 0.0;
    phi_ = (o.trend == TrendType::AdditiveDamped) ? (o.phi >= 0.0 ? o.phi : best_params[idx++]) : 1.0;

    fitted_.assign(n_, 0.0);
    residuals_.assign(n_, 0.0);
    for (std::size_t t = 0; t < warmup_; ++t) fitted_[t] = y[t]; // residual left at 0

    State final_state;
    sse_ = run(o, y, warmup_, init, alpha_, beta_, gamma_, phi_, &fitted_, &residuals_, &final_state);

    final_level_ = final_state.level;
    final_trend_ = final_state.trend;
    final_season_ = final_state.season;

    const std::size_t n_used = n_ - warmup_;
    sigma2_ = sse_ / static_cast<double>(n_used);
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    log_likelihood_ = -0.5 * static_cast<double>(n_used) * (std::log(kTwoPi) + std::log(sigma2_) + 1.0);
    const auto k_params = static_cast<double>(n_free + 1); // + sigma2
    aic_ = -2.0 * log_likelihood_ + 2.0 * k_params;
    bic_ = -2.0 * log_likelihood_ + std::log(static_cast<double>(n_used)) * k_params;
}

std::vector<double> ExponentialSmoothing::forecast(std::size_t horizon) const {
    const auto& o = options_;
    const std::size_t m = o.seasonal_period;
    std::vector<double> result(horizon);
    // The damped trend component at lead time h is phi + phi^2 + ... + phi^h (h * phi when
    // phi == 1, the undamped case).
    double damped_cumulative = 0.0;
    double phi_power = 1.0;
    for (std::size_t h = 1; h <= horizon; ++h) {
        phi_power *= phi_;
        damped_cumulative += phi_power;
        const double trend_component = (o.trend == TrendType::None) ? 0.0
                                        : (o.trend == TrendType::AdditiveDamped) ? damped_cumulative * final_trend_
                                                                                  : static_cast<double>(h) * final_trend_;
        const double level_plus_trend = final_level_ + trend_component;
        if (o.seasonal == SeasonalType::None) {
            result[h - 1] = level_plus_trend;
        } else {
            const std::size_t phase = (n_ + h - 1) % m;
            result[h - 1] = (o.seasonal == SeasonalType::Additive) ? (level_plus_trend + final_season_[phase])
                                                                     : (level_plus_trend * final_season_[phase]);
        }
    }
    return result;
}

} // namespace datamunge::stats
