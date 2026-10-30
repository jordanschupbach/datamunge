/// @file bayesian_optimization.cpp
/// @brief The BayesianOptimization loop: random initial design, surrogate refits, random
///        candidates polished by compass search on the acquisition surface.

#include <datamunge/optim/bayesian_optimization.hpp>

#include <algorithm>
#include <random>
#include <stdexcept>

namespace datamunge::optim {

BayesianOptimization::BayesianOptimization(BayesianOptimizationOptions options) : options_(options) {}

BayesianOptimizationResult BayesianOptimization::run(ArbitraryFunction& function, const std::vector<double>& lower,
                                                     const std::vector<double>& upper,
                                                     BayesianSurrogate& surrogate) const {
    const std::size_t d = lower.size();
    if (d == 0 || upper.size() != d) throw std::invalid_argument("BayesianOptimization: bounds must be non-empty and match");
    for (std::size_t j = 0; j < d; ++j)
        if (!(lower[j] < upper[j])) throw std::invalid_argument("BayesianOptimization: invalid bounds");

    std::mt19937_64                        rng(options_.seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    auto draw = [&] {
        std::vector<double> q(d);
        for (std::size_t j = 0; j < d; ++j) q[j] = lower[j] + unit(rng) * (upper[j] - lower[j]);
        return q;
    };

    BayesianOptimizationResult r;
    auto record = [&](std::vector<double> q) {
        const double v = function.evaluate(q);
        r.points.push_back(std::move(q));
        r.values.push_back(v);
        r.best_values.push_back(r.best_values.empty() ? v : std::min(r.best_values.back(), v));
    };
    for (std::size_t i = 0; i < std::max<std::size_t>(1, options_.initial_samples); ++i) record(draw());

    for (std::size_t it = 0; it < options_.max_iterations; ++it) {
        surrogate.fit(r.points, r.values);
        const double        incumbent = r.best_values.back();
        std::vector<double> best_q;
        double              best_score = 0.0;
        for (std::size_t k = 0; k < std::max<std::size_t>(1, options_.candidates); ++k) {
            auto         q = draw();
            const double a = surrogate.acquisition(q, incumbent);
            if (best_q.empty() || a > best_score) best_score = a, best_q = std::move(q);
        }
        // Compass search on the acquisition, starting from the best candidate.
        std::vector<double> step(d);
        for (std::size_t j = 0; j < d; ++j) step[j] = 0.1 * (upper[j] - lower[j]);
        for (std::size_t h = 0; h < options_.refine_iterations; ++h) {
            bool moved = true;
            while (moved) {
                moved = false;
                for (std::size_t j = 0; j < d && !moved; ++j)
                    for (const double sign : {1.0, -1.0}) {
                        auto q = best_q;
                        q[j]   = std::clamp(q[j] + sign * step[j], lower[j], upper[j]);
                        if (q[j] == best_q[j]) continue;
                        const double a = surrogate.acquisition(q, incumbent);
                        if (a > best_score) {
                            best_score = a, best_q = std::move(q), moved = true;
                            break;
                        }
                    }
            }
            for (double& s : step) s *= 0.5;
        }
        r.acquisition_values.push_back(best_score);
        record(std::move(best_q));
    }
    const auto i = static_cast<std::size_t>(std::min_element(r.values.begin(), r.values.end()) - r.values.begin());
    r.best_point = r.points[i];
    r.best_value = r.values[i];
    return r;
}

double BayesianOptimization::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                                      const std::vector<double>& lower, const std::vector<double>& upper,
                                      BayesianSurrogate& surrogate) const {
    if (lower.size() != coordinates.size())
        throw std::invalid_argument("BayesianOptimization: bounds must match coordinates");
    auto r      = run(function, lower, upper, surrogate);
    coordinates = std::move(r.best_point);
    return r.best_value;
}

} // namespace datamunge::optim
