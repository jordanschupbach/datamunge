/// @file alopex.cpp
/// @brief ALOPEX correlation-driven stochastic minimizer and its correlation trace.

#include <datamunge/algorithms/alopex.hpp>

#include <cmath>
#include <random>
#include <stdexcept>

namespace datamunge::algorithms {

AlopexResult alopex_minimize(const std::function<double(const std::vector<double>&)>& cost,
                             std::size_t dim, const std::vector<double>& init,
                             const AlopexParameters& params) {
    if (init.size() != dim) throw std::invalid_argument("alopex_minimize: init must have length dim");
    std::mt19937_64                        rng(params.seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    std::vector<double> w = init;
    std::vector<double> prev_dw(dim, 0.0);
    double              prev_cost = cost(w);
    double              prev_dE   = 0.0;
    double              T         = params.temperature;

    AlopexResult r;
    r.best_parameters = w;
    r.best_cost       = prev_cost;
    r.initial_cost    = prev_cost;
    r.cost_history.reserve(params.iterations);
    r.temperature_history.reserve(params.iterations);
    if (params.record_path) r.parameter_history.reserve(params.iterations);

    std::vector<double> C(dim);
    std::vector<double> dw(dim);
    for (std::size_t n = 0; n < params.iterations; ++n) {
        // Auto-scale temperature from the average recent correlation magnitude, if requested.
        double auto_T = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            C[i] = prev_dw[i] * prev_dE;
            auto_T += std::fabs(C[i]);
        }
        auto_T /= static_cast<double>(dim > 0 ? dim : 1);
        const double Teff = (params.temperature <= 0.0) ? (auto_T > 0 ? auto_T : 1.0) : T;

        for (std::size_t i = 0; i < dim; ++i) {
            const double p = 1.0 / (1.0 + std::exp(C[i] / (Teff + 1e-12)));  // P(move +delta)
            dw[i]          = (unit(rng) < p) ? params.step : -params.step;
            w[i] += dw[i];
        }

        const double c = cost(w);
        prev_dE        = c - prev_cost;  // change in cost caused by dw
        prev_cost      = c;
        prev_dw        = dw;
        T *= params.cooling;

        if (c < r.best_cost) {
            r.best_cost       = c;
            r.best_parameters = w;
        }
        r.cost_history.push_back(c);
        r.temperature_history.push_back(Teff);
        if (params.record_path) r.parameter_history.push_back(w);
    }
    return r;
}

std::vector<std::vector<double>> alopex_correlations(const AlopexResult& result,
                                                     const std::vector<double>& init) {
    const auto& path = result.parameter_history;
    if (path.empty() && !result.cost_history.empty())
        throw std::invalid_argument("alopex_correlations: run was made without record_path");
    const std::size_t                dim = init.size();
    std::vector<std::vector<double>> C(path.size(), std::vector<double>(dim, 0.0));
    for (std::size_t n = 1; n < path.size(); ++n) {
        const auto&  before = (n == 1) ? init : path[n - 2];
        const double cost_before = (n == 1) ? result.initial_cost : result.cost_history[n - 2];
        const double dE   = result.cost_history[n - 1] - cost_before;
        for (std::size_t i = 0; i < dim; ++i) C[n][i] = (path[n - 1][i] - before[i]) * dE;
    }
    return C;
}

}  // namespace datamunge::algorithms
