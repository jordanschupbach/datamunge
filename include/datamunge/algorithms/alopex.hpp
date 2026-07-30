#pragma once

/// \file alopex.hpp
/// \brief ALOPEX (ALgorithm Of Pattern EXtraction): a correlation-based, gradient-free
///        stochastic optimizer (Harth & Tzanakou 1974; Unnikrishnan & Venugopal 1994).
///
/// ALOPEX minimizes a scalar cost \f$E(w)\f$ without any gradient. It updates every
/// parameter *in parallel* using only the *correlation* between each parameter's most
/// recent change and the resulting change in the cost, plus stochastic exploration:
/// \f[
///   C_i(n) = \Delta w_i(n-1)\,\Delta E(n-1),\qquad
///   p_i(n) = \frac{1}{1+e^{\,C_i(n)/T}},
/// \f]
/// then \f$w_i(n) = w_i(n-1) + \delta\f$ with probability \f$p_i\f$ and \f$-\delta\f$
/// otherwise (for *minimization*). Intuition: if increasing \f$w_i\f$ last step
/// *raised* the cost (\f$C_i>0\f$), make a decrease more likely; if it *lowered* the
/// cost (\f$C_i<0\f$), keep going that way. The "temperature" \f$T\f$ controls
/// exploration and is annealed (often set from the recent average \f$|C_i|\f$). Because
/// it needs only cost *evaluations* -- no derivatives, no per-parameter credit
/// assignment -- ALOPEX applies to non-differentiable, noisy, or black-box objectives,
/// and was proposed as a biologically plausible learning rule.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

namespace datamunge::algorithms {

/// Result of an ALOPEX optimization run.
struct AlopexResult {
    std::vector<double> best_parameters;  ///< Parameters achieving the lowest cost seen.
    double              best_cost = 0.0;   ///< Lowest cost seen.
    std::vector<double> cost_history;      ///< Cost at each iteration.
};

/// Hyper-parameters for ALOPEX.
struct AlopexParameters {
    double        step        = 0.05;   ///< Perturbation magnitude \f$\delta\f$.
    double        temperature = 1.0;    ///< Initial temperature \f$T\f$ (auto-scaled from |C| if <=0).
    double        cooling     = 0.995;  ///< Multiplicative anneal per iteration.
    std::size_t   iterations  = 2000;
    std::uint64_t seed        = 0;
};

/// \brief Minimize \p cost over \p dim parameters with ALOPEX.
///
/// \param cost    Black-box scalar objective to minimize.
/// \param dim     Number of parameters.
/// \param init    Starting point.
/// \param params  Hyper-parameters.
inline AlopexResult alopex_minimize(const std::function<double(const std::vector<double>&)>& cost,
                                    std::size_t dim, const std::vector<double>& init,
                                    const AlopexParameters& params) {
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
    r.cost_history.reserve(params.iterations);

    for (std::size_t n = 0; n < params.iterations; ++n) {
        // Auto-scale temperature from the average recent correlation magnitude, if requested.
        double auto_T = 0.0;
        std::vector<double> C(dim);
        for (std::size_t i = 0; i < dim; ++i) {
            C[i] = prev_dw[i] * prev_dE;
            auto_T += std::fabs(C[i]);
        }
        auto_T /= (dim > 0 ? dim : 1);
        const double Teff = (params.temperature <= 0.0) ? (auto_T > 0 ? auto_T : 1.0) : T;

        std::vector<double> dw(dim);
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
    }
    return r;
}

}  // namespace datamunge::algorithms
