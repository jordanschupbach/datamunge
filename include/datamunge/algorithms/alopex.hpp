#pragma once

/// \file alopex.hpp
/// \brief ALOPEX (ALgorithm Of Pattern EXtraction): a correlation-based, gradient-free
///        stochastic optimizer (Harth & Tzanakou 1974; Unnikrishnan & Venugopal 1994).
///
/// ALOPEX minimizes a scalar cost \f$E(w)\f$ without any gradient. It updates every
/// parameter *in parallel* using only the *correlation* between each parameter's most
/// recent change and the resulting change in the cost, plus stochastic exploration:
/// the correlation is \f$C_i(n) = \Delta w_i(n-1)\,\Delta E(n-1)\f$, the move probability is
/// \f$p_i(n) = 1/(1+e^{C_i(n)/T})\f$, and
/// then \f$w_i(n) = w_i(n-1) + \delta\f$ with probability \f$p_i\f$ and \f$-\delta\f$
/// otherwise (for *minimization*). Intuition: if increasing \f$w_i\f$ last step
/// *raised* the cost (\f$C_i>0\f$), make a decrease more likely; if it *lowered* the
/// cost (\f$C_i<0\f$), keep going that way. The "temperature" \f$T\f$ controls
/// exploration and is annealed (often set from the recent average \f$|C_i|\f$). Because
/// it needs only cost *evaluations* -- no derivatives, no per-parameter credit
/// assignment -- ALOPEX applies to non-differentiable, noisy, or black-box objectives,
/// and was proposed as a biologically plausible learning rule.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace datamunge::algorithms {

/// Result of an ALOPEX optimization run.
struct AlopexResult {
    std::vector<double> best_parameters;  ///< Parameters achieving the lowest cost seen.
    double              best_cost = 0.0;   ///< Lowest cost seen.
    std::vector<double> cost_history;      ///< Cost after each iteration.
    double              initial_cost = 0.0;  ///< Cost at the starting point.
    std::vector<double> temperature_history;  ///< Temperature used at each iteration (auto-scaled when AlopexParameters::temperature <= 0).
    std::vector<std::vector<double>> parameter_history;  ///< Iterates after each step (only with AlopexParameters::record_path).
};

/// Hyper-parameters for ALOPEX.
struct AlopexParameters {
    double        step        = 0.05;   ///< Perturbation magnitude \f$\delta\f$.
    double        temperature = 1.0;    ///< Initial temperature \f$T\f$ (auto-scaled from |C| if <=0).
    double        cooling     = 0.995;  ///< Multiplicative anneal per iteration.
    std::size_t   iterations  = 2000;   ///< Number of parallel moves (one cost evaluation each).
    std::uint64_t seed        = 0;      ///< Seed for the move coin flips.
    bool          record_path = false;  ///< Store every iterate in AlopexResult::parameter_history.
};

/// \brief Minimize \p cost over \p dim parameters with ALOPEX.
///
/// Each iteration flips one biased coin per parameter, moves all parameters at once and
/// evaluates the cost once. The first move has \f$C_i=0\f$, so it is a fair coin flip.
/// \param cost    Black-box scalar objective to minimize.
/// \param dim     Number of parameters.
/// \param init    Starting point (length \p dim).
/// \param params  Hyper-parameters.
/// \return Best point, its cost and the per-iteration history.
/// \throws std::invalid_argument  If \p init does not have length \p dim.
AlopexResult alopex_minimize(const std::function<double(const std::vector<double>&)>& cost,
                             std::size_t dim, const std::vector<double>& init,
                             const AlopexParameters& params);

/// \brief Recovers the correlations \f$C_i(n)=\Delta w_i(n-1)\,\Delta E(n-1)\f$ that drove a recorded run.
///
/// Row \f$n\f$ (for \f$n\ge 1\f$) holds the correlation used to choose move \f$n\f$; row 0 is all
/// zeros. Useful for checking the drift law \f$\mathbb{E}[C_i]\approx\delta^2\,\partial E/\partial w_i\f$.
/// \param result  A run made with AlopexParameters::record_path set.
/// \param init    The starting point passed to alopex_minimize.
/// \return One row per iteration, one column per parameter.
/// \throws std::invalid_argument  If the run has no recorded path.
std::vector<std::vector<double>> alopex_correlations(const AlopexResult& result,
                                                     const std::vector<double>& init);

}  // namespace datamunge::algorithms
