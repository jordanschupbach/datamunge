#pragma once
/// @file grover_analysis.hpp
/// @brief Explicit single-marked-state theory and unnormalized native state-mass diagnostics.
#include <cmath>
#include <datamunge/algorithms/grover.hpp>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Evaluate the ideal single-marked-state rotation formula, separately from simulation.
/// Returns sin²((2k+1)asin(1/sqrt(2^n))) using ordinary scalar floating math.
/// n=0 returns one; n=1 returns one half, avoiding artificial numerical differences
/// in these constant ideal cases. This is a theoretical reference, not an actual
/// state, oracle execution, sampled outcome, interval bound or high-precision result.
/// @param n Nonnegative qubit count with container-representable state size.
/// @param iterations Nonnegative complete step count.
/// @return Floating evaluation of the ideal marked-state probability.
/// @throws std::invalid_argument On invalid n or negative iterations.
/// @throws std::length_error On state/count representability.
/// @par Example
/// @code
/// std::print("{:.6f}\n",dm::algorithms::grover_ideal_probability(2,1)); // 1.000000
/// @endcode
/// @par References
/// - L. K. Grover (1996), https://arxiv.org/abs/quant-ph/9605043 .
/// @see grover_run(), grover_first_peak_iterations()
inline double grover_ideal_probability(int n, int iterations) {
  if (iterations < 0)
    throw std::invalid_argument("Grover theory needs nonnegative steps");
  (void)grover_first_peak_iterations(n);
  if (n == 0)
    return 1;
  if (n == 1)
    return .5;
  const double theta = std::asin(1 / std::sqrt(std::ldexp(1., n)));
  const double p     = std::sin((2. * iterations + 1) * theta);
  return p * p;
}
/// @brief Sum actual squared complex amplitudes without normalization or sampling.
/// Sequential double sums expose ordinary roundoff, not certified bounds. Empty
/// states give zero. O(N) diagnostic work, excluded from producing run counters;
/// no input mutation, references or allocations.
/// @param state Borrowed finite amplitudes, not necessarily normalized.
/// @return Actual sequential sum of std::norm for the supplied state.
/// @throws std::invalid_argument On nonfinite input components.
/// @throws std::overflow_error On nonfinite probability/sum arithmetic.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::grover_state_mass({{1,0},{0,0}})); // 1
/// @endcode
/// @see GroverRun
inline double grover_state_mass(const QuantumState& state) {
  double mass = 0;
  for (auto z : state) {
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
      throw std::invalid_argument("state mass needs finite input");
    mass += std::norm(z);
    if (!std::isfinite(mass))
      throw std::overflow_error("state mass overflow");
  }
  return mass;
}
} // namespace datamunge::algorithms
