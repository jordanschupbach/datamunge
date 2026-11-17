/// @file phase_estimation.cpp
/// @brief Actual counting-register state preparation, kickback and inverse-QFT observation.
#include <cmath>
#include <datamunge/algorithms/phase_estimation.hpp>
#include <datamunge/algorithms/qft.hpp>
#include <limits>
#include <numbers>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Shared validated simulator kernel and optional stage-boundary observation.
namespace {
/// @brief Execute native gates once; snapshots are taken only at actual producing sites.
PhaseEstimationTrace run(int t, double phi, PhaseEstimationOptions options, bool capture) {
  if (t < 0 || t >= std::numeric_limits<std::size_t>::digits || !std::isfinite(phi) || phi < 0 ||
      phi >= 1)
    throw std::invalid_argument("phase estimation qubits/phase");
  const auto n = std::size_t{1} << t;
  if (n > QuantumState().max_size() || (options.max_basis_states && n > options.max_basis_states))
    throw std::length_error("phase estimation basis policy");
  if (capture && (n > std::numeric_limits<std::size_t>::max() / 4 ||
                  (options.max_trace_entries && 4 * n > options.max_trace_entries)))
    throw std::length_error("phase estimation history policy");
  PhaseEstimationTrace out;
  out.qubits = t;
  out.phase  = phi;
  auto state = qstate_zero(t);
  for (int q = 0; q < t; ++q) {
    h_gate(state, q);
    ++out.hadamards;
  }
  if (capture)
    out.after_hadamards = state;
  for (int q = 0; q < t; ++q) {
    phase_gate(state, q, 2 * std::numbers::pi * phi * static_cast<double>(std::size_t{1} << q));
    ++out.phase_gates;
  }
  if (capture)
    out.after_kickback = state;
  state        = qft_run(state, t, true, QFTOptions{0, 0}).state;
  const auto m = most_probable(state);
  out.result   = {static_cast<double>(m) / static_cast<double>(n), m, std::norm(state[m])};
  if (capture) {
    out.distribution      = probabilities(state);
    out.after_inverse_qft = std::move(state);
  }
  return out;
}
} // namespace
PhaseEstimationResult phase_estimation(int t, double phi) {
  return run(t, phi, {}, false).result;
}
PhaseEstimationTrace phase_estimation_trace(int t, double phi, PhaseEstimationOptions options) {
  return run(t, phi, options, true);
}
} // namespace datamunge::algorithms
