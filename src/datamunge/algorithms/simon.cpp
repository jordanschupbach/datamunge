/// @file simon.cpp
/// @brief Actual two-register evolution, probability marginal and support-row elimination.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/quantum_sim.hpp>
#include <datamunge/algorithms/simon.hpp>
#include <limits>
#include <stdexcept>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked helpers.
namespace {
/// @brief Evolve and eliminate exactly once.
/// @param n Input width.
/// @param period Promised period.
/// @param o Policies.
/// @param capture Retain actual reduction matrices.
/// @return Owned actual result.
SimonRun run(int n, std::size_t period, const SimonOptions& o, bool capture) {
  if (n < 1 || n > 15 || 2 * n >= std::numeric_limits<std::size_t>::digits)
    throw std::invalid_argument("simon: bit width");
  const std::size_t inputs = std::size_t{1} << n, amplitudes = std::size_t{1} << (2 * n);
  if (!period || period >= inputs)
    throw std::invalid_argument("simon: nonzero in-range period");
  if ((o.max_amplitudes && amplitudes > o.max_amplitudes) || amplitudes > QuantumState{}.max_size())
    throw std::length_error("simon: amplitudes");
  SimonRun r;
  r.amplitudes = amplitudes;
  auto state   = qstate_zero(2 * n);
  for (int q = 0; q < n; ++q) {
    h_gate(state, q);
    ++r.hadamard_calls;
  }
  QuantumState output(amplitudes, Complex{0, 0});
  for (std::size_t x = 0; x < inputs; ++x) {
    const auto fx         = std::min(x, x ^ period);
    output[x | (fx << n)] = state[x];
    ++r.oracle_writes;
  }
  state = std::move(output);
  for (int q = 0; q < n; ++q) {
    h_gate(state, q);
    ++r.hadamard_calls;
  }
  r.probabilities.assign(inputs, 0);
  for (std::size_t i = 0; i < state.size(); ++i)
    r.probabilities[i & (inputs - 1)] += std::norm(state[i]);
  for (std::size_t y = 1; y < inputs; ++y)
    if (r.probabilities[y] > 1e-9)
      r.result.observed_y.push_back(y);
  auto             rows = r.result.observed_y;
  std::vector<int> where(n, -1);
  std::size_t      retained = 0;
  for (int col = 0; col < n && r.rank < static_cast<int>(rows.size()); ++col) {
    std::size_t selected = rows.size();
    for (std::size_t i = r.rank; i < rows.size(); ++i)
      if ((rows[i] >> col) & 1) {
        selected = i;
        break;
      }
    if (selected == rows.size())
      continue;
    std::swap(rows[r.rank], rows[selected]);
    for (std::size_t i = 0; i < rows.size(); ++i)
      if (i != static_cast<std::size_t>(r.rank) && ((rows[i] >> col) & 1)) {
        rows[i] ^= rows[r.rank];
        ++r.row_xors;
      }
    where[col] = r.rank;
    if (capture) {
      if (rows.size() == std::numeric_limits<std::size_t>::max() ||
          retained > std::numeric_limits<std::size_t>::max() - rows.size() - 1)
        throw std::length_error("simon: history capacity");
      retained += rows.size() + 1;
      if (o.max_history_values && retained > o.max_history_values)
        throw std::length_error("simon: history");
      r.reduction.push_back({col, static_cast<std::size_t>(r.rank), rows});
    }
    ++r.rank;
  }
  for (int free = 0; free < n; ++free)
    if (where[free] < 0) {
      std::size_t solution = std::size_t{1} << free;
      for (int col = 0; col < n; ++col)
        if (where[col] >= 0 && ((rows[where[col]] >> free) & 1))
          solution |= std::size_t{1} << col;
      r.result.recovered_period = solution;
      break;
    }
  return r;
}
} // namespace
SimonRun simon_run(int n, std::size_t p, const SimonOptions& o) {
  return run(n, p, o, false);
}
SimonRun simon_trace(int n, std::size_t p, const SimonOptions& o) {
  return run(n, p, o, true);
}
SimonResult simon(int n, std::size_t p) {
  return simon_run(n, p).result;
}
} // namespace datamunge::algorithms
