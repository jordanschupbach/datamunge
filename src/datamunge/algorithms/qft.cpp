/// @file qft.cpp
/// @brief Shared checked forward/inverse gate ordering and actual post-gate state capture.
#include <datamunge/algorithms/qft.hpp>
#include <cmath>
#include <numbers>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Complete state validation and the single actual Fourier gate producer.
namespace fourier_detail {
/// @brief Reject any actual nonfinite complex component.
/// @param s Borrowed complete state.
/// @throws std::overflow_error If an amplitude is nonfinite.
void finite(const QuantumState& s) {
  for (auto v : s)
    if (!std::isfinite(v.real()) || !std::isfinite(v.imag()))
      throw std::overflow_error("Fourier nonfinite amplitude");
}
/// @brief Validate and execute the native gate order once with optional capture.
/// @param s Borrowed complete input.
/// @param n Counting qubits.
/// @param inverse Whether to execute the inverse network.
/// @param options Policies.
/// @param capture Whether post-gate histories are retained.
/// @return Owned actual output with optional histories.
/// @throws std::invalid_argument If n is invalid, state length differs from2^n, or an input amplitude is nonfinite.
/// @throws std::length_error If basis/history policy, size arithmetic or representable storage is exceeded.
/// @throws std::overflow_error If a native gate produces a nonfinite amplitude.
QFTTrace produce(const QuantumState& s, int n, bool inverse, QFTOptions options, bool capture) {
  if (n < 0 || n >= int(std::numeric_limits<std::size_t>::digits))
    throw std::invalid_argument("Fourier qubit count");
  const auto N = std::size_t{1} << n;
  if (s.size() != N)
    throw std::invalid_argument("Fourier state shape");
  for (auto v : s)
    if (!std::isfinite(v.real()) || !std::isfinite(v.imag()))
      throw std::invalid_argument("Fourier input finite");
  if (options.max_basis_states && N > options.max_basis_states)
    throw std::length_error("Fourier basis cap");
  std::size_t entries = 0;
  QFTTrace    out;
  if (capture) {
    if (N > std::numeric_limits<std::size_t>::max() / 2)
      throw std::length_error("Fourier history size");
    entries = 2 * N;
    if (options.max_trace_entries && entries > options.max_trace_entries)
      throw std::length_error("Fourier history cap");
    out.initial = s;
  }
  out.run.state = s;
  auto& state   = out.run.state;
  /// @brief Execute one actual simulator gate after admitting its exact retained snapshot.
  /// @param kind Gate selector.
  /// @param control Actual control/first swap bit.
  /// @param target Actual target/second swap bit.
  /// @param angle Actual controlled phase angle or zero.
  /// @throws std::length_error On history policy or size exhaustion.
  /// @throws std::overflow_error On nonfinite produced amplitudes.
  auto gate = [&](FourierGateKind kind, int control, int target, double angle) {
    if (capture) {
      if (N == std::numeric_limits<std::size_t>::max() ||
          N + 1 > std::numeric_limits<std::size_t>::max() - entries ||
          (options.max_trace_entries &&
           (entries > options.max_trace_entries || N + 1 > options.max_trace_entries - entries)))
        throw std::length_error("Fourier history cap");
      entries += N + 1;
    }
    if (kind == FourierGateKind::Hadamard) {
      h_gate(state, target);
      ++out.run.hadamards;
    } else if (kind == FourierGateKind::ControlledPhase) {
      controlled_phase(state, control, target, angle);
      ++out.run.phase_gates;
    } else {
      swap_qubits(state, control, target);
      ++out.run.swaps;
    }
    finite(state);
    if (capture)
      out.gates.push_back({kind, control, target, angle, state});
  };
  if (!inverse) {
    for (int j = n - 1; j >= 0; --j) {
      gate(FourierGateKind::Hadamard, j, j, 0);
      for (int k = j - 1; k >= 0; --k)
        gate(
            FourierGateKind::ControlledPhase,
            k,
            j,
            std::numbers::pi / double(std::size_t{1} << (j - k)));
    }
    for (int i = 0; i < n / 2; ++i)
      gate(FourierGateKind::Swap, i, n - 1 - i, 0);
  } else {
    for (int i = 0; i < n / 2; ++i)
      gate(FourierGateKind::Swap, i, n - 1 - i, 0);
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < j; ++k)
        gate(
            FourierGateKind::ControlledPhase,
            k,
            j,
            -std::numbers::pi / double(std::size_t{1} << (j - k)));
      gate(FourierGateKind::Hadamard, j, j, 0);
    }
  }
  return out;
}
} // namespace fourier_detail
void qft(QuantumState& s, int n) {
  s = fourier_detail::produce(s, n, false, {}, false).run.state;
}
void inverse_qft(QuantumState& s, int n) {
  s = fourier_detail::produce(s, n, true, {}, false).run.state;
}
QFTRun qft_run(const QuantumState& s, int n, bool inverse, QFTOptions options) {
  return fourier_detail::produce(s, n, inverse, options, false).run;
}
QFTTrace qft_trace(const QuantumState& s, int n, bool inverse, QFTOptions options) {
  return fourier_detail::produce(s, n, inverse, options, true);
}
QuantumState qft_comb(int n, std::size_t period, std::size_t offset, QFTOptions options) {
  if (n < 0 || n >= int(std::numeric_limits<std::size_t>::digits))
    throw std::invalid_argument("Fourier comb qubits");
  auto N = std::size_t{1} << n;
  if (!period || offset >= N)
    throw std::invalid_argument("Fourier comb support");
  if (options.max_basis_states && N > options.max_basis_states)
    throw std::length_error("Fourier comb basis cap");
  auto         count = 1 + (N - 1 - offset) / period;
  QuantumState s(N, Complex{0, 0});
  double       amplitude = 1 / std::sqrt(double(count));
  for (std::size_t k = 0; k < count; ++k)
    s[offset + k * period] = {amplitude, 0};
  return s;
}
} // namespace datamunge::algorithms
