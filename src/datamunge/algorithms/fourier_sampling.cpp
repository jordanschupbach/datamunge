/// @file fourier_sampling.cpp
/// @brief Fourier sampling circuit, fast Walsh-Hadamard transform and measurement sampling.

#include <datamunge/algorithms/fourier_sampling.hpp>

#include <algorithm>
#include <bit>
#include <random>
#include <stdexcept>
#include <string>

namespace datamunge::algorithms {

/// @brief Helpers for fourier_sampling.cpp.
namespace {

/// Number of qubits for a truth table, which must have 2^n entries.
int qubits_for(std::size_t size, const char* who) {
  if (size == 0 || !std::has_single_bit(size)) throw std::invalid_argument(std::string(who) + ": the truth table size must be a power of two");
  return std::countr_zero(size);
}

}  // namespace

std::vector<int> parity_truth_table(int n, std::size_t secret) {
  if (n < 0 || n > 30) throw std::invalid_argument("parity_truth_table: n must be 0 to 30");
  if (secret >> n) throw std::invalid_argument("parity_truth_table: secret has bits above n");
  std::vector<int> f(std::size_t{1} << n);
  for (std::size_t x = 0; x < f.size(); ++x) f[x] = std::popcount(x & secret) & 1;
  return f;
}

QuantumState fourier_sampling_state(const std::vector<int>& truth_table) {
  const int n = qubits_for(truth_table.size(), "fourier_sampling_state");
  QuantumState psi = qstate_zero(n);
  for (int q = 0; q < n; ++q) h_gate(psi, q);
  for (std::size_t x = 0; x < psi.size(); ++x)
    if (truth_table[x] & 1) psi[x] = -psi[x];
  for (int q = 0; q < n; ++q) h_gate(psi, q);
  return psi;
}

std::vector<double> walsh_spectrum(const std::vector<int>& truth_table) {
  const int n = qubits_for(truth_table.size(), "walsh_spectrum");
  std::vector<double> F(truth_table.size());
  for (std::size_t x = 0; x < F.size(); ++x) F[x] = (truth_table[x] & 1) ? -1.0 : 1.0;
  for (std::size_t h = 1; h < F.size(); h <<= 1)
    for (std::size_t i = 0; i < F.size(); i += 2 * h)
      for (std::size_t j = i; j < i + h; ++j) {
        const double a = F[j], b = F[j + h];
        F[j] = a + b;
        F[j + h] = a - b;
      }
  const double scale = std::ldexp(1.0, -n);
  for (auto& v : F) v *= scale;
  return F;
}

std::vector<std::size_t> measure_shots(const QuantumState& state, std::size_t shots, std::uint64_t seed) {
  std::vector<double> cumulative(state.size());
  double total = 0.0;
  for (std::size_t i = 0; i < state.size(); ++i) cumulative[i] = total += std::norm(state[i]);
  if (!(total > 0.0)) throw std::invalid_argument("measure_shots: the state has zero norm");
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(0.0, total);
  std::vector<std::size_t> out(shots);
  for (auto& o : out) {
    const auto it = std::upper_bound(cumulative.begin(), cumulative.end(), u(rng));
    o = static_cast<std::size_t>(std::min<std::ptrdiff_t>(it - cumulative.begin(), static_cast<std::ptrdiff_t>(state.size()) - 1));
  }
  return out;
}

std::size_t bernstein_vazirani_classical(int n, const std::function<int(std::size_t)>& oracle) {
  std::size_t s = 0;
  for (int i = 0; i < n; ++i)
    if (oracle(std::size_t{1} << i) & 1) s |= std::size_t{1} << i;
  return s;
}

}  // namespace datamunge::algorithms
