#pragma once

/// @file fourier_sampling.hpp
/// @brief Quantum Fourier sampling over \f$\mathbb Z_2^n\f$: the Bernstein-Vazirani circuit on an
///        arbitrary Boolean oracle, the matching classical Walsh-Hadamard spectrum, and shot sampling.
///
/// The circuit \f$H^{\otimes n}U_fH^{\otimes n}\f$ with the phase oracle
/// \f$U_f|x\rangle=(-1)^{f(x)}|x\rangle\f$ leaves the register in
/// \f$\sum_s\hat F(s)|s\rangle\f$, where \f$\hat F(s)=2^{-n}\sum_x(-1)^{f(x)\oplus s\cdot x}\f$ is the
/// Walsh-Hadamard (Fourier) coefficient of \f$F=(-1)^f\f$. A measurement returns \f$s\f$ with
/// probability \f$\hat F(s)^2\f$. For a parity \f$f(x)=s\cdot x\f$ the spectrum is a single spike,
/// which is the Bernstein-Vazirani algorithm.

#include <datamunge/algorithms/quantum_sim.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace datamunge::algorithms {

/// @brief Truth table of the parity \f$f(x)=s\cdot x \bmod 2\f$ on n bits.
///
/// @param n Number of input bits, 0 to 30.
/// @param secret The mask s, below 2^n.
/// @return 2^n values; entry x is the parity of `x & secret`.
/// @throws std::invalid_argument if n is out of range or secret has bits above n.
///
/// @par Example
/// @code
/// const auto f = dm::algorithms::parity_truth_table(2, 0b11);   // 0 1 1 0
/// @endcode
/// @see fourier_sampling_state()
std::vector<int> parity_truth_table(int n, std::size_t secret);

/// @brief Simulates \f$H^{\otimes n}U_fH^{\otimes n}|0\cdots0\rangle\f$ gate by gate on the state-vector
///        simulator, for the phase oracle of the Boolean function @p truth_table.
///
/// @param truth_table f(x) for x = 0 .. 2^n - 1 (entries 0 or 1); its size must be a power of two.
/// @return The final state; amplitude s equals walsh_spectrum(truth_table)[s] up to rounding.
/// @throws std::invalid_argument if the size is not a power of two.
///
/// @par Example
/// @code
/// const auto psi = dm::algorithms::fourier_sampling_state(dm::algorithms::parity_truth_table(6, 45));
/// // psi[45] == 1: the register is exactly |s> (Bernstein-Vazirani)
/// @endcode
/// @see walsh_spectrum(), measure_shots(), bernstein_vazirani()
QuantumState fourier_sampling_state(const std::vector<int>& truth_table);

/// @brief Normalised Walsh-Hadamard spectrum \f$\hat F(s)=2^{-n}\sum_x(-1)^{f(x)\oplus s\cdot x}\f$ by
///        the fast transform in \f$O(n2^n)\f$ additions.
///
/// By Parseval, \f$\sum_s\hat F(s)^2=1\f$.
///
/// @param truth_table f(x) for x = 0 .. 2^n - 1 (entries 0 or 1); its size must be a power of two.
/// @return The 2^n coefficients, indexed by s.
/// @throws std::invalid_argument if the size is not a power of two.
///
/// @par Example
/// @code
/// const auto F = dm::algorithms::walsh_spectrum({0, 0, 0, 1});   // AND: 0.5 0.5 0.5 -0.5
/// @endcode
/// @see fourier_sampling_state()
std::vector<double> walsh_spectrum(const std::vector<int>& truth_table);

/// @brief Simulates @p shots independent measurements of @p state in the computational basis.
///
/// @param state Any state; probabilities are renormalised to sum to one.
/// @param shots Number of measurements.
/// @param seed Seed for the random generator, so results are reproducible.
/// @return The measured basis indices, in order.
/// @throws std::invalid_argument if the state has zero norm.
///
/// @par Example
/// @code
/// const auto outcomes = dm::algorithms::measure_shots(psi, 1000, 42);
/// @endcode
/// @see probabilities()
std::vector<std::size_t> measure_shots(const QuantumState& state, std::size_t shots, std::uint64_t seed);

/// @brief The classical algorithm for a parity oracle: query \f$f(e_i)=s_i\f$ for each bit.
///
/// Any classical algorithm needs n queries in the worst case, since each answer is one bit and
/// s has n unknown bits.
///
/// @param n Number of bits.
/// @param oracle The black box, called exactly n times.
/// @return The recovered s.
///
/// @par Example
/// @code
/// const auto s = dm::algorithms::bernstein_vazirani_classical(
///     6, [](std::size_t x) { return std::popcount(x & 45u) & 1; });   // 45
/// @endcode
/// @see bernstein_vazirani()
std::size_t bernstein_vazirani_classical(int n, const std::function<int(std::size_t)>& oracle);

}  // namespace datamunge::algorithms
