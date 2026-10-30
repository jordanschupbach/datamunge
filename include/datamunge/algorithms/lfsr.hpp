#pragma once

/// @file lfsr.hpp
/// @brief Binary linear feedback shift registers and the statistics of linear complexity.
///
/// Companions to algorithms::berlekamp_massey(): generating LFSR output, the linear complexity
/// profile of a sequence in one pass, the expected profile of a random sequence, the Geffe
/// nonlinear combiner, and bit/byte conversion for stream-cipher experiments. A connection
/// polynomial \f$C(x)=1+c_1x+\dots+c_Lx^L\f$ is stored as `{1, c_1, ..., c_L}` and defines the
/// recurrence \f$s_i=c_1s_{i-1}\oplus\dots\oplus c_Ls_{i-L}\f$ over GF(2).

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace datamunge::algorithms {

/// @brief Output of the LFSR with connection polynomial @p connection started from @p seed.
///
/// The first L terms are the seed; every later term is
/// \f$s_i=\bigoplus_{j=1}^{L}c_js_{i-j}\f$.
///
/// @param connection Coefficients {1, c_1, ..., c_L} with entries 0 or 1.
/// @param seed The first L output bits.
/// @param n How many bits to return.
/// @return The first @p n output bits.
/// @throws std::invalid_argument if @p connection is empty, does not start with 1, or
///         @p seed does not hold exactly L bits.
///
/// @par Example
/// @code
/// const auto s = dm::algorithms::lfsr_sequence({1, 1, 0, 1}, {0, 0, 1}, 14);
/// // 0 0 1 1 1 0 1 0 0 1 1 1 0 1  (period 7 from 1 + x + x^3)
/// @endcode
/// @see berlekamp_massey(), lfsr_period()
std::vector<int> lfsr_sequence(const std::vector<int>& connection, const std::vector<int>& seed, std::size_t n);

/// @brief Period of the LFSR state sequence started from @p seed, by stepping until the state
///        repeats.
///
/// A primitive connection polynomial of degree L gives the maximum period \f$2^L-1\f$ from every
/// non-zero seed. The search stops after @p max_steps steps.
///
/// @param connection Coefficients {1, c_1, ..., c_L} with c_L = 1, L at most 62.
/// @param seed The initial L bits, not all zero.
/// @param max_steps Upper limit on the steps taken.
/// @return The period, or 0 if the state did not return within @p max_steps steps.
/// @throws std::invalid_argument if c_L is 0, L exceeds 62, the seed has the wrong length or is
///         all zero.
///
/// @par Example
/// @code
/// const auto T = dm::algorithms::lfsr_period({1, 1, 0, 1}, {0, 0, 1});   // 7
/// @endcode
/// @see lfsr_sequence()
long long lfsr_period(const std::vector<int>& connection, const std::vector<int>& seed,
                      long long max_steps = 1LL << 32);

/// @brief Seed bits encoded by the integer @p code: bit i of @p code is \f$s_i\f$, i < @p length.
///
/// @param code The seed as an integer, 0 <= code < 2^length.
/// @param length The number of register stages L.
/// @return The L seed bits.
///
/// @par Example
/// @code
/// const auto seed = dm::algorithms::lfsr_seed(6, 3);   // 0 1 1
/// @endcode
/// @see lfsr_correlation_scan()
std::vector<int> lfsr_seed(long long code, int length);

/// @brief Number of positions where each possible seed's LFSR output agrees with @p observed.
///
/// Entry `code` counts the i with `observed[i]` equal to bit i of lfsr_sequence(connection,
/// lfsr_seed(code, L), n); entries of @p observed other than 0 and 1 are skipped. This is the
/// exhaustive search of a correlation attack: when a keystream agrees with one register's
/// output with probability \f$q\ne 1/2\f$, the true seed stands out by about
/// \f$|2q-1|\sqrt n\f$ standard deviations, and only that register's \f$2^L\f$ seeds are tried.
///
/// @param connection Coefficients {1, c_1, ..., c_L} with L at most 30.
/// @param observed Known bits; any value other than 0 or 1 marks an unknown position.
/// @return \f$2^L\f$ agreement counts indexed by seed code (code 0, the all-zero seed, included).
/// @throws std::invalid_argument if the polynomial does not start with 1 or L is 0 or above 30.
///
/// @par Example
/// @code
/// const auto score = dm::algorithms::lfsr_correlation_scan({1, 1, 0, 1}, {0, 0, 1, 1, 1, 0, 1});
/// // score[4] == 7: the seed 0 0 1 (code 4) reproduces all seven bits
/// @endcode
/// @see lfsr_seed(), geffe_sequence()
std::vector<int> lfsr_correlation_scan(const std::vector<int>& connection, const std::vector<int>& observed);

/// @brief Linear complexity \f$L_k\f$ of every prefix \f$s_0\dots s_{k-1}\f$, k = 1..n.
///
/// One Berlekamp-Massey pass gives the whole profile in \f$O(n^2)\f$ bit operations, because
/// the register length after step k is the linear complexity of the first k bits.
///
/// @param s The sequence, entries 0 or 1.
/// @return A vector of length s.size() whose entry k-1 is \f$L_k\f$.
///
/// @par Example
/// @code
/// const auto profile = dm::algorithms::linear_complexity_profile({0, 0, 1, 1, 1, 0, 1});
/// // 0 0 3 3 3 3 3
/// @endcode
/// @see berlekamp_massey(), expected_linear_complexity()
std::vector<int> linear_complexity_profile(const std::vector<int>& s);

/// @brief Expected linear complexity of n independent fair bits (Rueppel, 1986).
///
/// \f[
/// \mathbb E[L_n]=\frac n2+\frac{4+(n\bmod 2)}{18}-2^{-n}\Big(\frac n3+\frac29\Big).
/// \f]
///
/// @param n The sequence length.
/// @return \f$\mathbb E[L_n]\f$ (0 when n is 0).
///
/// @par Example
/// @code
/// const double e = dm::algorithms::expected_linear_complexity(100);   // 50.222...
/// @endcode
/// @see linear_complexity_profile()
double expected_linear_complexity(std::size_t n);

/// @brief The Geffe generator's output \f$z=x_1x_2\oplus(1\oplus x_2)x_3\f$: x2 selects x1 or x3.
///
/// With three maximum-length LFSRs of pairwise coprime lengths L1, L2, L3 the output has linear
/// complexity \f$L_1L_2+L_2L_3+L_3\f$, yet agrees with x1 and with x3 on 3/4 of the bits.
///
/// @param x1 Bits chosen when x2 is 1.
/// @param x2 The selector bits.
/// @param x3 Bits chosen when x2 is 0.
/// @return The combined bits, as long as the shortest input.
///
/// @par Example
/// @code
/// const auto z = dm::algorithms::geffe_sequence({1, 1, 0}, {1, 0, 0}, {0, 0, 1});   // 1 0 1
/// @endcode
/// @see lfsr_sequence()
std::vector<int> geffe_sequence(const std::vector<int>& x1, const std::vector<int>& x2, const std::vector<int>& x3);

/// @brief Bits of @p text, eight per byte, most significant bit first.
///
/// @param text Any bytes.
/// @return 8 * text.size() bits.
///
/// @par Example
/// @code
/// const auto bits = dm::algorithms::bytes_to_bits("A");   // 0 1 0 0 0 0 0 1
/// @endcode
/// @see bits_to_bytes()
std::vector<int> bytes_to_bits(std::string_view text);

/// @brief Packs bits back into bytes, most significant bit first; the inverse of bytes_to_bits().
///
/// @param bits The bits; a final partial byte is padded with zeros.
/// @return ceil(bits.size() / 8) bytes.
///
/// @par Example
/// @code
/// const auto text = dm::algorithms::bits_to_bytes({0, 1, 0, 0, 0, 0, 0, 1});   // "A"
/// @endcode
/// @see bytes_to_bits()
std::string bits_to_bytes(const std::vector<int>& bits);

/// @brief Elementwise XOR of two bit vectors, as long as the shorter one.
///
/// @param a First bits.
/// @param b Second bits.
/// @return \f$a_i\oplus b_i\f$.
///
/// @par Example
/// @code
/// const auto c = dm::algorithms::xor_bits({1, 0, 1}, {1, 1, 0});   // 0 1 1
/// @endcode
std::vector<int> xor_bits(const std::vector<int>& a, const std::vector<int>& b);

}  // namespace datamunge::algorithms
