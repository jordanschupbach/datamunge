#pragma once

/// @file addition_chain.hpp
/// @brief Addition chains for exponentiation: the binary and sliding-window chains, a validator,
///        provable lower bounds on the shortest length, and powering along a chain.
///
/// An addition chain for @c e is a sequence \f$1=c_0<c_1<\dots<c_L=e\f$ in which every term is
/// the sum of two (not necessarily distinct) earlier terms. Computing \f$x^{c_k}=x^{c_i}x^{c_j}\f$
/// costs one multiplication per term, so \f$x^e\f$ takes \f$L\f$ multiplications. The shortest
/// chain is algorithms::addition_chain() (in arithmetic_extra.hpp).
///
/// @code
/// namespace alg = dm::algorithms;
/// const auto best = alg::addition_chain(15);        // 1 2 4 5 10 15: 5 multiplications
/// const auto bin = alg::binary_chain(15);           // 1 2 3 6 7 14 15: 6 multiplications
/// std::print("{}\n", alg::mod_pow_chain(3, best, 1'000'003) == alg::mod_pow(3, 15, 1'000'003));
/// @endcode

#include <datamunge/algorithms/arithmetic_extra.hpp>

#include <cstdint>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief The left-to-right binary ("square-and-multiply") chain for @p e: one doubling per bit
///        below the leading one, plus one step for each further 1 bit, so its length is
///        \f$\lfloor\log_2 e\rfloor + \nu(e) - 1\f$ with \f$\nu\f$ the number of 1 bits.
/// @throws std::invalid_argument if @p e is 0.
///
/// @par Example
/// @code
/// const auto c = dm::algorithms::binary_chain(13);   // 1 2 3 6 12 13
/// @endcode
std::vector<std::uint64_t> binary_chain(std::uint64_t e);

/// @brief The left-to-right sliding-window chain for @p e with windows of up to @p k bits.
///
/// The odd values \f$1, 3, \dots, w_{\max}\f$ up to the largest window used are computed first
/// (from \f$2\f$); then the exponent is scanned from its top bit, doubling for each bit and adding
/// a precomputed odd window value at the end of each window. With \f$\lambda=\lfloor\log_2e\rfloor\f$, a
/// random exponent needs about \f$(\lambda-k+1) + (\lambda+1-k)/(k+1) + 2^{k-1}\f$ multiplications
/// (doublings, window additions, precomputation). `k = 1` gives binary_chain().
/// @param e The exponent, at least 1.
/// @param k Window width, 1..16.
/// @return A valid addition chain, sorted and without repeats.
/// @throws std::invalid_argument if @p e is 0 or @p k is not in [1, 16].
///
/// @par Example
/// @code
/// const auto c = dm::algorithms::window_chain(0xFFFF'FFFF'FFFF'FFFFULL, 4);
/// std::print("{} multiplications\n", c.size() - 1);
/// @endcode
std::vector<std::uint64_t> window_chain(std::uint64_t e, int k);

/// @brief true if @p chain starts at 1, increases strictly, and every later term is the sum of two
///        earlier terms.
[[nodiscard]] bool is_addition_chain(const std::vector<std::uint64_t>& chain);

/// @brief For each term after the first, the indices (i, j), i >= j, of two earlier terms that sum
///        to it: the multiplications \f$x^{c_k}=x^{c_i}x^{c_j}\f$ to perform.
/// @throws std::invalid_argument if @p chain is not an addition chain.
std::vector<std::pair<std::size_t, std::size_t>> chain_steps(const std::vector<std::uint64_t>& chain);

/// @brief \f$x^{e} \bmod m\f$ for the last term @c e of @p chain, using one modular multiplication per
///        chain step (128-bit products, so any modulus up to \f$2^{64}-1\f$ works).
/// @throws std::invalid_argument if @p chain is not an addition chain or @p modulus is 0.
///
/// @par Example
/// @code
/// const auto r = dm::algorithms::mod_pow_chain(2, dm::algorithms::addition_chain(65537), 1'000'000'007);
/// @endcode
[[nodiscard]] std::uint64_t mod_pow_chain(std::uint64_t base, const std::vector<std::uint64_t>& chain,
                                          std::uint64_t modulus);

/// @brief A proven lower bound on the length \f$\ell(e)\f$ of the shortest chain:
///        \f$\lambda(e) + [\nu(e)\ge 2] + [\nu(e)\ge 3]\f$, with \f$\lambda(e)=\lfloor\log_2 e\rfloor\f$.
///
/// The first term holds because a chain term can at most double each step; a chain of length
/// \f$\lambda(e)\f$ must double every time and so reaches only a power of two; three or more 1
/// bits need a further step (Gioia, Subbarao and Sugunamma, 1962).
/// @throws std::invalid_argument if @p e is 0.
[[nodiscard]] int chain_lower_bound(std::uint64_t e);

} // namespace datamunge::algorithms
