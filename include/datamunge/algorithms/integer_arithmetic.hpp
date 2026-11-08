#pragma once
/// @file integer_arithmetic.hpp
/// @brief Exact unsigned square roots and decimal text for compiler-supported 128-bit integers.
#include <cstdint>
#include <string>
namespace datamunge::algorithms {
/// @brief Compute floor(sqrt(n)) exactly without floating arithmetic or squared overflow.
/// Binary search uses division comparisons over [0,2^32), taking at most 32
/// iterations and constant storage. Accepts every uint64, including zero and
/// UINT64_MAX; does not mutate shared state or allocate.
/// @param n Nonnegative integer whose square root is required.
/// @return Largest uint64 r with r*r<=n; at most 4294967295.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::integer_sqrt_u64(18446744073709551615ULL)); // 4294967295
/// @endcode
/// @see chakravala(), uint128_text()
std::uint64_t integer_sqrt_u64(std::uint64_t n) noexcept;
/// @brief Compute the exact floor square root of any unsigned 128-bit integer.
/// Like Python math.isqrt's mathematical contract, with input restricted to the
/// compiler-supported unsigned 128-bit type. Bit-width-bracketed binary search
/// compares by division, never squares a sentinel or converts to floating point.
/// At most63 division comparisons and constant storage for this fixed width;
/// no allocation, mutation or shared state. Zero returns zero and 2^128-1 returns
/// UINT64_MAX. The exclusive upper bound can be 2^64 in the wider type.
/// @param n Any unsigned 128-bit pattern, including zero and the maximum.
/// @return Largest uint64 r whose mathematical square is at most n.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::integer_sqrt_u128(~static_cast<unsigned __int128>(0))); // 18446744073709551615
/// @endcode
/// @par References
/// - Python 3.14 math.isqrt mathematical contract (independent implementation):
///   https://docs.python.org/3.14/library/math.html#math.isqrt
/// @see integer_sqrt_u64(), uint128_text(), fermat_factor_work()
std::uint64_t integer_sqrt_u128(unsigned __int128 n) noexcept;
/// @brief Convert an unsigned compiler-supported 128-bit integer to exact decimal text.
/// Returns owned digits with no padding or locale dependence. Zero gives "0";
/// all 128-bit patterns are supported. At most 39 division steps and 39 digits;
/// independent calls share no mutable state.
/// @param value Unsigned integer to format.
/// @return Owned decimal string without a sign.
/// @throws std::bad_alloc If string allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::uint128_text(static_cast<unsigned __int128>(1)<<100)); // 1267650600228229401496703205376
/// @endcode
/// @see int128_text()
std::string uint128_text(unsigned __int128 value);
/// @brief Convert a signed compiler-supported 128-bit integer to exact decimal text.
/// Handles the minimum signed value without negating it in its signed type.
/// Returns owned locale-independent digits and a leading minus only for negatives.
/// At most 39 division steps plus a sign; independent calls share no mutable state.
/// @param value Signed integer, including -2^127.
/// @return Owned decimal string; zero is "0".
/// @throws std::bad_alloc If string allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::int128_text(-static_cast<__int128>(1))); // -1
/// @endcode
/// @see uint128_text(), PellSolution
std::string int128_text(__int128 value);
} // namespace datamunge::algorithms
