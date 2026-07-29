#pragma once

#include <cstdint>

namespace datamunge::algorithms {

/// @brief Modular exponentiation by squaring -- computes @f$base^{exponent} \bmod modulus@f$ in
///        @f$O(\log exponent)@f$ modular multiplications rather than the @f$O(exponent)@f$ of
///        naive repeated multiplication. The exponent is read as a binary number
///        @f$b = \sum_i b_i 2^i@f$, so @f$a^b = \prod_{i:\,b_i=1} a^{2^i}@f$: we repeatedly square
///        a running base to obtain the @f$a^{2^i}@f$ and fold in the ones whose bit is set. Every
///        product is reduced modulo @p modulus at each step, keeping the operands bounded so the
///        result is exact even for cryptographic-sized exponents. Each modular product is formed
///        via a 128-bit intermediate (@c unsigned @c __int128), so the routine stays correct for
///        moduli all the way up to @f$2^{64}-1@f$, where a plain 64-bit multiply would overflow.
///
/// @param base     the base @f$a@f$.
/// @param exponent the exponent @f$b@f$ (may be 0).
/// @param modulus  the modulus @f$m@f$; must be nonzero.
/// @return @f$base^{exponent} \bmod modulus@f$. When @p modulus is 1 the result is 0; when
///         @p exponent is 0 the result is @f$modulus > 1 ? 1 : 0@f$.
/// @throws std::invalid_argument if @p modulus is 0.
std::uint64_t mod_pow(std::uint64_t base, std::uint64_t exponent, std::uint64_t modulus);

} // namespace datamunge::algorithms
