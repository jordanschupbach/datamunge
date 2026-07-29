#pragma once

#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// @brief The result of a Chinese Remainder Theorem solve.
struct CrtSolution {
    /// @brief True iff the system of congruences is consistent (has a solution).
    bool solvable{false};
    /// @brief When @c solvable, the unique solution's residue in [0, @c modulus): the unique @c x
    ///        with 0 <= x < modulus satisfying every input congruence. Unspecified otherwise.
    std::int64_t remainder{0};
    /// @brief When @c solvable, the modulus of the merged congruence, equal to lcm of all input
    ///        moduli: the solution is unique modulo this value. Unspecified otherwise.
    std::int64_t modulus{1};
};

/// @brief General Chinese Remainder Theorem solver: find every @c x satisfying the simultaneous
///        congruences @c x congruent-to remainders[i] (mod moduli[i]) for all @c i. Unlike the
///        classical CRT, the moduli need **not** be pairwise coprime.
///
///        The system is solved by *pairwise merging*: two congruences @c x==r1 (mod m1) and
///        @c x==r2 (mod m2) are combined into a single congruence modulo @c lcm(m1,m2) via the
///        extended Euclidean algorithm. Such a pair is consistent iff @c gcd(m1,m2) divides
///        @c (r1 - r2); if any pair is inconsistent the whole system is unsolvable and the
///        returned @c CrtSolution has @c solvable == false. When solvable, the solution is unique
///        modulo @c lcm of all moduli and is returned normalized into @c [0, modulus).
///
///        Empty input is treated as the (vacuously true) empty system, whose solution is
///        @c x congruent-to 0 (mod 1) -- i.e. every integer -- reported as @c {true, 0, 1}.
///
/// @param remainders the target residues r_i (may be negative; each is normalized into
///        @c [0, moduli[i]) before solving).
/// @param moduli the moduli m_i; every modulus must be strictly positive.
/// @return the merged solution (see @c CrtSolution).
/// @throws std::invalid_argument if @p remainders and @p moduli differ in length, or if any
///         modulus is not strictly positive.
///
/// @note Overflow safety: intermediate products (the merged residue and the running lcm) are
///       formed in 128-bit arithmetic, so the solver is exact as long as the final modulus --
///       @c lcm of all input moduli -- fits in @c std::int64_t (i.e. lcm <= 2^63 - 1).
CrtSolution chinese_remainder(const std::vector<std::int64_t>& remainders,
                              const std::vector<std::int64_t>& moduli);

} // namespace datamunge::algorithms
