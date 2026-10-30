#pragma once

/// @file aks.hpp
/// @brief The Agrawal-Kayal-Saxena primality test, step by step: the choice of r, the
///        polynomial congruence in \f$(\mathbb Z/n)[x]/(x^r-1)\f$, and a trace of which step
///        decides each n.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief The multiplicative order of @p n modulo @p r: the least \f$k\ge1\f$ with
///        \f$n^k\equiv1\pmod r\f$, or 0 when \f$\gcd(n,r)\ne1\f$.
/// @throws std::invalid_argument if @p r is 0.
[[nodiscard]] std::uint64_t multiplicative_order(std::uint64_t n, std::uint64_t r);

/// @brief Euler's totient \f$\varphi(n)\f$, the count of \f$1\le k\le n\f$ coprime to @p n, by trial
///        division (\f$O(\sqrt n)\f$). \f$\varphi(0)=0\f$.
[[nodiscard]] std::uint64_t euler_totient(std::uint64_t n);

/// @brief If @p n is a perfect power \f$b^e\f$ with \f$e\ge2\f$, the pair (b, e) with the
///        largest e; otherwise (n, 1).
[[nodiscard]] std::pair<std::uint64_t, int> perfect_power(std::uint64_t n);

/// @brief The AKS modulus: the least r with \f$\gcd(n,r)=1\f$ and
///        \f$\operatorname{ord}_r(n) > (\log_2 n)^2\f$, or the least r reaching @p n.
///
/// Agrawal, Kayal and Saxena prove \f$r\le\max(3,\lceil\log_2^5 n\rceil)\f$.
/// @throws std::invalid_argument if @p n is below 2.
[[nodiscard]] std::uint64_t aks_r(std::uint64_t n);

/// @brief The coefficients \f$c_0,\dots,c_{r-1}\f$ of \f$(x+a)^n\f$ in \f$(\mathbb Z/n)[x]/(x^r-1)\f$,
///        by repeated squaring: \f$O(r^2\log n)\f$ coefficient operations.
/// @throws std::invalid_argument if @p n is below 2 or @p r is 0.
///
/// @par Example
/// @code
/// const auto c = dm::algorithms::aks_power(1, 7, 5);   // (x+1)^7 mod (x^5-1, 7) = x^2 + 1
/// @endcode
[[nodiscard]] std::vector<std::uint64_t> aks_power(std::uint64_t a, std::uint64_t n, std::uint64_t r);

/// @brief true if \f$(x+a)^n \equiv x^{n} + a\f$ in \f$(\mathbb Z/n)[x]/(x^r-1)\f$.
[[nodiscard]] bool aks_congruence_holds(std::uint64_t n, std::uint64_t r, std::uint64_t a);

/// @brief What each step of the AKS test found for one n; see aks_trace().
struct AksTrace {
    std::uint64_t n{0};                    ///< The number tested.
    bool is_prime{false};                  ///< The verdict.
    std::string decided_by;                ///< "n < 2", "perfect power", "small factor", "n <= r",
                                           ///< "congruence fails", "all congruences hold" or
                                           ///< "congruences not checked".
    std::uint64_t power_base{0};           ///< b when n = b^e (perfect power), else 0.
    int power_exponent{0};                 ///< e when n = b^e (perfect power), else 0.
    std::uint64_t r{0};                    ///< The modulus r (0 if not reached).
    std::uint64_t order{0};                ///< \f$\operatorname{ord}_r(n)\f$ (0 if not reached).
    std::uint64_t factor{0};               ///< A factor found by the gcd step, else 0.
    std::uint64_t a_max{0};                ///< \f$\lfloor\sqrt{\varphi(r)}\log_2 n\rfloor\f$, the congruences required.
    std::uint64_t congruences_checked{0};  ///< Congruences evaluated before the verdict.
    std::uint64_t witness_a{0};            ///< The first a whose congruence fails, else 0.
};

/// @brief Runs the AKS test on @p n and records how it reached its verdict.
///
/// Steps: (1) reject perfect powers; (2) find r = aks_r(n); (3) reject n with a factor
/// \f$1<\gcd(a,n)<n\f$ for some \f$a\le r\f$; (4) accept if \f$n\le r\f$; (5) check
/// \f$(x+a)^n\equiv x^n+a\f$ modulo \f$(x^r-1, n)\f$ for \f$a=1,\dots,\lfloor\sqrt{\varphi(r)}\log_2n\rfloor\f$,
/// accepting if all hold. Deterministic and polynomial in \f$\log n\f$, but slow: use
/// Miller-Rabin in practice.
/// @param n The number to test.
/// @param check_congruences If false, stop before step 5: the trace then has r, the order and
///        a_max (the work step 5 would do) but no congruences checked; a number reaching step 5
///        is reported as "congruences not checked" with is_prime false.
///
/// @par Example
/// @code
/// const auto t = dm::algorithms::aks_trace(1'000'003);
/// std::print("{}: r = {}, {} congruences\n", t.decided_by, t.r, t.congruences_checked);
/// @endcode
[[nodiscard]] AksTrace aks_trace(std::uint64_t n, bool check_congruences = true);

} // namespace datamunge::algorithms
