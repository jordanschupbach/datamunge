#pragma once

/// @file sieve_of_atkin.hpp
/// @brief The quadratic forms behind the Sieve of Atkin, operation counts for prime sieves, and
///        analytic approximations of the prime-counting function.
///
/// Atkin and Bernstein's sieve marks a squarefree \f$n>3\f$ as prime when the number of
/// positive solutions of one quadratic form is odd: \f$4x^2+y^2=n\f$ for \f$n\equiv1,5\pmod{12}\f$,
/// \f$3x^2+y^2=n\f$ for \f$n\equiv7\pmod{12}\f$, and \f$3x^2-y^2=n\f$ with \f$x>y\f$ for
/// \f$n\equiv11\pmod{12}\f$. atkin_primes() (in primality.hpp) implements it;
/// atkin_sieve_work() runs the same sieve and counts its operations, and
/// eratosthenes_sieve_work() does the same for the Sieve of Eratosthenes.

#include <cstdint>

namespace datamunge::algorithms {

/// The three quadratic forms of the Sieve of Atkin.
enum class AtkinForm {
    FourXSquaredPlusYSquared,   ///< \f$4x^2+y^2\f$, used for \f$n\equiv1,5\pmod{12}\f$.
    ThreeXSquaredPlusYSquared,  ///< \f$3x^2+y^2\f$, used for \f$n\equiv7\pmod{12}\f$.
    ThreeXSquaredMinusYSquared  ///< \f$3x^2-y^2\f$ with \f$x>y\f$, used for \f$n\equiv11\pmod{12}\f$.
};

/// @brief Number of solutions of form(x, y) = n in positive integers (with x > y for the third form).
std::uint64_t atkin_form_representations(std::uint64_t n, AtkinForm form);

/// @brief The form that the Sieve of Atkin uses for @p n, chosen by n mod 12.
/// @return true and sets @p form when n mod 12 is 1, 5, 7 or 11; false otherwise.
bool atkin_form_for(std::uint64_t n, AtkinForm& form);

/// Operation counts of one run of a prime sieve.
struct SieveWork {
    std::uint64_t limit           = 0;  ///< Inclusive upper bound sieved.
    std::uint64_t primes          = 0;  ///< \f$\pi(\text{limit})\f$.
    std::uint64_t lattice_points  = 0;  ///< Atkin: (x, y) pairs visited across the three forms.
    std::uint64_t toggles         = 0;  ///< Atkin: flags flipped (a residue test passed).
    std::uint64_t square_removals = 0;  ///< Atkin: multiples of \f$p^2\f$ cleared.
    std::uint64_t crossings       = 0;  ///< Eratosthenes: multiples of primes crossed out.
};

/// @brief Runs the Sieve of Atkin up to @p limit, counting its operations.
///
/// Each form is enumerated only over its own region (for example \f$4x^2+y^2\le\text{limit}\f$),
/// so about \f$1.04\,\text{limit}\f$ lattice points are visited.
SieveWork atkin_sieve_work(std::uint64_t limit);

/// @brief Runs the Sieve of Eratosthenes (crossing out from \f$p^2\f$) up to @p limit, counting crossings.
SieveWork eratosthenes_sieve_work(std::uint64_t limit);

/// @brief The logarithmic integral \f$\mathrm{li}(x)=\int_0^x dt/\ln t\f$ (principal value), for x > 1.
/// @throws std::invalid_argument if x <= 1.
double logarithmic_integral(double x);

/// @brief Riemann's prime-counting approximation \f$R(x)=\sum_{n\ge1}\frac{\mu(n)}{n}\mathrm{li}(x^{1/n})\f$,
///        evaluated by the Gram series \f$1+\sum_{k\ge1}\frac{(\ln x)^k}{k\,k!\,\zeta(k+1)}\f$, for x >= 1.
/// @throws std::invalid_argument if x < 1.
double riemann_prime_count(double x);

}  // namespace datamunge::algorithms
