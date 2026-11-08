#pragma once
#include <datamunge/algorithms/binary_splitting.hpp>

#include <cstdint>
#include <string>

namespace datamunge::algorithms {

/// @brief A *spigot algorithm* for the decimal digits of @c e (Rabinowitz-Wagon style): it produces
///        the digits *one at a time* from a mixed-radix (factorial-base) representation, using only
///        small-integer arithmetic and *no* storage or knowledge of the digits already emitted -- the
///        defining property of a spigot. Each output digit is the carry that falls out of one pass of
///        "multiply the state by 10 and renormalise".
///
/// @param n_digits the number of digits after the decimal point to produce.
/// @return the string @c "2.7182818284..." with @p n_digits fractional digits.
std::string spigot_e(int n_digits);

/// @brief A *digit-by-digit square root* (the paper-and-pencil "long division" method for
///        @c sqrt(n)): process the radicand two decimal places at a time, and at each step find the
///        largest digit @c d with @c (20·root + d)·d <= remainder, append it to the running root, and
///        carry down the next pair. It yields the exact decimal expansion digit by digit, using only
///        integer multiply-compare-subtract -- one of the *methods of computing square roots*.
///
/// @param n the radicand (a nonnegative integer).
/// @param frac_digits the number of digits to produce after the decimal point.
/// @return the string @c "d.dddd" of @c sqrt(n) to @p frac_digits fractional digits.
std::string digit_by_digit_sqrt(std::uint64_t n, int frac_digits);

/// @brief The quotient and remainder produced by @ref long_division.
struct DivResult {
    std::uint64_t quotient{0};
    std::uint64_t remainder{0};
    int           iterations{0}; ///< bit steps performed (= number of dividend bits).
};

/// @brief *Long division* in binary (the schoolbook shift-and-subtract algorithm): process the
///        dividend one bit at a time from the most significant, shifting it into a remainder register
///        and subtracting the divisor whenever it fits (setting that quotient bit). It is the
///        hardware-free, exact way to divide, computing both quotient and remainder in @c O(bits)
///        compare-subtract steps -- the binary form of the decimal long division taught in school.
///
/// @param dividend the number to divide.
/// @param divisor the divisor (must be nonzero).
/// @return the @ref DivResult quotient, remainder, and bit-step count.
DivResult long_division(std::uint64_t dividend, std::uint64_t divisor);

} // namespace datamunge::algorithms
