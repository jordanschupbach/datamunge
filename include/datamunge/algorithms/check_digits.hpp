#pragma once

/// \file check_digits.hpp
/// \brief Two decimal check-digit schemes: the Luhn algorithm and the Verhoeff
///        algorithm, for catching data-entry errors in identification numbers.
///
/// A *check digit* is an extra digit appended to a number so that a recomputation can
/// flag most typing mistakes. Two schemes:
///   - *Luhn* (1954): double every second digit (subtracting 9 if the result exceeds
///     9), sum, and require the total to be a multiple of 10. It powers credit-card,
///     IMEI, and many account-number checks. It catches all single-digit errors and
///     *almost* all adjacent transpositions -- but misses the 09<->90 swap.
///   - *Verhoeff* (1969): uses the multiplication of the dihedral group \f$D_5\f$ plus a
///     position-dependent permutation, so it catches *all* single-digit errors *and
///     all* adjacent transpositions (the two most common human errors), which no
///     purely modular scheme over the integers can do.
/// Both are verified below against their standard reference numbers.

#include <array>
#include <cstddef>
#include <string>

namespace datamunge::algorithms {

// ---------------------------------------------------------------------------- Luhn

namespace detail {
/// Luhn weighted sum; \p double_last selects whether the rightmost digit is doubled.
inline int luhn_total(const std::string& s, bool double_last) {
    int  sum = 0;
    bool dbl = double_last;
    for (std::size_t i = s.size(); i-- > 0;) {
        int d = s[i] - '0';
        if (dbl) {
            d *= 2;
            if (d > 9) d -= 9;
        }
        sum += d;
        dbl = !dbl;
    }
    return sum;
}
}  // namespace detail

/// \brief The Luhn check digit for a payload (which does not yet include it).
///
/// luhn_check_digit("7992739871") == 3  (so "79927398713" is Luhn-valid).
inline int luhn_check_digit(const std::string& payload) {
    // The appended check digit occupies the not-doubled (rightmost) slot, so the
    // payload's own rightmost digit is doubled.
    const int sum = detail::luhn_total(payload, /*double_last=*/true);
    return (10 - (sum % 10)) % 10;
}

/// \brief Validate a full number whose last digit is the Luhn check digit.
inline bool luhn_validate(const std::string& full) {
    return detail::luhn_total(full, /*double_last=*/false) % 10 == 0;
}

// ------------------------------------------------------------------------- Verhoeff

namespace detail {
// Dihedral-group D5 multiplication table.
inline const std::array<std::array<int, 10>, 10>& verhoeff_d() {
    static const std::array<std::array<int, 10>, 10> d = {{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                           {1, 2, 3, 4, 0, 6, 7, 8, 9, 5},
                                                           {2, 3, 4, 0, 1, 7, 8, 9, 5, 6},
                                                           {3, 4, 0, 1, 2, 8, 9, 5, 6, 7},
                                                           {4, 0, 1, 2, 3, 9, 5, 6, 7, 8},
                                                           {5, 9, 8, 7, 6, 0, 4, 3, 2, 1},
                                                           {6, 5, 9, 8, 7, 1, 0, 4, 3, 2},
                                                           {7, 6, 5, 9, 8, 2, 1, 0, 4, 3},
                                                           {8, 7, 6, 5, 9, 3, 2, 1, 0, 4},
                                                           {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}}};
    return d;
}
// Position-dependent permutation table.
inline const std::array<std::array<int, 10>, 8>& verhoeff_p() {
    static const std::array<std::array<int, 10>, 8> p = {{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                          {1, 5, 7, 6, 2, 8, 3, 0, 9, 4},
                                                          {5, 8, 0, 3, 7, 9, 6, 1, 4, 2},
                                                          {8, 9, 1, 6, 0, 4, 3, 5, 2, 7},
                                                          {9, 4, 5, 3, 1, 2, 6, 8, 7, 0},
                                                          {4, 2, 8, 6, 5, 7, 3, 9, 0, 1},
                                                          {2, 7, 9, 3, 8, 0, 6, 4, 1, 5},
                                                          {7, 0, 4, 6, 9, 1, 3, 2, 5, 8}}};
    return p;
}
// Multiplicative inverse in D5.
inline const std::array<int, 10>& verhoeff_inv() {
    static const std::array<int, 10> inv = {0, 4, 3, 2, 1, 5, 6, 7, 8, 9};
    return inv;
}
}  // namespace detail

/// \brief The Verhoeff check digit for a payload.
///
/// verhoeff_check_digit("236") == 3  (so "2363" is Verhoeff-valid).
inline int verhoeff_check_digit(const std::string& payload) {
    const auto& d = detail::verhoeff_d();
    const auto& p = detail::verhoeff_p();
    int         c = 0;
    for (std::size_t i = 0; i < payload.size(); ++i) {
        const int digit = payload[payload.size() - 1 - i] - '0';
        c               = d[c][p[(i + 1) % 8][digit]];
    }
    return detail::verhoeff_inv()[c];
}

/// \brief Validate a full number whose last digit is the Verhoeff check digit.
inline bool verhoeff_validate(const std::string& full) {
    const auto& d = detail::verhoeff_d();
    const auto& p = detail::verhoeff_p();
    int         c = 0;
    for (std::size_t i = 0; i < full.size(); ++i) {
        const int digit = full[full.size() - 1 - i] - '0';
        c               = d[c][p[i % 8][digit]];
    }
    return c == 0;
}

}  // namespace datamunge::algorithms
