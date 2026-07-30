#pragma once

/// \file double_dabble.hpp
/// \brief The double-dabble algorithm: convert a binary number to binary-coded decimal
///        (BCD) using only shifts and adds.
///
/// Displaying a binary integer in decimal ordinarily needs division and modulo by 10,
/// which are expensive in hardware. *Double dabble* (a.k.a. shift-and-add-3) avoids them
/// entirely: it shifts the binary value left, one bit at a time, into a field of BCD
/// digits, and *before each shift* adds 3 to any BCD digit that is 5 or more. The
/// add-3 correction is the trick -- it pre-compensates so that the next left shift
/// (a multiply-by-2) carries correctly across the decimal-digit boundary, keeping every
/// nibble a valid decimal digit (0-9). After processing all input bits, the BCD field
/// holds the decimal digits. It is the standard way hardware drives 7-segment displays
/// and a neat illustration of doing decimal arithmetic with only binary shifts.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// \brief Convert \p value (using its low \p num_bits bits) to decimal BCD digits.
///
/// \return the decimal digits most-significant first (e.g. 243 -> {2, 4, 3}).
inline std::vector<int> double_dabble(std::uint64_t value, int num_bits) {
    // Number of decimal digits needed (ceil(num_bits * log10(2)) + 1 is safe).
    const int        num_digits = static_cast<int>(num_bits * 0.30103) + 1;
    std::vector<int> bcd(static_cast<std::size_t>(num_digits), 0);

    for (int bit = num_bits - 1; bit >= 0; --bit) {
        // Add 3 to any BCD digit that is >= 5 (pre-shift correction).
        for (int& d : bcd)
            if (d >= 5) d += 3;
        // Shift the whole BCD field left by one, bringing in the next input bit.
        int carry = (value >> bit) & 1ULL;
        for (std::size_t k = bcd.size(); k-- > 0;) {
            const int shifted = (bcd[k] << 1) | carry;
            bcd[k]            = shifted & 0xF;
            carry             = (shifted >> 4) & 1;
        }
    }

    // Strip leading zeros (keep at least one digit).
    std::size_t start = 0;
    while (start + 1 < bcd.size() && bcd[start] == 0) ++start;
    return std::vector<int>(bcd.begin() + static_cast<std::ptrdiff_t>(start), bcd.end());
}

/// \brief Convenience: the decimal string of \p value via double dabble.
inline std::string double_dabble_string(std::uint64_t value, int num_bits) {
    std::string out;
    for (int d : double_dabble(value, num_bits)) out += static_cast<char>('0' + d);
    return out;
}

}  // namespace datamunge::algorithms
