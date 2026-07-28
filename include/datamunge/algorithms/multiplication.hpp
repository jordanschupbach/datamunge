#pragma once

// Fast integer multiplication algorithms.
//
// (Karatsuba lives in number_theory.hpp as karatsuba_multiply; this header adds
// the faster Toom-Cook and Schoenhage-Strassen big-integer multipliers plus
// Booth's signed hardware multiplication.)
//
// Big non-negative integers are passed and returned as ordinary base-10 decimal
// strings (no sign, no leading zeros except the single "0"). Booth's algorithm
// operates on machine words and multiplies signed values.

#include <cstdint>
#include <string>

namespace datamunge::algorithms {

// Toom-Cook (Toom-3) multiplication: O(n^log3 5) ~ O(n^1.465) via five
// evaluate-multiply-interpolate products of third-size operands.
std::string toom3_mul(const std::string& a, const std::string& b);

// Schoenhage-Strassen-style multiplication via a number-theoretic transform:
// the product's digit convolution is computed by NTT in O(n log n).
std::string ntt_mul(const std::string& a, const std::string& b);

// Booth's multiplication: multiplies two signed two's-complement values by
// radix-2 recoding of the multiplier. Returns the exact 64-bit product.
struct BoothResult {
    std::int64_t product{0};
    int          additions{0};    // add steps triggered by 01 transitions
    int          subtractions{0}; // subtract steps triggered by 10 transitions
};
BoothResult booth_multiply(std::int32_t multiplicand, std::int32_t multiplier);

} // namespace datamunge::algorithms
