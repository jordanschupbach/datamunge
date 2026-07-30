#include <gtest/gtest.h>

#include <datamunge/algorithms/check_digits.hpp>
#include <datamunge/algorithms/checksums.hpp>
#include <datamunge/algorithms/hamming_code.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <utility>

using datamunge::algorithms::adler32;
using datamunge::algorithms::crc32;
using datamunge::algorithms::fletcher16;
using datamunge::algorithms::hamming74_decode;
using datamunge::algorithms::hamming74_encode;
using datamunge::algorithms::luhn_check_digit;
using datamunge::algorithms::luhn_validate;
using datamunge::algorithms::verhoeff_check_digit;
using datamunge::algorithms::verhoeff_validate;

TEST(Checksums, MatchReferenceValues) {
    EXPECT_EQ(crc32("123456789"), 0xCBF43926u);
    EXPECT_EQ(adler32("Wikipedia"), 0x11E60398u);
    EXPECT_EQ(fletcher16("abcde"), 0xC8F0u);
    // Different data -> different checksum (detection).
    EXPECT_NE(crc32("123456789"), crc32("123456780"));
}

TEST(Hamming74, CorrectsEverySingleBitError) {
    for (int v = 0; v < 16; ++v) {
        std::array<int, 4> data = {(v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1};
        auto               code = hamming74_encode(data);
        // No error: decodes cleanly.
        auto clean = hamming74_decode(code);
        EXPECT_EQ(clean.data, data);
        EXPECT_FALSE(clean.corrected);
        // Flip each of the 7 positions: the syndrome locates it and it is corrected.
        for (int pos = 0; pos < 7; ++pos) {
            auto corrupted = code;
            corrupted[pos] ^= 1;
            auto dec = hamming74_decode(corrupted);
            EXPECT_TRUE(dec.corrected);
            EXPECT_EQ(dec.error_position, pos + 1);
            EXPECT_EQ(dec.data, data) << "value " << v << " pos " << pos;
        }
    }
}

TEST(Luhn, ValidatesAndComputesCheckDigit) {
    EXPECT_TRUE(luhn_validate("79927398713"));
    EXPECT_FALSE(luhn_validate("79927398710"));
    EXPECT_EQ(luhn_check_digit("7992739871"), 3);
    // Catches a single-digit error.
    EXPECT_FALSE(luhn_validate("79927398723"));
}

TEST(Verhoeff, CatchesAllTranspositionsLuhnMisses) {
    EXPECT_EQ(verhoeff_check_digit("236"), 3);
    EXPECT_TRUE(verhoeff_validate("2363"));
    EXPECT_FALSE(verhoeff_validate("2364"));

    // Verhoeff catches every adjacent transposition, including 09<->90 which Luhn misses.
    // Build a valid number ending in ...09 vs ...90 and check Verhoeff rejects the swap.
    std::string base = "1090";  // arbitrary payload containing 09
    std::string full = base + std::to_string(verhoeff_check_digit(base));
    EXPECT_TRUE(verhoeff_validate(full));
    // Transpose the "09" -> "90" in the payload region; Verhoeff must reject.
    std::string swapped = full;
    std::swap(swapped[1], swapped[2]);  // "1090x" -> "1900x"... transpose two adjacent payload digits
    if (swapped != full) EXPECT_FALSE(verhoeff_validate(swapped));
}
