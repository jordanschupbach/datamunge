#include <gtest/gtest.h>

#include <datamunge/algorithms/multiplication.hpp>
#include <datamunge/algorithms/number_theory.hpp> // karatsuba_multiply reference

#include <cstdint>
#include <random>
#include <string>

using namespace datamunge::algorithms;

namespace {

std::string trim_zeros(std::string s) {
    std::size_t i = 0;
    while (i + 1 < s.size() && s[i] == '0') ++i;
    return s.substr(i);
}

std::string u128_to_string(__uint128_t v) {
    if (v == 0) return "0";
    std::string s;
    while (v > 0) {
        s.push_back(static_cast<char>('0' + static_cast<int>(v % 10)));
        v /= 10;
    }
    return {s.rbegin(), s.rend()};
}

std::string random_digits(std::mt19937_64& rng, int len) {
    std::string s;
    s.reserve(len);
    s.push_back(static_cast<char>('1' + rng() % 9)); // no leading zero
    for (int i = 1; i < len; ++i) s.push_back(static_cast<char>('0' + rng() % 10));
    return s;
}

} // namespace

TEST(Multiplication, SmallAgainstInt128) {
    std::mt19937_64 rng(1);
    for (int t = 0; t < 20000; ++t) {
        const std::uint64_t a = rng() % 1000000000ULL;
        const std::uint64_t b = rng() % 1000000000ULL;
        const std::string   want = u128_to_string(static_cast<__uint128_t>(a) * b);
        const std::string   sa = std::to_string(a), sb = std::to_string(b);
        EXPECT_EQ(trim_zeros(toom3_mul(sa, sb)), want) << a << "*" << b;
        EXPECT_EQ(trim_zeros(ntt_mul(sa, sb)), want) << a << "*" << b;
    }
}

TEST(Multiplication, BigOperandsCrossCheck) {
    std::mt19937_64 rng(2);
    for (int t = 0; t < 200; ++t) {
        const int         la = 1 + static_cast<int>(rng() % 600);
        const int         lb = 1 + static_cast<int>(rng() % 600);
        const std::string a = random_digits(rng, la);
        const std::string b = random_digits(rng, lb);
        const std::string ref = karatsuba_multiply(a, b); // known-good reference
        EXPECT_EQ(toom3_mul(a, b), ref) << "len " << la << "," << lb;
        EXPECT_EQ(ntt_mul(a, b), ref) << "len " << la << "," << lb;
    }
}

TEST(Multiplication, KnownProducts) {
    EXPECT_EQ(trim_zeros(toom3_mul("0", "12345")), "0");
    EXPECT_EQ(trim_zeros(ntt_mul("12345", "0")), "0");
    EXPECT_EQ(toom3_mul("999999999999", "999999999999"), "999999999998000000000001");
    EXPECT_EQ(ntt_mul("999999999999", "999999999999"), "999999999998000000000001");
    // Repunit squares are base-10 palindromes.
    EXPECT_EQ(toom3_mul("11111", "11111"), "123454321");
    EXPECT_EQ(ntt_mul("1111111", "1111111"), "1234567654321");
}

TEST(Multiplication, BoothMatchesBuiltin) {
    std::mt19937_64 rng(3);
    for (int t = 0; t < 200000; ++t) {
        const std::int32_t a = static_cast<std::int32_t>(rng());
        const std::int32_t b = static_cast<std::int32_t>(rng());
        const auto         r = booth_multiply(a, b);
        EXPECT_EQ(r.product, static_cast<std::int64_t>(a) * static_cast<std::int64_t>(b))
            << a << "*" << b;
    }
    // Signed corners.
    EXPECT_EQ(booth_multiply(-7, 6).product, -42);
    EXPECT_EQ(booth_multiply(-7, -6).product, 42);
    EXPECT_EQ(booth_multiply(123456, -789).product, -97406784);
    EXPECT_EQ(booth_multiply(0, 12345).product, 0);
    // Booth recodes runs of 1s: multiplier 15 (0b1111) is one add + one subtract.
    const auto r15 = booth_multiply(1, 15);
    EXPECT_EQ(r15.product, 15);
    EXPECT_EQ(r15.additions, 1);
    EXPECT_EQ(r15.subtractions, 1);
}
