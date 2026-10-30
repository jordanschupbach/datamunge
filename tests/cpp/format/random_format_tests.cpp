// std::format / std::print support for datamunge::random types.
#include <datamunge/format/random.hpp>

#include <gtest/gtest.h>

#include <format>
#include <string>

namespace rnd = datamunge::random;

static_assert(std::formattable<rnd::SplitMix64, char>);
static_assert(std::formattable<rnd::Acorn64, char>);
static_assert(std::formattable<rnd::BlumBlumShub, char>);
static_assert(std::formattable<rnd::LaggedFibonacci64, char>);
static_assert(std::formattable<rnd::LinearCongruential64, char>);
static_assert(std::formattable<rnd::MersenneTwister64, char>);
static_assert(std::formattable<rnd::Pcg32, char>);
static_assert(std::formattable<rnd::Xoroshiro128Plus, char>);
static_assert(std::formattable<rnd::Xoshiro256StarStar, char>);
static_assert(std::formattable<rnd::Sfc64, char>);
static_assert(std::formattable<rnd::ChaCha20, char>);

TEST(RandomFormat, GeneratorsPrintTheirEngineName) {
  EXPECT_EQ(std::format("{}", rnd::SplitMix64(1)), "SplitMix64");
  EXPECT_EQ(std::format("{}", rnd::Acorn64(1)), "Acorn64(order=12)");
  EXPECT_EQ(std::format("{}", rnd::Acorn64(1, 5)), "Acorn64(order=5)");
  EXPECT_EQ(std::format("{}", rnd::BlumBlumShub(1)), "BlumBlumShub(modulus=2001917005733)");
  EXPECT_EQ(std::format("{}", rnd::BlumBlumShub(1, 7, 11)), "BlumBlumShub(modulus=77)");
  EXPECT_EQ(std::format("{}", rnd::LaggedFibonacci64(1)), "LaggedFibonacci64(lags=24,55)");
  EXPECT_EQ(std::format("{}", rnd::LaggedFibonacci64(1, 5, 17)), "LaggedFibonacci64(lags=5,17)");
  EXPECT_EQ(std::format("{}", rnd::LinearCongruential64(1)),
            "LinearCongruential64(a=6364136223846793005, c=1442695040888963407)");
  EXPECT_EQ(std::format("{}", rnd::LinearCongruential64(1, 5, 3)), "LinearCongruential64(a=5, c=3)");
  EXPECT_EQ(std::format("{}", rnd::Pcg32(1, 0)), "Pcg32(increment=0x1)");
  EXPECT_EQ(std::format("{}", rnd::Pcg32(1, 3)), "Pcg32(increment=0x7)");
  EXPECT_EQ(std::format("{}", rnd::MersenneTwister64(1)), "MersenneTwister64");
  EXPECT_EQ(std::format("{}", rnd::Xoroshiro128Plus(1)), "Xoroshiro128Plus");
  EXPECT_EQ(std::format("{}", rnd::Xoshiro256StarStar(1)), "Xoshiro256StarStar");
  EXPECT_EQ(std::format("{}", rnd::Sfc64(1)), "Sfc64");
  EXPECT_EQ(std::format("{}", rnd::ChaCha20(1)), "ChaCha20");
  EXPECT_EQ(std::format("{:>8}", rnd::Sfc64(1)), "   Sfc64");
}

TEST(RandomFormat, FormattingDoesNotAdvanceTheGenerator) {
  rnd::Pcg32 a(42), b(42);
  EXPECT_EQ(std::format("{}", a), std::format("Pcg32(increment={:#x})", a.increment()));
  EXPECT_EQ(a.next_u64(), b.next_u64());
}
