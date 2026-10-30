#pragma once

/// @file random.hpp
/// @brief std::format / std::print support for datamunge::random.
/// Included by <datamunge/format.hpp>; requires C++20 <format>.
///
/// The generators keep their seed and state private (most do not store the seed
/// at all), so each prints its engine name plus any configuration it was built
/// with (order, lags, modulus, LCG constants). Formatting never advances the
/// generator.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::random::LaggedFibonacci64(42));
/// // LaggedFibonacci64(lags=24,55)
/// @endcode

#include <datamunge/format/text_formatter.hpp>
#include <datamunge/random/random.hpp>

#include <format>
#include <string>

/// @brief Prints a SplitMix64 generator as "SplitMix64" (its seed and state are private).
template <>
struct std::formatter<datamunge::random::SplitMix64> : datamunge::format_detail::text_formatter<datamunge::random::SplitMix64> {
  /// @brief The engine name.
  /// @param rng The generator (not advanced).
  /// @return "SplitMix64".
  static std::string text([[maybe_unused]] const datamunge::random::SplitMix64& rng) { return "SplitMix64"; }
};

/// @brief Prints a ACORN generator as "Acorn64(order=k)" (its seed and state are private).
template <>
struct std::formatter<datamunge::random::Acorn64> : datamunge::format_detail::text_formatter<datamunge::random::Acorn64> {
  /// @brief The engine name and order.
  /// @param rng The generator (not advanced).
  /// @return E.g. "Acorn64(order=12)".
  static std::string text(const datamunge::random::Acorn64& rng) { return std::format("Acorn64(order={})", rng.order()); }
};

/// @brief Prints a Blum-Blum-Shub generator as "BlumBlumShub(modulus=M)" (its seed and state are private).
template <>
struct std::formatter<datamunge::random::BlumBlumShub> : datamunge::format_detail::text_formatter<datamunge::random::BlumBlumShub> {
  /// @brief The engine name and modulus p*q.
  /// @param rng The generator (not advanced).
  /// @return E.g. "BlumBlumShub(modulus=2001917005733)".
  static std::string text(const datamunge::random::BlumBlumShub& rng) {
    return std::format("BlumBlumShub(modulus={})", rng.modulus());
  }
};

/// @brief Prints a lagged-Fibonacci generator as "LaggedFibonacci64(lags=j,k)" (its seed and state are private).
template <>
struct std::formatter<datamunge::random::LaggedFibonacci64> : datamunge::format_detail::text_formatter<datamunge::random::LaggedFibonacci64> {
  /// @brief The engine name and its two lags.
  /// @param rng The generator (not advanced).
  /// @return E.g. "LaggedFibonacci64(lags=24,55)".
  static std::string text(const datamunge::random::LaggedFibonacci64& rng) {
    return std::format("LaggedFibonacci64(lags={},{})", rng.short_lag(), rng.long_lag());
  }
};

/// @brief Prints a 64-bit LCG as "LinearCongruential64(a=..., c=...)" (its seed and state are private).
template <>
struct std::formatter<datamunge::random::LinearCongruential64> : datamunge::format_detail::text_formatter<datamunge::random::LinearCongruential64> {
  /// @brief The engine name, multiplier and increment.
  /// @param rng The generator (not advanced).
  /// @return E.g. "LinearCongruential64(a=6364136223846793005, c=1442695040888963407)".
  static std::string text(const datamunge::random::LinearCongruential64& rng) {
    return std::format("LinearCongruential64(a={}, c={})", rng.multiplier(), rng.increment());
  }
};

/// @brief Prints a MT19937-64 Mersenne Twister generator as "MersenneTwister64" (its seed and state are private).
template <>
struct std::formatter<datamunge::random::MersenneTwister64> : datamunge::format_detail::text_formatter<datamunge::random::MersenneTwister64> {
  /// @brief The engine name.
  /// @param rng The generator (not advanced).
  /// @return "MersenneTwister64".
  static std::string text([[maybe_unused]] const datamunge::random::MersenneTwister64& rng) { return "MersenneTwister64"; }
};

/// @brief Prints a PCG32 generator as "Pcg32(increment=0x...)" (its seed and state are private).
template <>
struct std::formatter<datamunge::random::Pcg32> : datamunge::format_detail::text_formatter<datamunge::random::Pcg32> {
  /// @brief The engine name and its stream-selecting increment, in hex.
  /// @param rng The generator (not advanced).
  /// @return E.g. "Pcg32(increment=0xb47c73972972b7b7)".
  static std::string text(const datamunge::random::Pcg32& rng) { return std::format("Pcg32(increment={:#x})", rng.increment()); }
};

/// @brief Prints a xoroshiro128+ generator as "Xoroshiro128Plus", without displaying or advancing its state.
template <>
struct std::formatter<datamunge::random::Xoroshiro128Plus> : datamunge::format_detail::text_formatter<datamunge::random::Xoroshiro128Plus> {
  /// @brief The engine name.
  /// @param rng The generator (not advanced).
  /// @return "Xoroshiro128Plus".
  static std::string text([[maybe_unused]] const datamunge::random::Xoroshiro128Plus& rng) { return "Xoroshiro128Plus"; }
};

/// @brief Prints a xoshiro256** generator as "Xoshiro256StarStar", without displaying or advancing its state.
template <>
struct std::formatter<datamunge::random::Xoshiro256StarStar> : datamunge::format_detail::text_formatter<datamunge::random::Xoshiro256StarStar> {
  /// @brief The engine name.
  /// @param rng The generator (not advanced).
  /// @return "Xoshiro256StarStar".
  static std::string text([[maybe_unused]] const datamunge::random::Xoshiro256StarStar& rng) { return "Xoshiro256StarStar"; }
};

/// @brief Prints a SFC64 generator as "Sfc64" (its seed and state are private).
template <>
struct std::formatter<datamunge::random::Sfc64> : datamunge::format_detail::text_formatter<datamunge::random::Sfc64> {
  /// @brief The engine name.
  /// @param rng The generator (not advanced).
  /// @return "Sfc64".
  static std::string text([[maybe_unused]] const datamunge::random::Sfc64& rng) { return "Sfc64"; }
};

/// @brief Prints a ChaCha20 generator as "ChaCha20" (its seed and state are private).
template <>
struct std::formatter<datamunge::random::ChaCha20> : datamunge::format_detail::text_formatter<datamunge::random::ChaCha20> {
  /// @brief The engine name.
  /// @param rng The generator (not advanced).
  /// @return "ChaCha20".
  static std::string text([[maybe_unused]] const datamunge::random::ChaCha20& rng) { return "ChaCha20"; }
};
