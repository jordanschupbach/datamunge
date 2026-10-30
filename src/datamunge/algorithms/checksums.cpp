/// @file checksums.cpp
/// @brief Adler-32: incremental update, combination, rolling windows and a collision model.

#include <datamunge/algorithms/checksums.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace datamunge::algorithms {

std::uint32_t adler32_update(std::uint32_t adler, std::string_view data) {
  std::uint32_t a = adler & 0xFFFFU, b = adler >> 16;
  const auto* byte = reinterpret_cast<const unsigned char*>(data.data());
  std::size_t left = data.size();
  while (left > 0) {
    const std::size_t run = std::min(left, adler32_nmax);
    for (std::size_t i = 0; i < run; ++i) {
      a += byte[i];
      b += a;
    }
    a %= adler32_modulus;
    b %= adler32_modulus;
    byte += run;
    left -= run;
  }
  return (b << 16) | a;
}

std::uint32_t adler32_combine(std::uint32_t adler1, std::uint32_t adler2, std::uint64_t length2) {
  constexpr std::uint64_t p = adler32_modulus;
  const std::uint64_t a1 = adler1 & 0xFFFFU, b1 = adler1 >> 16;
  const std::uint64_t a2 = adler2 & 0xFFFFU, b2 = adler2 >> 16;
  const std::uint64_t m = length2 % p;
  const std::uint64_t a = (a1 + a2 + p - 1) % p;
  const std::uint64_t b = (b1 + b2 + m * ((a1 + p - 1) % p)) % p;
  return static_cast<std::uint32_t>((b << 16) | a);
}

std::uint32_t adler32_roll(std::uint32_t adler, std::size_t window, unsigned char out, unsigned char in) {
  constexpr std::uint64_t p = adler32_modulus;
  const std::uint64_t a = adler & 0xFFFFU, b = adler >> 16;
  const std::uint64_t a_new = (a + p - out + in) % p;
  const std::uint64_t drop = (window % p) * out % p;
  const std::uint64_t b_new = (b + 2 * p - drop + a_new - 1) % p;  // a_new - 1 >= -1, so +2p is enough
  return static_cast<std::uint32_t>((b_new << 16) | a_new);
}

std::vector<std::uint32_t> adler32_windows(std::string_view data, std::size_t window) {
  if (window == 0) throw std::invalid_argument("adler32_windows: window must be at least 1");
  std::vector<std::uint32_t> sums;
  if (data.size() < window) return sums;
  sums.reserve(data.size() - window + 1);
  sums.push_back(adler32(data.substr(0, window)));
  for (std::size_t s = 1; s + window <= data.size(); ++s)
    sums.push_back(adler32_roll(sums.back(), window, static_cast<unsigned char>(data[s - 1]),
                                static_cast<unsigned char>(data[s + window - 1])));
  return sums;
}

/// @brief Helpers for the Adler-32 collision model.
namespace {

/// @brief P(X is a multiple of 65521) for integer-valued X ~ N(0, variance), as a wrapped
///        normal: sums the density over the multiples directly, or uses Poisson summation once
///        the spread exceeds the modulus.
double multiple_of_modulus_probability(double variance) {
  constexpr double p = adler32_modulus;
  const double sd = std::sqrt(variance);
  if (sd < p) {
    double total = 0.0;
    for (int k = -6; k <= 6; ++k) total += std::exp(-0.5 * (k * p) * (k * p) / variance);
    return total / (sd * std::sqrt(2.0 * std::numbers::pi));
  }
  double total = 1.0;
  for (int m = 1; m <= 6; ++m) total += 2.0 * std::exp(-2.0 * std::numbers::pi * std::numbers::pi * m * m * variance / (p * p));
  return total / p;
}

} // namespace

double adler32_collision_probability(std::uint64_t length, double byte_variance) {
  if (length < 3) throw std::invalid_argument("adler32_collision_probability: length must be at least 3");
  if (!(byte_variance > 0.0)) throw std::invalid_argument("adler32_collision_probability: byte_variance must be positive");
  const double n = static_cast<double>(length), v = byte_variance;
  const double var_a = n * v;
  const double var_b = v * n * (n + 1.0) * (2.0 * n + 1.0) / 6.0;
  const double cov = v * n * (n + 1.0) / 2.0;
  const double var_b_given_a = var_b - cov * cov / var_a;
  // Differences of two independent messages: twice the variances.
  return multiple_of_modulus_probability(2.0 * var_a) * multiple_of_modulus_probability(2.0 * var_b_given_a);
}

} // namespace datamunge::algorithms
