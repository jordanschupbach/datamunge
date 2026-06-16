#include <datamunge/random/random.hpp>

#include <cmath>

namespace datamunge::random {

SplitMix64::SplitMix64(std::uint64_t seed) : state_(seed) {}

std::uint64_t SplitMix64::next_u64() {
  std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
  z               = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  z               = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31U);
}

double SplitMix64::uniform01() {
  return static_cast<double>(next_u64() >> 11U) * (1.0 / 9007199254740992.0);
}

double SplitMix64::normal() {
  if (has_spare_) {
    has_spare_ = false;
    return spare_;
  }

  double u1 = 0.0;
  while (u1 <= 0.0) {
    u1 = uniform01();
  }

  const double u2     = uniform01();
  const double radius = std::sqrt(-2.0 * std::log(u1));
  const double theta  = 6.28318530717958647692 * u2;

  spare_     = radius * std::sin(theta);
  has_spare_ = true;
  return radius * std::cos(theta);
}

} // namespace datamunge::random
