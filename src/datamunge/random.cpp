#include <datamunge/random/random.hpp>

#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace datamunge::random {
namespace {

constexpr std::uint64_t rotl(std::uint64_t value, int shift) {
  return (value << shift) | (value >> (64 - shift));
}

constexpr std::uint32_t rotl32(std::uint32_t value, int shift) {
  return (value << shift) | (value >> (32 - shift));
}

std::uint64_t splitmix64_step(std::uint64_t& state) {
  std::uint64_t value = (state += 0x9E3779B97F4A7C15ULL);
  value               = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value               = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

} // namespace

double RandomGenerator::uniform01() {
  return static_cast<double>(next_u64() >> 11U) * (1.0 / 9007199254740992.0);
}

double RandomGenerator::uniform(double lower, double upper) {
  if (!(std::isfinite(lower) && std::isfinite(upper) && upper > lower)) {
    throw std::invalid_argument("uniform: bounds must be finite and upper must be greater than lower");
  }
  return lower + (upper - lower) * uniform01();
}

std::uint64_t RandomGenerator::uniform_u64(std::uint64_t lower, std::uint64_t upper) {
  if (upper < lower) {
    throw std::invalid_argument("uniform_u64: upper must not be less than lower");
  }

  const std::uint64_t range = upper - lower + 1U;
  if (range == 0U) return next_u64();

  const std::uint64_t threshold = -range % range;
  std::uint64_t       value;
  do {
    value = next_u64();
  } while (value < threshold);
  return lower + value % range;
}

bool RandomGenerator::bernoulli(double probability) {
  if (!(probability >= 0.0 && probability <= 1.0)) {
    throw std::invalid_argument("bernoulli: probability must be in [0, 1]");
  }
  return uniform01() < probability;
}

double RandomGenerator::exponential(double rate) {
  if (!(std::isfinite(rate) && rate > 0.0)) {
    throw std::invalid_argument("exponential: rate must be finite and greater than zero");
  }
  return -std::log1p(-uniform01()) / rate;
}

double RandomGenerator::normal() {
  if (has_spare_) {
    has_spare_ = false;
    return spare_;
  }

  double u1 = 0.0;
  while (u1 <= 0.0) u1 = uniform01();

  const double u2     = uniform01();
  const double radius = std::sqrt(-2.0 * std::log(u1));
  const double theta  = 6.28318530717958647692 * u2;

  spare_     = radius * std::sin(theta);
  has_spare_ = true;
  return radius * std::cos(theta);
}

double RandomGenerator::normal(double mean, double standard_deviation) {
  if (!(std::isfinite(mean) && std::isfinite(standard_deviation) && standard_deviation > 0.0)) {
    throw std::invalid_argument("normal: mean must be finite and standard_deviation must be finite and greater than zero");
  }
  return mean + standard_deviation * normal();
}

SplitMix64::SplitMix64(std::uint64_t seed) : state_(seed) {}

std::uint64_t SplitMix64::next_u64() {
  return splitmix64_step(state_);
}

Acorn64::Acorn64(std::uint64_t seed, std::size_t order) {
  if (order == 0) {
    throw std::invalid_argument("Acorn64: order must be positive");
  }
  SplitMix64 seeder(seed);
  state_.resize(order + 1);
  for (auto& word : state_) word = seeder.next_u64();
  state_[0] |= 1ULL;
}

Acorn64::Acorn64(std::vector<std::uint64_t> state) : state_(std::move(state)) {
  if (state_.size() < 2) {
    throw std::invalid_argument("Acorn64: state must contain an increment and an output word");
  }
  if ((state_[0] & 1ULL) == 0) {
    throw std::invalid_argument("Acorn64: the first state word must be odd");
  }
}

std::uint64_t Acorn64::next_u64() {
  for (std::size_t i = 1; i < state_.size(); ++i) state_[i] += state_[i - 1];
  return state_.back();
}

BlumBlumShub::BlumBlumShub(std::uint64_t seed, std::uint64_t p, std::uint64_t q) {
  if (p < 3 || q < 3 || p == q || p % 4 != 3 || q % 4 != 3) {
    throw std::invalid_argument("BlumBlumShub: p and q must be distinct and congruent to 3 mod 4");
  }
  const auto product = static_cast<unsigned __int128>(p) * q;
  if (product > std::numeric_limits<std::uint64_t>::max()) {
    throw std::invalid_argument("BlumBlumShub: p*q does not fit in uint64_t");
  }
  modulus_ = static_cast<std::uint64_t>(product);
  if (std::gcd(seed, modulus_) != 1) {
    throw std::invalid_argument("BlumBlumShub: seed must be coprime to p*q");
  }
  state_ = static_cast<std::uint64_t>(
      (static_cast<unsigned __int128>(seed % modulus_) * (seed % modulus_)) % modulus_);
}

std::uint64_t BlumBlumShub::step() {
  state_ = static_cast<std::uint64_t>(
      (static_cast<unsigned __int128>(state_) * state_) % modulus_);
  return state_;
}

std::uint64_t BlumBlumShub::next_u64() {
  std::uint64_t word = 0;
  for (unsigned bit = 0; bit < 64; ++bit) word |= (step() & 1ULL) << bit;
  return word;
}

LaggedFibonacci64::LaggedFibonacci64(std::uint64_t seed, std::size_t short_lag,
                                     std::size_t long_lag)
    : short_lag_(short_lag) {
  if (short_lag == 0 || short_lag >= long_lag) {
    throw std::invalid_argument("LaggedFibonacci64: require 0 < short_lag < long_lag");
  }
  SplitMix64 seeder(seed);
  state_.resize(long_lag);
  for (auto& word : state_) word = seeder.next_u64();
}

std::uint64_t LaggedFibonacci64::next_u64() {
  const std::size_t long_lag = state_.size();
  const std::size_t short_index = (index_ + long_lag - short_lag_) % long_lag;
  state_[index_] += state_[short_index];
  const std::uint64_t result = state_[index_];
  index_ = (index_ + 1) % long_lag;
  return result;
}

LinearCongruential64::LinearCongruential64(std::uint64_t seed, std::uint64_t multiplier,
                                           std::uint64_t increment)
    : state_(seed), multiplier_(multiplier), increment_(increment) {
  if ((multiplier & 3ULL) != 1ULL || (increment & 1ULL) == 0) {
    throw std::invalid_argument(
        "LinearCongruential64: require multiplier = 1 mod 4 and an odd increment");
  }
}

std::uint64_t LinearCongruential64::next_u64() {
  state_ = multiplier_ * state_ + increment_;
  return state_;
}

MersenneTwister64::MersenneTwister64(std::uint64_t seed) {
  state_[0] = seed;
  for (std::size_t i = 1; i < state_.size(); ++i) {
    state_[i] = 6364136223846793005ULL * (state_[i - 1] ^ (state_[i - 1] >> 62U)) + i;
  }
}

void MersenneTwister64::twist() {
  constexpr std::uint64_t upper_mask = 0xFFFFFFFF80000000ULL;
  constexpr std::uint64_t lower_mask = 0x7FFFFFFFULL;
  constexpr std::uint64_t matrix_a   = 0xB5026F5AA96619E9ULL;
  for (std::size_t i = 0; i < state_.size(); ++i) {
    const std::uint64_t value = (state_[i] & upper_mask) | (state_[(i + 1) % state_.size()] & lower_mask);
    state_[i] = state_[(i + 156) % state_.size()] ^ (value >> 1U) ^ ((value & 1U) ? matrix_a : 0U);
  }
  index_ = 0;
}

std::uint64_t MersenneTwister64::next_u64() {
  if (index_ == state_.size()) twist();
  std::uint64_t value = state_[index_++];
  value ^= (value >> 29U) & 0x5555555555555555ULL;
  value ^= (value << 17U) & 0x71D67FFFEDA60000ULL;
  value ^= (value << 37U) & 0xFFF7EEE000000000ULL;
  return value ^ (value >> 43U);
}

Pcg32::Pcg32(std::uint64_t seed, std::uint64_t sequence) : increment_((sequence << 1U) | 1U) {
  next_u32();
  state_ += seed;
  next_u32();
}

std::uint32_t Pcg32::next_u32() {
  const std::uint64_t old_state = state_;
  state_ = old_state * 6364136223846793005ULL + increment_;
  const std::uint32_t xorshifted = static_cast<std::uint32_t>(((old_state >> 18U) ^ old_state) >> 27U);
  const std::uint32_t rotation = old_state >> 59U;
  return (xorshifted >> rotation) | (xorshifted << ((-rotation) & 31U));
}

std::uint64_t Pcg32::next_u64() {
  return (static_cast<std::uint64_t>(next_u32()) << 32U) | next_u32();
}

Xoroshiro128Plus::Xoroshiro128Plus(std::uint64_t seed) {
  state0_ = splitmix64_step(seed);
  state1_ = splitmix64_step(seed);
}

std::uint64_t Xoroshiro128Plus::next_u64() {
  const std::uint64_t result = state0_ + state1_;
  state1_ ^= state0_;
  state0_ = rotl(state0_, 24) ^ state1_ ^ (state1_ << 16U);
  state1_ = rotl(state1_, 37);
  return result;
}

Xoshiro256StarStar::Xoshiro256StarStar(std::uint64_t seed) {
  for (auto& value : state_) value = splitmix64_step(seed);
}

std::uint64_t Xoshiro256StarStar::next_u64() {
  const std::uint64_t result = rotl(state_[1] * 5ULL, 7) * 9ULL;
  const std::uint64_t t      = state_[1] << 17U;

  state_[2] ^= state_[0];
  state_[3] ^= state_[1];
  state_[1] ^= state_[2];
  state_[0] ^= state_[3];
  state_[2] ^= t;
  state_[3] = rotl(state_[3], 45);
  return result;
}

Sfc64::Sfc64(std::uint64_t seed) {
  state0_ = splitmix64_step(seed);
  state1_ = splitmix64_step(seed);
  state2_ = splitmix64_step(seed);
}

std::uint64_t Sfc64::next_u64() {
  const std::uint64_t result = state0_ + state1_ + counter_++;
  state0_ = state1_ ^ (state1_ >> 11U);
  state1_ = state2_ + (state2_ << 3U);
  state2_ = rotl(state2_, 24) + result;
  return result;
}

ChaCha20::ChaCha20(std::uint64_t seed) {
  std::array<std::uint32_t, 8> key{};
  for (std::size_t i = 0; i < key.size(); i += 2) {
    const std::uint64_t word = splitmix64_step(seed);
    key[i] = static_cast<std::uint32_t>(word);
    key[i + 1] = static_cast<std::uint32_t>(word >> 32U);
  }
  state_ = {0x61707865U, 0x3320646eU, 0x79622d32U, 0x6b206574U,
            key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
            0U, 0U, 0U, 0U};
}

ChaCha20::ChaCha20(const std::array<std::uint32_t, 8>& key,
                   const std::array<std::uint32_t, 3>& nonce,
                   std::uint32_t initial_counter)
    : state_{0x61707865U, 0x3320646eU, 0x79622d32U, 0x6b206574U,
             key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
             initial_counter, nonce[0], nonce[1], nonce[2]} {}

void ChaCha20::refill() {
  auto work = state_;
  const auto quarter_round = [](std::uint32_t& a, std::uint32_t& b, std::uint32_t& c, std::uint32_t& d) {
    a += b; d ^= a; d = rotl32(d, 16);
    c += d; b ^= c; b = rotl32(b, 12);
    a += b; d ^= a; d = rotl32(d, 8);
    c += d; b ^= c; b = rotl32(b, 7);
  };
  for (int round = 0; round < 10; ++round) {
    quarter_round(work[0], work[4], work[8], work[12]);
    quarter_round(work[1], work[5], work[9], work[13]);
    quarter_round(work[2], work[6], work[10], work[14]);
    quarter_round(work[3], work[7], work[11], work[15]);
    quarter_round(work[0], work[5], work[10], work[15]);
    quarter_round(work[1], work[6], work[11], work[12]);
    quarter_round(work[2], work[7], work[8], work[13]);
    quarter_round(work[3], work[4], work[9], work[14]);
  }
  for (std::size_t i = 0; i < work.size(); ++i) buffer_[i] = work[i] + state_[i];
  if (++state_[12] == 0U) ++state_[13];
  buffer_index_ = 0;
}

std::uint64_t ChaCha20::next_u64() {
  if (buffer_index_ == buffer_.size()) refill();
  const std::uint64_t low = buffer_[buffer_index_++];
  const std::uint64_t high = buffer_[buffer_index_++];
  return low | (high << 32U);
}

} // namespace datamunge::random
