#pragma once

/// @file mt19937.hpp
/// @brief Binding-friendly standard-library random streams for the teaching examples.
/// These engines are separate from Generator's xoshiro stream. Distribution and
/// shuffle sequences follow the active C++ standard library implementation.
#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace datamunge::nn {
/// @brief Mutable std::mt19937 stream with independent float and double normal caches.
/// Instances are unsynchronized; callers serialize access if shared across threads.
class Mt19937 {
 public:
  /// @brief Seeds the 32-bit Mersenne Twister and starts both normal caches empty.
  explicit Mt19937(std::uint32_t seed = 0)
      : engine_(seed), normal32_(0.0f, 1.0f), normal64_(0.0, 1.0) {}

  /// @brief Draws from a float-precision normal distribution, widening its result to double.
  /// @pre Parameters remain valid after float conversion.
  /// @throws std::invalid_argument If stddev is not positive.
  double normal32(double mean = 0.0, double stddev = 1.0) {
    if (!(stddev > 0.0)) throw std::invalid_argument("normal32: positive standard deviation required");
    return normal32_(engine_, decltype(normal32_)::param_type(static_cast<float>(mean), static_cast<float>(stddev)));
  }
  /// @brief Draws a double-precision normal sample using the persistent double normal cache.
  /// @throws std::invalid_argument If stddev is not positive.
  double normal64(double mean = 0.0, double stddev = 1.0) {
    if (!(stddev > 0.0)) throw std::invalid_argument("normal64: positive standard deviation required");
    return normal64_(engine_, decltype(normal64_)::param_type(mean, stddev));
  }
  /// @brief Draws a float-precision uniform value in [low, high), widening the result to double.
  /// @pre Bounds are finite and remain ordered after float conversion.
  /// @throws std::invalid_argument If high does not exceed low.
  double uniform32(double low = 0.0, double high = 1.0) {
    if (!(high > low)) throw std::invalid_argument("uniform32: high must exceed low");
    return std::uniform_real_distribution<float>(static_cast<float>(low), static_cast<float>(high))(engine_);
  }
  /// @brief Draws a double-precision uniform value in [low, high) from the shared engine.
  /// @pre Bounds are finite.
  /// @throws std::invalid_argument If high does not exceed low.
  double uniform64(double low = 0.0, double high = 1.0) {
    if (!(high > low)) throw std::invalid_argument("uniform64: high must exceed low");
    return std::uniform_real_distribution<double>(low, high)(engine_);
  }
  // Bulk draws keep scalar stream/cache semantics while avoiding one FFI call
  // per element in statistical examples.
  /// @brief Returns count successive normal32() draws, retaining scalar stream and cache order.
  /// @throws std::invalid_argument For a negative count or a nonpositive deviation.
  std::vector<float> normal32_values(std::int64_t count, double mean = 0.0, double stddev = 1.0) {
    if (count < 0 || !(stddev > 0.0)) throw std::invalid_argument("normal32_values: nonnegative count and positive stddev required");
    std::vector<float> values(static_cast<std::size_t>(count));
    for (auto& value : values) value = static_cast<float>(normal32(mean, stddev));
    return values;
  }
  /// @brief Returns count successive normal64() draws, retaining scalar stream and cache order.
  /// @throws std::invalid_argument For a negative count or a nonpositive deviation.
  std::vector<double> normal64_values(std::int64_t count, double mean = 0.0, double stddev = 1.0) {
    if (count < 0 || !(stddev > 0.0)) throw std::invalid_argument("normal64_values: nonnegative count and positive stddev required");
    std::vector<double> values(static_cast<std::size_t>(count));
    for (auto& value : values) value = normal64(mean, stddev);
    return values;
  }
  /// @brief Shuffles a by-value vector with the shared engine and returns the shuffled copy.
  std::vector<std::int64_t> shuffle(std::vector<std::int64_t> values) {
    std::shuffle(values.begin(), values.end(), engine_);
    return values;
  }
  /// @brief Returns one raw engine draw modulo divisor, preserving example stream semantics.
  /// The remainder can be biased; this is not a rejection-sampled uniform integer draw.
  /// @throws std::invalid_argument If divisor is nonpositive.
  std::int64_t modulo(std::int64_t divisor) {
    if (divisor <= 0) throw std::invalid_argument("modulo: positive divisor required");
    return static_cast<std::int64_t>(engine_() % static_cast<std::uint64_t>(divisor));
  }
  /// @brief Discards both normal spare caches without reseeding or rewinding the raw engine.
  void reset_normal() { normal32_.reset(); normal64_.reset(); }

 private:
  /// @brief Shared raw 32-bit engine advanced by every distribution, shuffle, and modulo draw.
  std::mt19937 engine_;
  /// @brief Stateful float normal sampler, retaining its own cached spare draw.
  std::normal_distribution<float> normal32_;
  /// @brief Stateful double normal sampler, retaining a cache separate from normal32_.
  std::normal_distribution<double> normal64_;
};
/// @brief Mutable std::mt19937_64 stream with float/double normal caches and a distinct 64-bit sequence.
/// Instances are unsynchronized; the normal and shuffle algorithms follow the active standard library.
class Mt19937_64 {
 public:
  /// @brief Seeds the 64-bit Mersenne Twister and starts both normal caches empty.
  explicit Mt19937_64(std::uint64_t seed = 0)
      : engine_(seed), normal32_(0.0f, 1.0f), normal64_(0.0, 1.0) {}

  /// @brief Draws from a float-precision normal distribution, widening its result to double.
  /// @pre Parameters remain valid after float conversion.
  /// @throws std::invalid_argument If stddev is not positive.
  double normal32(double mean = 0.0, double stddev = 1.0) {
    if (!(stddev > 0.0)) throw std::invalid_argument("normal32: positive standard deviation required");
    return normal32_(engine_, decltype(normal32_)::param_type(static_cast<float>(mean), static_cast<float>(stddev)));
  }
  /// @brief Draws a double-precision normal sample using the persistent double normal cache.
  /// @throws std::invalid_argument If stddev is not positive.
  double normal64(double mean = 0.0, double stddev = 1.0) {
    if (!(stddev > 0.0)) throw std::invalid_argument("normal64: positive standard deviation required");
    return normal64_(engine_, decltype(normal64_)::param_type(mean, stddev));
  }
  /// @brief Draws a float-precision uniform value in [low, high), widening the result to double.
  /// @pre Bounds are finite and remain ordered after float conversion.
  /// @throws std::invalid_argument If high does not exceed low.
  double uniform32(double low = 0.0, double high = 1.0) {
    if (!(high > low)) throw std::invalid_argument("uniform32: high must exceed low");
    return std::uniform_real_distribution<float>(static_cast<float>(low), static_cast<float>(high))(engine_);
  }
  /// @brief Draws a double-precision uniform value in [low, high) from the shared engine.
  /// @pre Bounds are finite.
  /// @throws std::invalid_argument If high does not exceed low.
  double uniform64(double low = 0.0, double high = 1.0) {
    if (!(high > low)) throw std::invalid_argument("uniform64: high must exceed low");
    return std::uniform_real_distribution<double>(low, high)(engine_);
  }
  // Bulk draws keep scalar stream/cache semantics while avoiding one FFI call
  // per element in statistical examples.
  /// @brief Returns count successive normal32() draws, retaining scalar stream and cache order.
  /// @throws std::invalid_argument For a negative count or a nonpositive deviation.
  std::vector<float> normal32_values(std::int64_t count, double mean = 0.0, double stddev = 1.0) {
    if (count < 0 || !(stddev > 0.0)) throw std::invalid_argument("normal32_values: nonnegative count and positive stddev required");
    std::vector<float> values(static_cast<std::size_t>(count));
    for (auto& value : values) value = static_cast<float>(normal32(mean, stddev));
    return values;
  }
  /// @brief Returns count successive normal64() draws, retaining scalar stream and cache order.
  /// @throws std::invalid_argument For a negative count or a nonpositive deviation.
  std::vector<double> normal64_values(std::int64_t count, double mean = 0.0, double stddev = 1.0) {
    if (count < 0 || !(stddev > 0.0)) throw std::invalid_argument("normal64_values: nonnegative count and positive stddev required");
    std::vector<double> values(static_cast<std::size_t>(count));
    for (auto& value : values) value = normal64(mean, stddev);
    return values;
  }
  /// @brief Shuffles a by-value vector with the shared engine and returns the shuffled copy.
  std::vector<std::int64_t> shuffle(std::vector<std::int64_t> values) {
    std::shuffle(values.begin(), values.end(), engine_);
    return values;
  }
  /// @brief Returns one raw engine draw modulo divisor, preserving example stream semantics.
  /// The remainder can be biased; this is not a rejection-sampled uniform integer draw.
  /// @throws std::invalid_argument If divisor is nonpositive.
  std::int64_t modulo(std::int64_t divisor) {
    if (divisor <= 0) throw std::invalid_argument("modulo: positive divisor required");
    return static_cast<std::int64_t>(engine_() % static_cast<std::uint64_t>(divisor));
  }
  /// @brief Discards both normal spare caches without reseeding or rewinding the raw engine.
  void reset_normal() { normal32_.reset(); normal64_.reset(); }

 private:
  /// @brief Shared raw 64-bit engine advanced by every distribution, shuffle, and modulo draw.
  std::mt19937_64 engine_;
  /// @brief Float normal distribution with a cached spare independent of normal64_.
  std::normal_distribution<float> normal32_;
  /// @brief Double normal distribution with its own cached spare draw.
  std::normal_distribution<double> normal64_;
};
}  // namespace datamunge::nn
