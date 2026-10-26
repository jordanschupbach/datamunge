#pragma once

// Binding-friendly access to the standard-library random streams used by the
// teaching examples. This is separate from Generator: initializers and ordinary
// NN random operations retain their existing xoshiro stream and API.
#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace datamunge::nn {
class Mt19937 {
 public:
  explicit Mt19937(std::uint32_t seed = 0)
      : engine_(seed), normal32_(0.0f, 1.0f), normal64_(0.0, 1.0) {}

  double normal32(double mean = 0.0, double stddev = 1.0) {
    if (!(stddev > 0.0)) throw std::invalid_argument("normal32: positive standard deviation required");
    return normal32_(engine_, decltype(normal32_)::param_type(static_cast<float>(mean), static_cast<float>(stddev)));
  }
  double normal64(double mean = 0.0, double stddev = 1.0) {
    if (!(stddev > 0.0)) throw std::invalid_argument("normal64: positive standard deviation required");
    return normal64_(engine_, decltype(normal64_)::param_type(mean, stddev));
  }
  double uniform32(double low = 0.0, double high = 1.0) {
    if (!(high > low)) throw std::invalid_argument("uniform32: high must exceed low");
    return std::uniform_real_distribution<float>(static_cast<float>(low), static_cast<float>(high))(engine_);
  }
  double uniform64(double low = 0.0, double high = 1.0) {
    if (!(high > low)) throw std::invalid_argument("uniform64: high must exceed low");
    return std::uniform_real_distribution<double>(low, high)(engine_);
  }
  // Bulk draws keep scalar stream/cache semantics while avoiding one FFI call
  // per element in statistical examples.
  std::vector<float> normal32_values(std::int64_t count, double mean = 0.0, double stddev = 1.0) {
    if (count < 0 || !(stddev > 0.0)) throw std::invalid_argument("normal32_values: nonnegative count and positive stddev required");
    std::vector<float> values(static_cast<std::size_t>(count));
    for (auto& value : values) value = static_cast<float>(normal32(mean, stddev));
    return values;
  }
  std::vector<double> normal64_values(std::int64_t count, double mean = 0.0, double stddev = 1.0) {
    if (count < 0 || !(stddev > 0.0)) throw std::invalid_argument("normal64_values: nonnegative count and positive stddev required");
    std::vector<double> values(static_cast<std::size_t>(count));
    for (auto& value : values) value = normal64(mean, stddev);
    return values;
  }
  std::vector<std::int64_t> shuffle(std::vector<std::int64_t> values) {
    std::shuffle(values.begin(), values.end(), engine_);
    return values;
  }
  std::int64_t modulo(std::int64_t divisor) {
    if (divisor <= 0) throw std::invalid_argument("modulo: positive divisor required");
    return static_cast<std::int64_t>(engine_() % static_cast<std::uint64_t>(divisor));
  }
  void reset_normal() { normal32_.reset(); normal64_.reset(); }

 private:
  std::mt19937 engine_;
  std::normal_distribution<float> normal32_;
  std::normal_distribution<double> normal64_;
};
// The 64-bit standard engine has a distinct stream and shuffle sequence.
class Mt19937_64 {
 public:
  explicit Mt19937_64(std::uint64_t seed = 0)
      : engine_(seed), normal32_(0.0f, 1.0f), normal64_(0.0, 1.0) {}

  double normal32(double mean = 0.0, double stddev = 1.0) {
    if (!(stddev > 0.0)) throw std::invalid_argument("normal32: positive standard deviation required");
    return normal32_(engine_, decltype(normal32_)::param_type(static_cast<float>(mean), static_cast<float>(stddev)));
  }
  double normal64(double mean = 0.0, double stddev = 1.0) {
    if (!(stddev > 0.0)) throw std::invalid_argument("normal64: positive standard deviation required");
    return normal64_(engine_, decltype(normal64_)::param_type(mean, stddev));
  }
  double uniform32(double low = 0.0, double high = 1.0) {
    if (!(high > low)) throw std::invalid_argument("uniform32: high must exceed low");
    return std::uniform_real_distribution<float>(static_cast<float>(low), static_cast<float>(high))(engine_);
  }
  double uniform64(double low = 0.0, double high = 1.0) {
    if (!(high > low)) throw std::invalid_argument("uniform64: high must exceed low");
    return std::uniform_real_distribution<double>(low, high)(engine_);
  }
  // Bulk draws keep scalar stream/cache semantics while avoiding one FFI call
  // per element in statistical examples.
  std::vector<float> normal32_values(std::int64_t count, double mean = 0.0, double stddev = 1.0) {
    if (count < 0 || !(stddev > 0.0)) throw std::invalid_argument("normal32_values: nonnegative count and positive stddev required");
    std::vector<float> values(static_cast<std::size_t>(count));
    for (auto& value : values) value = static_cast<float>(normal32(mean, stddev));
    return values;
  }
  std::vector<double> normal64_values(std::int64_t count, double mean = 0.0, double stddev = 1.0) {
    if (count < 0 || !(stddev > 0.0)) throw std::invalid_argument("normal64_values: nonnegative count and positive stddev required");
    std::vector<double> values(static_cast<std::size_t>(count));
    for (auto& value : values) value = normal64(mean, stddev);
    return values;
  }
  std::vector<std::int64_t> shuffle(std::vector<std::int64_t> values) {
    std::shuffle(values.begin(), values.end(), engine_);
    return values;
  }
  std::int64_t modulo(std::int64_t divisor) {
    if (divisor <= 0) throw std::invalid_argument("modulo: positive divisor required");
    return static_cast<std::int64_t>(engine_() % static_cast<std::uint64_t>(divisor));
  }
  void reset_normal() { normal32_.reset(); normal64_.reset(); }

 private:
  std::mt19937_64 engine_;
  std::normal_distribution<float> normal32_;
  std::normal_distribution<double> normal64_;
};
}  // namespace datamunge::nn
