/// @file vector.cpp
/// @brief Element-wise vector math, summary statistics, and the shared random sampling engine.

#include <datamunge/random/random.hpp>
#include <datamunge/vector.hpp>

#include <charconv>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>

namespace datamunge {

std::string vector_detail::scalar_text(double value) {
  if (std::isnan(value)) return "NaN";
  if (std::isinf(value)) return value < 0 ? "-Inf" : "Inf";
  if (value == 0) return "0";
  char buffer[64];
  const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
  if (result.ec != std::errc{}) throw std::runtime_error("c: cannot format scalar");
  return std::string(buffer, result.ptr);
}

Vector<> c(std::initializer_list<double> values) { return Vector<>(values); }

Vector<int> factor(const Vector<std::string>& values) {
  auto levels = values.values();
  std::sort(levels.begin(), levels.end());
  levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
  if (levels.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("factor: too many levels");
  std::vector<int> result;
  result.reserve(values.size());
  for (const auto& value : values)
    result.push_back(static_cast<int>(std::lower_bound(levels.begin(), levels.end(), value) - levels.begin()) + 1);
  return result;
}

/// @brief Numeric element transforms and process-wide sampling state.
namespace {

/// @brief Replaces every element of a by-value vector with its transformed value.
/// @tparam F Callable accepting one double and returning its replacement.
/// @return The transformed vector, reusing the parameter's owned storage.
template <typename F>
Vector<> transform_in_place(Vector<> v, F f) {
  for (double& x : v) x = f(x);
  return v;
}

/// @brief Combines two random_device draws into the initial seed for vector sampling.
std::uint64_t random_seed() {
  std::random_device rd;
  return (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
}

/// @brief Borrows the process-wide SplitMix64 engine; callers synchronize concurrent access.
random::SplitMix64& generator() {
  /// @brief Persistent sampling state, initialized once on first use and reset by set_seed().
  static random::SplitMix64 rng(random_seed());
  return rng;
}

} // namespace

Vector<> operator-(Vector<> v) { return transform_in_place(std::move(v), [](double x) { return -x; }); }

Vector<> operator+(Vector<> lhs, const Vector<>& rhs) { return lhs += rhs; }
Vector<> operator-(Vector<> lhs, const Vector<>& rhs) { return lhs -= rhs; }
Vector<> operator*(Vector<> lhs, const Vector<>& rhs) { return lhs *= rhs; }
Vector<> operator/(Vector<> lhs, const Vector<>& rhs) { return lhs /= rhs; }

Vector<> operator+(Vector<> lhs, double rhs) { return lhs += rhs; }
Vector<> operator-(Vector<> lhs, double rhs) { return lhs -= rhs; }
Vector<> operator*(Vector<> lhs, double rhs) { return lhs *= rhs; }
Vector<> operator/(Vector<> lhs, double rhs) { return lhs /= rhs; }

Vector<> operator+(double lhs, Vector<> rhs) { return rhs += lhs; }
Vector<> operator-(double lhs, Vector<> rhs) {
  return transform_in_place(std::move(rhs), [lhs](double x) { return lhs - x; });
}
Vector<> operator*(double lhs, Vector<> rhs) { return rhs *= lhs; }
Vector<> operator/(double lhs, Vector<> rhs) {
  return transform_in_place(std::move(rhs), [lhs](double x) { return lhs / x; });
}

Vector<> sin(Vector<> v) { return transform_in_place(std::move(v), [](double x) { return std::sin(x); }); }
Vector<> cos(Vector<> v) { return transform_in_place(std::move(v), [](double x) { return std::cos(x); }); }
Vector<> tan(Vector<> v) { return transform_in_place(std::move(v), [](double x) { return std::tan(x); }); }
Vector<> exp(Vector<> v) { return transform_in_place(std::move(v), [](double x) { return std::exp(x); }); }
Vector<> log(Vector<> v) { return transform_in_place(std::move(v), [](double x) { return std::log(x); }); }
Vector<> sqrt(Vector<> v) { return transform_in_place(std::move(v), [](double x) { return std::sqrt(x); }); }
Vector<> abs(Vector<> v) { return transform_in_place(std::move(v), [](double x) { return std::abs(x); }); }
Vector<> pow(Vector<> v, double exponent) {
  return transform_in_place(std::move(v), [exponent](double x) { return std::pow(x, exponent); });
}

double round(double value, int digits) {
  if (!std::isfinite(value) || value == 0) return value;
  // R's algorithm (R >= 4.0.0): of the two decimal candidates bracketing the value, return the
  // one nearer to the value as stored; an exact tie goes to the even last digit. So round(2.5)
  // is 2 and round(0.15, 1) is 0.1 (0.15 is stored slightly below 0.15), exactly as in R.
  const double scale = std::pow(10.0, std::abs(static_cast<double>(digits)));
  if (!std::isfinite(scale)) return digits >= 0 ? value : std::copysign(0.0, value);
  const double scaled = digits >= 0 ? value * scale : value / scale;
  if (!std::isfinite(scaled)) return value;
  const double lower = std::floor(scaled);
  const double upper = std::ceil(scaled);
  if (lower == upper) return value;  // already has at most `digits` decimals
  const auto unscale = [&](double k) { return digits >= 0 ? k / scale : k * scale; };
  const double down = unscale(lower);
  const double up = unscale(upper);
  const double below = value - down;
  const double above = up - value;
  double chosen = 0;
  if (below < above) {
    chosen = down;
  } else if (above < below) {
    chosen = up;
  } else {
    chosen = std::fmod(lower, 2.0) == 0 ? down : up;
  }
  return chosen == 0 ? std::copysign(0.0, value) : chosen;
}

Vector<> round(Vector<> v, int digits) {
  return transform_in_place(std::move(v), [digits](double x) { return datamunge::round(x, digits); });
}

Vector<> seql(double from, double to, std::size_t n) {
  Vector<> v(n);
  if (n == 1) {
    v[0] = from;
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      v[i] = from + (to - from) * static_cast<double>(i) / static_cast<double>(n - 1);
    }
  }
  return v;
}

double sum(const Vector<>& v) { return std::accumulate(v.begin(), v.end(), 0.0); }

double mean(const Vector<>& v) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  return sum(v) / static_cast<double>(v.size());
}

double sd(const Vector<>& v) {
  if (v.size() < 2) return std::numeric_limits<double>::quiet_NaN();
  // Welford's centered update avoids subtracting large, nearly equal raw moments.
  long double center = 0;
  long double squared_deviations = 0;
  std::size_t count = 0;
  for (double value : v) {
    if (!std::isfinite(value)) return std::numeric_limits<double>::quiet_NaN();
    const long double delta = static_cast<long double>(value) - center;
    center += delta / ++count;
    squared_deviations += delta * (static_cast<long double>(value) - center);
  }
  return static_cast<double>(std::sqrt(squared_deviations / (v.size() - 1)));
}

void set_seed(std::uint64_t seed) { generator() = random::SplitMix64(seed); }

Vector<> unif(double lower, double upper, std::size_t n) {
  Vector<> v(n);
  for (double& x : v) x = generator().uniform(lower, upper);
  return v;
}

Vector<> runif(double lower, double upper, std::size_t n) { return unif(lower, upper, n); }

Vector<> rnorm(double mean, double sd, std::size_t n) {
  Vector<> v(n);
  for (double& x : v) x = generator().normal(mean, sd);
  return v;
}

} // namespace datamunge
