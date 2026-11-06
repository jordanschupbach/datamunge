/// @file vector.cpp
/// @brief Legacy eager vector ABI entry points, scalar kernels, reductions and random sampling.

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
/// @param v Owned vector transformed in place.
/// @param f Numeric element transform.
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

/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param v Numeric operand consumed by this legacy call.
/// @return Owned eager numeric result.
Vector<> operator-(Vector<> v) { return transform_in_place(std::move(v), [](double x) { return -x; }); }

/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param lhs Numeric operand consumed by this legacy call.
/// @param rhs Numeric operand consumed by this legacy call.
/// @return Owned eager numeric result.
Vector<> operator+(Vector<> lhs, const Vector<>& rhs) { return lhs += rhs; }
/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param lhs Numeric operand consumed by this legacy call.
/// @param rhs Numeric operand consumed by this legacy call.
/// @return Owned eager numeric result.
Vector<> operator-(Vector<> lhs, const Vector<>& rhs) { return lhs -= rhs; }
/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param lhs Numeric operand consumed by this legacy call.
/// @param rhs Numeric operand consumed by this legacy call.
/// @return Owned eager numeric result.
Vector<> operator*(Vector<> lhs, const Vector<>& rhs) { return lhs *= rhs; }
/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param lhs Numeric operand consumed by this legacy call.
/// @param rhs Numeric operand consumed by this legacy call.
/// @return Owned eager numeric result.
Vector<> operator/(Vector<> lhs, const Vector<>& rhs) { return lhs /= rhs; }

/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param lhs Numeric operand consumed by this legacy call.
/// @param rhs Numeric operand consumed by this legacy call.
/// @return Owned eager numeric result.
Vector<> operator+(Vector<> lhs, double rhs) { return lhs += rhs; }
/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param lhs Numeric operand consumed by this legacy call.
/// @param rhs Numeric operand consumed by this legacy call.
/// @return Owned eager numeric result.
Vector<> operator-(Vector<> lhs, double rhs) { return lhs -= rhs; }
/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param lhs Numeric operand consumed by this legacy call.
/// @param rhs Numeric operand consumed by this legacy call.
/// @return Owned eager numeric result.
Vector<> operator*(Vector<> lhs, double rhs) { return lhs *= rhs; }
/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param lhs Numeric operand consumed by this legacy call.
/// @param rhs Numeric operand consumed by this legacy call.
/// @return Owned eager numeric result.
Vector<> operator/(Vector<> lhs, double rhs) { return lhs /= rhs; }

/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param lhs Numeric operand consumed by this legacy call.
/// @param rhs Numeric operand consumed by this legacy call.
/// @return Owned eager numeric result.
Vector<> operator+(double lhs, Vector<> rhs) { return rhs += lhs; }
/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param lhs Numeric operand consumed by this legacy call.
/// @param rhs Numeric operand consumed by this legacy call.
/// @return Owned eager numeric result.
Vector<> operator-(double lhs, Vector<> rhs) {
  return transform_in_place(std::move(rhs), [lhs](double x) { return lhs - x; });
}
/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param lhs Numeric operand consumed by this legacy call.
/// @param rhs Numeric operand consumed by this legacy call.
/// @return Owned eager numeric result.
Vector<> operator*(double lhs, Vector<> rhs) { return rhs *= lhs; }
/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param lhs Numeric operand consumed by this legacy call.
/// @param rhs Numeric operand consumed by this legacy call.
/// @return Owned eager numeric result.
Vector<> operator/(double lhs, Vector<> rhs) {
  return transform_in_place(std::move(rhs), [lhs](double x) { return lhs / x; });
}

/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param v Numeric operand consumed by this legacy call.
/// @return Owned eager numeric result.
Vector<> sin(Vector<> v) { return transform_in_place(std::move(v), [](double x) { return std::sin(x); }); }
/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param v Numeric operand consumed by this legacy call.
/// @return Owned eager numeric result.
Vector<> cos(Vector<> v) { return transform_in_place(std::move(v), [](double x) { return std::cos(x); }); }
/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param v Numeric operand consumed by this legacy call.
/// @return Owned eager numeric result.
Vector<> tan(Vector<> v) { return transform_in_place(std::move(v), [](double x) { return std::tan(x); }); }
/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param v Numeric operand consumed by this legacy call.
/// @return Owned eager numeric result.
Vector<> exp(Vector<> v) { return transform_in_place(std::move(v), [](double x) { return std::exp(x); }); }
/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param v Numeric operand consumed by this legacy call.
/// @return Owned eager numeric result.
Vector<> log(Vector<> v) { return transform_in_place(std::move(v), [](double x) { return std::log(x); }); }
/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param v Numeric operand consumed by this legacy call.
/// @return Owned eager numeric result.
Vector<> sqrt(Vector<> v) { return transform_in_place(std::move(v), [](double x) { return std::sqrt(x); }); }
/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param v Numeric operand consumed by this legacy call.
/// @return Owned eager numeric result.
Vector<> abs(Vector<> v) { return transform_in_place(std::move(v), [](double x) { return std::abs(x); }); }
/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param v Numeric operand consumed by this legacy call.
/// @param exponent Numeric operand consumed by this legacy call.
/// @return Owned eager numeric result.
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

/// @brief Eager ABI compatibility entry point; new C++ calls use lazy header templates.
/// @param v Numeric operand consumed by this legacy call.
/// @param digits Decimal places for scalar rounding.
/// @return Owned eager numeric result.
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

double sum(const Vector<>& v) { return vector_expr::sum_values(v); }

double mean(const Vector<>& v) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  return sum(v) / static_cast<double>(v.size());
}

double sd(const Vector<>& v) { return vector_expr::sample_sd(v); }

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
