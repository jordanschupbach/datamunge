/// @file toom3.cpp
/// @brief Actual checked signed coefficient convolution, interpolation and decimal carry.
#include <algorithm>
#include <array>
#include <datamunge/algorithms/toom3.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private exact checked coefficient producers; no shared state.
namespace {
/// @brief Checked signed working coefficient, never exposed in the ABI.
using Wide = __int128;
/// @brief Owned little-endian coefficient polynomial.
using Coeffs = std::vector<Wide>;
/// @brief Convert one actual signed coefficient to decimal without narrowing.
std::string decimal(Wide x) {
  bool        neg = x < 0;
  __uint128_t u   = neg ? __uint128_t(-(x + 1)) + 1 : __uint128_t(x);
  std::string s;
  do {
    s.push_back(char('0' + u % 10));
    u /= 10;
  } while (u);
  if (neg)
    s.push_back('-');
  std::reverse(s.begin(), s.end());
  return s;
}
/// @brief Invocation-local selected-work, exact arithmetic and optional recursive state capture.
struct Context {
  const ToomOptions& o;          ///< Borrowed invocation policies.
  ToomRun            r;          ///< Actual result and selected event counts.
  std::size_t        retained{}; ///< Actual retained record/coefficient values.

  /// @brief Count one actual scalar coefficient arithmetic event before evaluating it.
  void count() {
    if (r.operations == std::numeric_limits<std::size_t>::max() ||
        (o.max_operations && r.operations >= o.max_operations))
      throw std::length_error("Toom arithmetic work");
    ++r.operations;
  }
  /// @brief Add actual coefficients with overflow detection.
  Wide plus(Wide a, Wide b) {
    count();
    Wide x;
    if (__builtin_add_overflow(a, b, &x))
      throw std::overflow_error("Toom coefficient addition");
    return x;
  }
  /// @brief Subtract actual coefficients with overflow detection.
  Wide minus(Wide a, Wide b) {
    count();
    Wide x;
    if (__builtin_sub_overflow(a, b, &x))
      throw std::overflow_error("Toom coefficient subtraction");
    return x;
  }
  /// @brief Multiply actual coefficients with overflow detection.
  Wide times(Wide a, Wide b) {
    count();
    Wide x;
    if (__builtin_mul_overflow(a, b, &x))
      throw std::overflow_error("Toom coefficient multiplication");
    return x;
  }
  /// @brief Count/check exact small-divisor interpolation before quotient evaluation.
  Wide divide(Wide a, int b) {
    count();
    if (a % b)
      throw std::logic_error("Toom nonexact interpolation");
    return a / b;
  }
  /// @brief Own actual coefficients as signed decimal strings under capture geometry.
  std::vector<std::string> strings(const Coeffs& a) {
    if (a.size() > std::numeric_limits<std::size_t>::max() - retained ||
        (o.max_history_values &&
         (retained > o.max_history_values || a.size() > o.max_history_values - retained)))
      throw std::length_error("Toom captured coefficients");
    retained += a.size();
    std::vector<std::string> s;
    for (auto x : a)
      s.push_back(decimal(x));
    return s;
  }
  /// @brief Capture the actual returned recursive state and already produced interpolation values.
  void record(
      const char*                kind,
      std::size_t                depth,
      std::size_t                split,
      const Coeffs&              a,
      const Coeffs&              b,
      const Coeffs&              out,
      const std::vector<Coeffs>& products     = {},
      const std::vector<Coeffs>& coefficients = {}) {
    if (o.max_records && r.steps.size() >= o.max_records)
      throw std::length_error("Toom records");
    if (retained == std::numeric_limits<std::size_t>::max() ||
        (o.max_history_values && retained >= o.max_history_values))
      throw std::length_error("Toom record units");
    ++retained;
    ToomStep s;
    s.kind  = kind;
    s.depth = depth;
    s.split = split;
    s.left  = strings(a);
    s.right = strings(b);
    for (const auto& x : products)
      s.products.push_back(strings(x));
    for (const auto& x : coefficients)
      s.interpolated.push_back(strings(x));
    s.result = strings(out);
    r.steps.push_back(std::move(s));
  }
};
/// @brief Parse actual validated decimal digits in little-endian order; empty means0.
Coeffs parse(const std::string& s) {
  Coeffs r;
  for (auto it = s.rbegin(); it != s.rend(); ++it)
    r.push_back(*it - '0');
  if (r.empty())
    r.push_back(0);
  return r;
}
/// @brief Own one actual coefficient slice, using zero for an empty chunk.
Coeffs slice(const Coeffs& a, std::size_t lo, std::size_t k) {
  if (lo >= a.size() || k == 0) return Coeffs{0};
  const auto length = std::min(k, a.size() - lo);
  return Coeffs(a.begin() + lo, a.begin() + lo + length);
}
/// @brief Add/subtract actual polynomial coefficients using checked scalar arithmetic.
Coeffs add(const Coeffs& a, const Coeffs& b, Context& c, bool subtract = false) {
  Coeffs r = a;
  r.resize(std::max(a.size(), b.size()), 0);
  for (std::size_t i = 0; i < b.size(); ++i)
    r[i] = subtract ? c.minus(r[i], b[i]) : c.plus(r[i], b[i]);
  return r;
}
/// @brief Apply an actual small integer scale to every coefficient.
Coeffs scale(const Coeffs& a, int b, Context& c) {
  auto r = a;
  for (auto& x : r)
    x = c.times(x, b);
  return r;
}
/// @brief Divide actual interpolation coefficients exactly by2 or3.
Coeffs divide(const Coeffs& a, int b, Context& c) {
  auto r = a;
  for (auto& x : r)
    x = c.divide(x, b);
  return r;
}
/// @brief Shift actual polynomial coefficient positions without arithmetic replay.
Coeffs shift(const Coeffs& a, std::size_t k) {
  if (k > std::numeric_limits<std::size_t>::max() - a.size())
    throw std::length_error("Toom shifted geometry");
  Coeffs r(a.size() + k, 0);
  std::copy(a.begin(), a.end(), r.begin() + k);
  return r;
}
/// @brief Execute an actual scalar convolution leaf.
Coeffs schoolbook(const Coeffs& a, const Coeffs& b, Context& c, std::size_t depth, bool capture) {
  if (a.size() > std::numeric_limits<std::size_t>::max() - b.size())
    throw std::length_error("Toom leaf geometry");
  ++c.r.schoolbook_calls;
  Coeffs r(a.size() + b.size());
  for (std::size_t i = 0; i < a.size(); ++i)
    for (std::size_t j = 0; j < b.size(); ++j) {
      auto product = c.times(a[i], b[j]);
      ++c.r.scalar_products;
      r[i + j] = c.plus(r[i + j], product);
    }
  if (capture)
    c.record("schoolbook", depth, 0, a, b, r);
  return r;
}
/// @brief Execute the actual three-product Karatsuba crossover producer.
Coeffs karatsuba(const Coeffs& a, const Coeffs& b, Context& c, std::size_t depth, bool capture) {
  auto n = std::max(a.size(), b.size());
  if (n <= c.o.karatsuba_leaf)
    return schoolbook(a, b, c, depth, capture);
  ++c.r.karatsuba_calls;
  auto half = n / 2;
  auto a0 = slice(a, 0, half), a1 = slice(a, half, n - half), b0 = slice(b, 0, half),
       b1 = slice(b, half, n - half);
  auto z0 = karatsuba(a0, b0, c, depth + 1, capture), z2 = karatsuba(a1, b1, c, depth + 1, capture),
       z1 = karatsuba(add(a0, a1, c), add(b0, b1, c), c, depth + 1, capture);
  z1      = add(add(z1, z0, c, true), z2, c, true);
  auto r  = add(add(z0, shift(z1, half), c), shift(z2, 2 * half), c);
  while (r.size() > 1 && r.back() == 0)
    r.pop_back();
  if (capture)
    c.record("karatsuba", depth, half, a, b, r);
  return r;
}
/// @brief Execute actual five point products and exact coefficient interpolation at0,1,-1,2,infinity.
Coeffs toom(const Coeffs& a, const Coeffs& b, Context& c, std::size_t depth, bool capture) {
  auto n = std::max(a.size(), b.size());
  if (n <= c.o.toom_leaf)
    return karatsuba(a, b, c, depth, capture);
  ++c.r.toom_calls;
  auto k  = n / 3 + (n % 3 != 0);
  auto a0 = slice(a, 0, k), a1 = slice(a, k, k), a2 = slice(a, 2 * k, k), b0 = slice(b, 0, k),
       b1 = slice(b, k, k), b2 = slice(b, 2 * k, k);
  auto p1 = add(add(a0, a1, c), a2, c), pm1 = add(add(a0, a1, c, true), a2, c),
       p2 = add(add(a0, scale(a1, 2, c), c), scale(a2, 4, c), c);
  auto q1 = add(add(b0, b1, c), b2, c), qm1 = add(add(b0, b1, c, true), b2, c),
       q2 = add(add(b0, scale(b1, 2, c), c), scale(b2, 4, c), c);
  auto w0 = toom(a0, b0, c, depth + 1, capture), w1 = toom(p1, q1, c, depth + 1, capture),
       w2 = toom(pm1, qm1, c, depth + 1, capture), w3 = toom(p2, q2, c, depth + 1, capture),
       w4 = toom(a2, b2, c, depth + 1, capture);
  auto c0 = w0, c4 = w4, c2 = add(add(divide(add(w1, w2, c), 2, c), c0, c, true), c4, c, true),
       s = divide(add(w1, w2, c, true), 2, c),
       t = divide(
           add(add(add(w3, c0, c, true), scale(c2, 4, c), c, true), scale(c4, 16, c), c, true),
           2,
           c),
       c3 = divide(add(t, s, c, true), 3, c), c1 = add(s, c3, c, true);
  auto r =
      add(add(add(add(c0, shift(c1, k), c), shift(c2, 2 * k), c), shift(c3, 3 * k), c),
          shift(c4, 4 * k),
          c);
  if (capture)
    c.record("toom3", depth, k, a, b, r, {w0, w1, w2, w3, w4}, {c0, c1, c2, c3, c4});
  return r;
}
/// @brief Execute complete validated parsing/multiplication and actual decimal carry.
ToomRun execute(const std::string& a, const std::string& b, const ToomOptions& o, bool capture) {
  if (!o.toom_leaf || !o.karatsuba_leaf)
    throw std::invalid_argument("Toom positive crossovers");
  if (o.max_digits && (a.size() > o.max_digits || b.size() > o.max_digits))
    throw std::length_error("Toom complete digits");
  for (const auto* s : {&a, &b})
    for (unsigned char ch : *s)
      if (ch < '0' || ch > '9')
        throw std::invalid_argument("Toom decimal bytes");
  Context     c{o};
  auto        coefficients = toom(parse(a), parse(b), c, 0, capture);
  Wide        carry        = 0;
  std::string reversed;
  for (auto v : coefficients) {
    auto total = c.plus(v, carry);
    Wide digit = total % 10;
    if (digit < 0)
      digit += 10;
    carry = c.minus(total, digit) / 10;
    reversed.push_back(char('0' + digit));
  }
  if (carry < 0)
    throw std::logic_error("Toom negative final carry");
  while (carry) {
    c.count();
    reversed.push_back(char('0' + carry % 10));
    carry /= 10;
  }
  while (reversed.size() > 1 && reversed.back() == '0')
    reversed.pop_back();
  std::reverse(reversed.begin(), reversed.end());
  c.r.value = std::move(reversed);
  return c.r;
}
} // namespace
ToomRun toom3_run(const std::string& a, const std::string& b, const ToomOptions& options) {
  return execute(a, b, options, false);
}
ToomRun toom3_trace(const std::string& a, const std::string& b, const ToomOptions& options) {
  return execute(a, b, options, true);
}
std::string toom3_mul(const std::string& a, const std::string& b) {
  return toom3_run(a, b).value;
}
} // namespace datamunge::algorithms
