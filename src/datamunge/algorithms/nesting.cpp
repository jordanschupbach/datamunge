/// @file nesting.cpp
/// @brief Canonical FFD with exact nonnegative dyadic capacity and volume arithmetic.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <datamunge/algorithms/nesting.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Fixed dyadic arithmetic and shared checked packing producer.
namespace {
/// @brief Nonnegative magnitude scaled by2^-1074,68 little-endian32-bit limbs.
/// Binary64 needs2098 bits; at most INT_MAX items need at most2129.2176 bits
/// suffice for sums and capacity times any candidate bin count<=INT_MAX.
using Magnitude = std::array<std::uint32_t, 68>;
static_assert(
    sizeof(double) == 8 && std::numeric_limits<double>::is_iec559 &&
    std::numeric_limits<double>::digits == 53);
/// @brief Validate complete packing geometry and int-sized item identities.
/// @param sizes Finite nonnegative item sizes no greater than capacity.
/// @param capacity Finite strictly positive bin capacity.
/// @throws std::invalid_argument If any size/capacity is invalid.
/// @throws std::length_error If item count exceeds INT_MAX.
void validate(const std::vector<double>& sizes, double capacity) {
  if (!std::isfinite(capacity) || capacity <= 0)
    throw std::invalid_argument("nesting: finite positive capacity required");
  for (auto size : sizes)
    if (!std::isfinite(size) || size < 0 || size > capacity)
      throw std::invalid_argument("nesting: items must lie in[0,capacity]");
  if (sizes.size() > std::size_t(std::numeric_limits<int>::max()))
    throw std::length_error("nesting: item IDs exceed int");
}
/// @brief Add one finite nonnegative binary64 value exactly to a proven-bounded magnitude.
/// @param value Exact accumulator, with enough range for at most INT_MAX items.
/// @param x Finite nonnegative value, including signed zero.
void add(Magnitude& value, double x) {
  const auto     bits     = std::bit_cast<std::uint64_t>(x);
  const auto     exponent = unsigned((bits >> 52) & 2047);
  const auto     mantissa = (bits & ((1ULL << 52) - 1)) | (exponent ? (1ULL << 52) : 0);
  const unsigned shift    = exponent ? exponent - 1 : 0;
  std::size_t    word     = shift / 32;
  __uint128_t    rest     = static_cast<__uint128_t>(mantissa) << (shift % 32);
  while (rest) {
    const auto sum = std::uint64_t(value[word]) + std::uint32_t(rest);
    value[word]    = std::uint32_t(sum);
    rest           = (rest >> 32) + (sum >> 32);
    ++word;
  }
}
/// @brief Compare complete exact nonnegative dyadic magnitudes.
/// @param a First scaled magnitude.
/// @param b Second scaled magnitude.
/// @return−1,0,1 for less/equal/greater.
int compare(const Magnitude& a, const Magnitude& b) {
  for (std::size_t i = a.size(); i-- > 0;)
    if (a[i] != b[i])
      return a[i] < b[i] ? -1 : 1;
  return 0;
}
/// @brief Multiply a capacity magnitude by a proven int-sized nonnegative count.
/// @param a Exact capacity magnitude.
/// @param n Nonnegative count at most INT_MAX.
/// @return Exact product within the fixed accumulator proof.
Magnitude multiply(const Magnitude& a, std::uint32_t n) {
  Magnitude     out{};
  std::uint64_t carry = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    auto value = std::uint64_t(a[i]) * n + carry;
    out[i]     = std::uint32_t(value);
    carry      = value >> 32;
  }
  return out;
}
/// @brief Correctly round a feasible exact load into binary64, ties to even.
/// @param a Exact nonnegative load no greater than finite capacity.
/// @return Correctly rounded finite presentation load.
double rounded(const Magnitude& a) {
  std::size_t word = a.size();
  while (word && a[word - 1] == 0)
    --word;
  if (!word)
    return 0;
  int           top      = int((word - 1) * 32 + std::bit_width(a[word - 1]) - 1);
  std::uint64_t mantissa = 0;
  const int     bottom   = std::max(0, top - 52);
  for (int bit = top; bit >= bottom; --bit)
    mantissa = (mantissa << 1) | ((a[bit / 32] >> (bit % 32)) & 1U);
  if (bottom) {
    const int  half     = bottom - 1;
    const bool half_bit = (a[half / 32] >> (half % 32)) & 1U;
    bool       sticky   = false;
    for (int i = 0; i < half / 32; ++i)
      sticky |= a[i] != 0;
    if (half % 32)
      sticky |= (a[half / 32] & ((1U << (half % 32)) - 1)) != 0;
    if (half_bit && (sticky || (mantissa & 1)))
      ++mantissa;
  }
  return std::ldexp(double(mantissa), bottom - 1074);
}
/// @brief Exact ceiling of total binary64 volume divided by capacity.
/// @param sizes Complete validated sizes.
/// @param capacity Finite positive capacity.
/// @return Exact volume lower bound, at most size count.
int bound(const std::vector<double>& sizes, double capacity) {
  Magnitude total{}, cap{};
  add(cap, capacity);
  for (auto size : sizes)
    add(total, size);
  std::uint32_t lo = 0, hi = static_cast<std::uint32_t>(sizes.size());
  while (lo < hi) {
    auto mid = lo + (hi - lo) / 2;
    if (compare(multiply(cap, mid), total) < 0)
      lo = mid + 1;
    else
      hi = mid;
  }
  return int(lo);
}
/// @brief Execute the canonical exact-admission decreasing first-fit producer.
/// @param sizes Complete validated input sizes.
/// @param capacity Finite positive capacity.
/// @param out Optional actual-history sink.
/// @param quota Optional history cap, zero disables.
/// @return Actual result, order, bound and trial work.
/// @throws std::length_error If work or capture quota exceeds its limit.
NestingAnalysis
run(const std::vector<double>& sizes, double capacity, NestingTrace* out, std::size_t quota) {
  /// @brief Admit actual history records before appending.
  auto admit = [&](std::size_t amount) {
    if (!out)
      return;
    if (amount > std::numeric_limits<std::size_t>::max() - out->captured_entries ||
        (quota && (out->captured_entries > quota || amount > quota - out->captured_entries)))
      throw std::length_error("nesting: capture quota exceeded");
    out->captured_entries += amount;
  };
  NestingAnalysis a;
  a.order.resize(sizes.size());
  std::iota(a.order.begin(), a.order.end(), 0);
  std::sort(a.order.begin(), a.order.end(), [&](int i, int j) { return sizes[i] > sizes[j]; });
  a.volume_lower_bound = bound(sizes, capacity);
  Magnitude cap{};
  add(cap, capacity);
  std::vector<Magnitude> loads;
  for (auto item : a.order) {
    int        placed = -1;
    Magnitude  candidate{};
    const auto first = a.bin_trials;
    for (std::size_t bin = 0; bin < loads.size(); ++bin) {
      candidate = loads[bin];
      add(candidate, sizes[item]);
      const auto comparison = compare(candidate, cap);
      if (a.bin_trials == std::numeric_limits<std::size_t>::max())
        throw std::length_error("nesting: work count overflow");
      ++a.bin_trials;
      if (out) {
        admit(1);
        out->trials.push_back({item, int(bin), a.result.loads[bin], comparison, comparison <= 0});
      }
      if (comparison <= 0) {
        placed = int(bin);
        break;
      }
    }
    const bool opened = placed < 0;
    if (opened) {
      placed    = int(loads.size());
      candidate = {};
      add(candidate, sizes[item]);
      loads.push_back(candidate);
      a.result.loads.push_back(0);
      a.result.assignment.emplace_back();
      if (out)
        admit(1);
    } else
      loads[placed] = candidate;
    a.result.loads[placed] = rounded(loads[placed]);
    a.result.assignment[placed].push_back(item);
    if (out)
      out->placements.push_back(
          {item, placed, opened, a.result.loads[placed], first, a.bin_trials - first});
  }
  a.result.bins = int(loads.size());
  return a;
}
} // namespace
NestingResult nest_first_fit_decreasing(const std::vector<double>& s, double c) {
  validate(s, c);
  return run(s, c, nullptr, 0).result;
}
int nesting_lower_bound(const std::vector<double>& s, double c) {
  validate(s, c);
  return bound(s, c);
}
NestingAnalysis nesting_analyze(const std::vector<double>& s, double c) {
  validate(s, c);
  return run(s, c, nullptr, 0);
}
NestingTrace nesting_trace(const std::vector<double>& s, double c, std::size_t quota) {
  validate(s, c);
  if (s.size() > (std::numeric_limits<std::size_t>::max() - 1) / 4)
    throw std::length_error("nesting: capture geometry overflow");
  NestingTrace t;
  t.captured_entries = 1 + 4 * s.size();
  if (quota && t.captured_entries > quota)
    throw std::length_error("nesting: capture quota exceeded");
  t.sizes    = s;
  t.capacity = c;
  t.analysis = run(s, c, &t, quota);
  return t;
}
bool nesting_feasible(
    const std::vector<double>&           s,
    double                               c,
    const std::vector<std::vector<int>>& assignment) {
  validate(s, c);
  std::vector<bool> seen(s.size());
  Magnitude         cap{};
  add(cap, c);
  std::size_t count = 0;
  for (auto& bin : assignment) {
    Magnitude load{};
    for (auto id : bin) {
      if (id < 0 || std::size_t(id) >= s.size() || seen[id])
        return false;
      seen[id] = true;
      ++count;
      add(load, s[id]);
    }
    if (compare(load, cap) > 0)
      return false;
  }
  return count == s.size();
}
std::vector<double> nesting_demo_sizes() {
  std::vector<double> s;
  std::uint32_t       state = 12345;
  for (int i = 0; i < 30; ++i) {
    state = state * 1103515245U + 12345U;
    s.push_back(.15 + (state % 1000) / 1000. * .7);
  }
  return s;
}
} // namespace datamunge::algorithms
