/// @file fibonacci_coding.cpp
/// @brief Checked complete-width Fibonacci weights, emission and strict actual-span parsing.
#include <algorithm>
#include <array>
#include <datamunge/algorithms/fibonacci_coding.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Constant complete coding table and shared preflight/emission/parser helpers.
namespace fibonacci_detail {
/// @brief Construct the92 representable weights, ending at F93 without computing F94.
/// @return Array1,2,3,5,...,12200160415121876738; all additions in range.
constexpr std::array<std::uint64_t, 92> make_weights() {
  std::array<std::uint64_t, 92> a{};
  a[0] = 1;
  a[1] = 2;
  for (std::size_t i = 2; i < a.size(); ++i)
    a[i] = a[i - 1] + a[i - 2];
  return a;
}
/// @brief Immutable full-width weights with no dynamic initialization or mutation.
constexpr auto weights = make_weights();
static_assert(weights.back() == 12200160415121876738ULL);
static_assert(weights.back() > std::numeric_limits<std::uint64_t>::max() - weights[90]);
/// @brief Add size counts after checking for wrap.
/// @param a Existing count.
/// @param b Additional count.
/// @return Exact size_t sum.
/// @throws std::length_error If sum exceeds size_t.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("Fibonacci count overflow");
  return a + b;
}
/// @brief Preflight positive values and exact output container size without output allocation.
/// @param values Borrowed positive values.
/// @return Complete output bit count.
/// @throws std::invalid_argument On zero.
/// @throws std::length_error On count or container limits.
std::size_t plan(const std::vector<std::uint64_t>& values) {
  std::size_t bits = 0;
  for (auto n : values)
    bits = add(bits, fibonacci_code_length(n));
  if (bits > std::vector<bool>{}.max_size())
    throw std::length_error("Fibonacci bit vector exceeds maximum size");
  return bits;
}
/// @brief Admit logical capture entries before their associated copies/appends.
/// @param trace Owned capture under construction.
/// @param count Additional logical entries.
/// @param cap Optional logical entry maximum; zero disables.
/// @throws std::length_error On count overflow or quota failure.
void capture(FibonacciTrace& trace, std::size_t count, std::size_t cap) {
  const auto c = add(trace.captured_entries, count);
  if (cap && c > cap)
    throw std::length_error("Fibonacci capture limit exceeded");
  trace.captured_entries = c;
}
/// @brief Count one actual complete word's payload and terminator.
/// @param work Checked counts in progress.
/// @param length Complete word length including the single appended terminator.
/// @throws std::length_error On count overflow.
void record(FibonacciWork& work, std::size_t length) {
  work.values          = add(work.values, 1);
  work.payload_bits    = add(work.payload_bits, length - 1);
  work.terminator_bits = add(work.terminator_bits, 1);
  work.bit_visits      = add(work.bit_visits, length);
}
/// @brief Append actual greedy digits to output with complete capacity already reserved.
/// @param n Positive value consumed by descending greedy subtraction.
/// @param length Preflight exact complete word length.
/// @param out Output with reserved room for the whole word.
void append(std::uint64_t n, std::size_t length, std::vector<bool>& out) {
  std::array<bool, 92> digits{};
  for (std::size_t i = length - 1; i > 0; --i)
    if (weights[i - 1] <= n) {
      digits[i - 1] = true;
      n -= weights[i - 1];
    }
  for (std::size_t i = 0; i < length - 1; ++i)
    out.push_back(digits[i]);
  out.push_back(true);
}
/// @brief Shared actual encoder, optionally capturing spans during emission.
/// @param values Validated positive inputs.
/// @param bits Preflight complete output size.
/// @param trace Null for no optional history; otherwise already admitted/reserved.
/// @return Owned output and actual work.
/// @throws std::bad_alloc If output storage fails.
FibonacciEncoded
emit(const std::vector<std::uint64_t>& values, std::size_t bits, FibonacciTrace* trace) {
  FibonacciEncoded out;
  out.bits.reserve(bits);
  for (auto n : values) {
    const auto begin = out.bits.size(), length = fibonacci_code_length(n);
    append(n, length, out.bits);
    if (trace)
      trace->words.push_back({n, begin, out.bits.size() - 2, out.bits.size()});
    record(out.work, length);
  }
  return out;
}
/// @brief Shared actual whole-frame parser with optional online completed-word spans.
/// @param bits Complete borrowed/owned bit frame, immutable during this call.
/// @param trace Optional capture already retaining the entire input bit vector.
/// @param cap Optional logical capture cap, zero disables.
/// @return Complete decoded values and actual work, with no partial result on exceptions.
/// @throws std::invalid_argument On unfinished trailing bits.
/// @throws std::overflow_error Before weight indexing or value addition can exceed uint64.
/// @throws std::length_error On capture/count/container limits.
/// @throws std::bad_alloc If output/history storage fails.
FibonacciDecoded parse(const std::vector<bool>& bits, FibonacciTrace* trace, std::size_t cap) {
  FibonacciDecoded out;
  std::size_t      index = 0, begin = 0;
  std::uint64_t    value    = 0;
  bool             previous = false;
  for (std::size_t p = 0; p < bits.size(); ++p) {
    const bool bit = bits[p];
    if (bit && previous) {
      if (trace)
        capture(*trace, 2, cap);
      out.values.push_back(value);
      if (trace)
        trace->words.push_back({value, begin, p - 1, p + 1});
      record(out.work, p + 1 - begin);
      begin    = p + 1;
      index    = 0;
      value    = 0;
      previous = false;
    } else {
      if (index >= weights.size())
        throw std::overflow_error("Fibonacci payload exceeds uint64 coding weights");
      if (bit) {
        if (weights[index] > std::numeric_limits<std::uint64_t>::max() - value)
          throw std::overflow_error("Fibonacci decoded value exceeds uint64");
        value += weights[index];
      }
      ++index;
      previous = bit;
    }
  }
  if (begin != bits.size())
    throw std::invalid_argument("unfinished Fibonacci codeword");
  return out;
}
} // namespace fibonacci_detail
std::uint64_t fibonacci_coding_weight(std::size_t index) {
  if (index >= fibonacci_detail::weights.size())
    throw std::out_of_range("Fibonacci coding weight index must be below92");
  return fibonacci_detail::weights[index];
}
std::size_t fibonacci_code_length(std::uint64_t n) {
  if (!n)
    throw std::invalid_argument("Fibonacci coding requires positive integers");
  return static_cast<std::size_t>(
             std::upper_bound(
                 fibonacci_detail::weights.begin(), fibonacci_detail::weights.end(), n) -
             fibonacci_detail::weights.begin()) +
         1;
}
void fibonacci_encode_value(std::uint64_t n, std::vector<bool>& out) {
  const auto length = fibonacci_code_length(n), total = fibonacci_detail::add(out.size(), length);
  if (total > out.max_size())
    throw std::length_error("Fibonacci append exceeds vector maximum size");
  out.reserve(total);
  fibonacci_detail::append(n, length, out);
}
std::vector<bool> fibonacci_encode(const std::vector<std::uint64_t>& values) {
  return fibonacci_encode_work(values).bits;
}
std::vector<std::uint64_t> fibonacci_decode(const std::vector<bool>& bits) {
  return fibonacci_decode_work(bits).values;
}
FibonacciEncoded fibonacci_encode_work(const std::vector<std::uint64_t>& values) {
  return fibonacci_detail::emit(values, fibonacci_detail::plan(values), nullptr);
}
FibonacciDecoded fibonacci_decode_work(const std::vector<bool>& bits) {
  return fibonacci_detail::parse(bits, nullptr, 0);
}
FibonacciTrace fibonacci_encode_trace(
    const std::vector<std::uint64_t>& values,
    std::size_t                       max_entries) {
  const auto     bits  = fibonacci_detail::plan(values);
  const auto     twice = fibonacci_detail::add(values.size(), values.size());
  FibonacciTrace trace;
  fibonacci_detail::capture(trace, fibonacci_detail::add(bits, twice), max_entries);
  if (values.size() > trace.words.max_size())
    throw std::length_error("Fibonacci word history exceeds vector maximum size");
  trace.encoding = true;
  trace.values   = values;
  trace.words.reserve(values.size());
  auto out   = fibonacci_detail::emit(trace.values, bits, &trace);
  trace.bits = std::move(out.bits);
  trace.work = out.work;
  return trace;
}
FibonacciTrace fibonacci_decode_trace(const std::vector<bool>& bits, std::size_t max_entries) {
  FibonacciTrace trace;
  fibonacci_detail::capture(trace, bits.size(), max_entries);
  trace.bits   = bits;
  auto out     = fibonacci_detail::parse(trace.bits, &trace, max_entries);
  trace.values = std::move(out.values);
  trace.work   = out.work;
  return trace;
}
} // namespace datamunge::algorithms
