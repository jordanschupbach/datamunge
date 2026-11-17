/// @file matrix_chain.cpp
/// @brief Canonical matrix chain API, producing histories and checked native contracts.

#include <algorithm>
#include <datamunge/algorithms/matrix_chain.hpp>
#include <limits>
#include <stdexcept>
#include <utility>

namespace datamunge::algorithms {
/// @brief Implementation-only helpers for canonical matrix chain production and capture.
namespace {
/// @brief Unsigned LLONG_MAX+1 saturation sentinel for nonnegative subproblem costs.
constexpr std::uint64_t cap = std::uint64_t{1} << 63;
/// @brief Add two complete logical geometry counts without wrap.
/// @param a First count.
/// @param b Additional count.
/// @return Exact sum.
/// @throws std::length_error On size_t overflow.
std::size_t add_size(std::size_t a, std::size_t b) {
  if (b > SIZE_MAX - a)
    throw std::length_error("matrix-chain geometry");
  return a + b;
}
/// @brief Multiply complete logical geometry counts without wrap.
/// @param a First factor.
/// @param b Second factor.
/// @return Exact product.
/// @throws std::length_error On size_t overflow.
std::size_t mul_size(std::size_t a, std::size_t b) {
  if (a && b > SIZE_MAX / a)
    throw std::length_error("matrix-chain geometry");
  return a * b;
}
/// @brief Validate every supplied dimension before any table work or capture.
/// @param d Original dimension list.
/// @throws std::invalid_argument On any nonpositive dimension.
void validate(const std::vector<int>& d) {
  for (int x : d)
    if (x <= 0)
      throw std::invalid_argument("matrix-chain dimensions must be positive");
}
/// @brief Saturate an exact wide nonnegative cost at the representability sentinel.
/// @param x Exact wide cost.
/// @return Exact smaller cost or cap.
std::uint64_t capped(unsigned __int128 x) {
  return x >= cap ? cap : static_cast<std::uint64_t>(x);
}
/// @brief Compute a positive three-dimension product with defined wide arithmetic and saturation.
/// @param a First positive dimension.
/// @param b Second positive dimension.
/// @param c Third positive dimension.
/// @return Capped exact product.
std::uint64_t product(int a, int b, int c) {
  return capped(static_cast<unsigned __int128>(a) * b * c);
}
/// @brief Add actual nonnegative subproblem/product costs with saturation.
/// @param a First actual capped cost.
/// @param b Second actual capped cost.
/// @param c Third actual capped cost.
/// @return Exact representable sum or sentinel.
std::uint64_t sum(std::uint64_t a, std::uint64_t b, std::uint64_t c) {
  return capped(static_cast<unsigned __int128>(a) + b + c);
}
/// @brief Recover a signed exact full-chain cost or reject the saturation sentinel.
/// @param x Actual full-chain capped optimum.
/// @return Exact signed cost.
/// @throws std::overflow_error On a saturated full-chain optimum.
long long exact(std::uint64_t x) {
  if (x == cap)
    throw std::overflow_error("matrix-chain full cost exceeds LLONG_MAX");
  return static_cast<long long>(x);
}
/// @brief Reconstruct the actual selected split tree with an explicit task stack.
/// @param s Actual complete selected split table.
/// @param n Actual matrix count.
/// @return Fully parenthesized product, empty for no matrices.
std::string expression(const std::vector<std::vector<std::size_t>>& s, std::size_t n) {
  if (!n)
    return {};
  // Explicit stack avoids recursion-depth failure for a skewed optimum.
  /// @brief One actual split interval or literal output delimiter pending reconstruction.
  struct Task {
    std::size_t i;       ///< First actual interval matrix, unused for a delimiter.
    std::size_t j;       ///< Last actual interval matrix, unused for a delimiter.
    char        literal; ///< Nonzero delimiter to append, or zero for an interval.
  };
  std::vector<Task> stack{{1, n, 0}};
  std::string       out;
  while (!stack.empty()) {
    auto t = stack.back();
    stack.pop_back();
    if (t.literal) {
      out += t.literal;
      continue;
    }
    if (t.i == t.j) {
      out += 'A';
      out += std::to_string(t.i);
      continue;
    }
    auto k = s[t.i][t.j];
    stack.push_back({0, 0, ')'});
    stack.push_back({k + 1, t.j, 0});
    stack.push_back({t.i, k, 0});
    stack.push_back({0, 0, '('});
  }
  return out;
}
/// @brief Canonical checked bottom-up producer, optionally capturing actual trials and commits.
/// @param d Original complete dimension list.
/// @param trace Optional owned capture target.
/// @param limit Optional complete logical cap, zero off.
/// @return Actual result, tables and work.
/// @throws std::invalid_argument On a nonpositive dimension.
/// @throws std::length_error On complete count/table/capture geometry overflow.
/// @throws std::overflow_error On an unrepresentable full optimum.
MatrixChainAnalysis produce(const std::vector<int>& d, MatrixChainTrace* trace, std::size_t limit) {
  validate(d);
  const auto n      = d.empty() ? 0 : d.size() - 1;
  const auto side   = add_size(n, 1);
  auto       square = mul_size(side, side);
  auto       pairs  = n ? mul_size(n, n - 1) / 2 : 0;
  // Divide the consecutive factors before multiplying to avoid false intermediate overflow.
  std::size_t a = n, b = n ? n - 1 : 0, c = side;
  if (a % 2 == 0)
    a /= 2;
  else
    b /= 2;
  if (a % 3 == 0)
    a /= 3;
  else if (b % 3 == 0)
    b /= 3;
  else
    c /= 3;
  auto       trials   = mul_size(mul_size(a, b), c);
  const auto geometry = add_size(add_size(d.size(), mul_size(2, square)), add_size(pairs, trials));
  if (trace && limit && geometry > limit)
    throw std::length_error("matrix-chain complete capture exceeds quota");
  MatrixChainAnalysis out;
  out.costs.assign(side, std::vector<std::uint64_t>(side));
  out.splits.assign(side, std::vector<std::size_t>(side));
  if (trace) {
    trace->captured_entries = geometry;
    trace->dimensions       = d;
    trace->trials.reserve(trials);
    trace->commits.reserve(pairs);
  }
  for (std::size_t len = 2; len <= n; ++len)
    for (std::size_t i = 1; i <= n - len + 1; ++i) {
      auto          j           = i + len - 1;
      std::uint64_t best        = cap;
      bool          initialized = false;
      for (std::size_t k = i; k < j; ++k) {
        auto l = out.costs[i][k], r = out.costs[k + 1][j], p = product(d[i - 1], d[k], d[j]),
             v        = sum(l, r, p);
        bool selected = !initialized || v < best;
        ++out.work.trials;
        if (selected) {
          initialized      = true;
          best             = v;
          out.splits[i][j] = k;
          ++out.work.improvements;
        }
        if (trace)
          trace->trials.push_back({i, j, k, l, r, p, v, selected});
      }
      out.costs[i][j] = best;
      ++out.work.cells;
      if (trace)
        trace->commits.push_back({i, j, out.splits[i][j], best});
    }
  out.result.cost             = n ? exact(out.costs[1][n]) : 0;
  out.result.parenthesization = expression(out.splits, n);
  return out;
}
} // namespace
MatrixChainResult matrix_chain_order(const std::vector<int>& dims) {
  return produce(dims, nullptr, 0).result;
}
MatrixChainAnalysis matrix_chain_analyze(const std::vector<int>& dims) {
  return produce(dims, nullptr, 0);
}
MatrixChainTrace matrix_chain_trace(const std::vector<int>& dims, std::size_t limit) {
  MatrixChainTrace out;
  out.analysis = produce(dims, &out, limit);
  return out;
}
long long matrix_chain_left_cost(const std::vector<int>& d) {
  validate(d);
  std::uint64_t total = 0;
  for (std::size_t j = 2; j < d.size(); ++j)
    total = sum(total, 0, product(d[0], d[j - 1], d[j]));
  return exact(total);
}
long long matrix_chain_right_cost(const std::vector<int>& d) {
  validate(d);
  std::uint64_t total = 0;
  if (d.size() > 2)
    for (std::size_t i = d.size() - 2; i > 0; --i)
      total = sum(total, 0, product(d[i - 1], d[i], d.back()));
  return exact(total);
}
std::vector<int> matrix_chain_demo_dimensions() {
  return {30, 35, 15, 5, 10, 20, 25};
}
} // namespace datamunge::algorithms
