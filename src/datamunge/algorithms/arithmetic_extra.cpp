#include <cmath>
#include <cstdint>
#include <datamunge/algorithms/arithmetic_extra.hpp>
#include <functional>
#include <vector>

namespace datamunge::algorithms {

std::vector<std::uint64_t> addition_chain(std::uint64_t e) {
  // Binary-method chain (always valid), used as-is for large e.
  auto binary_chain = [](std::uint64_t e) {
    std::vector<std::uint64_t> chain{1};
    if (e == 1)
      return chain;
    int hb = 63;
    while (!((e >> hb) & 1))
      --hb;
    std::uint64_t cur = 1;
    for (int b = hb - 1; b >= 0; --b) {
      cur *= 2;
      chain.push_back(cur);
      if ((e >> b) & 1) {
        cur += 1;
        chain.push_back(cur);
      }
    }
    return chain;
  };

  if (e <= 1)
    return {1};
  if (e > 4096)
    return binary_chain(e); // optimal search only for small e

  // Iterative-deepening DFS for a shortest chain.
  std::vector<std::uint64_t> chain{1};
  std::vector<std::uint64_t> result;
  std::function<bool(int)>   dfs = [&](int max_len) {
    const std::uint64_t last = chain.back();
    if (last == e) {
      result = chain;
      return true;
    }
    if (static_cast<int>(chain.size()) >= max_len)
      return false;
    // Optimistic bound: doubling each remaining step must be able to reach e.
    std::uint64_t reach = last;
    for (int k = static_cast<int>(chain.size()); k < max_len; ++k)
      reach <<= 1;
    if (reach < e)
      return false;
    // Try sums of pairs, larger first, keeping the chain strictly increasing.
    for (int i = static_cast<int>(chain.size()) - 1; i >= 0; --i)
      for (int j = i; j >= 0; --j) {
        const std::uint64_t s = chain[i] + chain[j];
        if (s <= last || s > e)
          continue;
        chain.push_back(s);
        if (dfs(max_len))
          return true;
        chain.pop_back();
      }
    return false;
  };

  const auto fallback = binary_chain(e);
  for (int len = 1; len <= static_cast<int>(fallback.size()); ++len) {
    chain.assign(1, 1);
    if (dfs(len))
      return result;
  }
  return fallback;
}

} // namespace datamunge::algorithms
