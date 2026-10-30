/// @file addition_chain.cpp
/// @brief Binary and sliding-window chains, chain validation and chain-driven modular powers.

#include <datamunge/algorithms/addition_chain.hpp>

#include <algorithm>
#include <bit>
#include <stdexcept>
#include <unordered_map>

namespace datamunge::algorithms {

std::vector<std::uint64_t> binary_chain(std::uint64_t e) { return window_chain(e, 1); }

std::vector<std::uint64_t> window_chain(std::uint64_t e, int k) {
  if (e == 0) throw std::invalid_argument("window_chain: e must be at least 1");
  if (k < 1 || k > 16) throw std::invalid_argument("window_chain: k must be in [1, 16]");
  const int top = 63 - std::countl_zero(e);

  // Split the bits into windows: (start bit, odd value) pairs, scanning from the top.
  std::vector<std::pair<int, std::uint64_t>> windows;
  for (int b = top; b >= 0;) {
    if (((e >> b) & 1U) == 0U) {
      --b;
      continue;
    }
    int low = std::max(b - k + 1, 0);
    while (((e >> low) & 1U) == 0U) ++low;  // a window ends in a 1 bit
    windows.emplace_back(low, (e >> low) & ((std::uint64_t{1} << (b - low + 1)) - 1U));
    b = low - 1;
  }
  std::uint64_t largest = 1;
  for (const auto& w : windows) largest = std::max(largest, w.second);

  std::vector<std::uint64_t> chain{1};
  if (largest > 1) {
    chain.push_back(2);
    for (std::uint64_t odd = 3; odd <= largest; odd += 2) chain.push_back(odd);
  }
  // Invariant: cur == e >> position. Each window shifts cur left to its low bit and adds its value.
  std::uint64_t cur = windows.front().second;
  int position = windows.front().first;
  for (std::size_t w = 1; w < windows.size(); ++w) {
    for (; position > windows[w].first; --position) chain.push_back(cur *= 2);
    chain.push_back(cur += windows[w].second);
  }
  for (; position > 0; --position) chain.push_back(cur *= 2);  // trailing zero bits
  std::sort(chain.begin(), chain.end());
  chain.erase(std::unique(chain.begin(), chain.end()), chain.end());
  return chain;
}

bool is_addition_chain(const std::vector<std::uint64_t>& chain) {
  if (chain.empty() || chain.front() != 1) return false;
  for (std::size_t k = 1; k < chain.size(); ++k) {
    if (chain[k] <= chain[k - 1]) return false;
    // Two-pointer search for chain[i] + chain[j] == chain[k] among the sorted earlier terms.
    std::size_t lo = 0, hi = k - 1;
    bool found = false;
    while (lo <= hi) {
      const std::uint64_t s = chain[lo] + chain[hi];
      if (s == chain[k]) {
        found = true;
        break;
      }
      if (s < chain[k]) ++lo;
      else if (hi == 0) break;
      else --hi;
    }
    if (!found) return false;
  }
  return true;
}

std::vector<std::pair<std::size_t, std::size_t>> chain_steps(const std::vector<std::uint64_t>& chain) {
  if (!is_addition_chain(chain)) throw std::invalid_argument("chain_steps: not an addition chain");
  std::unordered_map<std::uint64_t, std::size_t> index;
  std::vector<std::pair<std::size_t, std::size_t>> steps;
  index[chain[0]] = 0;
  for (std::size_t k = 1; k < chain.size(); ++k) {
    // The largest earlier term whose complement is also an earlier term.
    std::pair<std::size_t, std::size_t> step{k, k};
    for (std::size_t i = k; i-- > 0;) {
      const auto j = index.find(chain[k] - chain[i]);
      if (j != index.end() && j->second <= i) {
        step = {i, j->second};
        break;
      }
    }
    steps.push_back(step);
    index[chain[k]] = k;
  }
  return steps;
}

std::uint64_t mod_pow_chain(std::uint64_t base, const std::vector<std::uint64_t>& chain, std::uint64_t modulus) {
  if (modulus == 0) throw std::invalid_argument("mod_pow_chain: modulus must be positive");
  const auto steps = chain_steps(chain);
  std::vector<std::uint64_t> power{base % modulus};
  for (const auto& [i, j] : steps)
    power.push_back(static_cast<std::uint64_t>(static_cast<unsigned __int128>(power[i]) * power[j] % modulus));
  return modulus == 1 ? 0 : power.back();
}

int chain_lower_bound(std::uint64_t e) {
  if (e == 0) throw std::invalid_argument("chain_lower_bound: e must be at least 1");
  const int lambda = 63 - std::countl_zero(e), nu = std::popcount(e);
  return lambda + (nu >= 2 ? 1 : 0) + (nu >= 3 ? 1 : 0);
}

} // namespace datamunge::algorithms
