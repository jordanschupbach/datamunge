/// @file timsort.cpp
/// @brief Original deterministic prefix-shuffle fixture; generic sorting lives in the public header.
#include <datamunge/algorithms/timsort.hpp>
#include <numeric>
#include <random>
namespace datamunge::algorithms {
std::vector<std::vector<int>> timsort_example_arrays() {
  std::mt19937_64  rng(9);
  std::vector<int> base(1000);
  std::iota(base.begin(), base.end(), 0);
  std::vector<std::vector<int>> out;
  for (int pct = 0; pct <= 100; pct += 10) {
    auto a = base;
    std::shuffle(a.begin(), a.begin() + 1000 * pct / 100, rng);
    out.push_back(std::move(a));
  }
  return out;
}
} // namespace datamunge::algorithms
