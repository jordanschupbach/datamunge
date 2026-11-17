/// @file strand_sort.cpp
/// @brief Original controlled prefix-shuffle input cohort; sort kernels are generic in the header.
#include <datamunge/algorithms/strand_sort.hpp>
#include <numeric>
#include <random>
namespace datamunge::algorithms {
std::vector<std::vector<int>> strand_sort_example_arrays() {
  std::mt19937_64  rng(7);
  std::vector<int> base(400);
  std::iota(base.begin(), base.end(), 0);
  std::vector<std::vector<int>> out;
  for (int percent = 0; percent <= 100; percent += 10) {
    auto values = base;
    std::shuffle(values.begin(), values.begin() + 4 * percent, rng);
    out.push_back(std::move(values));
  }
  return out;
}
} // namespace datamunge::algorithms
