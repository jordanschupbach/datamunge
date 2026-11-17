/// @file samplesort.cpp
/// @brief Native reproducible distribution-sort workload.
#include <datamunge/algorithms/samplesort.hpp>
/// @brief Canonical checked native algorithm producers and owned outputs.
namespace datamunge::algorithms {
std::vector<int> samplesort_example_values(std::size_t n, std::uint64_t seed, std::size_t cap) {
  if ((cap && n > cap) || n > std::vector<int>().max_size())
    throw std::length_error("samplesort: workload capacity");
  std::mt19937_64  rng(seed);
  std::vector<int> a;
  a.reserve(n);
  for (std::size_t i = 0; i < n; ++i)
    a.push_back(static_cast<int>(rng() % 1000000));
  return a;
}
} // namespace datamunge::algorithms
