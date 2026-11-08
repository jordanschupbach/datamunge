/// @file
/// @brief Virtual class counts, exact class aggregation and original Flashsort sample generation.
#include <datamunge/algorithms/flashsort.hpp>
#include <random>

/// @brief Canonical numerical and combinatorial algorithms.
namespace datamunge::algorithms {
std::size_t flashsort_class_count(std::size_t n) {
  if (n > static_cast<std::size_t>(INT_MAX))
    throw std::length_error("Flashsort signed input length");
  if (n <= 1)
    return 0;
  return static_cast<std::size_t>(std::max(2, static_cast<int>(.43 * static_cast<int>(n))));
}
std::vector<FlashsortClassGroup> flashsort_class_groups(
    const std::vector<std::size_t>& counts,
    std::size_t                     groups,
    std::size_t                     max_groups) {
  if (!groups)
    throw std::invalid_argument("Flashsort groups must be positive");
  unsigned __int128 total = 0;
  for (auto count : counts) {
    total += count;
    if (total > std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("Flashsort occupancy total overflow");
  }
  if ((max_groups && groups > max_groups) || groups > std::vector<FlashsortClassGroup>{}.max_size())
    throw std::length_error("Flashsort group capacity");
  std::vector<FlashsortClassGroup> out;
  out.reserve(groups);
  for (std::size_t g = 0; g < groups; ++g) {
    const auto begin = static_cast<std::size_t>(
                   static_cast<unsigned __int128>(g) * counts.size() / groups),
               end = static_cast<std::size_t>(
                   static_cast<unsigned __int128>(g + 1) * counts.size() / groups);
    std::uint64_t sum = 0;
    for (std::size_t k = begin; k < end; ++k)
      sum += counts[k];
    out.push_back({begin, end, sum});
  }
  return out;
}
std::vector<int>
flashsort_uniform_sample(std::size_t count, std::uint64_t seed, int lower, int upper) {
  if (upper < lower)
    throw std::invalid_argument("Flashsort uniform sample bounds");
  if (count > std::vector<int>{}.max_size())
    throw std::length_error("Flashsort sample capacity");
  std::mt19937_64                    rng(seed);
  std::uniform_int_distribution<int> distribution(lower, upper);
  std::vector<int>                   out(count);
  for (auto& value : out)
    value = distribution(rng);
  return out;
}
} // namespace datamunge::algorithms
