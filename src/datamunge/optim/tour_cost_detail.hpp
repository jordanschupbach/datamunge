#pragma once

/// @file
/// @brief Shape and numeric validation shared by symmetric tour optimizers.

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace datamunge::optim::detail {

/// @brief Validate complete symmetric tour costs and obtain a normalization scale.
/// @param distances Square matrix with at least three rows, diagonal zero and positive edge costs.
/// @return Smallest positive edge cost.
/// @throws std::invalid_argument Malformed graph or costs outside the finite tour range.
inline long double validate_tour_costs(const std::vector<std::vector<double>>& distances) {
  const auto n = distances.size();
  if (n < 3) throw std::invalid_argument("Tour costs require at least three cities");
  for (const auto& row : distances) {
    if (row.size() != n) throw std::invalid_argument("Distances must be a square matrix");
  }
  long double minimum = std::numeric_limits<double>::max();
  for (std::size_t i = 0; i < n; ++i) {
    if (distances[i][i] != 0) throw std::invalid_argument("Distance diagonal must be zero");
    for (std::size_t j = i + 1; j < n; ++j) {
      const auto cost = distances[i][j];
      if (!(cost > 0 && static_cast<long double>(cost) * n <= std::numeric_limits<double>::max()) ||
          cost != distances[j][i]) {
        throw std::invalid_argument("Distances must be positive, symmetric and finite for a tour");
      }
      minimum = std::min(minimum, static_cast<long double>(cost));
    }
  }
  return minimum;
}

/// @brief Sum the length of a valid permutation including its closing edge.
/// @param distances Valid symmetric cost matrix.
/// @param tour Nonempty permutation of the city identifiers.
/// @return Complete tour length.
inline long double tour_cost(const std::vector<std::vector<double>>& distances,
                             const std::vector<std::size_t>& tour) {
  long double total = 0;
  auto previous = tour.back();
  for (const auto city : tour) {
    total += distances[previous][city];
    previous = city;
  }
  return total;
}

} // namespace datamunge::optim::detail
