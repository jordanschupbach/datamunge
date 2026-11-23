#pragma once
/// @file simplex_detail.hpp
/// @brief Parameter checks for simplex probabilities and sampling.

#include <cmath>
#include <stdexcept>
#include <vector>

/// @brief Validation shared by the simplex probability and sampling routines.
namespace datamunge::random::simplex_detail {

/// @brief Checks that the concentration vector is nonempty and positive and finite.
/// @param concentration Proposed shape for every category.
/// @throws std::invalid_argument Empty vector or invalid concentration.
inline void check_concentration(const std::vector<double>& concentration) {
  if (concentration.empty())
    throw std::invalid_argument("Dirichlet: need at least one concentration");
  for (const auto shape : concentration)
    if (!(shape > 0 && std::isfinite(shape)))
      throw std::invalid_argument("Dirichlet: concentrations must be finite and positive");
}

} // namespace datamunge::random::simplex_detail
