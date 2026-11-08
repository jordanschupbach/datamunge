#pragma once
/// @file gaussian_blobs.hpp
/// @brief Owned labelled spherical Gaussian clouds for small native examples.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include <utility>
namespace datamunge::datasets {
/// @brief Owned numeric features and zero-based centre labels for Gaussian clouds.
/// Rows are grouped by centre, then draw order; coordinates are unscaled. Copies are
/// independent. The generator stores no references to centres or shared RNG state.
/// @par Example
/// @code
/// const auto b=dm::datasets::gaussian_blobs({{0,0},{6,0}},2,0,5);
/// std::print("{} {}\n",b.features.size(),b.labels); // 4 [0, 0, 1, 1]
/// @endcode
/// @see gaussian_blobs()
struct GaussianBlobs {
  std::vector<std::vector<double>> features; ///< Owned row-major continuous coordinates.
  std::vector<int> labels; ///< Matching zero-based centre id for each row.
};
/// @brief Generate equally sized isotropic Gaussian clouds with a local seeded RNG.
/// Each coordinate is centre plus an independent normal draw of standard deviation
/// sd, in centre/row/coordinate order. Uses mt19937_64 and normal_distribution<double>;
/// same-toolchain repeats are identical, but normal draws may differ across standard
/// libraries. At sd zero, centres repeat exactly without draws. Empty centres or zero
/// rows return empty arrays after validation. Nonempty centres must have equal positive
/// dimension and finite values. Inputs are unchanged; owned result has O(k*n*d) work
/// and storage for k centres, n rows per centre and d coordinates. No shared state.
/// @param centres Finite rectangular centre coordinates, empty allowed.
/// @param rows_per_centre Rows generated for each centre; default sixty.
/// @param sd Finite nonnegative common coordinate standard deviation; default 0.6.
/// @param seed Local mt19937_64 seed; default five.
/// @return Owned features and integer labels in centre order.
/// @throws std::invalid_argument Invalid sd, ragged/zero-width or nonfinite centres.
/// @throws std::length_error Total rows overflow size_t or centre count exceeds int labels.
/// @throws std::overflow_error A generated coordinate is nonfinite.
/// @par Example
/// @code
/// const auto b=dm::datasets::gaussian_blobs({{0,0},{6,0},{3,6}});
/// std::print("{} {}\n",b.features.size(),b.features[0].size()); // 180 2
/// @endcode
/// @see GaussianBlobs
inline GaussianBlobs gaussian_blobs(const std::vector<std::vector<double>>& centres,
    std::size_t rows_per_centre=60,double sd=0.6,std::uint64_t seed=5) {
  if (!std::isfinite(sd) || sd<0) throw std::invalid_argument("gaussian_blobs: invalid sd");
  const auto dimension=centres.empty()?0:centres.front().size();
  for (const auto& centre:centres) {
    if (!dimension || centre.size()!=dimension) throw std::invalid_argument("gaussian_blobs: invalid centre shape");
    for (double value:centre) if(!std::isfinite(value)) throw std::invalid_argument("gaussian_blobs: nonfinite centre");
  }
  if (centres.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      (!centres.empty() && rows_per_centre>std::numeric_limits<std::size_t>::max()/centres.size()))
    throw std::length_error("gaussian_blobs: row/label capacity exceeded");
  GaussianBlobs result;const auto total=centres.size()*rows_per_centre;
  result.features.reserve(total);result.labels.reserve(total);
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> noise(0,sd>0?sd:1);
  for(std::size_t c=0;c<centres.size();++c) for(std::size_t i=0;i<rows_per_centre;++i) {
    auto row=centres[c];
    for(double& value:row) {
      if(sd>0) value+=noise(rng);
      if(!std::isfinite(value)) throw std::overflow_error("gaussian_blobs: nonfinite draw");
    }
    result.features.push_back(std::move(row));result.labels.push_back(static_cast<int>(c));
  }
  return result;
}
} // namespace datamunge::datasets
