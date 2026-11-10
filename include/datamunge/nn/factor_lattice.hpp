#pragma once
/// @file factor_lattice.hpp
/// @brief Exhaustive binary circle/square images with a known compositional holdout protocol.
#include <array>
#include <cstdint>
#include <vector>

#include "datamunge/nn/io.hpp"

namespace datamunge::nn::models {
/// @brief Generate all 294 original teaching-lattice images and their fixed compositional splits.
///
/// Enumerates shape=0(circle),1(square), size=0..2, x=0..6, y=0..6 in that
/// lexicographic order. Radius=size+2 and centre=(3*x+5,3*y+5) in 28x28 pixels.
/// Circle pixels satisfy dx*dx+dy*dy<=radius*radius; square pixels satisfy
/// abs(dx)<=radius and abs(dy)<=radius. Boundaries are inclusive; no antialiasing,
/// randomization, normalization, external data or generator state is involved.
/// Source ID=(((shape*3+size)*7+x)*7+y). Residue=(x+2*y+size+3*shape)%7 assigns
/// test=0, validation=1, fit=2..6, giving 210/42/42 unique binary images. Each
/// factor level is balanced within each split. Rows retain ascending source IDs.
/// This is a local procedural fixture, not a downloaded benchmark or dSprites.
///
/// Returns nine owning detached tensors: <split>_images Float32[N,28,28,1],
/// <split>_factors Int32[N,4] and <split>_source_ids Int32[N], with splits in
/// fit/validation/test order. Separate calls own independent storage and are
/// thread independent. Work/storage O(294*784); allocation/backend errors propagate.
/// @return Named image, factor-index and source-ID arrays for all three splits.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto data=dm::nn::models::KnownFactorLattice();
/// // Nine arrays; data[0] is fit_images [210,28,28,1], with binary Float32 pixels.
/// dm::nn::save_npz("build/known_factors.npz",data);
/// @endcode
/// @see datasets::prepare_example_data(), save_npz(), load_npz()
inline NamedTensors KnownFactorLattice() {
  std::array<std::vector<float>, 3>        images;
  std::array<std::vector<std::int32_t>, 3> factors, ids;
  for (int shape = 0; shape < 2; ++shape)
    for (int size = 0; size < 3; ++size)
      for (int x = 0; x < 7; ++x)
        for (int y = 0; y < 7; ++y) {
          const int residue = (x + 2 * y + size + 3 * shape) % 7;
          const int split   = residue == 0 ? 2 : residue == 1 ? 1 : 0;
          ids[split].push_back(((shape * 3 + size) * 7 + x) * 7 + y);
          factors[split].insert(factors[split].end(), {shape, size, x, y});
          const int radius = size + 2, cx = 3 * x + 5, cy = 3 * y + 5;
          for (int row = 0; row < 28; ++row)
            for (int column = 0; column < 28; ++column) {
              const int dx = column - cx, dy = row - cy;
              images[split].push_back(
                  shape == 0 ? dx * dx + dy * dy <= radius * radius
                             : dx >= -radius && dx <= radius && dy >= -radius && dy <= radius);
            }
        }
  NamedTensors result;
  for (int split = 0; split < 3; ++split) {
    const std::string prefix = split == 0 ? "fit_" : split == 1 ? "validation_" : "test_";
    const auto        count  = static_cast<std::int64_t>(ids[split].size());
    result.emplace_back(prefix + "images", from_vector(images[split], {count, 28, 28, 1}));
    result.emplace_back(prefix + "factors", from_vector(factors[split], {count, 4}));
    result.emplace_back(prefix + "source_ids", from_vector(ids[split], {count}));
  }
  return result;
}
} // namespace datamunge::nn::models
