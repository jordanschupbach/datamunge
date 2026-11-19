/// @file
/// @brief Two-dimensional behavior archive for MAP-Elites.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::rl {

/// @brief Retain the highest-scoring candidate in each behavior bin.
///
/// Both behavior axes are regularly spaced and start at zero. Values at or
/// above an axis's upper boundary enter its final bin. Earlier arrivals
/// win exact score ties. The archive owns each candidate's parameter vector.
/// @par Example
/// @code
/// dm::rl::MapElites archive({8,8},{0.25,0.15});
/// const auto [cell,replaced]=archive.insert({0.5,0.15},-2.,{0.,1.});
/// std::print("{}\n",cell); // 17
/// @endcode
/// @par References
/// - Mouret and Clune (2015), Illuminating search spaces by mapping elites.
///   https://arxiv.org/abs/1504.04909
class MapElites {
 public:
  /// @brief Candidate retained as the elite of one cell.
  /// @par Example
  /// @code
  /// dm::rl::MapElites archive({2,2},{1.,1.});
  /// archive.insert({0.,0.},1.,{0.5});
  /// const dm::rl::MapElites::Elite& elite=archive.cells().front().value();
  /// std::print("{:.1f}\n",elite.score); // 1.0
  /// @endcode
  struct Elite {
    std::vector<double> parameters;  ///< Owned candidate parameters.
    double score;                     ///< Finite objective to maximize.
    std::size_t id;                   ///< Zero-based index in the insertion stream.
  };

  /// @brief Create an empty regular grid.
  /// @param bins Number of cells on each axis, both positive.
  /// @param widths Finite positive bin widths.
  /// @throws std::invalid_argument Invalid geometry.
  /// @par Example
  /// @code
  /// const dm::rl::MapElites archive({8,8},{0.25,0.15});
  /// std::print("{}\n",archive.cells().size()); // 64
  /// @endcode
  MapElites(std::array<std::size_t,2> bins,std::array<double,2> widths)
      : bins_(bins),widths_(widths) {
    if (bins[0]==0 || bins[1]==0 || bins[0]>cells_.max_size()/bins[1] ||
        !(widths[0]>0 && std::isfinite(widths[0]) && widths[1]>0 && std::isfinite(widths[1]))) {
      throw std::invalid_argument("MapElites: expected positive bin counts and widths");
    }
    cells_.resize(bins[0]*bins[1]);
  }

  /// @brief Compare a candidate with the incumbent in its behavior cell.
  /// @param behavior Finite nonnegative descriptor values.
  /// @param score Finite objective to maximize.
  /// @param parameters Candidate parameter vector, moved into the cell on improvement.
  /// @return Flattened cell index and whether the candidate became its elite.
  /// @throws std::invalid_argument Nonfinite score or invalid descriptor.
  /// @throws std::overflow_error The insertion stream exhausted its index range.
  /// @par Example
  /// @code
  /// dm::rl::MapElites archive({2,2},{1.,1.});
  /// archive.insert({0.,0.},2.,{0.5});
  /// const auto [cell,replaced]=archive.insert({0.,0.},1.,{1.});
  /// std::print("{}\n",replaced); // false
  /// @endcode
  std::pair<std::size_t,bool> insert(std::array<double,2> behavior,double score,
                                     std::vector<double> parameters) {
    if (!(behavior[0]>=0 && std::isfinite(behavior[0]) &&
          behavior[1]>=0 && std::isfinite(behavior[1]) && std::isfinite(score))) {
      throw std::invalid_argument("MapElites: invalid descriptor or score");
    }
    if (next_id_==std::numeric_limits<std::size_t>::max()) {
      throw std::overflow_error("MapElites: exhausted candidate IDs");
    }
    std::array<std::size_t,2> coordinates{};
    for (std::size_t axis=0;axis<coordinates.size();++axis) {
      const double index=behavior[axis]/widths_[axis];
      coordinates[axis]=index>=static_cast<double>(bins_[axis]-1) ?
          bins_[axis]-1 : static_cast<std::size_t>(index);
    }
    const auto cell=coordinates[0]*bins_[1]+coordinates[1];
    auto& incumbent=cells_[cell];
    const auto id=next_id_++;
    if (incumbent && incumbent->score>=score) return {cell,false};
    incumbent=Elite{std::move(parameters),score,id};
    return {cell,true};
  }

  /// @brief Access archive cells in row-major behavior order.
  /// @return Borrowed cells with empty optionals for unoccupied bins. Insertion can replace an elite.
  [[nodiscard]] const std::vector<std::optional<Elite>>& cells() const { return cells_; }

 private:
  std::array<std::size_t,2> bins_;     ///< Cell count for each behavior axis.
  std::array<double,2> widths_;       ///< Spatial width of cells on each axis.
  std::vector<std::optional<Elite>> cells_; ///< Grid with the second axis varying fastest.
  std::size_t next_id_=0;             ///< Index of the next valid candidate.
};

} // namespace datamunge::rl
