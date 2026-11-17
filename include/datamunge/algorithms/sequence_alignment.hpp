#pragma once
/// @file sequence_alignment.hpp
/// @brief Compatibility aggregation for dedicated checked native producers.
#include <cstddef>
#include <datamunge/algorithms/alignment_types.hpp>
#include <datamunge/algorithms/hirschberg.hpp>
#include <datamunge/algorithms/needleman_wunsch.hpp>
#include <datamunge/algorithms/smith_waterman.hpp>
#include <string>
#include <utility>
#include <vector>

/// @brief Canonical checked native producers and owned values.
namespace datamunge::algorithms {

// Note: Dynamic Time Warping, the fourth member of the "sequence alignment" family, lives in the
// geometry module as datamunge::geometry::dynamic_time_warping (it aligns numeric series, not
// character strings). See include/datamunge/geometry/dtw.hpp.

} // namespace datamunge::algorithms
