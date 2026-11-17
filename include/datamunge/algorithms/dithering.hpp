#pragma once

/// @file dithering.hpp
/// @brief Binary-valued halftoning: native raster error diffusion and ordered Bayer thresholds.
/// Floyd-Steinberg uses an owned working copy and discards outside error shares;
/// arithmetic tone equality and a universal perceptual/speed ranking are not implied.

#include <datamunge/algorithms/floyd_steinberg.hpp>
#include <datamunge/algorithms/ordered_dither.hpp>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

}  // namespace datamunge::algorithms
