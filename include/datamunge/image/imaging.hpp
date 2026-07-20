#pragma once

// Convenience umbrella header -- includes the whole image-processing module: the core Image
// class and pixel type, color-space conversion, geometric transforms (resize/crop/rotate/
// flip), convolution-based filters and point operations, ImageDraw-style drawing primitives,
// and file I/O for Netpbm (PPM/PGM/PBM), BMP, and PNG.
//
// Named "imaging.hpp" rather than "image.hpp" (this module's usual umbrella-matches-directory-
// name convention, e.g. geometry/geometry.hpp) to avoid colliding with image/image.hpp, which
// is already taken by the core Image class itself.

#include <datamunge/image/bmp.hpp>
#include <datamunge/image/color.hpp>
#include <datamunge/image/draw.hpp>
#include <datamunge/image/filters.hpp>
#include <datamunge/image/image.hpp>
#include <datamunge/image/netpbm.hpp>
#include <datamunge/image/png.hpp>
#include <datamunge/image/transform.hpp>
