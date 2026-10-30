#pragma once

/// @file gradient.hpp
/// @brief Sobel image gradients on a plain row-major grey-level grid.

#include <vector>

namespace datamunge::cv {

/// @brief Per-pixel image derivatives, row-major like the input.
struct GradientField {
    int                 width{0};   ///< columns.
    int                 height{0};  ///< rows.
    std::vector<double> gx;         ///< horizontal derivative (positive when intensity rises to the right).
    std::vector<double> gy;         ///< vertical derivative (positive when intensity rises downwards).
};

/// @brief Sobel derivatives of a grey-level grid, scaled to estimate the true partial derivatives.
///
/// Uses the 3x3 Sobel kernels divided by 8, so a linear ramp of slope 1 gives a derivative of 1.
/// Edges are handled by clamping (replicating the border pixels).
/// @param gray row-major intensities, @c width*height values.
/// @param width columns (at least 1).
/// @param height rows (at least 1).
/// @return the gradient field.
/// @throws std::invalid_argument if the sizes do not match.
GradientField sobel_gradient(const std::vector<double>& gray, int width, int height);

}  // namespace datamunge::cv
