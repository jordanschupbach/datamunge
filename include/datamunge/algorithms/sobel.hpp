#pragma once

/// \file sobel.hpp
/// \brief The Sobel operator: detect edges by estimating the image intensity gradient
///        with two 3x3 convolution kernels.
///
/// Edges are where image intensity changes sharply -- i.e. where the *gradient* is large.
/// The *Sobel operator* estimates the horizontal and vertical gradients with two 3x3
/// kernels that combine differentiation in one direction with light smoothing in the
/// other (the smoothing suppresses noise):
/// \f[
///   G_x=\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix},\quad
///   G_y=\begin{bmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{bmatrix}.
/// \f]
/// The gradient *magnitude* \f$\sqrt{G_x^2+G_y^2}\f$ is large on edges and near zero in
/// flat regions, giving an edge map; the direction \f$\operatorname{atan2}(G_y,G_x)\f$
/// gives edge orientation. Sobel is the classic first-derivative edge detector and the
/// gradient stage inside Canny.

#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// Sobel gradients and magnitude of a grayscale image.
struct SobelResult {
    std::vector<std::vector<double>> gx;         ///< Horizontal gradient.
    std::vector<std::vector<double>> gy;         ///< Vertical gradient.
    std::vector<std::vector<double>> magnitude;  ///< sqrt(gx^2 + gy^2).
};

/// \brief Apply the Sobel operator to a grayscale image (border pixels get zero gradient).
inline SobelResult sobel(const std::vector<std::vector<double>>& img) {
    const std::size_t rows = img.size();
    const std::size_t cols = rows ? img.front().size() : 0;
    SobelResult       r;
    r.gx.assign(rows, std::vector<double>(cols, 0.0));
    r.gy.assign(rows, std::vector<double>(cols, 0.0));
    r.magnitude.assign(rows, std::vector<double>(cols, 0.0));
    static const int kx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    static const int ky[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    for (std::size_t i = 1; i + 1 < rows; ++i)
        for (std::size_t j = 1; j + 1 < cols; ++j) {
            double sx = 0.0, sy = 0.0;
            for (int di = -1; di <= 1; ++di)
                for (int dj = -1; dj <= 1; ++dj) {
                    const double v = img[i + di][j + dj];
                    sx += kx[di + 1][dj + 1] * v;
                    sy += ky[di + 1][dj + 1] * v;
                }
            r.gx[i][j]        = sx;
            r.gy[i][j]        = sy;
            r.magnitude[i][j] = std::sqrt(sx * sx + sy * sy);
        }
    return r;
}

}  // namespace datamunge::algorithms
