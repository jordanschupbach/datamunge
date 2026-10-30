/// @file gradient.cpp
/// @brief Sobel gradient field on a grey-level grid.

#include <datamunge/cv/gradient.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace datamunge::cv {

GradientField sobel_gradient(const std::vector<double>& gray, int width, int height) {
    if (width < 1 || height < 1 || gray.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(height))
        throw std::invalid_argument("sobel_gradient: gray must hold width*height values");
    GradientField g{width, height, std::vector<double>(gray.size()), std::vector<double>(gray.size())};
    auto at = [&](int x, int y) {
        x = std::clamp(x, 0, width - 1);
        y = std::clamp(y, 0, height - 1);
        return gray[static_cast<std::size_t>(y) * width + x];
    };
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            const double dx = (at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)) -
                              (at(x - 1, y - 1) + 2 * at(x - 1, y) + at(x - 1, y + 1));
            const double dy = (at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)) -
                              (at(x - 1, y - 1) + 2 * at(x, y - 1) + at(x + 1, y - 1));
            const std::size_t i = static_cast<std::size_t>(y) * width + x;
            g.gx[i]             = dx / 8.0;
            g.gy[i]             = dy / 8.0;
        }
    return g;
}

}  // namespace datamunge::cv
