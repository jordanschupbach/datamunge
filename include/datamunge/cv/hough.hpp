#pragma once

#include <datamunge/cv/detail/gaussian_double.hpp>
#include <datamunge/image/image.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::cv {

/// @brief A detected line in the classical (rho, theta) normal-form parameterization: every
///        point (x, y) on the line satisfies x*cos(theta) + y*sin(theta) = rho. Represents an
///        INFINITE line, not a segment -- see line_endpoints() to clip it to an image.
struct HoughLine {
    double rho{0.0};
    double theta{0.0};
    int votes{0};
};

/// @brief The two points where the infinite line @p line crosses the border of a @p
///        image_width x @p image_height image, for drawing/visualization. Returns false (and
///        leaves the output points unchanged) if the line doesn't intersect the image
///        rectangle at all (only possible for a line entirely outside it).
[[nodiscard]] inline bool line_endpoints(const HoughLine& line, int image_width, int image_height, std::pair<int, int>& p1,
                                          std::pair<int, int>& p2) {
    const double cos_t = std::cos(line.theta);
    const double sin_t = std::sin(line.theta);
    std::vector<std::pair<double, double>> hits;

    // Intersect with all four border lines and keep the ones landing inside the image.
    if (std::abs(sin_t) > 1e-9) {
        for (const int x : {0, image_width - 1}) {
            const double y = (line.rho - x * cos_t) / sin_t;
            if (y >= 0.0 && y <= image_height - 1) hits.emplace_back(x, y);
        }
    }
    if (std::abs(cos_t) > 1e-9) {
        for (const int y : {0, image_height - 1}) {
            const double x = (line.rho - y * sin_t) / cos_t;
            if (x >= 0.0 && x <= image_width - 1) hits.emplace_back(x, y);
        }
    }
    if (hits.size() < 2) {
        return false;
    }
    p1 = {static_cast<int>(std::lround(hits.front().first)), static_cast<int>(std::lround(hits.front().second))};
    p2 = {static_cast<int>(std::lround(hits.back().first)), static_cast<int>(std::lround(hits.back().second))};
    return true;
}

/// @brief Standard Hough line transform: every pixel of @p img whose grayscale luma is >= @p
///        edge_threshold votes, in a rho-theta accumulator (theta stepped every degree over
///        [0, pi), rho stepped every pixel over the image's diagonal range), for every line
///        that could pass through it. Returns every accumulator cell with at least @p
///        min_votes, non-max-suppressed within a (2*nms_radius+1) rho-theta cell window so a
///        single real line doesn't produce a cluster of near-duplicate near-identical results.
///        Intended input is an edge map (e.g. image::sobel_edges()'s output), not a raw photo.
[[nodiscard]] inline std::vector<HoughLine> hough_lines(const image::Image& img, int edge_threshold = 128, int min_votes = 50,
                                                          int nms_radius = 5) {
    const int w = img.width();
    const int h = img.height();
    const std::vector<double> gray = detail::to_double_luma(img);

    constexpr int kNumAngles = 180;
    constexpr double kPi = 3.14159265358979323846;
    const double diagonal = std::sqrt(static_cast<double>(w) * w + static_cast<double>(h) * h);
    const int num_rhos = 2 * static_cast<int>(std::ceil(diagonal)) + 1;
    const int rho_offset = num_rhos / 2;

    std::vector<double> cos_table(kNumAngles), sin_table(kNumAngles);
    for (int t = 0; t < kNumAngles; ++t) {
        const double theta = t * kPi / kNumAngles;
        cos_table[static_cast<std::size_t>(t)] = std::cos(theta);
        sin_table[static_cast<std::size_t>(t)] = std::sin(theta);
    }

    std::vector<int> accumulator(static_cast<std::size_t>(kNumAngles) * static_cast<std::size_t>(num_rhos), 0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (gray[static_cast<std::size_t>(y) * w + static_cast<std::size_t>(x)] < edge_threshold) continue;
            for (int t = 0; t < kNumAngles; ++t) {
                const double rho = x * cos_table[static_cast<std::size_t>(t)] + y * sin_table[static_cast<std::size_t>(t)];
                const int rho_idx = static_cast<int>(std::lround(rho)) + rho_offset;
                if (rho_idx >= 0 && rho_idx < num_rhos) {
                    ++accumulator[static_cast<std::size_t>(t) * num_rhos + static_cast<std::size_t>(rho_idx)];
                }
            }
        }
    }

    std::vector<HoughLine> lines;
    for (int t = 0; t < kNumAngles; ++t) {
        for (int r = 0; r < num_rhos; ++r) {
            const int votes = accumulator[static_cast<std::size_t>(t) * num_rhos + static_cast<std::size_t>(r)];
            if (votes < min_votes) continue;

            bool is_max = true;
            for (int dt = -nms_radius; dt <= nms_radius && is_max; ++dt) {
                const int nt = t + dt;
                if (nt < 0 || nt >= kNumAngles) continue;
                for (int dr = -nms_radius; dr <= nms_radius; ++dr) {
                    if (dt == 0 && dr == 0) continue;
                    const int nr = r + dr;
                    if (nr < 0 || nr >= num_rhos) continue;
                    if (accumulator[static_cast<std::size_t>(nt) * num_rhos + static_cast<std::size_t>(nr)] > votes) {
                        is_max = false;
                        break;
                    }
                }
            }
            if (is_max) {
                lines.push_back(HoughLine{static_cast<double>(r - rho_offset), t * kPi / kNumAngles, votes});
            }
        }
    }
    std::sort(lines.begin(), lines.end(), [](const HoughLine& a, const HoughLine& b) { return a.votes > b.votes; });
    return lines;
}

/// @brief A detected circle: center, radius, and the raw vote count that found it.
struct HoughCircle {
    int cx{0};
    int cy{0};
    int radius{0};
    int votes{0};
};

/// @brief Gradient-direction-guided Hough circle transform: for every edge pixel (Sobel
///        gradient magnitude >= @p gradient_threshold on @p img's grayscale conversion), and
///        for every candidate radius in [@p min_radius, @p max_radius], votes for the two
///        candidate centers at that distance along the pixel's gradient line (handles both
///        bright-disk-on-dark and dark-disk-on-bright edge polarity). Each radius gets an
///        independent 2D (cx, cy) accumulator and non-max suppression pass -- simpler to
///        reason about than a combined 3D (cx, cy, r) accumulator, at the cost of not
///        suppressing near-duplicate detections across adjacent radii.
[[nodiscard]] inline std::vector<HoughCircle> hough_circles(const image::Image& img, int min_radius, int max_radius,
                                                              double gradient_threshold = 50.0, int min_votes = 20, int nms_radius = 10) {
    if (min_radius <= 0 || max_radius < min_radius) {
        throw std::invalid_argument("hough_circles: require 0 < min_radius <= max_radius");
    }
    const int w = img.width();
    const int h = img.height();
    const std::vector<double> gray = detail::to_double_luma(img);

    static constexpr int kGx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    static constexpr int kGy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    struct EdgePoint {
        int x, y;
        double angle;
    };
    std::vector<EdgePoint> edges;
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            double gx = 0.0, gy = 0.0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    const double v = gray[static_cast<std::size_t>(y + ky) * w + static_cast<std::size_t>(x + kx)];
                    gx += kGx[ky + 1][kx + 1] * v;
                    gy += kGy[ky + 1][kx + 1] * v;
                }
            }
            const double magnitude = std::sqrt(gx * gx + gy * gy);
            if (magnitude >= gradient_threshold) {
                edges.push_back(EdgePoint{x, y, std::atan2(gy, gx)});
            }
        }
    }

    std::vector<HoughCircle> results;
    for (int radius = min_radius; radius <= max_radius; ++radius) {
        std::vector<int> accumulator(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
        const auto vote = [&](int cx, int cy) {
            if (cx >= 0 && cx < w && cy >= 0 && cy < h) ++accumulator[static_cast<std::size_t>(cy) * w + static_cast<std::size_t>(cx)];
        };
        for (const auto& e : edges) {
            const double dx = radius * std::cos(e.angle);
            const double dy = radius * std::sin(e.angle);
            vote(static_cast<int>(std::lround(e.x - dx)), static_cast<int>(std::lround(e.y - dy)));
            vote(static_cast<int>(std::lround(e.x + dx)), static_cast<int>(std::lround(e.y + dy)));
        }

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int votes = accumulator[static_cast<std::size_t>(y) * w + static_cast<std::size_t>(x)];
                if (votes < min_votes) continue;
                bool is_max = true;
                for (int dy = -nms_radius; dy <= nms_radius && is_max; ++dy) {
                    const int ny = y + dy;
                    if (ny < 0 || ny >= h) continue;
                    for (int dx = -nms_radius; dx <= nms_radius; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        const int nx = x + dx;
                        if (nx < 0 || nx >= w) continue;
                        if (accumulator[static_cast<std::size_t>(ny) * w + static_cast<std::size_t>(nx)] > votes) {
                            is_max = false;
                            break;
                        }
                    }
                }
                if (is_max) results.push_back(HoughCircle{x, y, radius, votes});
            }
        }
    }
    std::sort(results.begin(), results.end(), [](const HoughCircle& a, const HoughCircle& b) { return a.votes > b.votes; });
    return results;
}

} // namespace datamunge::cv
