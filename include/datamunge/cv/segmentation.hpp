#pragma once

#include <datamunge/image/color.hpp>
#include <datamunge/image/image.hpp>

#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace datamunge::cv {

/// @brief The result of connected_components(): a per-pixel label image (row-major, same
///        layout as image::Image::data() but one int per pixel rather than per channel byte).
///        Label 0 is always background; foreground components are numbered 1..count.
struct LabelMap {
    std::vector<int> labels;
    int width{0};
    int height{0};
    int count{0};
};

/// @brief Labels every foreground (luma >= 128) connected region of @p img with a distinct
///        positive integer via flood fill (simple breadth-first flood fill per unvisited
///        foreground pixel, rather than the classical two-pass union-find labeling algorithm
///        -- same asymptotic complexity, much easier to verify correct, matching this module's
///        established simplicity-over-cleverness bias). @p eight_connected selects 8-
///        neighborhood (diagonals count) vs. 4-neighborhood (orthogonal only) connectivity.
[[nodiscard]] inline LabelMap connected_components(const image::Image& img, bool eight_connected = true) {
    const image::Image gray = image::to_grayscale(img);
    const int w = gray.width();
    const int h = gray.height();

    LabelMap result;
    result.width = w;
    result.height = h;
    result.labels.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);

    const auto idx = [&](int x, int y) { return static_cast<std::size_t>(y) * w + static_cast<std::size_t>(x); };
    std::vector<std::pair<int, int>> neighbors4 = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    std::vector<std::pair<int, int>> neighbors8 = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    const auto& neighbors = eight_connected ? neighbors8 : neighbors4;

    int next_label = 0;
    std::vector<std::pair<int, int>> stack;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (gray.get_pixel(x, y).r < 128 || result.labels[idx(x, y)] != 0) continue;

            ++next_label;
            stack.clear();
            stack.emplace_back(x, y);
            result.labels[idx(x, y)] = next_label;
            while (!stack.empty()) {
                const auto [cx, cy] = stack.back();
                stack.pop_back();
                for (const auto& [dx, dy] : neighbors) {
                    const int nx = cx + dx, ny = cy + dy;
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                    if (result.labels[idx(nx, ny)] != 0 || gray.get_pixel(nx, ny).r < 128) continue;
                    result.labels[idx(nx, ny)] = next_label;
                    stack.emplace_back(nx, ny);
                }
            }
        }
    }
    result.count = next_label;
    return result;
}

/// @brief Otsu's method: the grayscale threshold (0-255) that maximizes the between-class
///        variance of the pixels it would split into "below" and "at-or-above" groups --
///        the standard automatic threshold selection when a fixed threshold isn't known in
///        advance. Pairs directly with image::threshold(img, otsu_threshold(img)).
[[nodiscard]] inline int otsu_threshold(const image::Image& img) {
    const image::Image gray = image::to_grayscale(img);
    std::array<long, 256> histogram{};
    for (int y = 0; y < gray.height(); ++y) {
        for (int x = 0; x < gray.width(); ++x) {
            ++histogram[gray.get_pixel(x, y).r];
        }
    }

    const long total = static_cast<long>(gray.width()) * static_cast<long>(gray.height());
    if (total == 0) {
        throw std::invalid_argument("otsu_threshold: image must be non-empty");
    }

    double sum_all = 0.0;
    for (int i = 0; i < 256; ++i) sum_all += i * static_cast<double>(histogram[static_cast<std::size_t>(i)]);

    long weight_bg = 0;
    double sum_bg = 0.0;
    double best_variance = -1.0;
    int best_threshold = 0;
    for (int t = 0; t < 256; ++t) {
        weight_bg += histogram[static_cast<std::size_t>(t)];
        if (weight_bg == 0) continue;
        const long weight_fg = total - weight_bg;
        if (weight_fg == 0) break;

        sum_bg += t * static_cast<double>(histogram[static_cast<std::size_t>(t)]);
        const double mean_bg = sum_bg / weight_bg;
        const double mean_fg = (sum_all - sum_bg) / weight_fg;
        const double between_class_variance =
            static_cast<double>(weight_bg) * static_cast<double>(weight_fg) * (mean_bg - mean_fg) * (mean_bg - mean_fg);
        if (between_class_variance > best_variance) {
            best_variance = between_class_variance;
            best_threshold = t + 1; // the split point: pixels >= this value are foreground
        }
    }
    return best_threshold;
}

/// @brief K-means color segmentation: clusters every pixel of @p img by its RGB value into @p
///        k clusters (Lloyd's algorithm, deterministic fixed-seed initialization by sampling k
///        distinct pixels) and returns a new RGB image where every pixel is replaced by its
///        assigned cluster's mean color. Throws std::invalid_argument if @p k exceeds the
///        number of distinct pixels available to seed from.
[[nodiscard]] inline image::Image kmeans_segment(const image::Image& img, int k, int max_iterations = 20, unsigned seed = 42) {
    if (k <= 0) {
        throw std::invalid_argument("kmeans_segment: k must be positive");
    }
    const image::Image rgb = image::to_rgb(img);
    const int w = rgb.width();
    const int h = rgb.height();
    const std::size_t n = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    if (static_cast<std::size_t>(k) > n) {
        throw std::invalid_argument("kmeans_segment: k cannot exceed the number of pixels");
    }

    std::vector<image::Pixel> pixels(n);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) pixels[static_cast<std::size_t>(y) * w + static_cast<std::size_t>(x)] = rgb.get_pixel(x, y);

    // k-means++ initialization: the first centroid is uniform-random, then each subsequent one
    // is chosen with probability proportional to its squared distance from the nearest
    // already-chosen centroid. Plain uniform-random selection of k pixels can (and, on
    // low-cardinality-color synthetic images especially, often does) seed two centroids into
    // the SAME true cluster, leaving another cluster empty for every iteration thereafter --
    // k-means++ makes that failure far less likely by actively favoring well-separated seeds.
    std::mt19937 rng(seed);
    std::vector<std::array<double, 3>> centroids;
    std::vector<double> min_sq_dist(n, -1.0);

    std::uniform_int_distribution<std::size_t> first_pick(0, n - 1);
    {
        const image::Pixel& p = pixels[first_pick(rng)];
        centroids.push_back({static_cast<double>(p.r), static_cast<double>(p.g), static_cast<double>(p.b)});
    }
    for (int c = 1; c < k; ++c) {
        const auto& newest = centroids.back();
        double total_weight = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double dr = pixels[i].r - newest[0], dg = pixels[i].g - newest[1], db = pixels[i].b - newest[2];
            const double dist = dr * dr + dg * dg + db * db;
            if (min_sq_dist[i] < 0.0 || dist < min_sq_dist[i]) min_sq_dist[i] = dist;
            total_weight += min_sq_dist[i];
        }
        std::uniform_real_distribution<double> pick(0.0, total_weight);
        double target = pick(rng);
        std::size_t chosen = n - 1;
        for (std::size_t i = 0; i < n; ++i) {
            target -= min_sq_dist[i];
            if (target <= 0.0) {
                chosen = i;
                break;
            }
        }
        const image::Pixel& p = pixels[chosen];
        centroids.push_back({static_cast<double>(p.r), static_cast<double>(p.g), static_cast<double>(p.b)});
    }

    std::vector<int> assignment(n, 0);
    for (int iter = 0; iter < max_iterations; ++iter) {
        bool changed = false;
        for (std::size_t i = 0; i < n; ++i) {
            const image::Pixel& p = pixels[i];
            int best = 0;
            double best_dist = -1.0;
            for (int c = 0; c < k; ++c) {
                const auto& centroid = centroids[static_cast<std::size_t>(c)];
                const double dr = p.r - centroid[0], dg = p.g - centroid[1], db = p.b - centroid[2];
                const double dist = dr * dr + dg * dg + db * db;
                if (best_dist < 0.0 || dist < best_dist) {
                    best_dist = dist;
                    best = c;
                }
            }
            if (assignment[i] != best) {
                assignment[i] = best;
                changed = true;
            }
        }

        std::vector<std::array<double, 3>> sums(static_cast<std::size_t>(k), {0.0, 0.0, 0.0});
        std::vector<long> counts(static_cast<std::size_t>(k), 0);
        for (std::size_t i = 0; i < n; ++i) {
            const int c = assignment[i];
            sums[static_cast<std::size_t>(c)][0] += pixels[i].r;
            sums[static_cast<std::size_t>(c)][1] += pixels[i].g;
            sums[static_cast<std::size_t>(c)][2] += pixels[i].b;
            ++counts[static_cast<std::size_t>(c)];
        }
        for (int c = 0; c < k; ++c) {
            if (counts[static_cast<std::size_t>(c)] == 0) continue; // keep the previous centroid for an empty cluster
            centroids[static_cast<std::size_t>(c)] = {sums[static_cast<std::size_t>(c)][0] / counts[static_cast<std::size_t>(c)],
                                                        sums[static_cast<std::size_t>(c)][1] / counts[static_cast<std::size_t>(c)],
                                                        sums[static_cast<std::size_t>(c)][2] / counts[static_cast<std::size_t>(c)]};
        }
        if (!changed) break;
    }

    image::Image out(w, h, image::ImageMode::RGB);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const auto& c = centroids[static_cast<std::size_t>(assignment[static_cast<std::size_t>(y) * w + static_cast<std::size_t>(x)])];
            out.set_pixel(x, y,
                          image::Pixel{static_cast<std::uint8_t>(std::lround(c[0])), static_cast<std::uint8_t>(std::lround(c[1])),
                                       static_cast<std::uint8_t>(std::lround(c[2])), 255});
        }
    }
    return out;
}

} // namespace datamunge::cv
