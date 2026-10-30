/// @file generators.cpp
/// @brief Synthetic datasets with known structure.

#include <datamunge/datasets/generators.hpp>

#include <datamunge/random/random.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::datasets {

/// Draws all \f$(t,h)\f$ pairs first, then the noise, so the same seed gives the same sheet with
/// or without noise.
dstruct::DataFrame swiss_roll(std::size_t n, double noise, std::uint64_t seed) {
    if (!(noise >= 0.0)) throw std::invalid_argument("swiss_roll: noise must be non-negative");
    random::Xoshiro256StarStar rng(seed);
    std::vector<double>        x(n), y(n), z(n), t(n), h(n);
    for (std::size_t i = 0; i < n; ++i) {
        t[i] = 1.5 * std::numbers::pi * (1.0 + 2.0 * rng.uniform01());
        h[i] = 21.0 * rng.uniform01();
    }
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = t[i] * std::cos(t[i]);
        y[i] = h[i];
        z[i] = t[i] * std::sin(t[i]);
        if (noise > 0.0) {
            x[i] += rng.normal(0.0, noise);
            y[i] += rng.normal(0.0, noise);
            z[i] += rng.normal(0.0, noise);
        }
    }
    dstruct::DataFrame out;
    out.add_column("x", x);
    out.add_column("y", y);
    out.add_column("z", z);
    out.add_column("t", t);
    out.add_column("height", h);
    std::vector<double> s(n);
    for (std::size_t i = 0; i < n; ++i) s[i] = 0.5 * (t[i] * std::sqrt(1.0 + t[i] * t[i]) + std::asinh(t[i]));
    out.add_column("arc_length", s);
    return out;
}

/// @brief Shared code of the two-class benchmark generators.
namespace {

/// @brief Builds the frame `x1` ... `xd`, `class` from columns and labels.
/// @param columns  One vector per predictor.
/// @param labels   Class per row.
/// @return The frame.
dstruct::DataFrame labelled_frame(const std::vector<std::vector<double>>& columns, const std::vector<std::string>& labels) {
    dstruct::DataFrame out;
    for (std::size_t j = 0; j < columns.size(); ++j) out.add_column("x" + std::to_string(j + 1), columns[j]);
    out.add_column("class", labels);
    return out;
}

/// @brief Draws a two-class Gaussian benchmark: the class by a fair coin, then each coordinate
///        as \f$N(\mu_j,\sigma^2)\f$ with \f$(\mu_j,\sigma)\f$ given by @p law.
/// @param n     Rows.
/// @param d     Predictors.
/// @param seed  Seed.
/// @param name  Function name, for errors.
/// @param law   `(class_one, mixture_sign, j) -> {mean, sd}`.
/// @return The frame.
template <typename Law>
dstruct::DataFrame gaussian_benchmark(std::size_t n, std::size_t d, std::uint64_t seed, const char* name, Law law) {
    if (d == 0) throw std::invalid_argument(std::string(name) + ": d must be at least 1");
    random::Xoshiro256StarStar       rng(seed);
    std::vector<std::vector<double>> columns(d, std::vector<double>(n));
    std::vector<std::string>         labels(n);
    for (std::size_t i = 0; i < n; ++i) {
        const bool   one  = rng.uniform01() < 0.5;
        const double sign = rng.uniform01() < 0.5 ? 1.0 : -1.0;
        labels[i]         = one ? "1" : "2";
        for (std::size_t j = 0; j < d; ++j) {
            const auto [mean, sd] = law(one, sign, j);
            columns[j][i]         = rng.normal(mean, sd);
        }
    }
    return labelled_frame(columns, labels);
}

} // namespace

dstruct::DataFrame twonorm(std::size_t n, std::size_t d, std::uint64_t seed) {
    const double a = 2.0 / std::sqrt(static_cast<double>(std::max<std::size_t>(d, 1)));
    return gaussian_benchmark(n, d, seed, "twonorm", [a](bool one, double, std::size_t) {
        return std::pair{one ? a : -a, 1.0};
    });
}

dstruct::DataFrame threenorm(std::size_t n, std::size_t d, std::uint64_t seed) {
    const double a = 2.0 / std::sqrt(static_cast<double>(std::max<std::size_t>(d, 1)));
    return gaussian_benchmark(n, d, seed, "threenorm", [a](bool one, double sign, std::size_t j) {
        return std::pair{one ? sign * a : (j % 2 == 0 ? a : -a), 1.0};
    });
}

dstruct::DataFrame ringnorm(std::size_t n, std::size_t d, std::uint64_t seed) {
    const double a = 1.0 / std::sqrt(static_cast<double>(std::max<std::size_t>(d, 1)));
    return gaussian_benchmark(n, d, seed, "ringnorm", [a](bool one, double, std::size_t) {
        return one ? std::pair{0.0, 2.0} : std::pair{a, 1.0};
    });
}

dstruct::DataFrame circle(std::size_t n, std::size_t d, double flip, std::uint64_t seed) {
    if (d == 0) throw std::invalid_argument("circle: d must be at least 1");
    if (!(flip >= 0.0 && flip <= 1.0)) throw std::invalid_argument("circle: flip must be in [0, 1]");
    const double dd = static_cast<double>(d);
    // pi^(d/2) r^d / Gamma(d/2 + 1) = 2^(d - 1)
    const double log_r = ((dd - 1.0) * std::log(2.0) + std::lgamma(dd / 2.0 + 1.0) - dd / 2.0 * std::log(std::numbers::pi)) / dd;
    const double r2    = std::exp(2.0 * log_r);
    random::Xoshiro256StarStar       rng(seed);
    std::vector<std::vector<double>> columns(d, std::vector<double>(n));
    std::vector<std::string>         labels(n);
    for (std::size_t i = 0; i < n; ++i) {
        double ss = 0.0;
        for (std::size_t j = 0; j < d; ++j) {
            columns[j][i] = -1.0 + 2.0 * rng.uniform01();
            ss += columns[j][i] * columns[j][i];
        }
        bool inside = ss < r2;
        if (rng.uniform01() < flip) inside = !inside;
        labels[i] = inside ? "1" : "2";
    }
    return labelled_frame(columns, labels);
}

} // namespace datamunge::datasets
