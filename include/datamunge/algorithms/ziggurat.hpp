#pragma once

/// \file ziggurat.hpp
/// \brief The Ziggurat algorithm for fast sampling from the standard normal
///        distribution (Marsaglia & Tsang 2000; setup after Doornik 2005).
///
/// The ziggurat method draws from a non-uniform density -- here the standard normal
/// \f$f(x)\propto e^{-x^2/2}\f$ -- extremely fast by covering the density with \f$N\f$
/// equal-area horizontal strips: \f$N-1\f$ rectangles stacked under the curve plus a
/// base strip that includes the tail. To sample, pick a strip uniformly and a point
/// uniformly along it; if the point falls in the rectangle's inner region (the common
/// case, needing only a comparison), accept immediately. Only near the curved edge or
/// in the tail is a slower fallback used. Because the fast path is a single multiply
/// and compare, the ziggurat generates normal variates several times faster than the
/// Box-Muller or inverse-CDF methods, which is why it is the default normal sampler in
/// many numerical libraries.
///
/// This implementation precomputes the strip boundaries \f$x_i\f$ once and then draws
/// variates from a user-supplied uniform bit source (\c std::mt19937_64).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace datamunge::algorithms {

/// A precomputed ziggurat sampler for the standard normal distribution.
class ZigguratNormal {
 public:
    /// Build the ziggurat tables. \p layers is the number of strips (128 or 256 are standard).
    explicit ZigguratNormal(std::size_t layers = 256) : n_(layers), x_(layers + 1), ratio_(layers) {
        // Tail boundary r and common strip area v for the chosen layer count.
        const double r = (layers == 128) ? 3.442619855899 : 3.6541528853610088;
        const double v = (layers == 128) ? 9.91256303526217e-3 : 0.00492867323399;
        double       f = std::exp(-0.5 * r * r);
        x_[0]          = v / f;  // width of the base strip
        x_[1]          = r;
        x_[layers]     = 0.0;
        for (std::size_t i = 2; i < layers; ++i) {
            x_[i] = std::sqrt(-2.0 * std::log(v / x_[i - 1] + f));
            f     = std::exp(-0.5 * x_[i] * x_[i]);
        }
        for (std::size_t i = 0; i < layers; ++i) ratio_[i] = x_[i + 1] / x_[i];
    }

    /// Draw one standard-normal variate using bits from \p rng.
    double sample(std::mt19937_64& rng) const {
        std::uniform_real_distribution<double>     unit(0.0, 1.0);
        std::uniform_int_distribution<std::size_t> layer(0, n_ - 1);
        for (;;) {
            const double      u = 2.0 * unit(rng) - 1.0;  // uniform in (-1, 1)
            const std::size_t i = layer(rng);
            if (std::abs(u) < ratio_[i]) return u * x_[i];  // fast accept (inner rectangle)
            if (i == 0) return tail(rng, u < 0.0);          // base strip -> sample the tail
            const double xr = u * x_[i];
            const double f0 = std::exp(-0.5 * (x_[i] * x_[i] - xr * xr));
            const double f1 = std::exp(-0.5 * (x_[i + 1] * x_[i + 1] - xr * xr));
            if (f1 + unit(rng) * (f0 - f1) < 1.0) return xr;  // under the curve
        }
    }

 private:
    /// Sample the exponential tail beyond r by Marsaglia's rejection method.
    double tail(std::mt19937_64& rng, bool negative) const {
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        const double                           r = x_[1];
        double                                 xx, yy;
        do {
            xx = std::log(unit(rng)) / r;
            yy = std::log(unit(rng));
        } while (-2.0 * yy < xx * xx);
        return negative ? xx - r : r - xx;
    }

    std::size_t         n_;
    std::vector<double> x_;      // strip boundaries x_[0..n]
    std::vector<double> ratio_;  // x_[i+1]/x_[i], the fast-accept threshold
};

}  // namespace datamunge::algorithms
