#pragma once

// Lanczos resampling: reconstruct a uniformly-sampled 1-D signal at arbitrary
// positions using a windowed-sinc kernel. Ideal (bandlimited) reconstruction
// convolves the samples with sinc, which has infinite support; Lanczos windows
// sinc with a wider sinc lobe to get a finite 2a-wide kernel,
//
//   L(x) = sinc(x) * sinc(x/a)   for |x| < a,   else 0,   sinc(x) = sin(pi x)/(pi x),
//
// with a small integer a (typically 2 or 3). The value at a fractional sample
// position p is the sum of nearby samples weighted by L(p - k). Lanczos passes
// exactly through the original samples (L(0)=1, L(nonzero integer)=0) and, for
// smooth/bandlimited data, reconstructs far more faithfully than linear
// interpolation -- it is the high-quality resampler behind image and audio
// scaling.

#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

namespace detail {

inline double lanczos_sinc(double x) {
    if (std::fabs(x) < 1e-12) return 1.0;
    const double px = M_PI * x;
    return std::sin(px) / px;
}

// The Lanczos kernel with lobe parameter a.
inline double lanczos_kernel(double x, int a) {
    if (x <= -a || x >= a) return 0.0;
    return lanczos_sinc(x) * lanczos_sinc(x / a);
}

} // namespace detail

// Value of the signal `samples` (taken at integer positions 0..n-1) resampled at
// fractional position `p`, using a Lanczos-`a` kernel. Positions outside the
// sample range are clamped to the border samples.
inline double lanczos_resample_at(const std::vector<double>& samples, double p, int a = 3) {
    const long   n     = static_cast<long>(samples.size());
    const long   floor = static_cast<long>(std::floor(p));
    double       acc   = 0.0;
    for (long k = floor - a + 1; k <= floor + a; ++k) {
        const double w  = detail::lanczos_kernel(p - static_cast<double>(k), a);
        const long   kc = k < 0 ? 0 : (k >= n ? n - 1 : k); // clamp to border
        acc += samples[static_cast<std::size_t>(kc)] * w;
    }
    return acc;
}

// Resample `samples` at each fractional position in `positions`.
inline std::vector<double> lanczos_resample(const std::vector<double>& samples,
                                            const std::vector<double>& positions, int a = 3) {
    std::vector<double> out;
    out.reserve(positions.size());
    for (double p : positions) out.push_back(lanczos_resample_at(samples, p, a));
    return out;
}

} // namespace datamunge::algorithms
