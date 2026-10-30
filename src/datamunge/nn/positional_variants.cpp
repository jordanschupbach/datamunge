/// @file positional_variants.cpp
/// @brief rope_inverse_frequencies(), apply_rotary_at(), sinusoidal_encoding_2d(), fourier_features() and
///        gaussian_fourier_features().

#include "datamunge/nn/positional_variants.hpp"

#include "datamunge/nn/attention.hpp"
#include "datamunge/nn/ops.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace datamunge::nn {

std::vector<double> rope_inverse_frequencies(std::int64_t d, double base, RopeScaling scaling, double factor) {
    if (d < 2 || d % 2 != 0) throw std::invalid_argument("rope_inverse_frequencies: d must be even and >= 2");
    if (!(base > 0)) throw std::invalid_argument("rope_inverse_frequencies: base must be > 0");
    if (scaling != RopeScaling::None && !(factor >= 1)) throw std::invalid_argument("rope_inverse_frequencies: factor must be >= 1");
    if (scaling == RopeScaling::Ntk && d < 4) throw std::invalid_argument("rope_inverse_frequencies: NTK scaling needs d >= 4");
    double b = base;
    if (scaling == RopeScaling::Ntk) b = base * std::pow(factor, static_cast<double>(d) / static_cast<double>(d - 2));
    std::vector<double> theta(static_cast<std::size_t>(d / 2));
    for (std::int64_t j = 0; j < d / 2; ++j) {
        theta[static_cast<std::size_t>(j)] = std::pow(b, -2.0 * static_cast<double>(j) / static_cast<double>(d));
        if (scaling == RopeScaling::PositionInterpolation) theta[static_cast<std::size_t>(j)] /= factor;
    }
    return theta;
}

Tensor apply_rotary_at(const Tensor& x, const std::vector<double>& positions, const std::vector<double>& inv_freq) {
    if (x.dim() < 2) throw std::invalid_argument("apply_rotary_at: x must be [..., T, d]");
    const std::int64_t T = x.size(x.dim() - 2), d = x.size(x.dim() - 1), half = d / 2;
    if (d % 2 != 0 || static_cast<std::int64_t>(inv_freq.size()) != half)
        throw std::invalid_argument("apply_rotary_at: inv_freq must have d / 2 entries for even d");
    if (static_cast<std::int64_t>(positions.size()) != T) throw std::invalid_argument("apply_rotary_at: need one position per row");
    std::vector<double> c(static_cast<std::size_t>(T * half)), s(c.size());
    for (std::int64_t t = 0; t < T; ++t)
        for (std::int64_t j = 0; j < half; ++j) {
            const double a = positions[static_cast<std::size_t>(t)] * inv_freq[static_cast<std::size_t>(j)];
            c[static_cast<std::size_t>(t * half + j)] = std::cos(a);
            s[static_cast<std::size_t>(t * half + j)] = std::sin(a);
        }
    const DType  ft  = to_floating(x.dtype());
    const Tensor cos_t = cast(from_vector(c, {T, half}), ft), sin_t = cast(from_vector(s, {T, half}), ft);
    Shape pairs = x.shape();
    pairs.back() = half;
    pairs.push_back(2);
    const Tensor xp = reshape(cast(x, ft), pairs);
    const Tensor x0 = select(xp, xp.dim() - 1, 0), x1 = select(xp, xp.dim() - 1, 1);
    return reshape(stack({x0 * cos_t - x1 * sin_t, x0 * sin_t + x1 * cos_t}, -1), x.shape());
}

Tensor sinusoidal_encoding_2d(std::int64_t H, std::int64_t W, std::int64_t d, double base, DType dtype) {
    if (d < 4 || d % 4 != 0) throw std::invalid_argument("sinusoidal_encoding_2d: d must be a positive multiple of 4");
    if (H < 1 || W < 1) throw std::invalid_argument("sinusoidal_encoding_2d: H and W must be >= 1");
    const std::int64_t h = d / 2;
    const Tensor rows = reshape(sinusoidal_encoding(H, h, base, dtype), {H, 1, h});
    const Tensor cols = reshape(sinusoidal_encoding(W, h, base, dtype), {1, W, h});
    const Tensor r = rows + zeros({1, W, h}, dtype), c = cols + zeros({H, 1, h}, dtype);
    return reshape(concat({r, c}, 2), {H * W, d});
}

Tensor fourier_features(const Tensor& x, std::int64_t num_frequencies, bool include_input) {
    if (num_frequencies < 1) throw std::invalid_argument("fourier_features: num_frequencies must be >= 1");
    if (x.dim() < 1) throw std::invalid_argument("fourier_features: x must have a coordinate axis");
    const Tensor xf = cast(x, to_floating(x.dtype()));
    std::vector<Tensor> parts;
    if (include_input) parts.push_back(xf);
    for (std::int64_t k = 0; k < num_frequencies; ++k) {
        const Tensor a = xf * (std::ldexp(1.0, static_cast<int>(k)) * std::numbers::pi);
        parts.push_back(sin(a));
        parts.push_back(cos(a));
    }
    return concat(parts, -1);
}

Tensor gaussian_fourier_features(const Tensor& x, const Tensor& B) {
    if (x.dim() != 2 || B.dim() != 2 || B.size(0) != x.size(1))
        throw std::invalid_argument("gaussian_fourier_features: need x [N, D] and B [D, m]");
    const Tensor a = matmul(cast(x, to_floating(x.dtype())), cast(B, to_floating(x.dtype()))) * (2.0 * std::numbers::pi);
    return concat({cos(a), sin(a)}, 1);
}

}  // namespace datamunge::nn
