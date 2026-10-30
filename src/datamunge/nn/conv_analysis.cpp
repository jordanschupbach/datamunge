/// @file conv_analysis.cpp
/// @brief receptive_fields(), conv2d_matrix() and conv2d_circular_singular_values().

#include "datamunge/nn/conv_analysis.hpp"

#include "datamunge/nn/linalg.hpp"
#include "datamunge/nn/ops.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numbers>
#include <stdexcept>

namespace datamunge::nn {

std::vector<ReceptiveField> receptive_fields(const std::vector<ConvGeometry>& layers) {
    std::vector<ReceptiveField> out;
    ReceptiveField              rf;
    for (const auto& g : layers) {
        if (g.kernel < 1 || g.stride < 1 || g.dilation < 1)
            throw std::invalid_argument("receptive_fields: kernel, stride and dilation must be at least 1");
        const std::int64_t k = g.dilation * (g.kernel - 1) + 1;
        rf.start += 0.5 * static_cast<double>((k - 1) * rf.jump);
        rf.size += (k - 1) * rf.jump;
        rf.jump *= g.stride;
        out.push_back(rf);
    }
    return out;
}

/// @brief Helpers private to this file.
namespace {

/// @brief Output length and padding before the data along one axis, for conv2d_matrix().
/// @param in  Input length.
/// @param k  Kernel length.
/// @param stride  Stride.
/// @param dilation  Dilation.
/// @param padding  Padding kind.
/// @param explicit_before  Explicit padding before the data.
/// @param explicit_after  Explicit padding after the data.
/// @param circular  Use Same geometry (the input then wraps around).
/// @return {output length, padding before}.
std::pair<std::int64_t, std::int64_t> axis_geometry(std::int64_t in, std::int64_t k, std::int64_t stride, std::int64_t dilation,
                                                    Padding padding, std::int64_t explicit_before, std::int64_t explicit_after,
                                                    bool circular) {
    const Padding p = circular ? Padding::Same : padding;
    if (p == Padding::Explicit)
        return {conv_output_size(in, k, stride, dilation, p, explicit_before, explicit_after), explicit_before};
    const std::int64_t out = conv_output_size(in, k, stride, dilation, p);
    if (p == Padding::Valid) return {out, 0};
    const std::int64_t total = std::max<std::int64_t>(0, (out - 1) * stride + dilation * (k - 1) + 1 - in);
    return {out, total / 2};
}

}  // namespace

Tensor conv2d_matrix(std::int64_t height, std::int64_t width, const Tensor& kernel, const Conv2DOptions& options, bool circular) {
    if (kernel.dim() != 4) throw std::invalid_argument("conv2d_matrix: kernel must be [kh, kw, Cin / groups, Cout]");
    const std::int64_t kh = kernel.size(0), kw = kernel.size(1), cig = kernel.size(2), cout = kernel.size(3);
    const std::int64_t groups = options.groups;
    if (groups < 1 || cout % groups != 0) throw std::invalid_argument("conv2d_matrix: Cout must split into the groups");
    const std::int64_t cin = cig * groups, cog = cout / groups;
    const auto [ho, ph] = axis_geometry(height, kh, options.strides[0], options.dilations[0], options.padding,
                                        options.explicit_paddings[0], options.explicit_paddings[1], circular);
    const auto [wo, pw] = axis_geometry(width, kw, options.strides[1], options.dilations[1], options.padding,
                                        options.explicit_paddings[2], options.explicit_paddings[3], circular);
    const auto          K    = cast(kernel.detach(), DType::Float64).contiguous().to_vector<double>();
    const std::int64_t  cols = height * width * cin;
    std::vector<double> M(static_cast<std::size_t>(ho * wo * cout * cols), 0.0);
    for (std::int64_t i = 0; i < ho; ++i)
        for (std::int64_t j = 0; j < wo; ++j)
            for (std::int64_t o = 0; o < cout; ++o) {
                const std::int64_t row = (i * wo + j) * cout + o, g = o / cog;
                for (std::int64_t a = 0; a < kh; ++a)
                    for (std::int64_t b = 0; b < kw; ++b) {
                        std::int64_t r = i * options.strides[0] + a * options.dilations[0] - ph;
                        std::int64_t c = j * options.strides[1] + b * options.dilations[1] - pw;
                        if (circular) {
                            r = ((r % height) + height) % height;
                            c = ((c % width) + width) % width;
                        } else if (r < 0 || r >= height || c < 0 || c >= width) {
                            continue;  // zero padding
                        }
                        for (std::int64_t q = 0; q < cig; ++q) {
                            const std::int64_t ci = g * cig + q;
                            M[static_cast<std::size_t>(row * cols + (r * width + c) * cin + ci)] +=
                                K[static_cast<std::size_t>(((a * kw + b) * cig + q) * cout + o)];
                        }
                    }
            }
    return from_vector(M, {ho * wo * cout, cols});
}

Tensor conv2d_circular_singular_values(const Tensor& kernel, std::int64_t height, std::int64_t width) {
    if (kernel.dim() != 4) throw std::invalid_argument("conv2d_circular_singular_values: kernel must be [kh, kw, Cin, Cout]");
    const std::int64_t kh = kernel.size(0), kw = kernel.size(1), cin = kernel.size(2), cout = kernel.size(3);
    if (kh > height || kw > width) throw std::invalid_argument("conv2d_circular_singular_values: kernel larger than the input");
    const auto          K = cast(kernel.detach(), DType::Float64).contiguous().to_vector<double>();
    const std::int64_t  r = std::min(cin, cout);
    std::vector<double> all;
    all.reserve(static_cast<std::size_t>(height * width * r));
    std::vector<double> block(static_cast<std::size_t>(4 * cin * cout));
    for (std::int64_t u = 0; u < height; ++u)
        for (std::int64_t v = 0; v < width; ++v) {
            // Real form [[Re, -Im], [Im, Re]] of K-hat(u, v): each singular value appears twice.
            std::fill(block.begin(), block.end(), 0.0);
            for (std::int64_t a = 0; a < kh; ++a)
                for (std::int64_t b = 0; b < kw; ++b) {
                    const double angle = -2.0 * std::numbers::pi *
                                         (static_cast<double>(a * u) / static_cast<double>(height) +
                                          static_cast<double>(b * v) / static_cast<double>(width));
                    const double cr = std::cos(angle), cs = std::sin(angle);
                    for (std::int64_t i = 0; i < cin; ++i)
                        for (std::int64_t o = 0; o < cout; ++o) {
                            const double w  = K[static_cast<std::size_t>(((a * kw + b) * cin + i) * cout + o)];
                            const auto   at = [&](std::int64_t row, std::int64_t col) -> double& {
                                return block[static_cast<std::size_t>(row * 2 * cout + col)];
                            };
                            at(i, o) += w * cr;
                            at(i, o + cout) -= w * cs;
                            at(i + cin, o) += w * cs;
                            at(i + cin, o + cout) += w * cr;
                        }
                }
            const auto s = svd(from_vector(block, {2 * cin, 2 * cout}), false).s.to_vector<double>();
            for (std::int64_t k = 0; k < r; ++k) all.push_back(s[static_cast<std::size_t>(2 * k)]);
        }
    std::sort(all.begin(), all.end(), std::greater<>());
    return from_vector(all, {static_cast<std::int64_t>(all.size())});
}

}  // namespace datamunge::nn
