#pragma once

#include <datamunge/image/image.hpp>
#include <datamunge/linalg/tensor.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace datamunge::cv {

/// @brief @p img as a [channels, height, width] Tensor with values normalized to [0, 1] --
///        the bridge between the image module and this basic neural-network inference
///        scaffolding (forward-pass building blocks only: no training, no autograd wiring, no
///        pretrained weights -- callers supply their own kernel/bias tensors).
[[nodiscard]] inline linalg::Tensor image_to_tensor(const image::Image& img) {
    const int channels = img.channels();
    const int h = img.height();
    const int w = img.width();
    linalg::Tensor t =
        linalg::Tensor::zeros({static_cast<std::size_t>(channels), static_cast<std::size_t>(h), static_cast<std::size_t>(w)});
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const image::Pixel p = img.get_pixel(x, y);
            const std::uint8_t fields[4] = {p.r, p.g, p.b, p.a};
            for (int c = 0; c < channels; ++c) {
                t.set({static_cast<std::size_t>(c), static_cast<std::size_t>(y), static_cast<std::size_t>(x)}, fields[c] / 255.0);
            }
        }
    }
    return t;
}

[[nodiscard]] inline linalg::Tensor relu(const linalg::Tensor& input) {
    return input.apply([](double x) { return x > 0.0 ? x : 0.0; });
}

[[nodiscard]] inline linalg::Tensor sigmoid(const linalg::Tensor& input) {
    return input.apply([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
}

/// @brief Numerically-stable softmax (subtract the max before exponentiating) over a 1D
///        tensor. Throws std::invalid_argument for any other rank -- flatten a multi-
///        dimensional activation first if needed.
[[nodiscard]] inline linalg::Tensor softmax(const linalg::Tensor& input) {
    if (input.ndim() != 1) {
        throw std::invalid_argument("softmax: expected a 1D tensor (flatten() first)");
    }
    const std::size_t n = input.size();
    const double max_val = input.max();
    std::vector<double> exps(n);
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        exps[i] = std::exp(input.at_flat(i) - max_val);
        sum += exps[i];
    }
    linalg::Tensor out = linalg::Tensor::zeros({n});
    for (std::size_t i = 0; i < n; ++i) out.set_flat(i, exps[i] / sum);
    return out;
}

/// @brief 2D convolution ("valid"/"same" via explicit @p padding) over a single [C_in, H, W]
///        input (no batch dimension -- run in a loop over a batch if needed) with a [C_out,
///        C_in, KH, KW] kernel and a [C_out] bias, producing a [C_out, H_out, W_out] output.
///        Direct nested-loop implementation (no im2col/FFT speedup) -- this is inference
///        SCAFFOLDING (matching the explicitly-scoped ask: forward pass only, no training, no
///        pretrained weights bundled), not a performance-tuned engine.
[[nodiscard]] inline linalg::Tensor conv2d(const linalg::Tensor& input, const linalg::Tensor& kernel, const linalg::Tensor& bias,
                                            int stride = 1, int padding = 0) {
    if (input.ndim() != 3 || kernel.ndim() != 4 || bias.ndim() != 1) {
        throw std::invalid_argument("conv2d: expected input [C_in,H,W], kernel [C_out,C_in,KH,KW], bias [C_out]");
    }
    if (stride < 1) {
        throw std::invalid_argument("conv2d: stride must be positive");
    }
    const auto& in_shape = input.shape();
    const auto& k_shape = kernel.shape();
    const std::size_t c_in = in_shape[0], h = in_shape[1], w = in_shape[2];
    const std::size_t c_out = k_shape[0], kh = k_shape[2], kw = k_shape[3];
    if (k_shape[1] != c_in) {
        throw std::invalid_argument("conv2d: kernel's C_in dimension must match the input's channel count");
    }
    if (bias.shape()[0] != c_out) {
        throw std::invalid_argument("conv2d: bias length must match the kernel's C_out dimension");
    }

    const std::size_t h_padded = h + 2 * static_cast<std::size_t>(padding);
    const std::size_t w_padded = w + 2 * static_cast<std::size_t>(padding);
    if (kh > h_padded || kw > w_padded) {
        throw std::invalid_argument("conv2d: kernel is larger than the (padded) input");
    }
    const std::size_t h_out = (h_padded - kh) / static_cast<std::size_t>(stride) + 1;
    const std::size_t w_out = (w_padded - kw) / static_cast<std::size_t>(stride) + 1;

    linalg::Tensor output = linalg::Tensor::zeros({c_out, h_out, w_out});
    for (std::size_t oc = 0; oc < c_out; ++oc) {
        for (std::size_t oy = 0; oy < h_out; ++oy) {
            for (std::size_t ox = 0; ox < w_out; ++ox) {
                double acc = bias.at_flat(oc);
                for (std::size_t ic = 0; ic < c_in; ++ic) {
                    for (std::size_t ky = 0; ky < kh; ++ky) {
                        const long iy = static_cast<long>(oy * static_cast<std::size_t>(stride) + ky) - padding;
                        if (iy < 0 || iy >= static_cast<long>(h)) continue;
                        for (std::size_t kx = 0; kx < kw; ++kx) {
                            const long ix = static_cast<long>(ox * static_cast<std::size_t>(stride) + kx) - padding;
                            if (ix < 0 || ix >= static_cast<long>(w)) continue;
                            acc += input.at({ic, static_cast<std::size_t>(iy), static_cast<std::size_t>(ix)}) *
                                   kernel.at({oc, ic, ky, kx});
                        }
                    }
                }
                output.set({oc, oy, ox}, acc);
            }
        }
    }
    return output;
}

enum class PoolMode { Max, Average };

namespace detail {
[[nodiscard]] inline linalg::Tensor pool2d(const linalg::Tensor& input, int pool_size, int stride, PoolMode mode) {
    if (input.ndim() != 3) {
        throw std::invalid_argument("pool2d: expected input [C,H,W]");
    }
    if (pool_size < 1) {
        throw std::invalid_argument("pool2d: pool_size must be positive");
    }
    if (stride < 1) stride = pool_size; // default: non-overlapping windows
    const auto& shape = input.shape();
    const std::size_t c = shape[0], h = shape[1], w = shape[2];
    if (static_cast<std::size_t>(pool_size) > h || static_cast<std::size_t>(pool_size) > w) {
        throw std::invalid_argument("pool2d: pool_size is larger than the input");
    }
    const std::size_t h_out = (h - static_cast<std::size_t>(pool_size)) / static_cast<std::size_t>(stride) + 1;
    const std::size_t w_out = (w - static_cast<std::size_t>(pool_size)) / static_cast<std::size_t>(stride) + 1;

    linalg::Tensor output = linalg::Tensor::zeros({c, h_out, w_out});
    for (std::size_t ch = 0; ch < c; ++ch) {
        for (std::size_t oy = 0; oy < h_out; ++oy) {
            for (std::size_t ox = 0; ox < w_out; ++ox) {
                double acc = (mode == PoolMode::Max) ? -std::numeric_limits<double>::infinity() : 0.0;
                for (int py = 0; py < pool_size; ++py) {
                    for (int px = 0; px < pool_size; ++px) {
                        const double v = input.at({ch, oy * static_cast<std::size_t>(stride) + static_cast<std::size_t>(py),
                                                    ox * static_cast<std::size_t>(stride) + static_cast<std::size_t>(px)});
                        if (mode == PoolMode::Max)
                            acc = std::max(acc, v);
                        else
                            acc += v;
                    }
                }
                if (mode == PoolMode::Average) acc /= (pool_size * pool_size);
                output.set({ch, oy, ox}, acc);
            }
        }
    }
    return output;
}
} // namespace detail

/// @brief Max pooling over a [C, H, W] tensor: each output pixel is the max of a @p pool_size
///        x @p pool_size window. @p stride <= 0 defaults to @p pool_size (non-overlapping
///        windows, the typical CNN pooling configuration).
[[nodiscard]] inline linalg::Tensor max_pool2d(const linalg::Tensor& input, int pool_size, int stride = -1) {
    return detail::pool2d(input, pool_size, stride, PoolMode::Max);
}

/// @brief Average pooling over a [C, H, W] tensor: each output pixel is the mean of a @p
///        pool_size x @p pool_size window. @p stride <= 0 defaults to @p pool_size.
[[nodiscard]] inline linalg::Tensor avg_pool2d(const linalg::Tensor& input, int pool_size, int stride = -1) {
    return detail::pool2d(input, pool_size, stride, PoolMode::Average);
}

} // namespace datamunge::cv
