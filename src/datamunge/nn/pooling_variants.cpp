/// @file pooling_variants.cpp
/// @brief power_mean_pool2d(), gem_pool2d(), logsumexp_pool2d(), adaptive pooling and blur_pool2d().

#include "datamunge/nn/pooling_variants.hpp"

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace datamunge::nn {

/// @brief Helpers private to this file.
namespace {

/// @brief Throws unless x is a rank-4 NHWC tensor.
/// @param x  The tensor to check.
/// @param who  Function name for the message.
void check_nhwc(const Tensor& x, const char* who) {
    if (x.dim() != 4) throw std::invalid_argument(std::string(who) + ": x must be [N, H, W, C]");
}

/// @brief Shared body of the adaptive pools.
/// @param x  Input [N, H, W, C].
/// @param out_h  Output height.
/// @param out_w  Output width.
/// @param use_max  Take the maximum of each window instead of the mean.
/// @param who  Function name for messages.
/// @return The pooled tensor [N, out_h, out_w, C].
Tensor adaptive_pool(const Tensor& x, std::int64_t out_h, std::int64_t out_w, bool use_max, const char* who) {
    check_nhwc(x, who);
    const std::int64_t n = x.size(0), h = x.size(1), w = x.size(2), c = x.size(3);
    if (out_h < 1 || out_h > h || out_w < 1 || out_w > w)
        throw std::invalid_argument(std::string(who) + ": output size must be between 1 and the input size");
    std::vector<Tensor> rows;
    rows.reserve(static_cast<std::size_t>(out_h));
    for (std::int64_t i = 0; i < out_h; ++i) {
        const std::int64_t h0 = i * h / out_h, h1 = ((i + 1) * h + out_h - 1) / out_h;
        std::vector<Tensor> cells;
        cells.reserve(static_cast<std::size_t>(out_w));
        for (std::int64_t j = 0; j < out_w; ++j) {
            const std::int64_t w0 = j * w / out_w, w1 = ((j + 1) * w + out_w - 1) / out_w;
            const Tensor win = slice(x, {0, h0, w0, 0}, {n, h1 - h0, w1 - w0, c});
            cells.push_back(use_max ? max(win, {1, 2}, true) : mean(win, {1, 2}, true));
        }
        rows.push_back(concat(cells, 2));
    }
    return concat(rows, 1);
}

}  // namespace

Tensor power_mean_pool2d(const Tensor& x, double p, std::int64_t window, std::int64_t stride, Padding padding) {
    if (!(p > 0)) throw std::invalid_argument("power_mean_pool2d: p must be > 0");
    // Zero entries and all-zero windows (common after ReLU) would give infinite derivatives of t^p (p < 1) and of
    // t^(1/p); they map to 0 with zero gradient by evaluating the power only on a safe stand-in value.
    const Tensor ax = abs(x), nz = greater(ax, 0.0);
    const Tensor xp = p == 1.0 ? ax : where(nz, pow(where(nz, ax, 1.0), p), 0.0);
    const Tensor m = avg_pool2d(xp, window, stride, padding), mz = greater(m, 0.0);
    return where(mz, pow(where(mz, m, 1.0), 1.0 / p), 0.0);
}

Tensor gem_pool2d(const Tensor& x, const Tensor& p, double eps) {
    check_nhwc(x, "gem_pool2d");
    if (p.numel() != 1) throw std::invalid_argument("gem_pool2d: p must be a scalar");
    const Tensor ps = reshape(p, {});
    return pow(mean(pow(maximum(x, full_like(x, eps)), ps), {1, 2}), 1.0 / ps);
}

Tensor gem_pool2d(const Tensor& x, double p, double eps) {
    if (!(p > 0)) throw std::invalid_argument("gem_pool2d: p must be > 0");
    check_nhwc(x, "gem_pool2d");
    return pow(mean(pow(maximum(x, full_like(x, eps)), p), {1, 2}), 1.0 / p);
}

Tensor logsumexp_pool2d(const Tensor& x, double beta, std::int64_t window, std::int64_t stride, Padding padding) {
    if (!(beta > 0)) throw std::invalid_argument("logsumexp_pool2d: beta must be > 0");
    check_nhwc(x, "logsumexp_pool2d");
    if (window < 1 || stride < 1) throw std::invalid_argument("logsumexp_pool2d: window and stride must be >= 1");
    if (padding != Padding::Valid && padding != Padding::Same)
        throw std::invalid_argument("logsumexp_pool2d: padding must be Valid or Same");
    const std::int64_t n = x.size(0), h = x.size(1), w = x.size(2), c = x.size(3);
    const std::int64_t oh = conv_output_size(h, window, stride, 1, padding), ow = conv_output_size(w, window, stride, 1, padding);
    if (oh < 1 || ow < 1) throw std::invalid_argument("logsumexp_pool2d: window larger than the input");
    // SAME pads with -inf (TF convention: the extra entry goes after) and counts only real entries.
    const std::int64_t th = std::max<std::int64_t>((oh - 1) * stride + window - h, 0), tw = std::max<std::int64_t>((ow - 1) * stride + window - w, 0);
    const std::vector<std::pair<std::int64_t, std::int64_t>> pads{{0, 0}, {th / 2, th - th / 2}, {tw / 2, tw - tw / 2}, {0, 0}};
    const Tensor xp   = pad(x, pads, PadMode::Constant, -INFINITY);
    const Tensor real = pad(ones({1, h, w, 1}, x.dtype()), pads);
    // Stack the window x window offsets on a new last axis: [N, OH, OW, C, window^2].
    std::vector<Tensor> taps, counts;
    for (std::int64_t a = 0; a < window; ++a)
        for (std::int64_t b = 0; b < window; ++b) {
            const Shape begin{0, a, b, 0}, step{1, stride, stride, 1};
            taps.push_back(strided_slice(xp, begin, {n, a + (oh - 1) * stride + 1, b + (ow - 1) * stride + 1, c}, step));
            counts.push_back(strided_slice(real, begin, {1, a + (oh - 1) * stride + 1, b + (ow - 1) * stride + 1, 1}, step));
        }
    const Tensor t = stack(taps, 4);
    const Tensor m = stop_gradient(max(t, {4}, true));  // per-window maximum, finite since every window has a real entry
    const Tensor count = reshape(sum(stack(counts, 4), {4}), {1, oh, ow, 1});
    return reshape(m, {n, oh, ow, c}) + log(sum(exp((t - m) * beta), {4}) / count) / beta;
}

Tensor adaptive_avg_pool2d(const Tensor& x, std::int64_t out_h, std::int64_t out_w) {
    return adaptive_pool(x, out_h, out_w, false, "adaptive_avg_pool2d");
}

Tensor adaptive_max_pool2d(const Tensor& x, std::int64_t out_h, std::int64_t out_w) {
    return adaptive_pool(x, out_h, out_w, true, "adaptive_max_pool2d");
}

Tensor blur_pool2d(const Tensor& x, std::int64_t stride, std::int64_t filter_size) {
    check_nhwc(x, "blur_pool2d");
    if (stride < 1) throw std::invalid_argument("blur_pool2d: stride must be >= 1");
    if (filter_size < 1 || filter_size > 7) throw std::invalid_argument("blur_pool2d: filter_size must be in [1, 7]");
    // Binomial row by Pascal's rule, normalised to sum 1.
    std::vector<double> row{1.0};
    for (std::int64_t k = 1; k < filter_size; ++k) {
        std::vector<double> next(row.size() + 1, 0.0);
        for (std::size_t i = 0; i < row.size(); ++i) next[i] += row[i], next[i + 1] += row[i];
        row = next;
    }
    double total = 0;
    for (double v : row) total += v;
    const std::int64_t c = x.size(3), k = filter_size;
    std::vector<double> f(static_cast<std::size_t>(k * k * c));
    for (std::int64_t a = 0; a < k; ++a)
        for (std::int64_t b = 0; b < k; ++b)
            for (std::int64_t ch = 0; ch < c; ++ch) f[static_cast<std::size_t>((a * k + b) * c + ch)] = row[a] * row[b] / (total * total);
    const Tensor filter = cast(from_vector(f, {k, k, c, 1}), x.dtype());
    // Reflect-pad so that a stride-s VALID convolution gives ceil(H / s) outputs, as SAME pooling does.
    auto amount = [&](std::int64_t len) {
        const std::int64_t out = (len + stride - 1) / stride, need = std::max<std::int64_t>((out - 1) * stride + k - len, 0);
        return std::pair<std::int64_t, std::int64_t>{need / 2, need - need / 2};
    };
    const auto ph = amount(x.size(1)), pw = amount(x.size(2));
    const bool reflect = ph.first < x.size(1) && ph.second < x.size(1) && pw.first < x.size(2) && pw.second < x.size(2);
    const Tensor padded = pad(x, {{0, 0}, ph, pw, {0, 0}}, reflect ? PadMode::Reflect : PadMode::Symmetric);
    return depthwise_conv2d(padded, filter, conv_options(stride, Padding::Valid));
}

}  // namespace datamunge::nn
