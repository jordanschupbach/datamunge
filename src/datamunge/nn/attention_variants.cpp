/// @file attention_variants.cpp
/// @brief kernel_attention(), elu_feature_map(), positive_random_features(), orthogonal_gaussian_projection(),
///        local_keep_mask() and attention_entropy().

#include "datamunge/nn/attention_variants.hpp"

#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/params.hpp"

#include <cmath>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Helpers private to this file.
namespace {

/// @brief Rows offset .. offset + count - 1 of @p x along @p axis (negative counts from the end).
/// @param x  Input.
/// @param axis  Axis, negative from the end.
/// @param offset  First index.
/// @param count  Number of indices.
/// @return The slice.
Tensor narrow_axis(const Tensor& x, std::int64_t axis, std::int64_t offset, std::int64_t count) {
    const auto a     = static_cast<std::size_t>(axis < 0 ? axis + x.dim() : axis);
    Shape      begin(x.shape().size(), 0), size = x.shape();
    begin[a] = offset;
    size[a]  = count;
    return slice(x, begin, size);
}

}  // namespace

Tensor kernel_attention(const Tensor& phi_q, const Tensor& phi_k, const Tensor& v, bool causal, double eps) {
    if (phi_q.dim() < 2 || phi_k.dim() < 2 || v.dim() < 2)
        throw std::invalid_argument("kernel_attention: inputs need at least two axes");
    if (phi_q.size(-1) != phi_k.size(-1)) throw std::invalid_argument("kernel_attention: query and key features differ in length");
    if (phi_k.size(-2) != v.size(-2)) throw std::invalid_argument("kernel_attention: keys and values differ in length");
    const std::int64_t Tq = phi_q.size(-2), Tk = phi_k.size(-2);
    if (!causal) {
        const Tensor kv  = matmul(swapaxes(phi_k, -2, -1), v);    // [..., m, dv]
        const Tensor z   = sum(phi_k, {-2}, true);                  // [..., 1, m]
        const Tensor den = sum(mul(phi_q, z), {-1}, true);          // [..., Tq, 1]
        return div(matmul(phi_q, kv), add(den, eps));
    }
    if (Tq > Tk) throw std::invalid_argument("kernel_attention: causal needs Tq <= Tk");
    // Prefix sums over keys, then the rows aligned with the queries.
    const Tensor outer = mul(expand_dims(phi_k, -1), expand_dims(v, -2));         // [..., Tk, m, dv]
    const Tensor S     = narrow_axis(cumsum(outer, -3), -3, Tk - Tq, Tq);         // [..., Tq, m, dv]
    const Tensor Z     = narrow_axis(cumsum(phi_k, -2), -2, Tk - Tq, Tq);         // [..., Tq, m]
    const Tensor num   = squeeze(matmul(expand_dims(phi_q, -2), S), {-2});       // [..., Tq, dv]
    const Tensor den   = sum(mul(phi_q, Z), {-1}, true);
    return div(num, add(den, eps));
}

Tensor elu_feature_map(const Tensor& x) { return add(elu(x), 1.0); }

Tensor positive_random_features(const Tensor& x, const Tensor& w, double scale) {
    if (w.dim() != 2 || x.dim() < 1 || w.size(1) != x.size(-1))
        throw std::invalid_argument("positive_random_features: w must be [m, d] with d = x.size(-1)");
    const double s  = scale > 0 ? scale : 1.0 / std::sqrt(static_cast<double>(x.size(-1)));
    const Tensor xs = mul(x, std::sqrt(s));
    const Tensor proj = matmul(xs, swapaxes(w, 0, 1));                       // [..., T, m]
    const Tensor half = mul(sum(square(xs), {-1}, true), 0.5);                 // [..., T, 1]
    return mul(exp(sub(proj, half)), 1.0 / std::sqrt(static_cast<double>(w.size(0))));
}

Tensor orthogonal_gaussian_projection(std::int64_t m, std::int64_t d, Generator& gen, DType dtype) {
    if (m < 1 || d < 1) throw std::invalid_argument("orthogonal_gaussian_projection: m and d must be positive");
    std::vector<Tensor> blocks;
    for (std::int64_t done = 0; done < m; done += d) {
        const Tensor q     = init::orthogonal()({d, d}, dtype, gen);                      // orthonormal rows
        const Tensor norms = sqrt(sum(square(normal({d, d}, 0.0, 1.0, gen, dtype)), {-1}, true));  // chi_d lengths
        blocks.push_back(mul(q, norms));
    }
    const Tensor all = blocks.size() == 1 ? blocks[0] : concat(blocks, 0);
    return slice(all, {0, 0}, {m, d});
}

Tensor local_keep_mask(std::int64_t Tq, std::int64_t Tk, std::int64_t window, bool causal) {
    if (window < 1) throw std::invalid_argument("local_keep_mask: window must be at least 1");
    if (Tq < 0 || Tk < 0) throw std::invalid_argument("local_keep_mask: sizes must be non-negative");
    const Tensor rows = add(reshape(arange(0, static_cast<double>(Tq), 1, DType::Int64), {Tq, 1}),
                            scalar(static_cast<double>(Tk - Tq), DType::Int64));
    const Tensor cols = reshape(arange(0, static_cast<double>(Tk), 1, DType::Int64), {1, Tk});
    const Tensor diff = sub(rows, cols);  // i' - j
    const auto   w    = static_cast<double>(window);
    if (causal) return logical_and(greater_equal(diff, scalar(0.0, DType::Int64)), less(diff, scalar(w, DType::Int64)));
    return less(abs(diff), scalar(w, DType::Int64));
}

Tensor attention_entropy(const Tensor& weights) {
    const Tensor positive = greater(weights, 0.0);
    const Tensor safe     = where(positive, weights, 1.0);  // log(1) = 0 where the weight is zero
    return neg(sum(mul(weights, log(safe)), {-1}));
}

}  // namespace datamunge::nn
