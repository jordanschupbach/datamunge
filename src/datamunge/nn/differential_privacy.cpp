/// @file differential_privacy.cpp
/// @brief dense_per_example_grad_norms(), clip_factors(), rdp_subsampled_gaussian() and dp_sgd_epsilon().

#include "datamunge/nn/differential_privacy.hpp"

#include "datamunge/nn/ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace datamunge::nn {

Tensor dense_per_example_grad_norms(const Tensor& x, const Tensor& dy, bool bias) {
    if (x.dim() < 2 || dy.dim() < 2) throw std::invalid_argument("dense_per_example_grad_norms: inputs need rank >= 2");
    if (x.dim() != dy.dim()) throw std::invalid_argument("dense_per_example_grad_norms: x and dy ranks differ");
    for (std::int64_t a = 0; a + 1 < x.dim(); ++a)
        if (x.size(a) != dy.size(a)) throw std::invalid_argument("dense_per_example_grad_norms: leading axes differ");
    const std::int64_t n = x.size(0), in = x.shape().back(), out = dy.shape().back();
    const std::int64_t t  = n == 0 ? 0 : x.numel() / (n * in);
    const Tensor       X  = reshape(cast(x.detach(), DType::Float64), {n, t, in});
    const Tensor       D  = reshape(cast(dy.detach(), DType::Float64), {n, t, out});
    Tensor             sq;
    if (t == 1) {
        sq = sum(square(X), {1, 2}) * sum(square(D), {1, 2});
    } else if (t * t <= in * out) {
        sq = sum(matmul(X, X, false, true) * matmul(D, D, false, true), {1, 2});
    } else {
        sq = sum(square(matmul(X, D, true, false)), {1, 2});
    }
    if (bias) sq = sq + sum(square(sum(D, {1})), {1});
    return sqrt(sq);
}

Tensor clip_factors(const Tensor& norms, double max_norm) {
    if (!(max_norm > 0)) throw std::invalid_argument("clip_factors: max_norm must be positive");
    // max(norm, C) never divides by zero and gives exactly 1 below the threshold.
    return max_norm / maximum(norms.detach(), full_like(norms.detach(), max_norm));
}

double rdp_subsampled_gaussian(double q, double noise_multiplier, std::int64_t order) {
    if (!(q >= 0 && q <= 1)) throw std::invalid_argument("rdp_subsampled_gaussian: q must be in [0, 1]");
    if (!(noise_multiplier > 0)) throw std::invalid_argument("rdp_subsampled_gaussian: noise multiplier must be positive");
    if (order < 2) throw std::invalid_argument("rdp_subsampled_gaussian: order must be at least 2");
    const double a = static_cast<double>(order), s2 = noise_multiplier * noise_multiplier;
    if (q == 0) return 0.0;
    if (q == 1) return a / (2 * s2);
    // log of each binomial term, then log-sum-exp.
    std::vector<double> terms;
    terms.reserve(static_cast<std::size_t>(order + 1));
    for (std::int64_t k = 0; k <= order; ++k) {
        const double kd = static_cast<double>(k);
        terms.push_back(std::lgamma(a + 1) - std::lgamma(kd + 1) - std::lgamma(a - kd + 1) + (a - kd) * std::log1p(-q) + kd * std::log(q) +
                        (kd * kd - kd) / (2 * s2));
    }
    const double m = *std::max_element(terms.begin(), terms.end());
    double       s = 0;
    for (double v : terms) s += std::exp(v - m);
    return std::max(0.0, (m + std::log(s)) / (a - 1));
}

DpEpsilon dp_sgd_epsilon(double q, double noise_multiplier, std::int64_t steps, double delta, std::int64_t max_order) {
    if (steps < 0) throw std::invalid_argument("dp_sgd_epsilon: steps must be non-negative");
    if (!(delta > 0 && delta < 1)) throw std::invalid_argument("dp_sgd_epsilon: delta must be in (0, 1)");
    if (max_order < 2) throw std::invalid_argument("dp_sgd_epsilon: max_order must be at least 2");
    DpEpsilon best{std::numeric_limits<double>::infinity(), 0};
    for (std::int64_t a = 2; a <= max_order; ++a) {
        const double eps = static_cast<double>(steps) * rdp_subsampled_gaussian(q, noise_multiplier, a) - std::log(delta) / static_cast<double>(a - 1);
        if (eps < best.epsilon) best = {eps, a};
    }
    return best;
}

}  // namespace datamunge::nn
