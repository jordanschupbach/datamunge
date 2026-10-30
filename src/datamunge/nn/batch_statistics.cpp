/// @file batch_statistics.cpp
/// @brief batch_renorm(), BatchStatsAccumulator and fold_batch_norm().

#include "datamunge/nn/batch_statistics.hpp"

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/conv.hpp"
#include "datamunge/nn/ops.hpp"

#include <stdexcept>

namespace datamunge::nn {

Tensor batch_renorm(ParameterStore& params, ParameterStore& stats, const std::string& name, const Tensor& x, bool training,
                    const BatchRenormOptions& options) {
    if (x.dim() < 1) throw std::invalid_argument("batch_renorm: input needs rank >= 1");
    if (options.r_max < 1.0) throw std::invalid_argument("batch_renorm: r_max must be at least 1");
    if (options.d_max < 0.0) throw std::invalid_argument("batch_renorm: d_max must be non-negative");
    const std::int64_t c     = x.shape().back();
    const Tensor       gamma = params.get(name + "/gamma", {c}, init::ones());
    const Tensor       beta  = params.get(name + "/beta", {c}, init::zeros());
    const Tensor       mu    = stats.get(name + "/moving_mean", {c}, init::zeros());
    const Tensor       var   = stats.get(name + "/moving_var", {c}, init::ones());
    if (!training) return batch_norm_infer(x, gamma, beta, mu.detach(), var.detach(), options.eps);

    Axes axes;
    for (std::int64_t a = 0; a + 1 < x.dim(); ++a) axes.push_back(a);
    const Tensor mean_b = mean(x, axes);
    const Tensor var_b  = mean(square(x - mean_b), axes);  // biased
    const Tensor s_b    = sqrt(var_b + options.eps);
    Tensor       r, d;
    {
        NoGradGuard  guard;
        const Tensor s = sqrt(cast(var.detach(), x.dtype()) + options.eps);
        r = clip(s_b.detach() / s, 1.0 / options.r_max, options.r_max);
        d = clip((mean_b.detach() - cast(mu.detach(), x.dtype())) / s, -options.d_max, options.d_max);
        if (options.update_stats) {
            const double n  = static_cast<double>(x.numel()) / static_cast<double>(c);
            const double m  = options.momentum;
            const Tensor vu = var_b.detach() * (n > 1 ? n / (n - 1) : 1.0);  // unbiased, as nn::batch_norm stores
            stats.set(name + "/moving_mean", mu.detach() * m + cast(mean_b.detach(), mu.dtype()) * (1.0 - m));
            stats.set(name + "/moving_var", var.detach() * m + cast(vu, var.dtype()) * (1.0 - m));
        }
    }
    const Tensor xhat = (x - mean_b) / s_b * r + d;
    return xhat * gamma + beta;
}

void BatchStatsAccumulator::add(const Tensor& x) {
    if (x.dim() < 1) throw std::invalid_argument("BatchStatsAccumulator::add: input needs rank >= 1");
    const std::int64_t c = x.shape().back();
    if (count_ > 0 && static_cast<std::int64_t>(mean_.size()) != c)
        throw std::invalid_argument("BatchStatsAccumulator::add: channel count changed");
    const std::int64_t n = x.numel() / c;
    if (n == 0) return;
    const auto v = cast(x.detach(), DType::Float64).contiguous().to_vector<double>();
    if (count_ == 0) mean_.assign(static_cast<std::size_t>(c), 0.0), m2_.assign(static_cast<std::size_t>(c), 0.0);
    for (std::int64_t j = 0; j < c; ++j) {
        // Two-pass statistics of the batch, then Chan et al.'s merge with the running ones.
        double bm = 0;
        for (std::int64_t i = 0; i < n; ++i) bm += v[static_cast<std::size_t>(i * c + j)];
        bm /= static_cast<double>(n);
        double bm2 = 0;
        for (std::int64_t i = 0; i < n; ++i) {
            const double e = v[static_cast<std::size_t>(i * c + j)] - bm;
            bm2 += e * e;
        }
        const auto   k     = static_cast<std::size_t>(j);
        const double total = static_cast<double>(count_ + n);
        const double delta = bm - mean_[k];
        mean_[k] += delta * static_cast<double>(n) / total;
        m2_[k] += bm2 + delta * delta * static_cast<double>(count_) * static_cast<double>(n) / total;
    }
    count_ += n;
}

Tensor BatchStatsAccumulator::mean() const {
    if (count_ == 0) throw std::logic_error("BatchStatsAccumulator::mean: no data");
    return from_vector(mean_, {static_cast<std::int64_t>(mean_.size())});
}

Tensor BatchStatsAccumulator::variance(bool unbiased) const {
    if (count_ == 0) throw std::logic_error("BatchStatsAccumulator::variance: no data");
    if (unbiased && count_ < 2) throw std::logic_error("BatchStatsAccumulator::variance: unbiased needs two rows");
    std::vector<double> out(m2_.size());
    const double        div = static_cast<double>(unbiased ? count_ - 1 : count_);
    for (std::size_t j = 0; j < m2_.size(); ++j) out[j] = m2_[j] / div;
    return from_vector(out, {static_cast<std::int64_t>(out.size())});
}

void BatchStatsAccumulator::write(ParameterStore& stats, const std::string& name) const {
    const Shape shape{static_cast<std::int64_t>(mean_.size())};
    stats.get(name + "/moving_mean", shape, init::zeros());
    stats.get(name + "/moving_var", shape, init::ones());
    stats.set(name + "/moving_mean", cast(mean(), stats.dtype()));
    stats.set(name + "/moving_var", cast(variance(), stats.dtype()));
}

void BatchStatsAccumulator::reset() {
    count_ = 0;
    mean_.clear();
    m2_.clear();
}

FoldedLinear fold_batch_norm(const Tensor& weight, const Tensor& bias, const Tensor& gamma, const Tensor& beta,
                             const Tensor& mean, const Tensor& variance, double eps) {
    if (weight.dim() < 1) throw std::invalid_argument("fold_batch_norm: weight needs rank >= 1");
    const std::int64_t c = weight.shape().back();
    for (const Tensor* t : {&gamma, &beta, &mean, &variance, &bias})
        if (t->defined() && (t->dim() != 1 || t->size(0) != c))
            throw std::invalid_argument("fold_batch_norm: per-channel vectors must be [" + std::to_string(c) + "]");
    const Tensor s = gamma / sqrt(variance + eps);
    const Tensor b = bias.defined() ? bias : zeros({c}, weight.dtype());
    return {weight * s, (b - mean) * s + beta};
}

}  // namespace datamunge::nn
