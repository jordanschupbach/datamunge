/// @file normalization_variants.cpp
/// @brief normalize(), layer_norm_jacobian(), rms_norm_jacobian(), scale_norm(), dynamic_tanh() and
///        weight_standardize().

#include "datamunge/nn/normalization_variants.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace datamunge::nn {

/// @brief Helpers private to this file.
namespace {

/// @brief Throws unless x is a non-empty vector.
/// @param x  The tensor to check.
/// @param who  Function name for the message.
void check_vector(const Tensor& x, const char* who) {
    if (x.dim() != 1 || x.size(0) < 1) throw std::invalid_argument(std::string(who) + ": x must be a non-empty vector");
}

/// @brief Size of the last axis, throwing for scalars.
/// @param x  The input.
/// @param who  Function name for the message.
/// @return x.size(-1).
std::int64_t features(const Tensor& x, const char* who) {
    if (x.dim() < 1) throw std::invalid_argument(std::string(who) + ": x must have at least one axis");
    return x.size(x.dim() - 1);
}

}  // namespace

Tensor normalize(const Tensor& x, const Axes& axes, bool center, double eps) {
    if (axes.empty()) throw std::invalid_argument("normalize: axes must not be empty");
    if (!(eps >= 0)) throw std::invalid_argument("normalize: eps must be >= 0");
    if (center) {
        const Tensor mu  = mean(x, axes, true);
        const Tensor var = mean(squared_difference(x, mu), axes, true);
        return (x - mu) * rsqrt(var + eps);
    }
    return x * rsqrt(mean(x * x, axes, true) + eps);
}

Tensor layer_norm_jacobian(const Tensor& x, double eps) {
    check_vector(x, "layer_norm_jacobian");
    const Tensor xf    = cast(x, to_floating(x.dtype()));
    const auto   d     = static_cast<double>(xf.size(0));
    const Tensor c     = xf - mean(xf);
    const Tensor sigma = sqrt(mean(c * c) + eps);
    const Tensor yhat  = c / sigma;
    const Tensor ones1 = ones({xf.size(0)}, xf.dtype());
    return (eye(xf.size(0), xf.dtype()) - outer(ones1, ones1) / d - outer(yhat, yhat) / d) / sigma;
}

Tensor rms_norm_jacobian(const Tensor& x, double eps) {
    check_vector(x, "rms_norm_jacobian");
    const Tensor xf   = cast(x, to_floating(x.dtype()));
    const auto   d    = static_cast<double>(xf.size(0));
    const Tensor r    = sqrt(mean(xf * xf) + eps);
    const Tensor yhat = xf / r;
    return (eye(xf.size(0), xf.dtype()) - outer(yhat, yhat) / d) / r;
}

Tensor scale_norm(ParameterStore& store, const std::string& name, const Tensor& x, double eps) {
    const std::int64_t d = features(x, "scale_norm");
    const Tensor       g = store.get(name + "/g", {1}, init::constant(std::sqrt(static_cast<double>(d))));
    return x * rsqrt(sum(x * x, {-1}, true) + eps) * g;
}

Tensor dynamic_tanh(ParameterStore& store, const std::string& name, const Tensor& x, double alpha0) {
    const std::int64_t d     = features(x, "dynamic_tanh");
    const Tensor       alpha = store.get(name + "/alpha", {1}, init::constant(alpha0));
    const Tensor       gamma = store.get(name + "/gamma", {d}, init::ones());
    const Tensor       beta  = store.get(name + "/beta", {d}, init::zeros());
    return tanh(x * alpha) * gamma + beta;
}

Tensor weight_standardize(const Tensor& w, double eps) {
    if (w.dim() < 2) throw std::invalid_argument("weight_standardize: w must have rank >= 2");
    Axes axes;
    for (std::int64_t a = 0; a + 1 < w.dim(); ++a) axes.push_back(a);
    return normalize(w, axes, true, eps);
}

}  // namespace datamunge::nn
