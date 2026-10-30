/// @file activations.cpp
/// @brief glu() and activation_moments().

#include "datamunge/nn/activations.hpp"

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"
#include "detail.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace datamunge::nn {

Tensor glu(const Tensor& x, Activation gate, std::int64_t axis_in) {
    const std::int64_t axis = detail::normalize_axis(axis_in, x.dim(), "glu");
    const std::int64_t n    = x.shape()[static_cast<std::size_t>(axis)];
    if (n % 2 != 0) throw std::invalid_argument("glu: axis length " + std::to_string(n) + " is odd");
    const auto halves = split(x, 2, axis);
    return halves[0] * activate(halves[1], gate);
}

ActivationMoments activation_moments(Activation activation, double q, double alpha) {
    if (!(q > 0) || !std::isfinite(q)) throw std::invalid_argument("activation_moments: q must be positive and finite");
    // Midpoint rule over +-12 sd. Every kink of the built-in activations sits at an integer (0, +-1, +-3, 6),
    // and a kink on a cell boundary costs the midpoint rule nothing, so the cells are aligned to the integers
    // whenever the range reaches past +-1 (and cells of width <= 1 still fit in the budget); otherwise a
    // symmetric grid keeps 0 on a boundary.
    const double sd = std::sqrt(q), reach = 12.0 * sd;
    double       lo = -reach, h = 2.0 * reach / 24000;
    int          m  = 24000;
    if (reach >= 1.0 && reach <= 12000.0) {
        const double per_unit = std::ceil(24000 / (2.0 * reach));
        const double half     = std::ceil(reach);
        h  = 1.0 / per_unit;
        lo = -half;
        m  = static_cast<int>(2.0 * half * per_unit);
    }
    std::vector<double> zs(static_cast<std::size_t>(m)), w(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i) {
        zs[i] = lo + (i + 0.5) * h;
        w[i]  = h * std::exp(-0.5 * zs[i] * zs[i] / q) / (sd * std::sqrt(2 * std::numbers::pi));
    }
    EnableGradGuard grad_on;
    Tensor          z  = from_vector(zs, {m}).requires_grad_(true);
    const Tensor    f  = activate(z, activation, alpha);
    const auto      fv = f.detach().to_vector<double>();
    const auto      dv = grad({sum(f)}, {z})[0].to_vector<double>();
    ActivationMoments out;
    for (int i = 0; i < m; ++i) {
        out.mean += w[i] * fv[i];
        out.second_moment += w[i] * fv[i] * fv[i];
        out.derivative_mean += w[i] * dv[i];
        out.derivative_second_moment += w[i] * dv[i] * dv[i];
    }
    out.variance = out.second_moment - out.mean * out.mean;
    return out;
}

}  // namespace datamunge::nn
