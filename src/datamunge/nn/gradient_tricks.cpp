/// @file gradient_tricks.cpp
/// @brief straight_through(), round_ste(), fake_quantize(), gradient_reversal(), scale_gradient()
///        and clip_gradient().

#include "datamunge/nn/gradient_tricks.hpp"

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"

#include <stdexcept>

namespace datamunge::nn {

/// @brief Helpers private to this file.
namespace {

/// @brief Applies a one-input custom function whose backward maps the output gradient alone.
/// @param name  Node name.
/// @param x  Input.
/// @param forward  Forward map.
/// @param backward  Backward map from (input, output gradient) to the input gradient.
/// @return The output.
Tensor unary_function(const std::string& name, const Tensor& x, std::function<Tensor(const Tensor&)> forward,
                      std::function<Tensor(const Tensor&, const Tensor&)> backward) {
    return apply_function(
               name, {x}, [forward](const std::vector<Tensor>& in) { return std::vector<Tensor>{forward(in[0])}; },
               [backward](const std::vector<Tensor>& in, const std::vector<Tensor>&, const std::vector<Tensor>& g) {
                   return std::vector<Tensor>{backward(in[0], g[0])};
               })
        .at(0);
}

}  // namespace

Tensor straight_through(const Tensor& x, const Tensor& value) {
    if (x.shape() != value.shape()) throw std::invalid_argument("straight_through: shape mismatch");
    return x + stop_gradient(value - x);
}

Tensor round_ste(const Tensor& x) {
    return unary_function(
        "RoundSTE", x, [](const Tensor& in) { return round(in); },
        [](const Tensor&, const Tensor& g) { return g; });
}

Tensor stochastic_round_ste(const Tensor& x, Generator& gen) {
    const Tensor base = floor(x.detach());
    const Tensor up   = cast(less(uniform(x.shape(), 0.0, 1.0, gen, x.dtype()), x.detach() - base), x.dtype());
    return straight_through(x, base + up);
}

Tensor fake_quantize(const Tensor& x, double step, std::int64_t qmin, std::int64_t qmax) {
    if (!(step > 0.0)) throw std::invalid_argument("fake_quantize: step must be positive");
    if (qmax < qmin) throw std::invalid_argument("fake_quantize: qmax < qmin");
    const double lo = step * static_cast<double>(qmin), hi = step * static_cast<double>(qmax);
    return unary_function(
        "FakeQuantize", x,
        [=](const Tensor& in) {
            return clip(round(in / step), static_cast<double>(qmin), static_cast<double>(qmax)) * step;
        },
        [=](const Tensor& in, const Tensor& g) {
            // Pass the gradient where lo <= x <= hi, zero where the forward is flat.
            const Tensor inside = mul(cast(greater_equal(in, lo), g.dtype()), cast(less_equal(in, hi), g.dtype()));
            return mul(g, inside);
        });
}

Tensor gradient_reversal(const Tensor& x, double lambda) {
    return unary_function(
        "GradientReversal", x, [](const Tensor& in) { return in.clone(); },
        [lambda](const Tensor&, const Tensor& g) { return g * (-lambda); });
}

Tensor scale_gradient(const Tensor& x, double factor) {
    return unary_function(
        "ScaleGradient", x, [](const Tensor& in) { return in.clone(); },
        [factor](const Tensor&, const Tensor& g) { return g * factor; });
}

Tensor clip_gradient(const Tensor& x, double low, double high) {
    if (high < low) throw std::invalid_argument("clip_gradient: high < low");
    return unary_function(
        "ClipGradient", x, [](const Tensor& in) { return in.clone(); },
        [low, high](const Tensor&, const Tensor& g) { return clip(g, low, high); });
}

}  // namespace datamunge::nn
