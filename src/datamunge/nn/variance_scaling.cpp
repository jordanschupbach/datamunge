/// @file variance_scaling.cpp
/// @brief variance_scaling(), gain(), truncated_normal_std() and signal_propagation().

#include "datamunge/nn/variance_scaling.hpp"

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/random.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace datamunge::nn::init {

double truncated_normal_std(double bound) {
    if (!(bound > 0)) throw std::invalid_argument("truncated_normal_std: bound must be positive");
    const double pdf  = std::exp(-0.5 * bound * bound) / std::sqrt(2 * std::numbers::pi);
    const double mass = std::erf(bound / std::numbers::sqrt2);  // 2 Phi(b) - 1
    return std::sqrt(1.0 - 2.0 * bound * pdf / mass);
}

Initializer variance_scaling(double scale, FanMode mode, Distribution distribution) {
    if (scale < 0) throw std::invalid_argument("variance_scaling: scale must be non-negative");
    return [scale, mode, distribution](const Shape& s, DType d, Generator& g) {
        const auto   [in, out] = fans(s);
        const double n         = mode == FanMode::FanIn ? in : mode == FanMode::FanOut ? out : 0.5 * (in + out);
        const double variance  = scale / std::max(1.0, n);
        switch (distribution) {
            case Distribution::TruncatedNormal:
                return nn::truncated_normal(s, 0.0, std::sqrt(variance) / truncated_normal_std(2.0), g, d);
            case Distribution::Normal: return nn::normal(s, 0.0, std::sqrt(variance), g, d);
            case Distribution::Uniform: {
                const double a = std::sqrt(3.0 * variance);
                return nn::uniform(s, -a, a, g, d);
            }
        }
        throw std::logic_error("variance_scaling: unknown distribution");
    };
}

double gain(std::string_view nonlinearity, double param) {
    if (nonlinearity == "linear" || nonlinearity == "identity" || nonlinearity == "conv" || nonlinearity == "sigmoid") return 1.0;
    if (nonlinearity == "tanh") return 5.0 / 3.0;
    if (nonlinearity == "relu") return std::numbers::sqrt2;
    if (nonlinearity == "leaky_relu") return std::sqrt(2.0 / (1.0 + param * param));
    if (nonlinearity == "selu") return 0.75;
    throw std::invalid_argument("gain: unknown nonlinearity '" + std::string(nonlinearity) + "'");
}

SignalPropagation signal_propagation(const std::vector<std::int64_t>& widths, const Initializer& init,
                                     const std::function<Tensor(const Tensor&)>& activation, std::int64_t batch,
                                     Generator& gen, DType dtype) {
    if (widths.size() < 2) throw std::invalid_argument("signal_propagation: need at least two widths");
    if (batch < 1) throw std::invalid_argument("signal_propagation: batch must be positive");
    EnableGradGuard     grad_on;
    std::vector<Tensor> z;
    Tensor              h = nn::normal({batch, widths[0]}, 0.0, 1.0, gen, dtype);
    for (std::size_t l = 1; l < widths.size(); ++l) {
        const Tensor w = init({widths[l - 1], widths[l]}, dtype, gen);
        Tensor       zl = nn::matmul(h, w);
        if (!zl.requires_grad()) zl = zl.detach().requires_grad_(true);  // the first layer starts the graph
        z.push_back(zl);
        h = activation(zl);
    }
    const Tensor r    = nn::normal(h.shape(), 0.0, 1.0, gen, dtype);
    const auto   grads = nn::grad({nn::sum(h * r)}, z, {.allow_unused = true});
    SignalPropagation out;
    for (std::size_t l = 0; l < z.size(); ++l) {
        out.forward_rms.push_back(std::sqrt(nn::mean(nn::square(z[l].detach())).item()));
        out.backward_rms.push_back(grads[l].defined() ? std::sqrt(nn::mean(nn::square(grads[l])).item()) : 0.0);
    }
    return out;
}

}  // namespace datamunge::nn::init
