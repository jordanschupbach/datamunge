/// @file dropout_variants.cpp
/// @brief gaussian_dropout(), alpha_dropout(), spatial_dropout() and drop_path().

#include "datamunge/nn/dropout_variants.hpp"

#include "datamunge/nn/ops.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace datamunge::nn {

/// @brief Helpers private to this file.
namespace {

/// @brief Throws unless 0 <= rate < 1.
/// @param rate  The rate to check.
/// @param who  Function name for the message.
void check_rate(double rate, const char* who) {
    if (!(rate >= 0 && rate < 1)) throw std::invalid_argument(std::string(who) + ": rate must be in [0, 1)");
}

}  // namespace

Tensor gaussian_dropout(const Tensor& x, double rate, bool training, Generator& gen) {
    check_rate(rate, "gaussian_dropout");
    if (!training || rate == 0) return x;
    return mul(x, normal(x.shape(), 1.0, std::sqrt(rate / (1.0 - rate)), gen, to_floating(x.dtype())));
}

Tensor alpha_dropout(const Tensor& x, double rate, bool training, Generator& gen) {
    check_rate(rate, "alpha_dropout");
    if (!training || rate == 0) return x;
    constexpr double kAlpha = 1.6732632423543772848170429916717, kScale = 1.0507009873554804934193349852946;
    const double     ap = -kScale * kAlpha, keep = 1.0 - rate;
    const double     a = 1.0 / std::sqrt(keep * (1.0 + rate * ap * ap)), b = -a * rate * ap;
    const Tensor     m = bernoulli(x.shape(), keep, gen, to_floating(x.dtype()));
    return (mul(x, m) + (1.0 - m) * ap) * a + b;
}

Tensor spatial_dropout(const Tensor& x, double rate, bool training, Generator& gen) {
    check_rate(rate, "spatial_dropout");
    if (x.dim() < 2) throw std::invalid_argument("spatial_dropout: input needs rank >= 2 (batch first, channels last)");
    if (!training || rate == 0) return x;
    Shape shape(static_cast<std::size_t>(x.dim()), 1);
    shape.front() = x.size(0);
    shape.back()  = x.shape().back();
    const double keep = 1.0 - rate;
    return mul(x, bernoulli(shape, keep, gen, to_floating(x.dtype()))) / keep;
}

Tensor drop_path(const Tensor& x, double rate, bool training, Generator& gen) {
    check_rate(rate, "drop_path");
    if (x.dim() < 1) throw std::invalid_argument("drop_path: input needs rank >= 1");
    if (!training || rate == 0) return x;
    Shape shape(static_cast<std::size_t>(x.dim()), 1);
    shape.front()     = x.size(0);
    const double keep = 1.0 - rate;
    return mul(x, bernoulli(shape, keep, gen, to_floating(x.dtype()))) / keep;
}

}  // namespace datamunge::nn
