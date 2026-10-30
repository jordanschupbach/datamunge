#pragma once

/// @file variance_scaling.hpp
/// @brief The general variance-scaling initializer, recommended gains per nonlinearity, the
///        truncated-normal correction, and a signal-propagation diagnostic for deep networks.
///
/// Every fan-based initializer draws weights with variance scale / n, where n is the fan-in, the
/// fan-out or their mean. A dense layer y = W x then keeps Var(y) close to scale times Var(x)
/// (fan-in mode), so stacking layers neither shrinks nor grows the
/// signal when the scale undoes the activation's own gain. glorot_*, he_* and lecun_* in params.hpp are the
/// common special cases; variance_scaling() is the general one (Keras `VarianceScaling`).

#include "datamunge/nn/params.hpp"

#include <cstdint>
#include <functional>
#include <string_view>
#include <vector>

namespace datamunge::nn::init {

/// @brief Which fan sets the variance in variance_scaling().
enum class FanMode {
    FanIn,   ///< n = fan_in: preserves forward activations.
    FanOut,  ///< n = fan_out: preserves backward gradients.
    FanAvg,  ///< n = (fan_in + fan_out) / 2: the Glorot compromise.
};

/// @brief The distribution variance_scaling() draws from.
enum class Distribution {
    TruncatedNormal,  ///< Normal redrawn beyond two standard deviations, rescaled to the target variance.
    Normal,           ///< Untruncated normal.
    Uniform,          ///< U(-a, a) with a = sqrt(3 * variance).
};

/// @brief Standard deviation of a standard normal truncated to \f$[-b,b]\f$:
///        \f$\sqrt{1-2b\varphi(b)/(2\Phi(b)-1)}\f$.
/// @param bound  Truncation point \f$b>0\f$ (default 2, as in nn::truncated_normal()).
/// @return The standard deviation (0.87962566... for b = 2).
/// @throws std::invalid_argument  If bound is not positive.
[[nodiscard]] double truncated_normal_std(double bound = 2.0);

/// @brief Weights with variance scale / n (Keras `VarianceScaling`).
///
/// `variance_scaling(1, FanAvg, Uniform)` is glorot_uniform, `variance_scaling(2, FanIn, Normal)` is he_normal,
/// and `variance_scaling(1, FanIn, TruncatedNormal)` is Keras's lecun_normal. The truncated draw is divided by
/// truncated_normal_std() so that its variance is exactly scale / n.
///
/// @param scale  Variance multiplier, usually gain() squared.
/// @param mode  Which fan sets n.
/// @param distribution  Distribution of the draws.
/// @return The initializer.
/// @throws std::invalid_argument  If scale is negative.
[[nodiscard]] Initializer variance_scaling(double scale = 1.0, FanMode mode = FanMode::FanIn,
                                           Distribution distribution = Distribution::TruncatedNormal);

/// @brief The recommended gain for a nonlinearity (PyTorch `calculate_gain`): the factor by which to scale
///        a fan-in initializer's standard deviation so the activation keeps the signal's scale.
///
/// "linear", "identity", "conv" and "sigmoid" give 1; "tanh" 5/3; "relu" \f$\sqrt2\f$;
/// "leaky_relu" \f$\sqrt{2/(1+a^2)}\f$ with negative slope @p param; "selu" 3/4.
///
/// @param nonlinearity  Name of the activation.
/// @param param  Negative slope for "leaky_relu" (default 0.01); ignored otherwise.
/// @return The gain.
/// @throws std::invalid_argument  For an unknown name.
[[nodiscard]] double gain(std::string_view nonlinearity, double param = 0.01);

/// @brief Root-mean-square activations and gradients, layer by layer, of a freshly initialised network.
struct SignalPropagation {
    /// RMS of the pre-activations of each layer, \f$\sqrt{\operatorname{mean}(z_l^2)}\f$, l = 1..L.
    std::vector<double> forward_rms;
    /// RMS of the gradient of the loss with respect to each layer's pre-activations, l = 1..L, for the loss
    /// \f$\sum(\text{output}\odot r)\f$ with a fixed standard-normal r (so the output gradient has RMS 1).
    std::vector<double> backward_rms;
};

/// @brief Pushes a standard-normal batch through a bias-free multilayer perceptron with the given widths,
///        initializer and activation, and measures how the signal's scale changes with depth.
///
/// Layer l computes \f$z_l=h_{l-1}W_l\f$ and \f$h_l=\phi(z_l)\f$, with \f$h_0\f$ the input; the last layer's
/// activation is applied as well. Fresh weights are drawn from @p init with @p gen.
///
/// @param widths  Layer widths \f$n_0,n_1,\dots,n_L\f$ (at least two entries).
/// @param init  Weight initializer, called with shape [n_{l-1}, n_l].
/// @param activation  Elementwise activation \f$\phi\f$.
/// @param batch  Number of input rows.
/// @param gen  Random generator for the inputs, the weights and r.
/// @param dtype  Floating dtype (default float64).
/// @return Forward and backward RMS per layer.
/// @throws std::invalid_argument  If fewer than two widths are given or batch is not positive.
[[nodiscard]] SignalPropagation signal_propagation(const std::vector<std::int64_t>& widths, const Initializer& init,
                                                   const std::function<Tensor(const Tensor&)>& activation,
                                                   std::int64_t batch, Generator& gen, DType dtype = DType::Float64);

}  // namespace datamunge::nn::init
