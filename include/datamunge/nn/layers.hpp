#pragma once

/// @file layers.hpp
/// @brief Functional layers over a ParameterStore.
///
/// Each layer takes the store, a name and its input, and creates its
/// parameters under "<name>/..." on first use (tflow's `nn::Dense(scope, p,
/// name, x, ...)` without the scope). Input widths are read from the input,
/// so they are not passed in. Calling a layer again with the same name reuses
/// its weights.
///
/// plot_model() and Module::plot() group named dense, convolution and batch-normalization
/// calls into architecture nodes. This optional tracing does not change numerical results.

#include <cstdint>
#include <string>
#include <vector>

#include "datamunge/nn/conv.hpp"
#include "datamunge/nn/params.hpp"
#include "datamunge/nn/random.hpp"
#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

enum class Activation {
  None, Relu, Relu6, LeakyRelu, Elu, Selu, Gelu, GeluTanh, Silu, Mish, Softplus, Softsign, Sigmoid, Tanh,
  HardSigmoid,  ///< relu6(x + 3) / 6, PyTorch's and Keras 3's piecewise-linear sigmoid.
  HardSwish,    ///< x relu6(x + 3) / 6 (MobileNetV3).
  HardTanh,     ///< clip(x, -1, 1).
  LogSigmoid    ///< log sigmoid(x) = -softplus(-x).
};
/// The last enumerator, for loops over every activation.
inline constexpr Activation kLastActivation = Activation::LogSigmoid;

/// "relu", "gelu_tanh", ... ("identity" for None).
std::string activation_name(Activation activation);
/// Applies an activation; `alpha` is the LeakyReLU negative slope.
Tensor activate(const Tensor& x, Activation activation, double alpha = 0.2);

struct DenseOptions {
  Activation  activation{Activation::None};
  bool        bias{true};
  Initializer init{init::glorot_uniform()};
  double      bias_value{0.0};
  double      alpha{0.2};  // LeakyReLU slope
};

/// y = act(x W + b) for x [..., in]: W is "<name>/W" [in, units], b is
/// "<name>/b" [units]. Leading axes are kept.
Tensor dense(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t units,
             const DenseOptions& options = {});
Tensor dense(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t units,
             Activation activation);

/// A stack of dense layers "<name>/dense<i>"; returns the last hidden activation.
Tensor mlp(ParameterStore& store, const std::string& name, const Tensor& x, const std::vector<std::int64_t>& hidden,
           Activation activation = Activation::Relu);

/// Inverted dropout: zeroes each element with probability `rate` and scales
/// the rest by 1 / (1 - rate) while `training`; the identity otherwise.
Tensor dropout(const Tensor& x, double rate, bool training, Generator& gen);

/// Rows of the table "<name>/table" [vocab, dim] (initialised U(-limit, limit)) for integer `ids` of any shape.
Tensor embedding(ParameterStore& store, const std::string& name, const Tensor& ids, std::int64_t vocab,
                 std::int64_t dim, double limit = 0.05);

/// Mean over axis 1 of x [N, T, D] counting only positions where mask [N, T] is nonzero.
Tensor masked_mean(const Tensor& x, const Tensor& mask);

/// y = gamma (x - mean) / sqrt(var + eps) + beta over the last axis.
/// Parameters "<name>/gamma" (ones) and "<name>/beta" (zeros).
Tensor layer_norm(ParameterStore& store, const std::string& name, const Tensor& x, double eps = 1e-5);
/// y = gamma x / sqrt(mean(x^2) + eps) over the last axis; parameter "<name>/gamma".
Tensor rms_norm(ParameterStore& store, const std::string& name, const Tensor& x, double eps = 1e-6);

// ---- convolution and normalization layers ----------------------------------------------

struct ConvLayerOptions {
  std::int64_t stride{1};
  std::int64_t dilation{1};
  Padding      padding{Padding::Same};
  Activation   activation{Activation::None};
  bool         bias{true};
  Initializer  init{init::glorot_uniform()};
  double       alpha{0.2};  // LeakyReLU slope
};

/// act(conv2d(x, W) + b) for x [N, H, W, C]: W is "<name>/W" [k, k, C, filters],
/// b is "<name>/b" [filters] (tflow's Conv2DLayer).
Tensor conv2d(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t filters,
              std::int64_t kernel, const ConvLayerOptions& options = {});
/// 1-D convolution of x [N, L, C]: W is "<name>/W" [1, k, C, filters] (tflow's Conv1DLayer layout).
Tensor conv1d(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t filters,
              std::int64_t kernel, const ConvLayerOptions& options = {});
/// Depthwise convolution: W is "<name>/W" [k, k, C, multiplier], b "<name>/b" [C multiplier].
Tensor depthwise_conv2d(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t multiplier,
                        std::int64_t kernel, const ConvLayerOptions& options = {});
/// Depthwise ("<name>/depthwise/W", no bias or activation) then a 1x1
/// convolution ("<name>/pointwise/W", "<name>/pointwise/b") with the activation.
Tensor separable_conv2d(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t filters,
                        std::int64_t kernel, const ConvLayerOptions& options = {}, std::int64_t multiplier = 1);
/// Transposed convolution to `filters` channels: W is "<name>/W" [k, k, filters, C]
/// (initialised with the fans of the forward [k, k, C, filters] kernel, as tflow does).
/// The output is [N, H s, W s, filters] for SAME, [N, (H-1) s + k, ..] for VALID.
Tensor conv2d_transpose(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t filters,
                        std::int64_t kernel, const ConvLayerOptions& options = {});

/// act(conv3d(x, W) + b) for volumes x [N, D, H, W, C]: W is "<name>/W"
/// [k, k, k, C, filters], b is "<name>/b" [filters] (tflow's Conv3DLayer).
Tensor conv3d(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t filters,
              std::int64_t kernel, const ConvLayerOptions& options = {});
/// Transposed 3-D convolution to `filters` channels: W is "<name>/W"
/// [k, k, k, filters, C] (initialised with the fans of the forward
/// [k, k, k, C, filters] kernel). The output is [N, D s, H s, W s, filters]
/// for SAME, [N, (D-1) s + k, ..] for VALID.
Tensor conv3d_transpose(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t filters,
                        std::int64_t kernel, const ConvLayerOptions& options = {});

struct BatchNormOptions {
  double momentum{0.99};
  double eps{1e-3};
  /// Learn gamma ("<name>/gamma"); otherwise gamma is fixed at 1.
  bool scale{true};
  /// Learn beta ("<name>/beta"); otherwise beta is fixed at 0.
  bool center{true};
  /// Update the moving averages on a training-mode call. Turn off for extra
  /// passes within one step (e.g. a GAN's second discriminator call), since TF
  /// runs the update op once per step.
  bool update_stats{true};
};

/// Batch normalization over every axis but the last (tflow's BatchNorm).
/// While `training` it normalizes with the batch statistics and updates the
/// moving averages "<name>/moving_mean" and "<name>/moving_var" in `stats`
/// (a separate, non-trained store); otherwise it uses those averages.
///   mean <- m mean + (1 - m) batch_mean,  var <- m var + (1 - m) batch_var (unbiased)
Tensor batch_norm(ParameterStore& params, ParameterStore& stats, const std::string& name, const Tensor& x,
                  bool training, const BatchNormOptions& options = {});

/// Group normalization of x [N, H, W, C] over (H, W, C / groups) per example
/// and group; "<name>/gamma" and "<name>/beta" are [C].
Tensor group_norm(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t groups,
                  double eps = 1e-5);
/// Group norm with one group per channel.
Tensor instance_norm(ParameterStore& store, const std::string& name, const Tensor& x, double eps = 1e-5);

/// Noisy linear layer with factorised Gaussian noise (Fortunato et al. 2018):
///   W = mu_W + sigma_W (f(e_in) f(e_out)^T) s,  b = mu_b + sigma_b f(e_out) s
/// with f(e) = sign(e) sqrt|e|, fresh e ~ N(0, 1) from `noise` on every call
/// and s = `noise_scale` (1 to explore, 0 for the mean weights). Parameters
/// "<name>/mu_W", "<name>/sigma_W" [in, units], "<name>/mu_b", "<name>/sigma_b" [units].
Tensor noisy_dense(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t units,
                   Generator& noise, double noise_scale = 1.0, Activation activation = Activation::None,
                   double sigma0 = 0.5);

// ---- recurrent cells (one time step; unroll with a loop or `unroll`) -------------------

/// h' = tanh(x Wx + h Wh + b). x [B, in], h [B, units].
Tensor rnn_step(ParameterStore& store, const std::string& name, const Tensor& x, const Tensor& h);

struct LstmState {
  Tensor h, c;
};
/// One LSTM step with gates i, f, g, o packed in "<name>/Wx" [in, 4u], "<name>/Wh"
/// [u, 4u] and "<name>/b" [4u]; the forget-gate bias starts at `forget_bias`.
LstmState lstm_step(ParameterStore& store, const std::string& name, const Tensor& x, const LstmState& state,
                    double forget_bias = 1.0);
/// Zero state for `units` LSTM units and batch size `batch`.
LstmState lstm_zero_state(std::int64_t batch, std::int64_t units, DType dtype = DType::Float32);

/// One GRU step (Cho et al. 2014); the update-gate bias starts at `update_bias`.
Tensor gru_step(ParameterStore& store, const std::string& name, const Tensor& x, const Tensor& h,
                double update_bias = 0.0);

/// Splits x [B, T, ...] into T tensors [B, ...].
std::vector<Tensor> time_steps(const Tensor& x);
/// Stacks T tensors [B, ...] into [B, T, ...].
Tensor stack_time(const std::vector<Tensor>& steps);

enum class RnnKind { Simple, Lstm, Gru };
/// Runs a cell over x [B, T, in] from a zero state and returns every hidden
/// state as [B, T, units] (TF static unrolling).
Tensor recurrent(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t units, RnnKind kind);

}  // namespace datamunge::nn
