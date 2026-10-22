#pragma once

/// @file layers.hpp
/// @brief Functional layers over a ParameterStore.
///
/// Each layer takes the store, a name and its input, and creates its
/// parameters under "<name>/..." on first use (tflow's `nn::Dense(scope, p,
/// name, x, ...)` without the scope). Input widths are read from the input,
/// so they are not passed in. Calling a layer again with the same name reuses
/// its weights.

#include <cstdint>
#include <string>
#include <vector>

#include "datamunge/nn/params.hpp"
#include "datamunge/nn/random.hpp"
#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

enum class Activation {
  None, Relu, Relu6, LeakyRelu, Elu, Selu, Gelu, GeluTanh, Silu, Mish, Softplus, Softsign, Sigmoid, Tanh
};

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
