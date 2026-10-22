#include "datamunge/nn/layers.hpp"

#include <cmath>
#include <stdexcept>

#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

std::string activation_name(Activation activation) {
  switch (activation) {
    case Activation::None: return "identity";
    case Activation::Relu: return "relu";
    case Activation::Relu6: return "relu6";
    case Activation::LeakyRelu: return "leaky_relu";
    case Activation::Elu: return "elu";
    case Activation::Selu: return "selu";
    case Activation::Gelu: return "gelu";
    case Activation::GeluTanh: return "gelu_tanh";
    case Activation::Silu: return "silu";
    case Activation::Mish: return "mish";
    case Activation::Softplus: return "softplus";
    case Activation::Softsign: return "softsign";
    case Activation::Sigmoid: return "sigmoid";
    case Activation::Tanh: return "tanh";
  }
  return "?";
}

Tensor activate(const Tensor& x, Activation activation, double alpha) {
  switch (activation) {
    case Activation::None: return x;
    case Activation::Relu: return relu(x);
    case Activation::Relu6: return relu6(x);
    case Activation::LeakyRelu: return leaky_relu(x, alpha);
    case Activation::Elu: return elu(x);
    case Activation::Selu: return selu(x);
    case Activation::Gelu: return gelu(x);
    case Activation::GeluTanh: return gelu(x, true);
    case Activation::Silu: return silu(x);
    case Activation::Mish: return mish(x);
    case Activation::Softplus: return softplus(x);
    case Activation::Softsign: return softsign(x);
    case Activation::Sigmoid: return sigmoid(x);
    case Activation::Tanh: return tanh(x);
  }
  return x;
}

namespace {

std::int64_t last_dim(const Tensor& x, const char* what) {
  if (x.dim() < 1) throw std::invalid_argument(std::string(what) + ": input must have rank >= 1");
  return x.shape().back();
}

}  // namespace

Tensor dense(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t units,
             const DenseOptions& options) {
  const std::int64_t in = last_dim(x, "dense");
  const Tensor       W  = store.get(name + "/W", {in, units}, options.init);
  // Flatten [..., in] to [-1, in] for the product and restore the leading axes.
  Shape out_shape = x.shape();
  out_shape.back() = units;
  Tensor y         = x.dim() == 2 ? matmul(x, W) : reshape(matmul(reshape(x, {-1, in}), W), out_shape);
  if (options.bias) y = bias_add(y, store.get(name + "/b", {units}, init::constant(options.bias_value)));
  return activate(y, options.activation, options.alpha);
}

Tensor dense(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t units,
             Activation activation) {
  DenseOptions options;
  options.activation = activation;
  return dense(store, name, x, units, options);
}

Tensor mlp(ParameterStore& store, const std::string& name, const Tensor& x, const std::vector<std::int64_t>& hidden,
           Activation activation) {
  Tensor h = x;
  for (std::size_t i = 0; i < hidden.size(); ++i)
    h = dense(store, name + "/dense" + std::to_string(i), h, hidden[i], activation);
  return h;
}

Tensor dropout(const Tensor& x, double rate, bool training, Generator& gen) {
  if (!(rate >= 0 && rate < 1)) throw std::invalid_argument("dropout: rate must be in [0, 1)");
  if (!training || rate == 0) return x;
  const double keep = 1.0 - rate;
  const Tensor mask = bernoulli(x.shape(), keep, gen, to_floating(x.dtype()));
  return mul(x, mask) / keep;
}

Tensor embedding(ParameterStore& store, const std::string& name, const Tensor& ids, std::int64_t vocab,
                 std::int64_t dim, double limit) {
  const Tensor table = store.get(name + "/table", {vocab, dim}, init::uniform(limit));
  return gather(table, ids, 0);
}

Tensor masked_mean(const Tensor& x, const Tensor& mask) {
  const Tensor m     = expand_dims(cast(mask, x.dtype()), -1);
  const Tensor total = sum(mul(x, m), {1});
  const Tensor count = maximum(sum(m, {1}), scalar(1.0, x.dtype()));
  return div(total, count);
}

Tensor layer_norm(ParameterStore& store, const std::string& name, const Tensor& x, double eps) {
  const std::int64_t features = last_dim(x, "layer_norm");
  const Tensor       gamma    = store.get(name + "/gamma", {features}, init::ones());
  const Tensor       beta     = store.get(name + "/beta", {features}, init::zeros());
  const Tensor       mu       = mean(x, {-1}, true);
  const Tensor       var      = mean(square(sub(x, mu)), {-1}, true);
  return add(mul(mul(sub(x, mu), rsqrt(add(var, eps))), gamma), beta);
}

Tensor rms_norm(ParameterStore& store, const std::string& name, const Tensor& x, double eps) {
  const std::int64_t features = last_dim(x, "rms_norm");
  const Tensor       gamma    = store.get(name + "/gamma", {features}, init::ones());
  return mul(mul(x, rsqrt(add(mean(square(x), {-1}, true), eps))), gamma);
}

Tensor rnn_step(ParameterStore& store, const std::string& name, const Tensor& x, const Tensor& h) {
  const std::int64_t in = last_dim(x, "rnn_step"), units = last_dim(h, "rnn_step");
  const Tensor Wx = store.get(name + "/Wx", {in, units}, init::uniform(std::sqrt(6.0 / static_cast<double>(in + units))));
  const Tensor Wh = store.get(name + "/Wh", {units, units}, init::uniform(std::sqrt(6.0 / static_cast<double>(2 * units))));
  const Tensor b  = store.get(name + "/b", {units}, init::zeros());
  return tanh(bias_add(add(matmul(x, Wx), matmul(h, Wh)), b));
}

LstmState lstm_step(ParameterStore& store, const std::string& name, const Tensor& x, const LstmState& state,
                    double forget_bias) {
  const std::int64_t in = last_dim(x, "lstm_step"), u = last_dim(state.h, "lstm_step");
  const Tensor Wx = store.get(name + "/Wx", {in, 4 * u}, init::uniform(std::sqrt(6.0 / static_cast<double>(in + 4 * u))));
  const Tensor Wh = store.get(name + "/Wh", {u, 4 * u}, init::uniform(std::sqrt(6.0 / static_cast<double>(5 * u))));
  const Tensor b  = store.get(name + "/b", {4 * u}, [u, forget_bias](const Shape&, DType d, Generator&) {
    return concat({zeros({u}, d), full({u}, forget_bias, d), zeros({2 * u}, d)}, 0);
  });
  const Tensor gates = bias_add(add(matmul(x, Wx), matmul(state.h, Wh)), b);
  const auto   parts = split(gates, 4, 1);  // i, f, g, o
  const Tensor i = sigmoid(parts[0]), f = sigmoid(parts[1]), g = tanh(parts[2]), o = sigmoid(parts[3]);
  LstmState    next;
  next.c = add(mul(f, state.c), mul(i, g));
  next.h = mul(o, tanh(next.c));
  return next;
}

LstmState lstm_zero_state(std::int64_t batch, std::int64_t units, DType dtype) {
  return {zeros({batch, units}, dtype), zeros({batch, units}, dtype)};
}

Tensor gru_step(ParameterStore& store, const std::string& name, const Tensor& x, const Tensor& h, double update_bias) {
  const std::int64_t in = last_dim(x, "gru_step"), u = last_dim(h, "gru_step");
  const Tensor Wx = store.get(name + "/Wx_zr", {in, 2 * u}, init::uniform(std::sqrt(6.0 / static_cast<double>(in + 2 * u))));
  const Tensor Wh = store.get(name + "/Wh_zr", {u, 2 * u}, init::uniform(std::sqrt(6.0 / static_cast<double>(3 * u))));
  const Tensor b  = store.get(name + "/b_zr", {2 * u}, [u, update_bias](const Shape&, DType d, Generator&) {
    return concat({full({u}, update_bias, d), zeros({u}, d)}, 0);
  });
  const Tensor Wxn = store.get(name + "/Wx_n", {in, u}, init::uniform(std::sqrt(6.0 / static_cast<double>(in + u))));
  const Tensor Whn = store.get(name + "/Wh_n", {u, u}, init::uniform(std::sqrt(6.0 / static_cast<double>(2 * u))));
  const Tensor bn  = store.get(name + "/b_n", {u}, init::zeros());
  const auto   zr  = split(sigmoid(bias_add(add(matmul(x, Wx), matmul(h, Wh)), b)), 2, 1);
  const Tensor& z = zr[0];
  const Tensor& r = zr[1];
  const Tensor  n = tanh(bias_add(add(matmul(x, Wxn), matmul(mul(r, h), Whn)), bn));
  return add(mul(sub(1.0, z), n), mul(z, h));
}

std::vector<Tensor> time_steps(const Tensor& x) {
  if (x.dim() < 2) throw std::invalid_argument("time_steps: input must be [batch, time, ...]");
  return unstack(x, 1);
}

Tensor stack_time(const std::vector<Tensor>& steps) { return stack(steps, 1); }

Tensor recurrent(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t units, RnnKind kind) {
  if (x.dim() != 3) throw std::invalid_argument("recurrent: input must be [batch, time, features]");
  const std::int64_t  batch = x.shape()[0];
  std::vector<Tensor> outputs;
  Tensor              h     = zeros({batch, units}, to_floating(x.dtype()));
  LstmState           state = lstm_zero_state(batch, units, to_floating(x.dtype()));
  for (const Tensor& step : time_steps(x)) {
    switch (kind) {
      case RnnKind::Simple: h = rnn_step(store, name, step, h); break;
      case RnnKind::Gru: h = gru_step(store, name, step, h); break;
      case RnnKind::Lstm:
        state = lstm_step(store, name, step, state);
        h     = state.h;
        break;
    }
    outputs.push_back(h);
  }
  return stack_time(outputs);
}

}  // namespace datamunge::nn
