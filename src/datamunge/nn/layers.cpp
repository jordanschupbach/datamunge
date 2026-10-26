#include "datamunge/nn/layers.hpp"

#include <cmath>
#include <stdexcept>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"
#include "detail.hpp"
#include "vector_math.hpp"

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
  if (!is_grad_enabled() && features > 0 && is_floating(x.dtype()) && x.is_contiguous() &&
      gamma.is_contiguous() && beta.is_contiguous() && x.dtype() == gamma.dtype() &&
      x.dtype() == beta.dtype()) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    Tensor out = empty(x.shape(), x.dtype());
    detail::dispatch_float(x.dtype(), "layer_norm", [&](auto tag) {
      using T         = typename decltype(tag)::type;
      const T* source = x.data_ptr<T>();
      const T* scale  = gamma.data_ptr<T>();
      const T* bias   = beta.data_ptr<T>();
      T*       dest   = out.data_ptr<T>();
      parallel_for(
          0,
          x.numel() / features,
          std::max<std::int64_t>(1, detail::kParallelGrain / features),
          [&](std::int64_t lo, std::int64_t hi) {
            for (auto row = lo; row < hi; ++row) {
              const T* src = source + row * features;
              T*       dst = dest + row * features;
              const T  mu =
                  static_cast<T>(detail::sum_values(src, features)) / static_cast<T>(features);
              // Output doubles as row-local square scratch, preserving T rounding.
              for (std::int64_t i = 0; i < features; ++i) {
                const T d = src[i] - mu;
                dst[i]    = d * d;
              }
              const T var =
                  static_cast<T>(detail::sum_values(dst, features)) / static_cast<T>(features);
              const T inverse = T(1) / std::sqrt(var + static_cast<T>(eps));
              for (std::int64_t i = 0; i < features; ++i) {
                const T normalized = (src[i] - mu) * inverse;
                const T scaled     = normalized * scale[i];
                dst[i]             = scaled + bias[i];
              }
            }
          });
    });
    return out;
  }
  const Tensor mu  = mean(x, {-1}, true);
  const Tensor var = mean(square(sub(x, mu)), {-1}, true);
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
  const std::int64_t batch = x.shape()[0], steps = x.shape()[1], in = x.shape()[2];
  std::vector<Tensor> outputs;
  outputs.reserve(static_cast<std::size_t>(steps));
  Tensor h = zeros({batch, units}, to_floating(x.dtype()));
  if (steps == 0) return stack_time(outputs);
  if (kind != RnnKind::Simple && kind != RnnKind::Lstm && kind != RnnKind::Gru) {
    outputs.assign(static_cast<std::size_t>(steps), h);
    return stack_time(outputs);
  }

  // Preserve parameter names and initialization order from the single-step
  // cells, but fetch them once and project every input timestep in one GEMM.
  const std::int64_t gates = (kind == RnnKind::Lstm ? 4 : kind == RnnKind::Gru ? 2 : 1) * units;
  const std::string suffix = kind == RnnKind::Gru ? "_zr" : "";
  const Tensor Wx = store.get(name + "/Wx" + suffix, {in, gates}, init::uniform(std::sqrt(6.0 / static_cast<double>(in + gates))));
  const Tensor Wh = store.get(name + "/Wh" + suffix, {units, gates}, init::uniform(std::sqrt(6.0 / static_cast<double>(units + gates))));
  const Tensor b = store.get(name + "/b" + suffix, {gates}, [units, kind](const Shape& shape, DType dt, Generator&) {
    if (kind == RnnKind::Lstm) return concat({zeros({units}, dt), full({units}, 1.0, dt), zeros({2 * units}, dt)}, 0);
    return zeros(shape, dt);
  });
  Tensor Wxn, Whn, bn;
  if (kind == RnnKind::Gru) {
    Wxn = store.get(name + "/Wx_n", {in, units}, init::uniform(std::sqrt(6.0 / static_cast<double>(in + units))));
    Whn = store.get(name + "/Wh_n", {units, units}, init::uniform(std::sqrt(6.0 / static_cast<double>(2 * units))));
    bn = store.get(name + "/b_n", {units}, init::zeros());
  }
  // Small float32 cells favor per-step GEMM; larger batches benefit from
  // a time-major projection whose timestep views are contiguous.
  const bool batch_projection = batch > 1 || x.dtype() != DType::Float32;
  const auto projected = batch_projection ? unstack(matmul(swapaxes(x, 0, 1), Wx), 0) : time_steps(x);
  const auto projected_n = kind == RnnKind::Gru && batch_projection
                               ? unstack(matmul(swapaxes(x, 0, 1), Wxn), 0) : std::vector<Tensor>{};
  Tensor c;
  if (kind == RnnKind::Lstm) c = zeros({batch, units}, to_floating(x.dtype()));
  for (std::size_t t = 0; t < projected.size(); ++t) {
    const Tensor input_gate = batch_projection ? projected[t] : matmul(projected[t], Wx);
    const Tensor gate_values = bias_add(add(input_gate, matmul(h, Wh)), b);
    if (kind == RnnKind::Simple) {
      h = tanh(gate_values);
    } else if (kind == RnnKind::Lstm) {
      const auto parts = split(gate_values, 4, 1);
      const Tensor i = sigmoid(parts[0]), f = sigmoid(parts[1]), g = tanh(parts[2]), o = sigmoid(parts[3]);
      c = add(mul(f, c), mul(i, g));
      h = mul(o, tanh(c));
    } else {
      const auto zr = split(sigmoid(gate_values), 2, 1);
      const Tensor input_n = batch_projection ? projected_n[t] : matmul(projected[t], Wxn);
      const Tensor n = tanh(bias_add(add(input_n, matmul(mul(zr[1], h), Whn)), bn));
      h = add(mul(sub(1.0, zr[0]), n), mul(zr[0], h));
    }
    outputs.push_back(h);
  }
  return stack_time(outputs);
}

// ---- convolution and normalization layers -----------------------------------------------------

namespace {

std::int64_t channels(const Tensor& x, std::int64_t rank, const char* what) {
  if (x.dim() != rank)
    throw std::invalid_argument(std::string(what) + ": input must have rank " + std::to_string(rank) + ", got " +
                                shape_str(x.shape()));
  return x.shape().back();
}

Conv2DOptions layer_conv_options(const ConvLayerOptions& o) { return conv_options(o.stride, o.padding, o.dilation); }

Tensor finish(ParameterStore& store, const std::string& name, Tensor y, std::int64_t units,
              const ConvLayerOptions& o) {
  if (o.bias) y = bias_add(y, store.get(name + "/b", {units}, init::zeros()));
  return activate(y, o.activation, o.alpha);
}

}  // namespace

Tensor conv2d(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t filters,
              std::int64_t kernel, const ConvLayerOptions& options) {
  const std::int64_t in = channels(x, 4, "conv2d layer");
  const Tensor       w  = store.get(name + "/W", {kernel, kernel, in, filters}, options.init);
  return finish(store, name, conv2d(x, w, layer_conv_options(options)), filters, options);
}

Tensor conv1d(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t filters,
              std::int64_t kernel, const ConvLayerOptions& options) {
  const std::int64_t in = channels(x, 3, "conv1d layer");
  // tflow stores the kernel as [1, k, in, out]; draw it with the fans of [k, in, out].
  const Initializer init = options.init;
  const Tensor      w    = store.get(name + "/W", {1, kernel, in, filters}, [init](const Shape& s, DType dt, Generator& g) {
    return reshape(init({s[1], s[2], s[3]}, dt, g), s);
  });
  Conv2DOptions o;
  o.strides   = {1, options.stride};
  o.dilations = {1, options.dilation};
  o.padding   = options.padding;
  const Tensor y = squeeze(conv2d(expand_dims(x, 1), w, o), {1});
  return finish(store, name, y, filters, options);
}

Tensor depthwise_conv2d(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t multiplier,
                        std::int64_t kernel, const ConvLayerOptions& options) {
  const std::int64_t in = channels(x, 4, "depthwise_conv2d layer");
  // TF depthwise fans: fan_in = k*k, fan_out = k*k*multiplier.
  const Initializer init = options.init;
  const Tensor      w    = store.get(name + "/W", {kernel, kernel, in, multiplier}, [init](const Shape& s, DType dt, Generator& g) {
    std::vector<Tensor> per_channel;  // each [k, k, 1, M], so the fans are TF's
    for (std::int64_t c = 0; c < s[2]; ++c) per_channel.push_back(init({s[0], s[1], 1, s[3]}, dt, g));
    return concat(per_channel, 2);
  });
  return finish(store, name, depthwise_conv2d(x, w, layer_conv_options(options)), in * multiplier, options);
}

Tensor separable_conv2d(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t filters,
                        std::int64_t kernel, const ConvLayerOptions& options, std::int64_t multiplier) {
  ConvLayerOptions dw = options;
  dw.activation       = Activation::None;
  dw.bias             = false;
  const Tensor h      = depthwise_conv2d(store, name + "/depthwise", x, multiplier, kernel, dw);
  ConvLayerOptions pw = options;
  pw.stride = 1, pw.dilation = 1, pw.padding = Padding::Valid;
  return conv2d(store, name + "/pointwise", h, filters, 1, pw);
}

Tensor conv2d_transpose(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t filters,
                        std::int64_t kernel, const ConvLayerOptions& options) {
  const std::int64_t in   = channels(x, 4, "conv2d_transpose layer");
  const Initializer  init = options.init;
  const Tensor       w    = store.get(name + "/W", {kernel, kernel, filters, in}, [init](const Shape& s, DType dt, Generator& g) {
    return transpose(init({s[0], s[1], s[3], s[2]}, dt, g), {0, 1, 3, 2}).contiguous();
  });
  return finish(store, name, conv2d_transpose(x, w, layer_conv_options(options)), filters, options);
}

Tensor conv3d(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t filters,
              std::int64_t kernel, const ConvLayerOptions& options) {
  const std::int64_t in = channels(x, 5, "conv3d layer");
  const Tensor       w  = store.get(name + "/W", {kernel, kernel, kernel, in, filters}, options.init);
  return finish(store, name, conv3d(x, w, conv3d_options(options.stride, options.padding, options.dilation)), filters,
                options);
}

Tensor conv3d_transpose(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t filters,
                        std::int64_t kernel, const ConvLayerOptions& options) {
  const std::int64_t in   = channels(x, 5, "conv3d_transpose layer");
  const Initializer  init = options.init;
  const Tensor w = store.get(name + "/W", {kernel, kernel, kernel, filters, in}, [init](const Shape& s, DType dt, Generator& g) {
    return transpose(init({s[0], s[1], s[2], s[4], s[3]}, dt, g), {0, 1, 2, 4, 3}).contiguous();
  });
  const Conv3DOptions o = conv3d_options(options.stride, options.padding, options.dilation);
  return finish(store, name, conv3d_transpose(x, w, o), filters, options);
}

Tensor batch_norm(ParameterStore& params, ParameterStore& stats, const std::string& name, const Tensor& x,
                  bool training, const BatchNormOptions& options) {
  if (x.dim() < 1) throw std::invalid_argument("batch_norm layer: input needs rank >= 1");
  const std::int64_t c  = x.shape().back();
  const DType        dt = params.dtype();
  const Tensor gamma = options.scale ? params.get(name + "/gamma", {c}, init::ones()) : ones({c}, dt);
  const Tensor beta  = options.center ? params.get(name + "/beta", {c}, init::zeros()) : zeros({c}, dt);
  const Tensor mu    = stats.get(name + "/moving_mean", {c}, init::zeros());
  const Tensor var   = stats.get(name + "/moving_var", {c}, init::ones());
  if (!training) return batch_norm_infer(x, gamma, beta, mu.detach(), var.detach(), options.eps);
  const BatchNormResult r = batch_norm_train(x, gamma, beta, options.eps);
  if (options.update_stats) {
    NoGradGuard  guard;
    const double m = options.momentum;
    stats.set(name + "/moving_mean", mu.detach() * m + cast(r.mean, mu.dtype()) * (1.0 - m));
    stats.set(name + "/moving_var", var.detach() * m + cast(r.variance, var.dtype()) * (1.0 - m));
  }
  return r.y;
}

Tensor group_norm(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t groups, double eps) {
  const std::int64_t c = channels(x, 4, "group_norm");
  if (groups < 1 || c % groups != 0)
    throw std::invalid_argument("group_norm: " + std::to_string(c) + " channels do not split into " +
                                std::to_string(groups) + " groups");
  const Tensor gamma = store.get(name + "/gamma", {c}, init::ones());
  const Tensor beta  = store.get(name + "/beta", {c}, init::zeros());
  const auto&  s     = x.shape();
  const Tensor g     = reshape(x, {s[0], s[1], s[2], groups, c / groups});
  const Tensor mu    = mean(g, {1, 2, 4}, true);
  const Tensor var   = mean(squared_difference(g, mu), {1, 2, 4}, true);
  const Tensor xhat  = reshape((g - mu) * rsqrt(var + eps), s);
  return xhat * gamma + beta;
}

Tensor instance_norm(ParameterStore& store, const std::string& name, const Tensor& x, double eps) {
  return group_norm(store, name, x, channels(x, 4, "instance_norm"), eps);
}

Tensor noisy_dense(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t units,
                   Generator& noise, double noise_scale, Activation activation, double sigma0) {
  const std::int64_t in    = last_dim(x, "noisy_dense");
  const double       bound = 1.0 / std::sqrt(static_cast<double>(in));
  const Tensor mu_w    = store.get(name + "/mu_W", {in, units}, init::uniform(bound));
  const Tensor sigma_w = store.get(name + "/sigma_W", {in, units}, init::constant(sigma0 * bound));
  const Tensor mu_b    = store.get(name + "/mu_b", {units}, init::uniform(bound));
  const Tensor sigma_b = store.get(name + "/sigma_b", {units}, init::constant(sigma0 * bound));
  const DType  dt      = mu_w.dtype();
  const auto   f       = [](const Tensor& e) { return sign(e) * sqrt(abs(e)); };
  const Tensor e_in    = f(normal({in}, 0.0, 1.0, noise, dt));
  const Tensor e_out   = f(normal({units}, 0.0, 1.0, noise, dt));
  const Tensor w       = mu_w + sigma_w * (outer(e_in, e_out) * noise_scale);
  const Tensor b       = mu_b + sigma_b * (e_out * noise_scale);
  return activate(bias_add(matmul(x, w), b), activation);
}

}  // namespace datamunge::nn
