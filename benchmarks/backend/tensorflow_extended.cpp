// Extended matched fixtures: independent scalar references, not cross-library oracles.
#include <algorithm>
#include <cmath>
#include <datamunge/nn/attention.hpp>
#include <datamunge/nn/fft.hpp>
#include <datamunge/nn/memory.hpp>

#include "common.hpp"
namespace {
namespace nn = nb::nn;
using I      = std::int64_t;
enum class Op {
#define BASE(name, ...)
#define EXTRA(name, ...) name,
#include "../tensorflow_cases.def"
#undef BASE
#undef EXTRA
};
double value(I i) {
  return double(i % 251 - 125) / 128;
}
nn::Tensor data(const nn::Shape& shape, nn::DType dtype) {
  std::vector<double> v(static_cast<std::size_t>(nn::shape_numel(shape)));
  for (I i = 0; i < static_cast<I>(v.size()); ++i)
    v[static_cast<std::size_t>(i)] = value(i);
  return nn::cast(nn::from_vector(v, shape), dtype);
}
void BM_TensorFlowExtended(benchmark::State& state, Op op) {
  nb::Scope  scope;
  const I    n     = state.range(0);
  const auto dtype = state.range(1) ? nn::DType::Float64 : nn::DType::Float32;
  nn::Shape  shape{n, 128};
  const bool conv = op == Op::Conv2d || op == Op::ConvInputGradient;
  const bool pool = op == Op::MaxPool || op == Op::AvgPool;
  if (conv || pool)
    shape = {n, 16, 16, 4};
  if (op == Op::MatmulTranspose || op == Op::Solve || op == Op::Cholesky)
    shape = {n, n};
  if (op == Op::BatchMatmul)
    shape = {4, n, n};
  if (op == Op::MatmulGradient)
    shape = {n, 64};
  if (op == Op::Attention)
    shape = {2, n, 32};
  if (op == Op::FftPower)
    shape = {8, n};
  auto x = data(shape, dtype), w = data({shape.back()}, dtype);
  if (conv)
    w = data({3, 3, 4, 8}, dtype);
  if (op == Op::MatmulGradient)
    w = data({64, 32}, dtype).requires_grad_();
  if (op == Op::GatherRows) {
    std::vector<I> indices(static_cast<std::size_t>(n / 4));
    for (I i = 0; i < n / 4; ++i)
      indices[static_cast<std::size_t>(i)] = 3 * i;
    w = nn::from_vector(indices, {n / 4});
  }
  if (op == Op::Solve || op == Op::Cholesky) {
    std::vector<double> spd(static_cast<std::size_t>(n * n));
    for (I i = 0; i < n; ++i)
      for (I j = 0; j < n; ++j) {
        double total = i == j ? double(n) : 0;
        for (I k = 0; k < n; ++k)
          total += value(i * n + k) * value(j * n + k);
        spd[static_cast<std::size_t>(i * n + j)] = total;
      }
    x = nn::cast(nn::from_vector(spd, {n, n}), dtype);
    w = data({n, 4}, dtype);
  }
  const bool gradient =
      op == Op::SoftmaxGradient || op == Op::ConvInputGradient || op == Op::MatmulGradient;
  if (gradient && op != Op::MatmulGradient)
    x.requires_grad_();
  nn::ParameterStore parameters(42, dtype);
  auto               gamma = nn::ones({128}, dtype), beta = nn::zeros({128}, dtype);
  auto               run = [&]() -> nn::Tensor {
    switch (op) {
      case Op::SumFirst:
        return nn::sum(x, {0});
      case Op::MaxLast:
        return nn::max(x, {-1});
      case Op::Variance:
        return nn::variance(x, {-1});
      case Op::LogSumExp:
        return nn::logsumexp(x, {-1});
      case Op::Sigmoid:
        return nn::sigmoid(x);
      case Op::Tanh:
        return nn::tanh(x);
      case Op::Gelu:
        return nn::gelu(x);
      case Op::GatherRows:
        return nn::gather(x, w);
      case Op::TransposeAdd: {
        auto t = nn::transpose(x);
        return nn::add(t, t);
      }
      case Op::MatmulTranspose:
        return nn::matmul(x, x, true, false);
      case Op::BatchMatmul:
        return nn::matmul(x, x);
      case Op::MatmulGradient:
        return nn::grad(nn::sum(nn::square(nn::matmul(x, w))), {w})[0];
      case Op::Conv2d:
        return nn::conv2d(x, w, nn::conv_options(1, nn::Padding::Valid));
      case Op::ConvInputGradient:
        return nn::grad(nn::sum(nn::conv2d(x, w, nn::conv_options(1, nn::Padding::Valid))), {x})[0];
      case Op::MaxPool:
        return nn::max_pool2d(x, 2, 2, nn::Padding::Valid);
      case Op::AvgPool:
        return nn::avg_pool2d(x, 2, 2, nn::Padding::Valid);
      case Op::BatchNorm:
        return nn::batch_norm_train(x, gamma, beta, 1e-5).y;
      case Op::LayerNorm:
        return nn::layer_norm(parameters, "norm", x, 1e-5);
      case Op::Attention:
        return nn::scaled_dot_product_attention(x, x, x).output;
      case Op::FftPower:
        return nn::complex_abs_squared(nn::rfft(x));
      case Op::Solve:
        return nn::solve(x, w);
      case Op::Cholesky:
        return nn::cholesky(x);
      case Op::SoftmaxGradient:
        return nn::grad(nn::sum(nn::square(nn::softmax(x))), {x})[0];
    }
    throw std::logic_error("unhandled matched fixture");
  };
  const auto          v = x.to_vector<double>(), weights = w.to_vector<double>();
  std::vector<double> ref;
  nn::Shape           output_shape = shape;
  if (op == Op::SumFirst) {
    output_shape = {128};
    ref.assign(128, 0);
    for (I r = 0; r < n; ++r)
      for (I c = 0; c < 128; ++c)
        ref[c] += v[r * 128 + c];
  } else if (op == Op::GatherRows) {
    output_shape = {n / 4, 128};
    for (I r = 0; r < n / 4; ++r)
      for (I c = 0; c < 128; ++c)
        ref.push_back(v[3 * r * 128 + c]);
  } else if (op == Op::TransposeAdd) {
    output_shape = {128, n};
    for (I c = 0; c < 128; ++c)
      for (I r = 0; r < n; ++r)
        ref.push_back(2 * v[r * 128 + c]);
  } else if (op == Op::MatmulTranspose || op == Op::BatchMatmul) {
    for (I batch = 0; batch < (op == Op::BatchMatmul ? 4 : 1); ++batch)
      for (I i = 0; i < n; ++i)
        for (I j = 0; j < n; ++j) {
          double total = 0;
          for (I k = 0; k < n; ++k)
            total += op == Op::MatmulTranspose
                         ? v[k * n + i] * v[k * n + j]
                         : v[batch * n * n + i * n + k] * v[batch * n * n + k * n + j];
          ref.push_back(total);
        }
  } else if (op == Op::MatmulGradient) {
    output_shape = {64, 32};
    std::vector<double> y(static_cast<std::size_t>(n * 32), 0);
    for (I i = 0; i < n; ++i)
      for (I j = 0; j < 32; ++j)
        for (I k = 0; k < 64; ++k)
          y[i * 32 + j] += v[i * 64 + k] * weights[k * 32 + j];
    for (I k = 0; k < 64; ++k)
      for (I j = 0; j < 32; ++j) {
        double total = 0;
        for (I i = 0; i < n; ++i)
          total += v[i * 64 + k] * 2 * y[i * 32 + j];
        ref.push_back(total);
      }
  } else if (conv || pool) {
    if (conv) {
      output_shape = op == Op::Conv2d ? nn::Shape{n, 14, 14, 8} : shape;
      ref.assign(static_cast<std::size_t>(nn::shape_numel(output_shape)), 0);
      for (I b = 0; b < n; ++b)
        for (I y = 0; y < 14; ++y)
          for (I z = 0; z < 14; ++z)
            for (I ky = 0; ky < 3; ++ky)
              for (I kx = 0; kx < 3; ++kx)
                for (I c = 0; c < 4; ++c)
                  for (I out = 0; out < 8; ++out) {
                    const I      src    = ((b * 16 + y + ky) * 16 + z + kx) * 4 + c;
                    const double weight = weights[((ky * 3 + kx) * 4 + c) * 8 + out];
                    if (op == Op::Conv2d)
                      ref[((b * 14 + y) * 14 + z) * 8 + out] += v[src] * weight;
                    else
                      ref[src] += weight;
                  }
    } else {
      output_shape = {n, 8, 8, 4};
      for (I b = 0; b < n; ++b)
        for (I y = 0; y < 8; ++y)
          for (I z = 0; z < 8; ++z)
            for (I c = 0; c < 4; ++c) {
              double total = op == Op::MaxPool ? -INFINITY : 0;
              for (I ky = 0; ky < 2; ++ky)
                for (I kx = 0; kx < 2; ++kx) {
                  const double sample = v[((b * 16 + 2 * y + ky) * 16 + 2 * z + kx) * 4 + c];
                  total = op == Op::MaxPool ? std::max(total, sample) : total + sample;
                }
              ref.push_back(op == Op::MaxPool ? total : total / 4);
            }
    }
  } else if (op == Op::Attention) {
    output_shape = shape;
    ref.assign(v.size(), 0);
    std::vector<double> p(static_cast<std::size_t>(n));
    for (I b = 0; b < 2; ++b)
      for (I i = 0; i < n; ++i) {
        for (I j = 0; j < n; ++j) {
          double dot = 0;
          for (I k = 0; k < 32; ++k)
            dot += v[(b * n + i) * 32 + k] * v[(b * n + j) * 32 + k];
          p[j] = dot / std::sqrt(32.0);
        }
        const double largest = *std::max_element(p.begin(), p.end());
        double       total   = 0;
        for (auto& a : p) {
          a = std::exp(a - largest);
          total += a;
        }
        for (I k = 0; k < 32; ++k)
          for (I j = 0; j < n; ++j)
            ref[(b * n + i) * 32 + k] += p[j] / total * v[(b * n + j) * 32 + k];
      }
  } else if (op == Op::FftPower) {
    output_shape = {8, n / 2 + 1};
    for (I b = 0; b < 8; ++b)
      for (I k = 0; k <= n / 2; ++k) {
        double real = 0, imag = 0;
        for (I j = 0; j < n; ++j) {
          const double angle = 2 * std::acos(-1.0) * k * j / n;
          real += v[b * n + j] * std::cos(angle);
          imag -= v[b * n + j] * std::sin(angle);
        }
        ref.push_back(real * real + imag * imag);
      }
  } else if (op == Op::Solve || op == Op::Cholesky) {
    std::vector<double> l(static_cast<std::size_t>(n * n), 0);
    for (I i = 0; i < n; ++i)
      for (I j = 0; j <= i; ++j) {
        double a = v[i * n + j];
        for (I k = 0; k < j; ++k)
          a -= l[i * n + k] * l[j * n + k];
        l[i * n + j] = i == j ? std::sqrt(a) : a / l[j * n + j];
      }
    if (op == Op::Cholesky)
      ref = l;
    else {
      output_shape = {n, 4};
      ref          = weights;
      for (I i = 0; i < n; ++i)
        for (I c = 0; c < 4; ++c) {
          for (I k = 0; k < i; ++k)
            ref[i * 4 + c] -= l[i * n + k] * ref[k * 4 + c];
          ref[i * 4 + c] /= l[i * n + i];
        }
      for (I i = n; i-- > 0;)
        for (I c = 0; c < 4; ++c) {
          for (I k = i + 1; k < n; ++k)
            ref[i * 4 + c] -= l[k * n + i] * ref[k * 4 + c];
          ref[i * 4 + c] /= l[i * n + i];
        }
    }
  } else if (op == Op::BatchNorm) {
    std::vector<double> mu(128, 0), var(128, 0);
    for (I c = 0; c < 128; ++c) {
      for (I r = 0; r < n; ++r)
        mu[c] += v[r * 128 + c];
      mu[c] /= n;
      for (I r = 0; r < n; ++r) {
        const double d = v[r * 128 + c] - mu[c];
        var[c] += d * d;
      }
      var[c] /= n;
    }
    for (I r = 0; r < n; ++r)
      for (I c = 0; c < 128; ++c)
        ref.push_back((v[r * 128 + c] - mu[c]) / std::sqrt(var[c] + 1e-5));
  } else {
    const bool reduce = op == Op::MaxLast || op == Op::Variance || op == Op::LogSumExp;
    if (reduce)
      output_shape = {n};
    for (I r = 0; r < n; ++r) {
      double       mu = 0, var = 0, total = 0;
      const double largest = *std::max_element(v.begin() + r * 128, v.begin() + (r + 1) * 128);
      for (I c = 0; c < 128; ++c)
        mu += v[r * 128 + c];
      mu /= 128;
      for (I c = 0; c < 128; ++c) {
        const double d = v[r * 128 + c] - mu;
        var += d * d;
        total += std::exp(v[r * 128 + c] - largest);
      }
      var /= 128;
      if (reduce) {
        ref.push_back(
            op == Op::MaxLast    ? largest
            : op == Op::Variance ? var
                                 : largest + std::log(total));
        continue;
      }
      double squared = 0;
      for (I c = 0; c < 128; ++c) {
        const double p = std::exp(v[r * 128 + c] - largest) / total;
        squared += p * p;
      }
      for (I c = 0; c < 128; ++c) {
        const double a = v[r * 128 + c], p = std::exp(a - largest) / total;
        switch (op) {
          case Op::Sigmoid:
            ref.push_back(1 / (1 + std::exp(-a)));
            break;
          case Op::Tanh:
            ref.push_back(std::tanh(a));
            break;
          case Op::Gelu:
            ref.push_back(.5 * a * (1 + std::erf(a / std::sqrt(2.0))));
            break;
          case Op::LayerNorm:
            ref.push_back((a - mu) / std::sqrt(var + 1e-5));
            break;
          case Op::SoftmaxGradient:
            ref.push_back(2 * p * (p - squared));
            break;
          default:
            throw std::logic_error("missing scalar reference");
        }
      }
    }
  }
  const double tolerance = state.range(1)
                               ? (op == Op::Gelu ? 1e-8 : 1e-10)
                               : (op == Op::MatmulGradient || op == Op::FftPower ? 2e-4 : 2e-5);
  auto         check     = [&] {
    const auto result = run();
    nb::check(result.shape() == output_shape, "extended shape mismatch");
    const auto actual = result.to_vector<double>();
    nb::check(actual.size() == ref.size(), "extended size mismatch");
    for (std::size_t i = 0; i < ref.size(); ++i)
      nb::check(
          std::isfinite(actual[i]) &&
              std::abs(actual[i] - ref[i]) <= tolerance * (1 + std::abs(ref[i])),
          "extended independent reference failed");
  };
  if (gradient)
    check();
  else {
    nn::NoGradGuard guard;
    check();
  }
  state.counters["reference_checked"]   = 1;
  state.counters["reference_tolerance"] = tolerance;
  auto measure                          = [&] {
    {
      nn::PeakMemoryScope memory;
      nb::consume(run());
      state.counters[gradient ? "gradient_peak_tensor_bytes" : "peak_tensor_bytes"] =
          memory.peak_above_baseline();
    }
    for (auto _ : state)
      nb::consume(run());
  };
  if (gradient)
    measure();
  else {
    nn::NoGradGuard guard;
    measure();
  }
  state.SetItemsProcessed(state.iterations() * x.numel());
}
#define BASE(name, ...)
#define EXTRA(name, ...)                                   \
  BENCHMARK_CAPTURE(BM_TensorFlowExtended, name, Op::name) \
      ->Name("BM_TensorFlowMatched/" #name)                \
      ->ArgsProduct({{__VA_ARGS__}, {0, 1}})               \
      ->ArgNames({"rows", "f64"})                          \
      ->UseRealTime();
#include "../tensorflow_cases.def"
#undef BASE
#undef EXTRA
} // namespace
