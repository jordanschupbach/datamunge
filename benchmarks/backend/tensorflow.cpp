// Matched CPU fixtures shared with scripts/benchmark_tensorflow.py.
#include <algorithm>
#include <datamunge/nn/memory.hpp>
#include <limits>

#include "common.hpp"

namespace {
namespace nn = nb::nn;
double fixture(std::int64_t i) {
  return double(i % 251 - 125) / 128.0;
}
nn::Tensor input(nn::Shape shape, nn::DType dt) {
  std::vector<double> values(static_cast<std::size_t>(nn::shape_numel(shape)));
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = fixture(i);
  return nn::cast(nn::from_vector(values, shape), dt);
}
void BM_TensorFlowMatched(benchmark::State& s, int op) {
  nb::Scope  scope;
  const auto rows = s.range(0), cols = op == 10 ? rows : 128;
  const auto dt = s.range(1) ? nn::DType::Float64 : nn::DType::Float32;
  auto       x = input({rows, cols}, dt), bias = input({cols}, dt);
  if (op == 11)
    x.requires_grad_(true);
  auto run = [&]() -> nn::Tensor {
    switch (op) {
      case 0:
        return nn::add(x, x);
      case 1:
        return nn::add(x, bias);
      case 2:
        return nn::mul(x, x);
      case 3:
        return nn::sum(x, {-1});
      case 4:
        return nn::sum(x);
      case 5:
        return nn::mean(x);
      case 6:
        return nn::relu(x);
      case 7:
        return nn::exp(x);
      case 8:
        return nn::softmax(x);
      case 9:
        return nn::log_softmax(x);
      case 10:
        return nn::matmul(x, x);
      default:
        return nn::grad(nn::sum(nn::mul(x, x)), {x})[0];
    }
  };
  // Independent scalar references; fixture setup and validation are untimed.
  std::vector<double> reference;
  if (op == 4 || op == 5) {
    double total = 0;
    for (std::int64_t i = 0; i < rows * cols; ++i)
      total += fixture(i);
    reference.push_back(op == 5 ? total / (rows * cols) : total);
  } else {
    for (std::int64_t r = 0; r < rows; ++r) {
      double total = 0, largest = -std::numeric_limits<double>::infinity();
      for (std::int64_t c = 0; c < cols; ++c)
        largest = std::max(largest, fixture(r * cols + c));
      for (std::int64_t c = 0; c < cols; ++c)
        total +=
            op == 8 || op == 9 ? std::exp(fixture(r * cols + c) - largest) : fixture(r * cols + c);
      if (op == 3) {
        reference.push_back(total);
        continue;
      }
      for (std::int64_t c = 0; c < cols; ++c) {
        const auto v     = fixture(r * cols + c);
        double     value = 0;
        switch (op) {
          case 0:
          case 11:
            value = 2 * v;
            break;
          case 1:
            value = v + fixture(c);
            break;
          case 2:
            value = v * v;
            break;
          case 6:
            value = std::max(v, 0.0);
            break;
          case 7:
            value = std::exp(v);
            break;
          case 8:
            value = std::exp(v - largest) / total;
            break;
          case 9:
            value = v - largest - std::log(total);
            break;
          case 10:
            for (std::int64_t k = 0; k < cols; ++k)
              value += fixture(r * cols + k) * fixture(k * cols + c);
            break;
        }
        reference.push_back(value);
      }
    }
  }
  const auto actual = run().to_vector<double>();
  nb::check(actual.size() == reference.size(), "matched fixture shape mismatch");
  for (std::size_t i = 0; i < actual.size(); ++i)
    nb::check(
        std::abs(actual[i] - reference[i]) <=
            (s.range(1) ? 1e-10 : 2e-5) * (1 + std::abs(reference[i])),
        "matched fixture value mismatch");
  s.counters["reference_checked"] = 1;
  {
    nn::PeakMemoryScope memory;
    if (op == 11)
      nb::consume(run());
    else {
      nn::NoGradGuard guard;
      nb::consume(run());
    }
    s.counters[op == 11 ? "gradient_peak_tensor_bytes" : "peak_tensor_bytes"] =
        memory.peak_above_baseline();
  }
  if (op == 11) {
    for (auto _ : s)
      nb::consume(run());
  } else {
    nn::NoGradGuard guard;
    for (auto _ : s)
      nb::consume(run());
  }
  s.SetItemsProcessed(s.iterations() * rows * cols);
}
#define BASE(name, op, ...) \
  BENCHMARK_CAPTURE(BM_TensorFlowMatched, name, op) \
      ->ArgsProduct({{__VA_ARGS__}, {0, 1}})->ArgNames({"rows", "f64"})->UseRealTime();
#define EXTRA(name, ...)
#include "../tensorflow_cases.def"
#undef BASE
#undef EXTRA

// The public unary square is used by losses as well as the x*x expression in
// the matched TensorFlow fixture. Keep both sum and mean VJPs covered.
void BM_SquarePrimitiveVjp(benchmark::State& s) {
  nb::Scope  scope;
  const auto dtype = s.range(1) ? nn::DType::Float64 : nn::DType::Float32;
  auto       x     = input({s.range(0), 128}, dtype).requires_grad_();
  auto       run   = [&]() {
    auto squared = nn::square(x);
    return nn::grad(s.range(2) ? nn::mean(squared) : nn::sum(squared), {x})[0];
  };
  const auto scale = s.range(2) ? 2.0 / x.numel() : 2.0;
  nb::close(run(), nn::mul(x, scale), s.range(1) ? 1e-10 : 1e-5);
  {
    nn::PeakMemoryScope memory;
    nb::consume(run());
    s.counters["gradient_peak_tensor_bytes"] = memory.peak_above_baseline();
  }
  for (auto _ : s)
    nb::consume(run());
  s.SetItemsProcessed(s.iterations() * x.numel());
}
BENCHMARK(BM_SquarePrimitiveVjp)
    ->ArgsProduct({{512, 8192}, {0, 1}, {0, 1}})
    ->ArgNames({"rows", "f64", "mean"})
    ->UseRealTime();
} // namespace
