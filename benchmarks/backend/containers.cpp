#include <algorithm>
#include <random>

#include <datamunge/nn/fft.hpp>

#include "common.hpp"

namespace {
namespace nn = nb::nn;

std::vector<std::string> parameter_names(std::size_t count) {
  std::vector<std::string> names;
  for (std::size_t i = 0; i < count; ++i)
    names.push_back("encoder/block/" + std::to_string(i) + "/weight");
  std::shuffle(names.begin(), names.end(), std::mt19937(42));
  return names;
}

void BM_ContainerParameters(benchmark::State& s, bool create) {
  nb::Scope scope;
  const auto names = parameter_names(static_cast<std::size_t>(s.range(0)));
  const auto init = nn::init::constant(0.25);
  const nn::Shape shape{1};
  nn::ParameterStore store(42);
  for (const auto& name : names) store.get(name, shape, init);
  nb::check(store.names() == names, "parameter creation order changed");
  for (const auto& name : names)
    nb::check(store.at(name).item() == 0.25, "parameter lookup failed");
  for (auto _ : s) {
    if (create) {
      nn::ParameterStore fresh(42);
      for (const auto& name : names) benchmark::DoNotOptimize(fresh.get(name, shape, init));
    } else {
      for (const auto& name : names) benchmark::DoNotOptimize(store.get(name, shape, init));
    }
  }
  s.SetItemsProcessed(s.iterations() * s.range(0));
  s.counters["reference_checked"] = 1;
}
BENCHMARK_CAPTURE(BM_ContainerParameters, Lookup, false)->Arg(16)->Arg(256)->Arg(4096)->UseRealTime();
BENCHMARK_CAPTURE(BM_ContainerParameters, Create, true)->Arg(16)->Arg(256)->Arg(4096)->UseRealTime();

void BM_ContainerOptimizer(benchmark::State& s, bool first) {
  nb::Scope scope;
  std::vector<nn::Tensor> params, gradients;
  for (std::int64_t i = 0; i < s.range(0); ++i) {
    params.push_back(nn::ones({1}));
    params.back().requires_grad_(true);
    gradients.push_back(nn::full({1}, 0.25));
  }
  nn::OptimizerConfig config;
  config.kind = nn::OptimizerKind::AMSGrad;
  nn::Optimizer optimizer(params, 0, config);
  optimizer.step(gradients);
  nb::check(optimizer.state().size() == 2 + 3 * params.size(), "missing optimizer slots");
  for (auto _ : s) {
    if (first) {
      nn::Optimizer fresh(params, 0, config);
      fresh.step(gradients);
    } else {
      optimizer.step(gradients);
    }
  }
  for (const auto& p : params) nb::check(p.item() == 1, "zero-rate optimizer changed weight");
  s.SetItemsProcessed(s.iterations() * s.range(0));
  s.counters["reference_checked"] = 1;
}
BENCHMARK_CAPTURE(BM_ContainerOptimizer, First, true)->Arg(16)->Arg(256)->Arg(4096)->UseRealTime();
BENCHMARK_CAPTURE(BM_ContainerOptimizer, Warm, false)->Arg(16)->Arg(256)->Arg(4096)->UseRealTime();

void BM_ContainerBatchNorm(benchmark::State& s) {
  nb::Scope scope;
  nn::set_num_threads(static_cast<int>(s.range(1)));
  nn::NoGradGuard guard;
  auto x = nn::ones({1024, s.range(0)}, nn::DType::Float64);
  auto gamma = nn::ones({s.range(0)}, x.dtype()), beta = nn::zeros({s.range(0)}, x.dtype());
  auto run = [&] { return nn::batch_norm_train(x, gamma, beta); };
  auto result = run();
  nb::close(result.mean, gamma, 1e-12);
  nb::close(result.variance, beta, 1e-12);
  nb::close(result.y, nn::zeros(x.shape(), x.dtype()), 1e-12);
  for (auto _ : s) nb::consume(run().y);
  s.counters["reference_checked"] = 1;
}
BENCHMARK(BM_ContainerBatchNorm)->ArgsProduct({{1, 32, 256}, {1, 4}})->UseRealTime();
void BM_ContainerFFT(benchmark::State& s) {
  nb::Scope scope;
  nn::NoGradGuard guard;
  std::vector<nn::Tensor> inputs;
  for (std::int64_t i = 0; i < s.range(0); ++i)
    inputs.push_back(nn::ones({32 + i}, nn::DType::Float64));
  for (int warm = 0; warm < 2; ++warm)
    for (const auto& x : inputs) {
      auto spectrum = nn::rfft(x);
      nb::check(std::abs(spectrum.real.data_ptr<double>()[0] - x.numel()) < 1e-10,
                "FFT DC reference failed");
      nb::close(nn::irfft(spectrum, x.numel()), x, 1e-10);
    }
  for (auto _ : s)
    for (const auto& x : inputs) {
      auto result = nn::rfft(x);
      nb::consume(result.real);
      nb::consume(result.imag);
    }
  s.SetItemsProcessed(s.iterations() * s.range(0));
  s.counters["reference_checked"] = 1;
}
BENCHMARK(BM_ContainerFFT)->Arg(1)->Arg(16)->Arg(64)->UseRealTime();

void BM_ContainerGraph(benchmark::State& s) {
  nb::Scope scope;
  std::vector<nn::Tensor> leaves;
  auto loss = nn::scalar(0, nn::DType::Float64);
  for (std::int64_t i = 0; i < s.range(0); ++i) {
    leaves.push_back(nn::ones({1}, nn::DType::Float64).requires_grad_());
    loss = nn::add(loss, nn::sum(leaves.back()));
  }
  nn::GradOptions options;
  options.retain_graph = true;
  for (const auto& gradient : nn::grad(loss, leaves, options))
    nb::check(gradient.item() == 1, "wide graph gradient reference failed");
  for (auto _ : s) benchmark::DoNotOptimize(nn::grad(loss, leaves, options));
  s.SetItemsProcessed(s.iterations() * s.range(0));
  s.counters["reference_checked"] = 1;
}
BENCHMARK(BM_ContainerGraph)->Arg(16)->Arg(256)->Arg(4096)->UseRealTime();

} // namespace
