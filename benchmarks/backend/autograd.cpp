#include <datamunge/autodiff/autodiff.hpp>
#include <datamunge/nn/memory.hpp>

#include "common.hpp"
namespace {
namespace nn = nb::nn;
void BM_AutogradChain(benchmark::State& s, int mode) {
  nb::Scope scope;
  auto      x = nn::full({s.range(0)}, 0.1, nn::DType::Float64);
  x.requires_grad_(true);
  auto run = [&]() {
    auto y = x;
    for (int i = 0; i < s.range(1); ++i)
      y = nn::add(nn::mul(y, 0.99), 0.001);
    return nn::sum(y);
  };
  {
    auto g = nn::grad(run(), {x});
    nb::close(g[0], nn::full(x.shape(), std::pow(0.99, s.range(1)), x.dtype()), 1e-10);
  }
  std::int64_t peak     = 0;
  const auto   baseline = nn::live_memory_bytes();
  {
    nn::PeakMemoryScope mem;
    auto                g = nn::grad(run(), {x});
    peak                  = mem.peak_above_baseline();
  }
  nb::check(nn::live_memory_bytes() == baseline, "autograd retained tensor storage");
  s.counters["gradient_peak_tensor_bytes"] = double(peak);
  if (mode == 0) {
    nn::NoGradGuard guard;
    for (auto _ : s)
      nb::consume(run());
  } else if (mode == 1) {
    for (auto _ : s)
      nb::consume(run());
  } else if (mode == 2) {
    for (auto _ : s)
      benchmark::DoNotOptimize(nn::grad(run(), {x}));
  } else {
    auto            y = run();
    nn::GradOptions opts;
    opts.retain_graph = true;
    for (auto _ : s)
      benchmark::DoNotOptimize(nn::grad(y, {x}, opts));
  }
  s.SetItemsProcessed(s.iterations() * s.range(0) * s.range(1));
}
#define AG(name, mode)                            \
  BENCHMARK_CAPTURE(BM_AutogradChain, name, mode) \
      ->ArgsProduct({{32, 4096}, {1, 16, 128}})   \
      ->ArgNames({"elements", "depth"})           \
      ->UseRealTime()
AG(NoGrad, 0);
AG(Record, 1);
AG(ForwardBackward, 2);
AG(RetainedBackward, 3);
void BM_AutogradFanout(benchmark::State& s) {
  nb::Scope scope;
  auto      x = nn::ones({32, 32}, nn::DType::Float64), b = nn::ones({32}, nn::DType::Float64);
  x.requires_grad_(true);
  b.requires_grad_(true);
  auto run = [&]() {
    auto y = nn::add(x, b);
    for (int i = 0; i < s.range(0); ++i)
      y = nn::add(y, nn::mul(x, x));
    return nn::sum(y);
  };
  auto g = nn::grad(run(), {x, b});
  nb::close(g[0], nn::full(x.shape(), 1 + 2 * s.range(0), x.dtype()), 1e-10);
  nb::close(g[1], nn::full(b.shape(), 32, b.dtype()), 1e-10);
  for (auto _ : s)
    benchmark::DoNotOptimize(nn::grad(run(), {x, b}));
}
BENCHMARK(BM_AutogradFanout)->Arg(1)->Arg(16)->Arg(64)->UseRealTime();
void BM_AutogradLinalg(benchmark::State& s, int op) {
  nb::Scope scope;
  auto      n = s.range(0);
  auto      x = nb::random({n, n});
  auto      a = nn::add(nn::matmul(x, x, false, true), nn::eye(n, nn::DType::Float64));
  a.requires_grad_(true);
  auto b = nb::random({n, 4});
  b.requires_grad_(true);
  auto calc = [&](const nn::Tensor& p) {
    if (op == 0)
      return nn::sum(nn::solve(p, b));
    if (op == 1)
      return nn::sum(nn::cholesky(p));
    if (op == 2)
      return nn::sum(nn::qr(p).r);
    if (op == 3)
      return nn::sum(nn::svd(p).s);
    return nn::sum(nn::eigh(p).e);
  };
  auto run    = [&]() { return calc(a); };
  auto inputs = op == 0 ? std::vector<nn::Tensor>{a, b} : std::vector<nn::Tensor>{a};
  auto g      = nn::grad(run(), inputs);
  for (auto& t : g)
    nb::finite(t);
  // Directional finite difference in a symmetric direction, valid for both ops.
  const double eps       = 1e-5;
  auto         direction = nn::eye(n, nn::DType::Float64);
  auto         eval      = [&](double sign) {
    nn::NoGradGuard guard;
    auto            p = nn::add(a, nn::mul(direction, sign * eps));
    return calc(p).item();
  };
  nb::check(
      std::abs((eval(1) - eval(-1)) / (2 * eps) - nn::sum(nn::mul(g[0], direction)).item()) < 1e-5,
      "linear algebra gradient reference failed");
  for (auto _ : s)
    benchmark::DoNotOptimize(nn::grad(run(), inputs));
}
BENCHMARK_CAPTURE(BM_AutogradLinalg, Solve, 0)->Arg(16)->Arg(64)->UseRealTime();
BENCHMARK_CAPTURE(BM_AutogradLinalg, Cholesky, 1)->Arg(16)->Arg(64)->UseRealTime();
BENCHMARK_CAPTURE(BM_AutogradLinalg, QR, 2)->Arg(16)->Arg(64)->UseRealTime();
BENCHMARK_CAPTURE(BM_AutogradLinalg, SVD, 3)->Arg(16)->Arg(64)->UseRealTime();
BENCHMARK_CAPTURE(BM_AutogradLinalg, Eigh, 4)->Arg(16)->Arg(64)->UseRealTime();
void BM_AutodiffTape(benchmark::State& s, int mode) {
  namespace ad = datamunge::autodiff;
  auto build   = [&](ad::Tape& tape) {
    ad::Var y(tape, 0.1);
    for (int i = 0; i < s.range(0); ++i)
      y = y * 0.99 + 0.001;
    return y;
  };
  ad::Tape tape;
  auto     y = build(tape);
  nb::check(
      std::abs(tape.backward(y.index())[0] - std::pow(0.99, s.range(0))) < 1e-10,
      "tape reference failed");
  for (auto _ : s) {
    if (mode == 1)
      benchmark::DoNotOptimize(tape.backward(y.index()));
    else {
      ad::Tape t;
      auto     out = build(t);
      if (mode == 2)
        benchmark::DoNotOptimize(t.backward(out.index()));
      else
        benchmark::DoNotOptimize(out);
    }
  }
  s.SetItemsProcessed(s.iterations() * static_cast<std::int64_t>(tape.size()));
}
#define AD(name, mode) \
  BENCHMARK_CAPTURE(BM_AutodiffTape, name, mode)->Arg(16)->Arg(128)->Arg(1024)->UseRealTime()
AD(Record, 0);
AD(Backward, 1);
AD(ForwardBackward, 2);
void BM_AutodiffForward(benchmark::State& s, int mode) {
  namespace ad = datamunge::autodiff;
  std::vector<double> x(s.range(0), 0.25);
  auto                objective = [](const auto& v) {
    auto sum = v[0] * 0.0;
    for (const auto& t : v)
      sum = sum + t * t + t * t * t;
    return sum;
  };
  if (mode == 0) {
    auto g = ad::gradient_forward(objective, x);
    for (double v : g)
      nb::check(std::abs(v - 0.6875) < 1e-12, "dual reference failed");
    for (auto _ : s)
      benchmark::DoNotOptimize(ad::gradient_forward(objective, x));
  } else {
    auto h = ad::hessian(objective, x);
    for (std::size_t i = 0; i < x.size(); ++i)
      for (std::size_t j = 0; j < x.size(); ++j)
        nb::check(std::abs(h(i, j) - (i == j ? 3.5 : 0.0)) < 1e-12, "hyperdual reference failed");
    for (auto _ : s)
      benchmark::DoNotOptimize(ad::hessian(objective, x));
  }
}
BENCHMARK_CAPTURE(BM_AutodiffForward, DualGradient, 0)->Arg(4)->Arg(16)->Arg(64)->UseRealTime();
BENCHMARK_CAPTURE(BM_AutodiffForward, HyperdualHessian, 1)->Arg(4)->Arg(16)->Arg(64)->UseRealTime();

void BM_AutodiffBranch(benchmark::State& s) {
  namespace ad = datamunge::autodiff;
  auto run     = [&]() {
    ad::Tape tape;
    ad::Var  x(tape, 0.25);
    auto     y = x;
    for (int i = 0; i < s.range(0); ++i)
      y = y * 0.99 + x * x;
    return tape.backward(y.index())[x.index()];
  };
  double p = std::pow(0.99, s.range(0));
  nb::check(
      std::abs(run() - (p + 0.5 * (1 - p) / 0.01)) < 1e-9, "branching tape derivative failed");
  for (auto _ : s)
    benchmark::DoNotOptimize(run());
  s.SetItemsProcessed(s.iterations() * s.range(0));
}
BENCHMARK(BM_AutodiffBranch)->Arg(16)->Arg(128)->Arg(1024)->UseRealTime();
void BM_AutodiffJacobian(benchmark::State& s) {
  namespace ad = datamunge::autodiff;
  std::vector<double> x(static_cast<std::size_t>(s.range(0)), 0.25);
  auto objective = [](const auto& v) {
    auto a = v[0] * 0.0, b = v[0] * 0.0;
    for (const auto& t : v) { a = a + t * t; b = b + t * t * t; }
    return std::vector<decltype(a)>{a, b};
  };
  const auto reference = ad::jacobian_forward(objective, x);
  for (std::size_t j = 0; j < x.size(); ++j) {
    nb::check(std::abs(reference(0, j) - 0.5) < 1e-12, "Jacobian square reference failed");
    nb::check(std::abs(reference(1, j) - 0.1875) < 1e-12, "Jacobian cubic reference failed");
  }
  for (auto _ : s) benchmark::DoNotOptimize(ad::jacobian_forward(objective, x));
}
BENCHMARK(BM_AutodiffJacobian)->Arg(8)->Arg(32)->Arg(128)->UseRealTime();

} // namespace
