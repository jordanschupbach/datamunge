// Explicit cases completing the agreed coverage matrix; inputs and checks are untimed.
#include <datamunge/linalg/tensor.hpp>

#include "common.hpp"
namespace {
namespace nn = nb::nn;
void BM_TensorBroadcast(benchmark::State& s, int kind) {
  nb::Scope       scope;
  nn::NoGradGuard guard;
  auto            dt = s.range(0) ? nn::DType::Float64 : nn::DType::Float32;
  auto            x  = nn::ones({8, 8, 8, 32}, dt);
  auto            b  = nn::full(
      kind == 0   ? nn::Shape{}
      : kind == 1 ? nn::Shape{32}
                  : nn::Shape{1, 8, 1, 1},
      2,
      dt);
  nb::close(nn::add(x, b), nn::full(x.shape(), 3, dt), 1e-6);
  for (auto _ : s)
    nb::consume(nn::add(x, b));
  s.SetItemsProcessed(s.iterations() * x.numel());
}
BENCHMARK_CAPTURE(BM_TensorBroadcast, Scalar, 0)->Arg(0)->Arg(1)->UseRealTime();
BENCHMARK_CAPTURE(BM_TensorBroadcast, Channel, 1)->Arg(0)->Arg(1)->UseRealTime();
BENCHMARK_CAPTURE(BM_TensorBroadcast, Spatial, 2)->Arg(0)->Arg(1)->UseRealTime();
void BM_TensorMax(benchmark::State& s) {
  nb::Scope       scope;
  nn::NoGradGuard guard;
  auto x = nn::ones({s.range(0) / 32, 32}, s.range(1) ? nn::DType::Float64 : nn::DType::Float32);
  nb::close(nn::max(x, {-1}), nn::ones({s.range(0) / 32}, x.dtype()), 1e-6);
  for (auto _ : s)
    nb::consume(nn::max(x, {-1}));
  s.SetItemsProcessed(s.iterations() * x.numel());
}
BENCHMARK(BM_TensorMax)->ArgsProduct({{1024, 65536, 1048576}, {0, 1}})->UseRealTime();
void BM_LinalgTensorExtra(benchmark::State& s, int op) {
  using datamunge::linalg::Tensor;
  auto x   = Tensor::ones({static_cast<std::size_t>(s.range(0)) / 32, 32});
  auto run = [&]() {
    if (op == 0)
      return x.mean_axis(1);
    if (op == 1)
      return x.max_axis(1);
    return x.add_scalar(2);
  };
  auto ref = run();
  for (std::size_t i = 0; i < ref.size(); ++i)
    nb::check(ref.at_flat(i) == (op == 2 ? 3 : 1), "tensor reduction/broadcast reference failed");
  for (auto _ : s) {
    auto out = run();
    benchmark::DoNotOptimize(out);
    benchmark::ClobberMemory();
  }
  s.SetItemsProcessed(s.iterations() * x.size());
}
BENCHMARK_CAPTURE(BM_LinalgTensorExtra, Mean, 0)->Arg(1024)->Arg(65536)->UseRealTime();
BENCHMARK_CAPTURE(BM_LinalgTensorExtra, Max, 1)->Arg(1024)->Arg(65536)->UseRealTime();
BENCHMARK_CAPTURE(BM_LinalgTensorExtra, ScalarAdd, 2)->Arg(1024)->Arg(65536)->UseRealTime();
void BM_TensorSolveRhs(benchmark::State& s) {
  nb::Scope       scope;
  nn::NoGradGuard guard;
  auto            n = s.range(0), k = s.range(1);
  auto            x = nb::random({n, n});
  auto            a = nn::add(nn::matmul(x, x, false, true), nn::eye(n, nn::DType::Float64));
  auto            b = nb::random({n, k});
  nb::close(nn::matmul(a, nn::solve(a, b)), b, 1e-8);
  for (auto _ : s)
    nb::consume(nn::solve(a, b));
}
BENCHMARK(BM_TensorSolveRhs)
    ->ArgsProduct({{16, 64}, {1, 4}})
    ->ArgNames({"n", "rhs"})
    ->UseRealTime();
void BM_GeneralEigenvectors(benchmark::State& s) {
  nb::Scope       scope;
  nn::NoGradGuard guard;
  auto            n        = s.range(0);
  auto            diagonal = nn::add(nn::arange(0, n, 1, nn::DType::Float64), 1.0);
  auto a = nn::set_diag(nn::mul(nn::band_part(nb::random({n, n}), 0, -1), 0.01), diagonal);
  auto r = nn::eig(a);
  nb::close(nn::matmul(a, r.v.real), nn::matmul(r.v.real, nn::matrix_diag(r.e.real)), 1e-8);
  nb::close(r.e.imag, nn::zeros({n}, nn::DType::Float64), 1e-8);
  for (auto _ : s) {
    auto out = nn::eig(a);
    nb::consume(out.e.real);
    nb::consume(out.e.imag);
    nb::consume(out.v.real);
    nb::consume(out.v.imag);
  }
}
BENCHMARK(BM_GeneralEigenvectors)->Arg(16)->Arg(64)->UseRealTime();
void BM_LayerComponent(benchmark::State& s, int op) {
  nn::ParameterStore store(42, s.range(1) ? nn::DType::Float64 : nn::DType::Float32);
  nb::Scope          scope;
  auto               dt = s.range(1) ? nn::DType::Float64 : nn::DType::Float32;
  auto               x  = nb::random(
      (op >= 2 && op <= 4) ? nn::Shape{s.range(0), 8, 8, 4} : nn::Shape{s.range(0), 16, 32}, dt);
  x.requires_grad_(true);
  auto gamma = nn::ones({4}, dt), beta = nn::zeros({4}, dt), filter = nb::random({3, 3, 4, 8}, dt);
  gamma.requires_grad_(true);
  beta.requires_grad_(true);
  filter.requires_grad_(true);
  auto ids = nn::from_vector(std::vector<std::int64_t>(s.range(0) * 16, 2), {s.range(0), 16});
  auto run = [&]() {
    if (op == 0)
      return nn::layer_norm(store, "norm", x);
    if (op == 1)
      return nn::embedding(store, "embed", ids, 128, 32);
    if (op == 2)
      return nn::conv2d(x, filter);
    if (op == 3)
      return nn::max_pool2d(x, 2, 2);
    if (op >= 5)
      return nn::recurrent(
          store, "sequence", x, 32, op == 5 ? nn::RnnKind::Lstm : op == 6 ? nn::RnnKind::Gru : nn::RnnKind::Simple);
    return nn::batch_norm_train(x, gamma, beta).y;
  };
  nb::finite(run());
  std::vector<nn::Tensor> inputs = op == 1 ? store.parameters() : std::vector<nn::Tensor>{x};
  if (op == 0 || op >= 5) {
    auto p = store.parameters();
    inputs.insert(inputs.end(), p.begin(), p.end());
  }
  if (op == 2)
    inputs.push_back(filter);
  if (op == 4) {
    inputs.push_back(gamma);
    inputs.push_back(beta);
  }
  auto gradients = nn::grad(nn::sum(run()), inputs);
  for (auto& t : gradients)
    nb::finite(t);
  for (auto _ : s) {
    if (s.range(2))
      benchmark::DoNotOptimize(nn::grad(nn::sum(run()), inputs));
    else {
      nn::NoGradGuard guard;
      nb::consume(run());
    }
  }
  s.SetItemsProcessed(s.iterations() * s.range(0));
}
#define LAYER(name, op)                          \
  BENCHMARK_CAPTURE(BM_LayerComponent, name, op) \
      ->ArgsProduct({{1, 16}, {0, 1}, {0, 1}})   \
      ->ArgNames({"batch", "f64", "backward"})   \
      ->UseRealTime()
LAYER(LayerNorm, 0);
LAYER(Embedding, 1);
LAYER(Conv2D, 2);
LAYER(MaxPool, 3);
LAYER(BatchNorm, 4);
LAYER(LSTM, 5);
LAYER(GRU, 6);
LAYER(SimpleRNN, 7);
} // namespace
