#include <datamunge/linalg/tensor.hpp>

#include "common.hpp"
namespace {
namespace nn = nb::nn;
void tensor_args(benchmark::internal::Benchmark* b) {
  b->ArgsProduct({{1024, 65536, 1048576}, {0, 1}, {0, 1}})
      ->ArgNames({"elements", "f64", "transposed"})
      ->UseRealTime();
}
void BM_TensorOps(benchmark::State& s, int op) {
  nb::Scope       scope;
  nn::NoGradGuard guard;
  auto            n  = s.range(0);
  auto            dt = s.range(1) ? nn::DType::Float64 : nn::DType::Float32;
  auto            x  = nn::ones({n / 32, 32}, dt);
  if (s.range(2))
    x = nn::transpose(x);
  auto b       = nn::full({x.shape().back()}, 2.0, dt);
  auto indices = nn::from_vector(std::vector<std::int64_t>{0, 1, 2, 3}, {4});
  auto updates = nn::ones({4, x.shape().back()}, dt);
  auto run     = [&]() -> nn::Tensor {
    switch (op) {
      case 0:
        return nn::empty(x.shape(), dt);
      case 1:
        return nn::zeros(x.shape(), dt);
      case 2:
        return x.clone();
      case 3:
        return nn::transpose(x);
      case 4:
        return x.contiguous();
      case 5:
        return nn::add(x, b);
      case 6:
        return nn::mul(x, x);
      case 7:
        return nn::sum(x, {-1});
      case 8:
        return nn::mean(x);
      case 9:
        return nn::cast(x, s.range(1) ? nn::DType::Float32 : nn::DType::Float64);
      case 10:
        return nn::gather(x, indices, 0);
      case 11:
        return nn::concat({x, x}, 0);
      case 12:
        return nn::reshape(x, {n});
      case 13:
        return nn::slice_axis(x, 0, 0, 4);
      default:
        return nn::index_add(x, 0, indices, updates);
    }
  };
  // Allocation-only reads are deliberately excluded: empty() is uninitialized.
  if (op != 0) {
    auto r = run();
    nb::finite(r);
    double expected = op == 1 ? 0 : op == 5 ? 3 : op == 7 ? double(x.shape().back()) : 1;
    if (op == 14) {
      nb::close(nn::slice_axis(r, 0, 0, 4), nn::full({4, x.shape().back()}, 2, dt), 1e-5);
      nb::close(
          nn::slice_axis(r, 0, 4, x.shape()[0]),
          nn::ones({x.shape()[0] - 4, x.shape().back()}, dt),
          1e-5);
    } else
      nb::close(r, nn::full(r.shape(), expected, r.dtype()), 1e-5);
  }
  for (auto _ : s)
    nb::consume(run());
  s.SetItemsProcessed(
      s.iterations() * ((op == 10 || op == 13) ? 4 * x.shape().back()
                        : op == 11             ? 2 * n
                                               : n));
}
#define T(name, op) BENCHMARK_CAPTURE(BM_TensorOps, name, op)->Apply(tensor_args)
T(Empty, 0);
T(Zeros, 1);
T(Clone, 2);
T(Transpose, 3);
T(Contiguous, 4);
T(BroadcastAdd, 5);
T(Multiply, 6);
T(ReduceSum, 7);
T(Mean, 8);
T(Cast, 9);
T(Gather, 10);
T(Concat, 11);
T(Reshape, 12);
T(Slice, 13);
T(IndexAdd, 14);
void BM_LinalgTensor(benchmark::State& s, int op) {
  using datamunge::linalg::Tensor;
  auto n = static_cast<std::size_t>(s.range(0));
  auto x = Tensor::ones({n / 32, 32}), b = Tensor::full({32}, 2.0);
  auto run = [&]() {
    switch (op) {
      case 0:
        return Tensor::zeros({n / 32, 32});
      case 1:
        return x.reshape({n});
      case 2:
        return x.transpose();
      case 3:
        return x.add(b);
      case 4:
        return x.multiply(x);
      case 5:
        return x.slice(0, 0, 4);
      default:
        return Tensor::concatenate({x, x}, 0);
    }
  };
  auto r = run();
  nb::check(r.at_flat(0) == (op == 0 ? 0 : op == 3 ? 3 : 1), "linalg tensor reference failed");
  for (auto _ : s) {
    auto out = run();
    benchmark::DoNotOptimize(out);
    benchmark::ClobberMemory();
  }
  s.SetItemsProcessed(s.iterations() * static_cast<std::int64_t>(r.size()));
}
#define LT(name, op) \
  BENCHMARK_CAPTURE(BM_LinalgTensor, name, op)->Arg(1024)->Arg(65536)->Arg(1048576)->UseRealTime()
LT(Zeros, 0);
LT(Reshape, 1);
LT(Transpose, 2);
LT(BroadcastAdd, 3);
LT(Multiply, 4);
LT(Slice, 5);
LT(Concat, 6);
void BM_LinalgTensorSum(benchmark::State& s) {
  auto x = datamunge::linalg::Tensor::ones({static_cast<std::size_t>(s.range(0))});
  nb::check(x.sum() == s.range(0), "sum reference failed");
  for (auto _ : s)
    benchmark::DoNotOptimize(x.sum());
  s.SetItemsProcessed(s.iterations() * s.range(0));
}
BENCHMARK(BM_LinalgTensorSum)->Arg(1024)->Arg(65536)->Arg(1048576)->UseRealTime();
void BM_LinalgTensorMatmul(benchmark::State& s) {
  using datamunge::linalg::Tensor;
  auto m = static_cast<std::size_t>(s.range(0)), k = static_cast<std::size_t>(s.range(1)),
       n = static_cast<std::size_t>(s.range(2));
  auto a = Tensor::ones({m, k}), b = Tensor::ones({k, n});
  auto r = a.matmul(b);
  for (std::size_t i = 0; i < r.size(); ++i)
    nb::check(r.at_flat(i) == double(k), "linalg matmul reference failed");
  for (auto _ : s) {
    auto out = a.matmul(b);
    benchmark::DoNotOptimize(out);
    benchmark::ClobberMemory();
  }
  s.counters["FLOP/s"] =
      benchmark::Counter(2.0 * m * k * n, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_LinalgTensorMatmul)
    ->Args({16, 32, 16})
    ->Args({64, 64, 64})
    ->Args({128, 32, 64})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->UseRealTime();


void BM_LinalgTensorTransposeLayout(benchmark::State& s) {
  using datamunge::linalg::Tensor;
  const auto rows = static_cast<std::size_t>(s.range(0)), cols = static_cast<std::size_t>(s.range(1));
  auto x = Tensor::arange(0, double(rows * cols)).reshape({rows, cols});
  const bool identity = s.range(2);
  auto run = [&] { return x.transpose(identity ? std::vector<std::size_t>{0, 1} : std::vector<std::size_t>{1, 0}); };
  auto reference = run();
  nb::check(reference.at({0, 1}) == (identity ? 1.0 : double(cols)), "transpose layout reference failed");
  for (auto _ : s) { auto result = run(); benchmark::DoNotOptimize(result); benchmark::ClobberMemory(); }
  s.SetBytesProcessed(s.iterations() * rows * cols * sizeof(double) * 2);
}
BENCHMARK(BM_LinalgTensorTransposeLayout)
    ->ArgsProduct({{64, 256, 1024}, {64, 256, 1024}, {0, 1}})
    ->ArgNames({"rows", "cols", "identity"})->UseRealTime();

} // namespace
