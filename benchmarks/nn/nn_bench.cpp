// Speed benchmarks for datamunge::nn: GEMM, convolution, pooling, batch norm
// and a full cnn_mnist-sized training epoch on synthetic data.
//
// Run with `just bench-nn`. Thread count: DATAMUNGE_NUM_THREADS (default: all
// cores). Several benchmarks also take the thread count as their last
// argument, so the serial and threaded speeds are compared side by side.

#include <benchmark/benchmark.h>

#include <datamunge/nn/nn.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace nn = datamunge::nn;

namespace {

nn::Tensor random_tensor(const nn::Shape& shape, std::uint64_t seed) {
  nn::Generator gen(seed);
  return nn::normal(shape, 0.0, 1.0, gen);
}

/// Restores the default thread count when a benchmark finishes.
struct ThreadScope {
  explicit ThreadScope(std::int64_t threads) { nn::set_num_threads(static_cast<int>(threads)); }
  ~ThreadScope() { nn::set_num_threads(0); }
};

// ---- GEMM -------------------------------------------------------------------------

void BM_Matmul(benchmark::State& state) {
  const std::int64_t n = state.range(0);
  ThreadScope        threads(state.range(1));
  const nn::Tensor   a = random_tensor({n, n}, 1), b = random_tensor({n, n}, 2);
  for (auto _ : state) benchmark::DoNotOptimize(nn::matmul(a, b));
  state.counters["GFLOPS"] =
      benchmark::Counter(2.0 * n * n * n, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
  state.SetLabel(nn::gemm_backend());
}
BENCHMARK(BM_Matmul)->ArgsProduct({{128, 512, 1024}, {1, 0}})->Unit(benchmark::kMillisecond);

void BM_MatmulBatched(benchmark::State& state) {
  // Many small products, as in attention heads: [64, 32, 64] @ [64, 64, 32].
  const nn::Tensor a = random_tensor({64, 32, 64}, 3), b = random_tensor({64, 64, 32}, 4);
  for (auto _ : state) benchmark::DoNotOptimize(nn::matmul(a, b));
}
BENCHMARK(BM_MatmulBatched)->Unit(benchmark::kMicrosecond);

// ---- convolution (cnn_cifar10's second layer: [128, 16, 16, 32] * [3, 3, 32, 64]) ---

struct ConvShapes {
  nn::Shape x{128, 16, 16, 32}, filter{3, 3, 32, 64}, y{128, 16, 16, 64};
};

void BM_Conv2dForward(benchmark::State& state) {
  ThreadScope      threads(state.range(0));
  const ConvShapes s;
  const nn::Tensor x = random_tensor(s.x, 5), f = random_tensor(s.filter, 6);
  for (auto _ : state) benchmark::DoNotOptimize(nn::conv2d(x, f));
}
BENCHMARK(BM_Conv2dForward)->Arg(1)->Arg(0)->Unit(benchmark::kMillisecond);

void BM_Conv2dBackpropInput(benchmark::State& state) {
  ThreadScope      threads(state.range(0));
  const ConvShapes s;
  const nn::Tensor f = random_tensor(s.filter, 7), g = random_tensor(s.y, 8);
  for (auto _ : state) benchmark::DoNotOptimize(nn::conv2d_backprop_input(s.x, f, g));
}
BENCHMARK(BM_Conv2dBackpropInput)->Arg(1)->Arg(0)->Unit(benchmark::kMillisecond);

void BM_Conv2dBackpropFilter(benchmark::State& state) {
  ThreadScope      threads(state.range(0));
  const ConvShapes s;
  const nn::Tensor x = random_tensor(s.x, 9), g = random_tensor(s.y, 10);
  for (auto _ : state) benchmark::DoNotOptimize(nn::conv2d_backprop_filter(x, s.filter, g));
}
BENCHMARK(BM_Conv2dBackpropFilter)->Arg(1)->Arg(0)->Unit(benchmark::kMillisecond);

// ---- pooling, batch norm and elementwise ops on a [128, 32, 32, 32] activation --------

void BM_MaxPoolForwardBackward(benchmark::State& state) {
  nn::Tensor x = random_tensor({128, 32, 32, 32}, 11);
  x.requires_grad_(true);
  for (auto _ : state) benchmark::DoNotOptimize(nn::grad(nn::sum(nn::max_pool2d(x, 2, 2)), {x}));
}
BENCHMARK(BM_MaxPoolForwardBackward)->Unit(benchmark::kMillisecond);

void BM_BatchNormForwardBackward(benchmark::State& state) {
  nn::Tensor x = random_tensor({128, 32, 32, 32}, 12), gamma = nn::ones({32}), beta = nn::zeros({32});
  x.requires_grad_(true);
  gamma.requires_grad_(true);
  beta.requires_grad_(true);
  for (auto _ : state)
    benchmark::DoNotOptimize(nn::grad(nn::sum(nn::batch_norm_train(x, gamma, beta).y), {x, gamma, beta}));
}
BENCHMARK(BM_BatchNormForwardBackward)->Unit(benchmark::kMillisecond);

void BM_ReluBiasForwardBackward(benchmark::State& state) {
  ThreadScope threads(state.range(0));
  nn::Tensor  x = random_tensor({128, 32, 32, 32}, 13), b = nn::zeros({32});
  x.requires_grad_(true);
  b.requires_grad_(true);
  for (auto _ : state) benchmark::DoNotOptimize(nn::grad(nn::sum(nn::relu(nn::bias_add(x, b))), {x, b}));
}
BENCHMARK(BM_ReluBiasForwardBackward)->Arg(1)->Arg(0)->Unit(benchmark::kMillisecond);

// ---- a full cnn_mnist epoch -----------------------------------------------------------

/// tflow's MnistCnn: Conv(16)-pool-Conv(32)-pool-Dense(64)-10, Adam, batch 256,
/// over 55,000 synthetic 28x28 images (speed does not depend on the pixels).
void BM_CnnMnistEpoch(benchmark::State& state) {
  const std::int64_t rows = 55000, batch = 256;
  nn::Generator      gen(14);
  const nn::Tensor   images = nn::uniform({rows, 28, 28, 1}, 0.0, 1.0, gen);
  std::vector<std::int64_t> label_values(static_cast<std::size_t>(rows));
  for (std::int64_t i = 0; i < rows; ++i) label_values[static_cast<std::size_t>(i)] = i % 10;
  const nn::Tensor labels = nn::from_vector(label_values, {rows});

  nn::ParameterStore   store(15);
  nn::ConvLayerOptions relu;
  relu.activation   = nn::Activation::Relu;
  const auto logits = [&](const nn::Tensor& x) {
    nn::Tensor h = nn::max_pool2d(nn::conv2d(store, "conv1", x, 16, 3, relu), 2, 2);
    h            = nn::max_pool2d(nn::conv2d(store, "conv2", h, 32, 3, relu), 2, 2);
    h            = nn::dense(store, "dense", nn::reshape(h, {-1, 7 * 7 * 32}), 64, nn::Activation::Relu);
    return nn::dense(store, "output", h, 10);
  };
  {
    nn::NoGradGuard guard;
    logits(nn::slice_axis(images, 0, 0, 1));
  }
  nn::OptimizerConfig config;
  config.kind = nn::OptimizerKind::Adam;
  nn::Optimizer optimizer(store, 1e-3, config);
  for (auto _ : state) {
    for (std::int64_t start = 0; start < rows; start += batch) {
      const std::int64_t stop = std::min(rows, start + batch);
      const nn::Tensor   loss = nn::sparse_softmax_cross_entropy(
          logits(nn::slice_axis(images, 0, start, stop)), nn::slice_axis(labels, 0, start, stop));
      optimizer.step(nn::grad(loss, store.parameters()));
    }
  }
}
BENCHMARK(BM_CnnMnistEpoch)->Iterations(1)->Unit(benchmark::kSecond);

}  // namespace

BENCHMARK_MAIN();
