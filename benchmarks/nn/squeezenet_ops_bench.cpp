// Micro-benchmarks of every op class in one SqueezeNet v1.0 CIFAR-10 training
// step (batch 32, 32x32 images) at the exact layer shapes the report uses, so
// each item of the plan in benchmarks/SQUEEZENET.md can be measured alone.
//
// Build and run with `just bench-squeezenet-ops [--benchmark_filter=REGEX]`.
// The last benchmark argument is the thread count (1 and 16: one core, and
// the physical cores of the reference machine).

#include <benchmark/benchmark.h>

#include <datamunge/nn/augment.hpp>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/losses.hpp>
#include <datamunge/nn/nn.hpp>
#include <datamunge/nn/optim.hpp>
#include <datamunge/nn/parallel.hpp>
#include <datamunge/nn/squeezenet_models.hpp>
#include <datamunge/nn/training.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace nn = datamunge::nn;
using I64    = std::int64_t;

namespace {

nn::Tensor random_tensor(const nn::Shape& shape, std::uint64_t seed) {
  nn::Generator gen(seed);
  return nn::normal(shape, 0.0, 1.0, gen);
}

/// Restores the previous thread count when a benchmark finishes.
struct ThreadScope {
  int previous = nn::get_num_threads();
  explicit ThreadScope(I64 threads) { nn::set_num_threads(static_cast<int>(threads)); }
  ~ThreadScope() { nn::set_num_threads(previous); }
};

const std::vector<I64> kThreads{1, 8, 16};

// ---- convolution layers of SqueezeNet v1.0 (CIFAR stem) -------------------------------

struct ConvCase {
  const char* name;
  nn::Shape   x, filter;
};

// Every distinct (input, filter) geometry of the network; the fire modules not
// listed repeat one of these shapes with other widths of the same cost class.
const ConvCase kConvCases[] = {
    {"conv1", {32, 32, 32, 3}, {3, 3, 3, 96}},
    {"fire2_squeeze", {32, 16, 16, 96}, {1, 1, 96, 16}},
    {"fire2_expand1x1", {32, 16, 16, 16}, {1, 1, 16, 64}},
    {"fire2_expand3x3", {32, 16, 16, 16}, {3, 3, 16, 64}},
    {"fire4_expand3x3", {32, 16, 16, 32}, {3, 3, 32, 128}},
    {"fire5_squeeze", {32, 8, 8, 256}, {1, 1, 256, 32}},
    {"fire6_expand3x3", {32, 8, 8, 48}, {3, 3, 48, 192}},
    {"fire8_squeeze", {32, 8, 8, 384}, {1, 1, 384, 64}},
    {"fire8_expand3x3", {32, 8, 8, 64}, {3, 3, 64, 256}},
    {"fire9_expand3x3", {32, 4, 4, 64}, {3, 3, 64, 256}},
    {"conv10", {32, 4, 4, 512}, {1, 1, 512, 10}},
};
constexpr I64 kConvCount = sizeof(kConvCases) / sizeof(kConvCases[0]);

nn::Conv2DOptions same_padding() {
  nn::Conv2DOptions o;
  o.padding = nn::Padding::Same;
  return o;
}

double conv_flop(const ConvCase& c) {
  const double rows = static_cast<double>(c.x[0] * c.x[1] * c.x[2]);
  const double kc   = static_cast<double>(c.filter[0] * c.filter[1] * c.filter[2]);
  return 2.0 * rows * kc * static_cast<double>(c.filter[3]);
}

void conv_counters(benchmark::State& state, const ConvCase& c) {
  state.counters["FLOP/s"] = benchmark::Counter(conv_flop(c), benchmark::Counter::kIsIterationInvariantRate,
                                                benchmark::Counter::kIs1000);
  state.SetLabel(c.name);
}

void BM_Conv2dForward(benchmark::State& state) {
  const ConvCase&  c = kConvCases[state.range(0)];
  ThreadScope      threads(state.range(1));
  const nn::Tensor x = random_tensor(c.x, 1), f = random_tensor(c.filter, 2);
  const auto       o = same_padding();
  for (auto _ : state) benchmark::DoNotOptimize(nn::conv2d(x, f, o));
  conv_counters(state, c);
}
BENCHMARK(BM_Conv2dForward)->ArgsProduct({benchmark::CreateDenseRange(0, kConvCount - 1, 1), kThreads})->Unit(benchmark::kMicrosecond);

void BM_Conv2dBackpropInput(benchmark::State& state) {
  const ConvCase&  c = kConvCases[state.range(0)];
  ThreadScope      threads(state.range(1));
  const nn::Tensor f = random_tensor(c.filter, 2);
  const auto       o = same_padding();
  const nn::Tensor g = random_tensor(nn::conv2d(random_tensor(c.x, 1), f, o).shape(), 3);
  for (auto _ : state) benchmark::DoNotOptimize(nn::conv2d_backprop_input(c.x, f, g, o));
  conv_counters(state, c);
}
BENCHMARK(BM_Conv2dBackpropInput)->ArgsProduct({benchmark::CreateDenseRange(0, kConvCount - 1, 1), kThreads})->Unit(benchmark::kMicrosecond);

void BM_Conv2dBackpropFilter(benchmark::State& state) {
  const ConvCase&  c = kConvCases[state.range(0)];
  ThreadScope      threads(state.range(1));
  const nn::Tensor x = random_tensor(c.x, 1), f = random_tensor(c.filter, 2);
  const auto       o = same_padding();
  const nn::Tensor g = random_tensor(nn::conv2d(x, f, o).shape(), 3);
  for (auto _ : state) benchmark::DoNotOptimize(nn::conv2d_backprop_filter(x, c.filter, g, o));
  conv_counters(state, c);
}
BENCHMARK(BM_Conv2dBackpropFilter)->ArgsProduct({benchmark::CreateDenseRange(0, kConvCount - 1, 1), kThreads})->Unit(benchmark::kMicrosecond);

// ---- elementwise: bias, ReLU, their backward, dropout ---------------------------------

const nn::Shape kFireMap{32, 16, 16, 128};  // a fire2/3 output map

void BM_BiasAddRelu(benchmark::State& state) {
  ThreadScope      threads(state.range(0));
  const nn::Tensor x = random_tensor(kFireMap, 1), b = random_tensor({kFireMap[3]}, 2);
  for (auto _ : state) benchmark::DoNotOptimize(nn::relu(nn::bias_add(x, b)));
}
BENCHMARK(BM_BiasAddRelu)->Arg(1)->Arg(16)->Unit(benchmark::kMicrosecond);

void BM_BiasAddReluBackward(benchmark::State& state) {
  ThreadScope      threads(state.range(0));
  nn::Tensor       x = random_tensor(kFireMap, 1), b = random_tensor({kFireMap[3]}, 2);
  x.requires_grad_(true);
  b.requires_grad_(true);
  const nn::Tensor seed = nn::ones(kFireMap);
  for (auto _ : state) {
    const nn::Tensor y = nn::relu(nn::bias_add(x, b));
    benchmark::DoNotOptimize(nn::grad({y}, {x, b}, {seed}));
  }
}
BENCHMARK(BM_BiasAddReluBackward)->Arg(1)->Arg(16)->Unit(benchmark::kMicrosecond);

/// The fused convolution + bias + ReLU the conv layers use, at the fire4
/// expand3x3 geometry: forward, and forward + the single-pass backward.
void BM_ConvBiasReluFused(benchmark::State& state) {
  ThreadScope      threads(state.range(0));
  const ConvCase&  c = kConvCases[4];
  const nn::Tensor x = random_tensor(c.x, 1), w = random_tensor(c.filter, 2), b = random_tensor({c.filter[3]}, 3);
  for (auto _ : state) benchmark::DoNotOptimize(nn::conv2d_bias_relu(x, w, b, same_padding()));
}
BENCHMARK(BM_ConvBiasReluFused)->Arg(1)->Arg(16)->Unit(benchmark::kMicrosecond);

void BM_ConvBiasReluFusedBackward(benchmark::State& state) {
  ThreadScope     threads(state.range(0));
  const ConvCase& c = kConvCases[4];
  nn::Tensor      x = random_tensor(c.x, 1), w = random_tensor(c.filter, 2), b = random_tensor({c.filter[3]}, 3);
  x.requires_grad_(true);
  w.requires_grad_(true);
  b.requires_grad_(true);
  const nn::Tensor seed = nn::ones({c.x[0], c.x[1], c.x[2], c.filter[3]});
  for (auto _ : state) {
    const nn::Tensor y = nn::conv2d_bias_relu(x, w, b, same_padding());
    benchmark::DoNotOptimize(nn::grad({y}, {x, w, b}, {seed}));
  }
}
BENCHMARK(BM_ConvBiasReluFusedBackward)->Arg(1)->Arg(16)->Unit(benchmark::kMicrosecond);

/// The unfused composition at the same geometry, for comparison.
void BM_ConvBiasReluComposedBackward(benchmark::State& state) {
  ThreadScope     threads(state.range(0));
  const ConvCase& c = kConvCases[4];
  nn::Tensor      x = random_tensor(c.x, 1), w = random_tensor(c.filter, 2), b = random_tensor({c.filter[3]}, 3);
  x.requires_grad_(true);
  w.requires_grad_(true);
  b.requires_grad_(true);
  const nn::Tensor seed = nn::ones({c.x[0], c.x[1], c.x[2], c.filter[3]});
  for (auto _ : state) {
    const nn::Tensor y = nn::relu(nn::bias_add(nn::conv2d(x, w, same_padding()), b));
    benchmark::DoNotOptimize(nn::grad({y}, {x, w, b}, {seed}));
  }
}
BENCHMARK(BM_ConvBiasReluComposedBackward)->Arg(1)->Arg(16)->Unit(benchmark::kMicrosecond);

/// The fused op the conv layers use: one pass forward, one pass backward.
void BM_BiasAddReluFused(benchmark::State& state) {
  ThreadScope      threads(state.range(0));
  const nn::Tensor x = random_tensor(kFireMap, 1), b = random_tensor({kFireMap[3]}, 2);
  for (auto _ : state) benchmark::DoNotOptimize(nn::bias_add_relu(x, b));
}
BENCHMARK(BM_BiasAddReluFused)->Arg(1)->Arg(16)->Unit(benchmark::kMicrosecond);

void BM_BiasAddReluFusedBackward(benchmark::State& state) {
  ThreadScope      threads(state.range(0));
  nn::Tensor       x = random_tensor(kFireMap, 1), b = random_tensor({kFireMap[3]}, 2);
  x.requires_grad_(true);
  b.requires_grad_(true);
  const nn::Tensor seed = nn::ones(kFireMap);
  for (auto _ : state) {
    const nn::Tensor y = nn::bias_add_relu(x, b);
    benchmark::DoNotOptimize(nn::grad({y}, {x, b}, {seed}));
  }
}
BENCHMARK(BM_BiasAddReluFusedBackward)->Arg(1)->Arg(16)->Unit(benchmark::kMicrosecond);

/// ReLU backward with a contiguous seed: the squeeze activations.
void BM_ReluBackwardContiguous(benchmark::State& state) {
  ThreadScope threads(state.range(0));
  nn::Tensor  x = random_tensor(kFireMap, 1);
  x.requires_grad_(true);
  const nn::Tensor seed = nn::ones(kFireMap);
  for (auto _ : state) {
    const nn::Tensor y = nn::relu(x);
    benchmark::DoNotOptimize(nn::grad({y}, {x}, {seed}));
  }
}
BENCHMARK(BM_ReluBackwardContiguous)->Arg(1)->Arg(16)->Unit(benchmark::kMicrosecond);

/// ReLU backward with a strided seed: the expand activations receive a split
/// view of the concatenated fire-module gradient.
void BM_ReluBackwardStrided(benchmark::State& state) {
  ThreadScope threads(state.range(0));
  nn::Tensor  a = random_tensor({32, 16, 16, 64}, 1), b = random_tensor({32, 16, 16, 64}, 2);
  a.requires_grad_(true);
  b.requires_grad_(true);
  const nn::Tensor seed = nn::ones(kFireMap);
  for (auto _ : state) {
    const nn::Tensor y = nn::concat({nn::relu(a), nn::relu(b)}, 3);
    benchmark::DoNotOptimize(nn::grad({y}, {a, b}, {seed}));
  }
}
BENCHMARK(BM_ReluBackwardStrided)->Arg(1)->Arg(16)->Unit(benchmark::kMicrosecond);

void BM_Dropout(benchmark::State& state) {
  ThreadScope      threads(state.range(0));
  const nn::Tensor x = random_tensor({32, 4, 4, 512}, 1);
  nn::Generator    gen(101);
  for (auto _ : state) benchmark::DoNotOptimize(nn::dropout(x, 0.5, true, gen));
}
BENCHMARK(BM_Dropout)->Arg(1)->Arg(16)->Unit(benchmark::kMicrosecond);

// ---- pooling and concatenation ----------------------------------------------------------

void BM_MaxPoolForward(benchmark::State& state) {
  ThreadScope      threads(state.range(0));
  const nn::Tensor x = random_tensor({32, 33, 33, 96}, 1);  // conv1 map after the ceil pad
  for (auto _ : state) benchmark::DoNotOptimize(nn::max_pool2d(x, 3, 2, nn::Padding::Valid));
}
BENCHMARK(BM_MaxPoolForward)->Arg(1)->Arg(16)->Unit(benchmark::kMicrosecond);

void BM_MaxPoolBackward(benchmark::State& state) {
  ThreadScope threads(state.range(0));
  nn::Tensor  x = random_tensor({32, 33, 33, 96}, 1);
  x.requires_grad_(true);
  const nn::Tensor seed = nn::ones({32, 16, 16, 96});
  for (auto _ : state) {
    const nn::Tensor y = nn::max_pool2d(x, 3, 2, nn::Padding::Valid);
    benchmark::DoNotOptimize(nn::grad({y}, {x}, {seed}));
  }
}
BENCHMARK(BM_MaxPoolBackward)->Arg(1)->Arg(16)->Unit(benchmark::kMicrosecond);

void BM_ConcatForward(benchmark::State& state) {
  ThreadScope      threads(state.range(0));
  const nn::Tensor a = random_tensor({32, 16, 16, 64}, 1), b = random_tensor({32, 16, 16, 64}, 2);
  for (auto _ : state) benchmark::DoNotOptimize(nn::concat({a, b}, 3));
}
BENCHMARK(BM_ConcatForward)->Arg(1)->Arg(16)->Unit(benchmark::kMicrosecond);

// ---- the rest of the step: objective, optimizer, augmentation, pool overhead ------------

struct Model {
  nn::models::SqueezeNetConfig cfg;
  nn::ParameterStore           store{17};
  std::vector<nn::Tensor>      params;
  Model() {
    cfg.version    = 10;
    cfg.cifar_stem = true;
    cfg.image_size = 32;
    cfg.classes    = 10;
    nn::NoGradGuard guard;
    nn::models::SqueezeNet(store, nn::zeros({1, 32, 32, 3}), cfg, false);
    params = store.parameters();
  }
};

void BM_L2Penalty(benchmark::State& state) {
  ThreadScope threads(state.range(0));
  Model       m;
  for (auto _ : state) {
    nn::Tensor objective = nn::zeros({});
    for (const auto& w : m.params)
      if (w.dim() >= 2) objective = objective + .00005 * nn::sum(nn::square(w));
    nn::GradOptions options;
    options.allow_unused = true;
    benchmark::DoNotOptimize(nn::grad({objective}, m.params, {}, options));
  }
}
BENCHMARK(BM_L2Penalty)->Arg(1)->Arg(16)->Unit(benchmark::kMicrosecond);

void BM_AdamStep(benchmark::State& state) {
  ThreadScope         threads(state.range(0));
  Model               m;
  nn::OptimizerConfig config;
  config.kind = nn::OptimizerKind::Adam;
  nn::Optimizer           optimizer(m.store, .0005, config);
  std::vector<nn::Tensor> grads;
  for (std::size_t i = 0; i < m.params.size(); ++i) grads.push_back(random_tensor(m.params[i].shape(), 100 + i));
  for (auto _ : state) optimizer.step(grads);
}
BENCHMARK(BM_AdamStep)->Arg(1)->Arg(16)->Unit(benchmark::kMicrosecond);

/// The report's per-image pad / crop / flip pipeline for one batch.
void BM_Augment(benchmark::State& state) {
  ThreadScope      threads(state.range(0));
  const nn::Tensor x = random_tensor({32, 32, 32, 3}, 1);
  nn::Generator    gen(151);
  for (auto _ : state) {
    std::vector<nn::Tensor> transformed;
    for (I64 row = 0; row < x.size(0); ++row) {
      auto image = nn::pad(nn::slice_axis(x, 0, row, row + 1), {{0, 0}, {4, 4}, {4, 4}, {0, 0}});
      image      = nn::augment::random_crop(image, 32, 32, gen);
      transformed.push_back(nn::augment::random_flip_left_right(image, gen));
    }
    benchmark::DoNotOptimize(2 * nn::concat(transformed, 0) - 1);
  }
}
BENCHMARK(BM_Augment)->Arg(1)->Arg(16)->Unit(benchmark::kMicrosecond);

/// Dispatch cost of one parallel_for with 64 trivial chunks (wake-up latency of the pool).
void BM_ParallelForDispatch(benchmark::State& state) {
  ThreadScope threads(state.range(0));
  std::vector<float> v(64 * 1024, 1.0f);
  for (auto _ : state)
    nn::parallel_for(0, 64, 1, [&](I64 lo, I64 hi) {
      for (I64 c = lo; c < hi; ++c) v[c * 1024] += 1.0f;
    });
  benchmark::DoNotOptimize(v);
}
BENCHMARK(BM_ParallelForDispatch)->Arg(1)->Arg(16)->Unit(benchmark::kMicrosecond);

// ---- whole model --------------------------------------------------------------------------

void BM_ModelForward(benchmark::State& state) {
  ThreadScope      threads(state.range(0));
  Model            m;
  nn::Generator    dropout_rng(101);
  const nn::Tensor x = random_tensor({32, 32, 32, 3}, 1);
  for (auto _ : state) benchmark::DoNotOptimize(nn::models::SqueezeNet(m.store, x, m.cfg, true, &dropout_rng).logits);
}
BENCHMARK(BM_ModelForward)->Arg(1)->Arg(16)->Unit(benchmark::kMillisecond);

void BM_ModelForwardBackward(benchmark::State& state) {
  ThreadScope      threads(state.range(0));
  Model            m;
  nn::Generator    dropout_rng(101);
  const nn::Tensor x = random_tensor({32, 32, 32, 3}, 1);
  const nn::Tensor y = nn::full({32}, 3, nn::DType::Int64);
  for (auto _ : state) {
    const auto logits = nn::models::SqueezeNet(m.store, x, m.cfg, true, &dropout_rng).logits;
    nn::GradOptions options;
    options.allow_unused = true;
    benchmark::DoNotOptimize(nn::grad({nn::sparse_softmax_cross_entropy(logits, y)}, m.params, {}, options));
  }
}
BENCHMARK(BM_ModelForwardBackward)->Arg(1)->Arg(16)->Unit(benchmark::kMillisecond);

}  // namespace
