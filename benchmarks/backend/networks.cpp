#include <datamunge/nn/attention.hpp>
#include <datamunge/nn/memory.hpp>

#include "common.hpp"
namespace {
namespace nn = nb::nn;
// Model: 0 MLP, 1 compact CNN, 2 unrolled RNN, 3 compact transformer block.
void BM_Network(benchmark::State& s, int model, int mode) {
  nb::Scope          scope;
  nn::ParameterStore store(42);
  auto               batch = s.range(0);
  auto               x     = nb::random(
      model == 1   ? nn::Shape{batch, 8, 8, 1}
      : model >= 3 ? nn::Shape{batch, 16, 32}
                   : nn::Shape{batch, 32},
      nn::DType::Float32);
  auto forward = [&]() {
    if (model == 0)
      return nn::dense(store, "out", nn::dense(store, "hidden", x, 64, nn::Activation::Relu), 10);
    if (model == 1) {
      auto h = nn::max_pool2d(nn::conv2d(store, "conv", x, 8, 3), 2, 2);
      return nn::dense(store, "out", nn::reshape(h, {batch, -1}), 10);
    }
    if (model == 2) {
      auto h = nn::zeros({batch, 32});
      for (int i = 0; i < 16; ++i)
        h = nn::rnn_step(store, "rnn", x, h);
      return nn::dense(store, "out", h, 10);
    }
    if (model >= 4) {
      const auto kind = model == 4 ? nn::RnnKind::Simple : model == 5 ? nn::RnnKind::Lstm : nn::RnnKind::Gru;
      auto sequence = nn::recurrent(store, "sequence", x, 32, kind);
      return nn::dense(store, "out", nn::select(sequence, 1, 15), 10);
    }
    auto q = nn::dense(store, "q", x, 32), k = nn::dense(store, "k", x, 32),
         v = nn::dense(store, "v", x, 32);
    auto h = nn::layer_norm(
        store, "norm1", nn::add(x, nn::scaled_dot_product_attention(q, k, v).output));
    return nn::layer_norm(
        store,
        "norm2",
        nn::add(
            h, nn::dense(store, "ff2", nn::dense(store, "ff1", h, 64, nn::Activation::Relu), 32)));
  };
  {
    nn::NoGradGuard guard;
    nb::finite(forward());
  }
  auto                    params = store.parameters();
  std::vector<nn::Tensor> original;
  for (auto& p : params)
    original.push_back(p.detach().clone());
  nn::OptimizerConfig cfg;
  cfg.kind = nn::OptimizerKind::Adam;
  nn::Optimizer           optimizer(params, 1e-3, cfg);
  auto                    loss = [&]() { return nn::mean(nn::square(forward())); };
  nn::Tensor              fixed_logits, fixed_loss;
  std::vector<nn::Tensor> fixed_gradients;
  if (mode == 3)
    fixed_logits = forward().detach();
  if (mode == 4)
    fixed_loss = loss();
  if (mode == 5)
    fixed_gradients = nn::grad(loss(), params);
  auto step = [&]() {
    if (mode == 3) {
      nn::NoGradGuard guard;
      nb::consume(nn::mean(nn::square(fixed_logits)));
      return;
    }
    if (mode == 4) {
      benchmark::DoNotOptimize(nn::grad(fixed_loss, params, nn::GradOptions{true, false, false}));
      return;
    }
    if (mode == 5) {
      optimizer.step(fixed_gradients);
      return;
    }
    if (mode == 0) {
      nn::NoGradGuard guard;
      nb::consume(forward());
      return;
    }
    auto y = forward();
    auto l = nn::mean(nn::square(y));
    auto g = nn::grad(l, params);
    if (mode == 2 || mode == 5)
      optimizer.step(g);
    benchmark::DoNotOptimize(g);
  };
  {
    auto g = nn::grad(loss(), params);
    for (auto& t : g)
      nb::finite(t);
    if (mode == 2 || mode == 5)
      optimizer.step(g);
  }
  if (mode == 2 || mode == 5) {
    nn::NoGradGuard guard;
    for (std::size_t i = 0; i < params.size(); ++i)
      params[i].copy_(original[i]);
  }
  // Measure tensor storage separately, before timed updates; no global resets inside timing.
  std::int64_t peak = 0;
  {
    nn::PeakMemoryScope memory;
    step();
    peak = memory.peak_above_baseline();
  }
  // Snapshot warmed state by value, since optimizer slots mutate in place.
  std::vector<std::pair<std::string, nn::Tensor>> saved;
  for (auto& v : optimizer.state())
    saved.emplace_back(v.first, v.second.clone());
  for (auto _ : s) {
    if (mode == 2 || mode == 5) {
      s.PauseTiming();
      {
        nn::NoGradGuard guard;
        for (std::size_t i = 0; i < params.size(); ++i)
          params[i].copy_(original[i]);
        optimizer.load_state(saved);
      }
      s.ResumeTiming();
      step();
    } else if (mode == 0) {
      nn::NoGradGuard guard;
      step();
    } else
      step();
  }
  s.counters["peak_tensor_bytes"] = double(peak);
  s.SetItemsProcessed(s.iterations() * batch * (model == 3 ? 16 : 1));
}
#define NET(name, model)                                                                           \
  BENCHMARK_CAPTURE(BM_Network, name##_Inference, model, 0)->Arg(1)->Arg(32)->UseRealTime();       \
  BENCHMARK_CAPTURE(BM_Network, name##_ForwardBackward, model, 1)->Arg(1)->Arg(32)->UseRealTime(); \
  BENCHMARK_CAPTURE(BM_Network, name##_TrainingStep, model, 2)->Arg(1)->Arg(32)->UseRealTime()
NET(MLP, 0);
NET(CNN, 1);
NET(RNN, 2);
NET(Transformer, 3);
NET(RNNSequence, 4);
NET(LSTM, 5);
NET(GRU, 6);
#define COMPONENTS(name, model)                                                             \
  BENCHMARK_CAPTURE(BM_Network, name##_LossOnly, model, 3)->Arg(1)->Arg(32)->UseRealTime(); \
  BENCHMARK_CAPTURE(BM_Network, name##_RetainedBackward, model, 4)                          \
      ->Arg(1)                                                                              \
      ->Arg(32)                                                                             \
      ->UseRealTime();                                                                      \
  BENCHMARK_CAPTURE(BM_Network, name##_OptimizerOnly, model, 5)->Arg(1)->Arg(32)->UseRealTime()
COMPONENTS(MLP, 0);
COMPONENTS(CNN, 1);
COMPONENTS(RNN, 2);
COMPONENTS(Transformer, 3);
COMPONENTS(RNNSequence, 4);
COMPONENTS(LSTM, 5);
COMPONENTS(GRU, 6);

void BM_TensorOptimizer(benchmark::State& s, int phase, nn::DType dtype = nn::DType::Float32) {
  nb::Scope               scope;
  nn::NoGradGuard         guard;
  auto                    n = s.range(0), count = s.range(1);
  std::vector<nn::Tensor> p, g;
  for (int i = 0; i < count; ++i) {
    p.push_back(nn::ones(s.range(3) ? nn::Shape{n / count} : nn::Shape{n / count / 32, 32}, dtype));
    p.back().requires_grad_(true);
    g.push_back(nn::full(p.back().shape(), 0.01, dtype));
  }
  nn::OptimizerConfig config;
  config.kind = static_cast<nn::OptimizerKind>(s.range(2));
  nn::Optimizer optimizer(p, 1e-3, config);
  optimizer.step(g);
  for (auto& t : p)
    nb::finite(t);
  std::vector<std::pair<std::string, nn::Tensor>> saved;
  for (auto& v : optimizer.state())
    saved.emplace_back(v.first, v.second.clone());
  std::int64_t state_bytes = 0;
  for (auto& v : optimizer.state())
    if (v.first != "step" && v.first != "nadam_product")
      state_bytes += v.second.numel() * nn::dtype_size(v.second.dtype());
  std::int64_t peak;
  {
    nn::PeakMemoryScope mem;
    if (phase == 0) {
      nn::Optimizer fresh(p, 1e-3, config);
      fresh.step(g);
    } else
      optimizer.step(g);
    peak = mem.peak_above_baseline();
  }
  for (auto _ : s) {
    s.PauseTiming();
    for (auto& t : p)
      t.fill_(1);
    optimizer.load_state(saved);
    s.ResumeTiming();
    if (phase == 0) {
      nn::Optimizer fresh(p, 1e-3, config);
      fresh.step(g);
    } else
      optimizer.step(g);
    benchmark::ClobberMemory();
  }
  s.SetLabel(nn::optimizer_name(config.kind));
  s.SetItemsProcessed(s.iterations() * n);
  s.counters["update_peak_tensor_bytes"]      = double(peak);
  s.counters["persistent_state_tensor_bytes"] = double(state_bytes);
}
void optimizer_args(benchmark::internal::Benchmark* b) {
  for (int kind = 0; kind < 15; ++kind)
    for (int n : {4096, 65536})
      for (int count : {1, 16})
        b->Args({n, count, kind, 0});
  for (int kind = 0; kind < 15; ++kind)
    b->Args({4096, 1, kind, 1});
  b->ArgNames({"elements", "tensors", "kind", "vector"})->UseRealTime();
}
BENCHMARK_CAPTURE(BM_TensorOptimizer, Warmed, 1)->Apply(optimizer_args);
BENCHMARK_CAPTURE(BM_TensorOptimizer, FirstStep, 0)->Apply(optimizer_args);
void optimizer_double_args(benchmark::internal::Benchmark* b) {
  for (int kind = 0; kind < 15; ++kind) {
    b->Args({4096, 1, kind, 0});
    b->Args({65536, 1, kind, 0});
    b->Args({4096, 1, kind, 1});
  }
  b->ArgNames({"elements", "tensors", "kind", "vector"})->UseRealTime();
}
BENCHMARK_CAPTURE(BM_TensorOptimizer, DoubleWarmed, 1, nn::DType::Float64)
    ->Apply(optimizer_double_args);
BENCHMARK_CAPTURE(BM_TensorOptimizer, DoubleFirstStep, 0, nn::DType::Float64)
    ->Apply(optimizer_double_args);

void BM_GradientUtilities(benchmark::State& s, int op) {
  nb::Scope               scope;
  nn::NoGradGuard         guard;
  std::vector<nn::Tensor> g{nn::ones({s.range(0)})};
  if (op == 0) {
    auto r = nn::clip_by_global_norm(g, 1.0);
    nb::check(std::abs(nn::global_norm(r.first) - 1.0) < 1e-5, "clipping reference failed");
  } else {
    nn::GradientAccumulator acc;
    acc.add(g);
    acc.add(g);
    nb::close(acc.mean()[0], g[0], 1e-5);
  }
  std::int64_t peak_bytes = 0;
  {
    nn::PeakMemoryScope memory;
    if (op == 0)
      benchmark::DoNotOptimize(nn::clip_by_global_norm(g, 1.0));
    else {
      nn::GradientAccumulator acc;
      for (int i = 0; i < 4; ++i)
        acc.add(g);
      benchmark::DoNotOptimize(acc.mean());
    }
    peak_bytes = memory.peak_above_baseline();
  }
  s.counters["peak_tensor_bytes"] = double(peak_bytes);
  for (auto _ : s) {
    if (op == 0)
      benchmark::DoNotOptimize(nn::clip_by_global_norm(g, 1.0));
    else {
      nn::GradientAccumulator acc;
      for (int i = 0; i < 4; ++i)
        acc.add(g);
      benchmark::DoNotOptimize(acc.mean());
    }
  }
}
BENCHMARK_CAPTURE(BM_GradientUtilities, ClipGlobalNorm, 0)->Arg(4096)->Arg(65536)->UseRealTime();
BENCHMARK_CAPTURE(BM_GradientUtilities, Accumulate4, 1)->Arg(4096)->Arg(65536)->UseRealTime();
void BM_AttentionSmall(benchmark::State& s) {
  nb::Scope scope;
  auto q = nb::random({2, 2, s.range(0), 16}, nn::DType::Float32), k = q.clone(), v = q.clone();
  q.requires_grad_(true);
  k.requires_grad_(true);
  v.requires_grad_(true);
  nn::AttentionOptions plain, fused;
  plain.impl   = nn::AttentionImpl::Plain;
  fused.impl   = nn::AttentionImpl::Fused;
  plain.causal = fused.causal = s.range(1) != 0;
  auto run                    = [&](const nn::AttentionOptions& opts) {
    return nn::scaled_dot_product_attention(q, k, v, opts).output;
  };
  nb::close(run(plain), run(fused), 1e-4);
  auto pg = nn::grad(nn::sum(run(plain)), {q, k, v}), fg = nn::grad(nn::sum(run(fused)), {q, k, v});
  for (int i = 0; i < 3; ++i)
    nb::close(pg[i], fg[i], 2e-4);
  const auto&  opts = s.range(2) ? fused : plain;
  std::int64_t peak = 0;
  {
    nn::PeakMemoryScope mem;
    if (s.range(3)) {
      auto out = run(opts);
      benchmark::DoNotOptimize(nn::grad(nn::sum(out), {q, k, v}));
    } else {
      nn::NoGradGuard guard;
      nb::consume(run(opts));
    }
    peak = mem.peak_above_baseline();
  }
  for (auto _ : s) {
    if (s.range(3))
      benchmark::DoNotOptimize(nn::grad(nn::sum(run(opts)), {q, k, v}));
    else {
      nn::NoGradGuard guard;
      nb::consume(run(opts));
    }
  }
  s.counters["peak_tensor_bytes"] = double(peak);
  s.SetItemsProcessed(s.iterations() * 4 * s.range(0));
}
BENCHMARK(BM_AttentionSmall)
    ->ArgsProduct({{16, 64, 128}, {0, 1}, {0, 1}, {0, 1}})
    ->ArgNames({"tokens", "causal", "fused", "backward"})
    ->UseRealTime();
void BM_NeuralPrimitive(benchmark::State& s, int op) {
  nb::Scope scope;
  auto      x = nb::random({s.range(0), 32}, nn::DType::Float32);
  x.requires_grad_(true);
  auto labels = nn::from_vector(std::vector<std::int64_t>(s.range(0), 0), {s.range(0)});
  auto run    = [&]() {
    if (op == 0)
      return nn::relu(x);
    if (op == 1)
      return nn::softmax(x);
    return nn::sparse_softmax_cross_entropy(x, labels);
  };
  nb::finite(run());
  if (op == 1)
    nb::close(nn::sum(run(), {-1}), nn::ones({s.range(0)}), 1e-5);
  auto g = nn::grad(nn::sum(run()), {x});
  nb::finite(g[0]);
  for (auto _ : s) {
    if (s.range(1))
      benchmark::DoNotOptimize(nn::grad(nn::sum(run()), {x}));
    else {
      nn::NoGradGuard guard;
      nb::consume(run());
    }
  }
}
#define NP(name, op)                              \
  BENCHMARK_CAPTURE(BM_NeuralPrimitive, name, op) \
      ->ArgsProduct({{32, 256}, {0, 1}})          \
      ->ArgNames({"batch", "backward"})           \
      ->UseRealTime()
NP(Relu, 0);
NP(Softmax, 1);
NP(CrossEntropy, 2);

} // namespace
