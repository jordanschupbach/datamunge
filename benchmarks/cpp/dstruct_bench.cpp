#include <benchmark/benchmark.h>

#include <datamunge/dstruct/dstruct.hpp>

#include <cstddef>
#include <vector>

using datamunge::dstruct::ThreadSafeVector;

static void BM_ThreadSafeVectorPushBack(benchmark::State& state) {
  for (auto _ : state) {
    ThreadSafeVector<int> values;
    values.reserve(static_cast<std::size_t>(state.range(0)) * state.threads());
    for (int i = 0; i < state.range(0); ++i) {
      values.push_back(i);
    }
    benchmark::DoNotOptimize(values.size());
  }
  state.SetItemsProcessed(state.iterations() * state.range(0) * state.threads());
}
BENCHMARK(BM_ThreadSafeVectorPushBack)
    ->RangeMultiplier(4)
    ->Range(256, 16384)
    ->ThreadRange(1, 8)
    ->Unit(benchmark::kMicrosecond);

static void BM_ThreadSafeVectorSnapshot(benchmark::State& state) {
  ThreadSafeVector<int> values;
  std::vector<int> seed(static_cast<std::size_t>(state.range(0)), 42);
  values.append(seed);

  for (auto _ : state) {
    auto snapshot = values.snapshot();
    benchmark::DoNotOptimize(snapshot);
  }
  state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_ThreadSafeVectorSnapshot)
    ->RangeMultiplier(4)
    ->Range(256, 16384)
    ->Unit(benchmark::kMicrosecond);

static void BM_ThreadSafeVectorApply(benchmark::State& state) {
  ThreadSafeVector<int> values;
  std::vector<int> seed(static_cast<std::size_t>(state.range(0)), 7);
  values.append(seed);
  const auto ncores = static_cast<std::size_t>(state.threads());

  for (auto _ : state) {
    auto transformed = datamunge::dstruct::apply(values, [](const int value) { return value * 3; }, ncores, 256);
    benchmark::DoNotOptimize(transformed);
  }
  state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_ThreadSafeVectorApply)
    ->RangeMultiplier(4)
    ->Range(256, 16384)
    ->ThreadRange(1, 8)
    ->Unit(benchmark::kMicrosecond);
