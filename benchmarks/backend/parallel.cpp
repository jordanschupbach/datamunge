#include "common.hpp"
namespace {
namespace nn = nb::nn;

void BM_TensorParallelScaling(benchmark::State& s, int op) {
  nb::Scope scope;
  nn::set_num_threads(static_cast<int>(s.range(1)));
  nn::NoGradGuard guard;
  auto x = nn::full({s.range(0) / 32, 32}, 0.5);
  auto run = [&] { return op == 0 ? nn::relu(x) : nn::sum(x, {-1}); };
  nb::close(run(), op == 0 ? x : nn::full({s.range(0) / 32}, 16), 1e-6);
  for (auto _ : s) nb::consume(run());
  s.SetItemsProcessed(s.iterations() * s.range(0));
}
#define SCALE_KERNEL(name, op) BENCHMARK_CAPTURE(BM_TensorParallelScaling, name, op) \
    ->ArgsProduct({{1024, 32768, 65536, 1048576}, {1, 2, 4}}) \
    ->ArgNames({"elements", "threads"})->UseRealTime()
SCALE_KERNEL(Relu, 0);
SCALE_KERNEL(Reduce, 1);

void BM_ParallelSubmit(benchmark::State& s) {
  nb::Scope scope;
  nn::set_num_threads(static_cast<int>(s.range(1)));
  std::vector<std::int64_t> output(static_cast<std::size_t>(s.range(0)));
  auto fill = [&](std::int64_t lo, std::int64_t hi) {
    for (auto i = lo; i < hi; ++i) output[static_cast<std::size_t>(i)] = i;
  };
  nn::parallel_for(0, s.range(0), 1, fill);
  for (std::int64_t i = 0; i < s.range(0); ++i)
    nb::check(output[static_cast<std::size_t>(i)] == i, "parallel submission reference failed");
  for (auto _ : s) {
    nn::parallel_for(0, s.range(0), 1, fill);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
  s.SetItemsProcessed(s.iterations() * s.range(0));
}
BENCHMARK(BM_ParallelSubmit)->ArgsProduct({{64, 4096}, {1, 2, 4}})
    ->ArgNames({"elements", "threads"})->UseRealTime();

} // namespace
