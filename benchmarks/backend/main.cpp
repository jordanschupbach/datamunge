#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include <cstdlib>
#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#include "common.hpp"
#ifndef DATAMUNGE_BENCH_BUILD
#define DATAMUNGE_BENCH_BUILD "unknown"
#endif
int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  benchmark::AddCustomContext("build", DATAMUNGE_BENCH_BUILD);
#if defined(__unix__) || defined(__APPLE__)
  Dl_info core{};
  if (dladdr(reinterpret_cast<const void*>(&nb::nn::gemm_backend), &core) && core.dli_fname)
    benchmark::AddCustomContext("core_library_path", core.dli_fname);
#endif
  benchmark::AddCustomContext("gemm_backend", nb::nn::gemm_backend());
  benchmark::AddCustomContext("lapack_available", nb::nn::linalg_has_lapack() ? "true" : "false");
  benchmark::AddCustomContext("seed", "42");
  benchmark::AddCustomContext("nn_threads", "1 except explicit thread-scaling cases");
  for (const char* key : {"OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"}) {
    const auto* value = std::getenv(key);
    benchmark::AddCustomContext(key, value ? value : "unset");
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}
