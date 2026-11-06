// CUDA GEMM comparison: compute::Context::matmul with the library's own kernels
// (shaders/compute/blas.cu) against the same call routed to cuBLAS
// (DATAMUNGE_CUDA_GEMM=cublas, CUBLAS_PEDANTIC_MATH so no TF32), float and double,
// at square sizes and the SqueezeNet CIFAR-10 convolution shapes (NN, plus the NT
// input-gradient and TN filter-gradient products).
//
//   cuda_gemm_compare [--json out.json] [--quick]
//
// Both paths go through the same Context call, which synchronises after every
// product, so per-call overhead is identical; a 1x1x1 product measures it. The
// library must be built with DATAMUNGE_ENABLE_CUDA and DATAMUNGE_ENABLE_CUBLAS
// (`just bench-blas-cuda`). Times are medians of at least five calls after a warm-up.

#include <datamunge/compute/context.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace c = datamunge::compute;

namespace {

struct Case {
  std::string label;
  bool        ta{false}, tb{false};
  std::size_t m{0}, n{0}, k{0};
};

std::vector<Case> cases(bool quick) {
  std::vector<Case> out;
  for (std::size_t s : quick ? std::vector<std::size_t>{1024} : std::vector<std::size_t>{256, 512, 1024, 2048, 4096})
    out.push_back({"square " + std::to_string(s), false, false, s, s, s});
  out.push_back({"conv1 fwd 32768x64x27", false, false, 32768, 64, 27});
  out.push_back({"squeeze fwd 8192x16x128", false, false, 8192, 16, 128});
  out.push_back({"expand1x1 fwd 8192x64x16", false, false, 8192, 64, 16});
  out.push_back({"expand3x3 fwd 8192x64x144", false, false, 8192, 64, 144});
  out.push_back({"fire8 fwd 2048x256x576", false, false, 2048, 256, 576});
  out.push_back({"fire fwd 2048x32x512", false, false, 2048, 32, 512});
  out.push_back({"expand3x3 dx 8192x144x64 NT", false, true, 8192, 144, 64});
  out.push_back({"expand3x3 dw 144x64x8192 TN", true, false, 144, 64, 8192});
  return out;
}

template <typename Fn>
double median_seconds(Fn fn) {
  fn();
  std::vector<double> t;
  const auto          start = std::chrono::steady_clock::now();
  for (int r = 0; r < 2000; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (r >= 4 && std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > 0.3) break;
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

/// One backend: a context plus device buffers sized for the largest case.
struct Gpu {
  std::string                 name;
  std::unique_ptr<c::Context> context;
};

}  // namespace

int main(int argc, char** argv) {
  std::string json;
  bool        quick = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--json" && i + 1 < argc) json = argv[++i];
    else if (arg == "--quick") quick = true;
    else {
      std::fprintf(stderr, "usage: cuda_gemm_compare [--json out.json] [--quick]\n");
      return 2;
    }
  }
  const auto info = c::probe({c::Backend::CUDA, 0});
  if (!info.available) {
    std::fprintf(stderr, "no CUDA device: %s\n", info.diagnostic.c_str());
    return 1;
  }
  std::vector<Gpu> gpus;
  unsetenv("DATAMUNGE_CUDA_GEMM");
  gpus.push_back({"inhouse-cuda", std::make_unique<c::Context>(c::Device{c::Backend::CUDA, 0})});
  setenv("DATAMUNGE_CUDA_GEMM", "cublas", 1);
  gpus.push_back({"cublas", std::make_unique<c::Context>(c::Device{c::Backend::CUDA, 0})});
  unsetenv("DATAMUNGE_CUDA_GEMM");
  std::printf("# device: %s\n", info.name.c_str());
  std::printf("%-30s %-4s %-14s %12s %10s %10s\n", "case", "type", "backend", "us/call", "GFLOP/s", "max_err");

  std::ofstream out;
  if (!json.empty()) {
    out.open(json);
    out << "[\n";
  }
  bool         first = true;
  std::mt19937 rng(5);
  auto         all = cases(quick);
  all.insert(all.begin(), Case{"overhead 1x1x1", false, false, 1, 1, 1});
  for (const auto& cs : all)
    for (int precision = 0; precision < 2; ++precision) {
      const std::size_t width = precision == 0 ? 4 : 8;
      if (precision == 1 && cs.m * cs.n * cs.k > std::size_t(1) << 33) continue;  // f64 4096^3 is slow on consumer GPUs
      std::uniform_real_distribution<double> u(-1, 1);
      std::vector<double>                    a(cs.m * cs.k), b(cs.k * cs.n);
      for (auto& v : a) v = u(rng);
      for (auto& v : b) v = u(rng);
      std::vector<float> af(a.begin(), a.end()), bf(b.begin(), b.end());
      // Reference for a few spot entries (double sums of the rounded inputs).
      auto reference = [&](std::size_t i, std::size_t j) {
        double s = 0;
        for (std::size_t l = 0; l < cs.k; ++l) {
          const std::size_t ai = cs.ta ? l * cs.m + i : i * cs.k + l, bi = cs.tb ? j * cs.k + l : l * cs.n + j;
          s += precision == 0 ? double(af[ai]) * double(bf[bi]) : a[ai] * b[bi];
        }
        return s;
      };
      const c::MatrixProduct product{cs.m, cs.n, cs.k, precision == 0 ? c::MatrixType::Float32 : c::MatrixType::Float64,
                                     cs.ta, cs.tb};
      for (auto& g : gpus) {
        auto& ctx = *g.context;
        auto  da = ctx.allocate(a.size() * width), db = ctx.allocate(b.size() * width),
             dc = ctx.allocate(cs.m * cs.n * width);
        if (precision == 0) {
          ctx.upload(da, af.data(), af.size() * 4);
          ctx.upload(db, bf.data(), bf.size() * 4);
        } else {
          ctx.upload(da, a.data(), a.size() * 8);
          ctx.upload(db, b.data(), b.size() * 8);
        }
        const double seconds = median_seconds([&] { ctx.matmul(dc, da, db, product); });
        std::vector<double> result(cs.m * cs.n);
        if (precision == 0) {
          std::vector<float> rf(cs.m * cs.n);
          ctx.download(rf.data(), dc, rf.size() * 4);
          result.assign(rf.begin(), rf.end());
        } else {
          ctx.download(result.data(), dc, result.size() * 8);
        }
        double err = 0;
        for (int s = 0; s < 8; ++s) {
          const std::size_t i = (cs.m * std::size_t(s * 37 + 1) / 9) % cs.m, j = (cs.n * std::size_t(s * 11 + 3) / 7) % cs.n;
          const double      r = reference(i, j);
          err                 = std::max(err, std::fabs(r - result[i * cs.n + j]) / (1 + std::fabs(r)));
        }
        const double gflops = 2.0 * double(cs.m) * double(cs.n) * double(cs.k) / seconds / 1e9;
        std::printf("%-30s %-4s %-14s %12.1f %10.1f %10.1e\n", cs.label.c_str(), precision == 0 ? "f32" : "f64",
                    g.name.c_str(), seconds * 1e6, gflops, err);
        std::fflush(stdout);
        if (out) {
          out << (first ? "" : ",\n") << "  {\"op\": \"gemm\", \"case\": \"" << cs.label << "\", \"m\": " << cs.m
              << ", \"n\": " << cs.n << ", \"k\": " << cs.k << ", \"ta\": " << (cs.ta ? "true" : "false")
              << ", \"tb\": " << (cs.tb ? "true" : "false") << ", \"dtype\": \"" << (precision == 0 ? "f32" : "f64")
              << "\", \"threads\": 0, \"backend\": \"" << g.name << "\", \"seconds\": " << seconds
              << ", \"gflops\": " << gflops << ", \"gbps\": 0, \"max_err\": " << err << "}";
          first = false;
        }
      }
    }
  if (out) out << "\n]\n";
  return 0;
}
