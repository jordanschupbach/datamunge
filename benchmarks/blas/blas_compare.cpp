// Linear-algebra comparison: the in-house BLAS (datamunge/nn/blas.hpp, every
// instruction set this CPU can run) against external CBLAS libraries loaded
// with dlopen (OpenBLAS, BLIS, ATLAS, the netlib reference BLAS, ...), for
// GEMM (square and SqueezeNet convolution shapes, NN/NT/TN), GEMV, DOT, AXPY
// and SCAL in float and double at several thread counts.
//
//   blas_compare [--libs "name=path[:ilp64][:fixed-threads][:slow];..."] [--threads 1,8,16] [--no-inhouse]
//                (one process per op, thread count and library when libraries are given)
//                [--ops gemm,gemv,dot,axpy,scal] [--quick] [--json out.json]
//
// `just bench-blas` supplies the libraries from the blas-bench dev shell
// (DATAMUNGE_BLAS_LIBS). A library's `:ilp64` suffix says its CBLAS takes
// 64-bit integers (nixpkgs OpenBLAS and BLIS do); `:fixed-threads` marks one
// whose thread count cannot be set (ATLAS), so it only runs once; `:slow` skips
// GEMMs above 4 GFLOP (the unblocked netlib reference would take minutes). Libraries
// without a CBLAS layer (reference BLAS) are called through their Fortran
// entry points with the row-major/column-major swap. Reported numbers are the
// median of at least five calls after one warm-up call; the summary tables are
// produced by benchmarks/blas/summarize.py.

#include <datamunge/nn/blas.hpp>
#include <datamunge/nn/parallel.hpp>
#include "placement.hpp"

#include <dlfcn.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using I64 = std::int64_t;
namespace blas = datamunge::nn::blas;

/// One BLAS implementation with a uniform row-major interface.
struct Backend {
  std::string name;
  bool        fixed_threads{false};
  bool        slow{false};
  std::function<void(int)> set_threads;
  std::function<void(bool, bool, I64, I64, I64, float, const float*, I64, const float*, I64, float, float*, I64)>  sgemm;
  std::function<void(bool, bool, I64, I64, I64, double, const double*, I64, const double*, I64, double, double*, I64)> dgemm;
  std::function<void(bool, I64, I64, float, const float*, I64, const float*, I64, float, float*, I64)>    sgemv;
  std::function<void(bool, I64, I64, double, const double*, I64, const double*, I64, double, double*, I64)> dgemv;
  std::function<float(I64, const float*, I64, const float*, I64)>    sdot;
  std::function<double(I64, const double*, I64, const double*, I64)> ddot;
  std::function<void(I64, float, const float*, I64, float*, I64)>    saxpy;
  std::function<void(I64, double, const double*, I64, double*, I64)> daxpy;
  std::function<void(I64, float, float*, I64)>   sscal;
  std::function<void(I64, double, double*, I64)> dscal;
};

Backend inhouse_backend(const std::string& isa) {
  Backend b;
  b.name        = "inhouse-" + isa;
  b.set_threads = [isa](int t) {
    datamunge::nn::set_num_threads(t);
    blas::set_isa(isa);
  };
  b.sgemm = [](bool ta, bool tb, I64 m, I64 n, I64 k, float al, const float* a, I64 lda, const float* bb, I64 ldb,
               float be, float* c, I64 ldc) { blas::gemm(ta, tb, m, n, k, al, a, lda, bb, ldb, be, c, ldc); };
  b.dgemm = [](bool ta, bool tb, I64 m, I64 n, I64 k, double al, const double* a, I64 lda, const double* bb, I64 ldb,
               double be, double* c, I64 ldc) { blas::gemm(ta, tb, m, n, k, al, a, lda, bb, ldb, be, c, ldc); };
  b.sgemv = [](bool t, I64 m, I64 n, float al, const float* a, I64 lda, const float* x, I64 ix, float be, float* y,
               I64 iy) { blas::gemv(t, m, n, al, a, lda, x, ix, be, y, iy); };
  b.dgemv = [](bool t, I64 m, I64 n, double al, const double* a, I64 lda, const double* x, I64 ix, double be,
               double* y, I64 iy) { blas::gemv(t, m, n, al, a, lda, x, ix, be, y, iy); };
  b.sdot  = [](I64 n, const float* x, I64 ix, const float* y, I64 iy) { return blas::dot(n, x, ix, y, iy); };
  b.ddot  = [](I64 n, const double* x, I64 ix, const double* y, I64 iy) { return blas::dot(n, x, ix, y, iy); };
  b.saxpy = [](I64 n, float al, const float* x, I64 ix, float* y, I64 iy) { blas::axpy(n, al, x, ix, y, iy); };
  b.daxpy = [](I64 n, double al, const double* x, I64 ix, double* y, I64 iy) { blas::axpy(n, al, x, ix, y, iy); };
  b.sscal = [](I64 n, float al, float* x, I64 ix) { blas::scal(n, al, x, ix); };
  b.dscal = [](I64 n, double al, double* x, I64 ix) { blas::scal(n, al, x, ix); };
  return b;
}

/// CBLAS / Fortran BLAS entry points with either 32- or 64-bit integers.
template <typename Int>
Backend library_backend(const std::string& name, void* handle, bool fixed_threads) {
  Backend b;
  b.name          = name;
  b.fixed_threads = fixed_threads;
  auto sym        = [&](const char* s) { return dlsym(handle, s); };
  if (auto f = reinterpret_cast<void (*)(int)>(sym("openblas_set_num_threads"))) {
    auto get      = reinterpret_cast<int (*)()>(sym("openblas_get_num_threads"));
    b.set_threads = [f, get, name](int t) {
      f(t);
      if (get && get() != t)
        std::fprintf(stderr, "warning: %s runs %d threads, not %d (OPENBLAS_NUM_THREADS caps its pool)\n",
                     name.c_str(), get(), t);
    };
  }
  else if (auto g = reinterpret_cast<void (*)(Int)>(sym("bli_thread_set_num_threads")))
    b.set_threads = [g](int t) { g(static_cast<Int>(t)); };
  else
    b.set_threads = [](int) {};

  using csgemm_t = void (*)(int, int, int, Int, Int, Int, float, const float*, Int, const float*, Int, float, float*, Int);
  using cdgemm_t = void (*)(int, int, int, Int, Int, Int, double, const double*, Int, const double*, Int, double, double*, Int);
  using csgemv_t = void (*)(int, int, Int, Int, float, const float*, Int, const float*, Int, float, float*, Int);
  using cdgemv_t = void (*)(int, int, Int, Int, double, const double*, Int, const double*, Int, double, double*, Int);
  using csdot_t  = float (*)(Int, const float*, Int, const float*, Int);
  using cddot_t  = double (*)(Int, const double*, Int, const double*, Int);
  using csaxpy_t = void (*)(Int, float, const float*, Int, float*, Int);
  using cdaxpy_t = void (*)(Int, double, const double*, Int, double*, Int);
  using csscal_t = void (*)(Int, float, float*, Int);
  using cdscal_t = void (*)(Int, double, double*, Int);
  constexpr int RowMajor = 101, NoTrans = 111, Trans = 112;
  if (auto f = reinterpret_cast<csgemm_t>(sym("cblas_sgemm"))) {
    b.sgemm = [f](bool ta, bool tb, I64 m, I64 n, I64 k, float al, const float* a, I64 lda, const float* bb, I64 ldb,
                  float be, float* c, I64 ldc) {
      f(RowMajor, ta ? Trans : NoTrans, tb ? Trans : NoTrans, Int(m), Int(n), Int(k), al, a, Int(lda), bb, Int(ldb), be,
        c, Int(ldc));
    };
    auto fd = reinterpret_cast<cdgemm_t>(sym("cblas_dgemm"));
    b.dgemm = [fd](bool ta, bool tb, I64 m, I64 n, I64 k, double al, const double* a, I64 lda, const double* bb,
                   I64 ldb, double be, double* c, I64 ldc) {
      fd(RowMajor, ta ? Trans : NoTrans, tb ? Trans : NoTrans, Int(m), Int(n), Int(k), al, a, Int(lda), bb, Int(ldb),
         be, c, Int(ldc));
    };
    auto gs = reinterpret_cast<csgemv_t>(sym("cblas_sgemv"));
    auto gd = reinterpret_cast<cdgemv_t>(sym("cblas_dgemv"));
    b.sgemv = [gs](bool t, I64 m, I64 n, float al, const float* a, I64 lda, const float* x, I64 ix, float be, float* y,
                   I64 iy) { gs(RowMajor, t ? Trans : NoTrans, Int(m), Int(n), al, a, Int(lda), x, Int(ix), be, y, Int(iy)); };
    b.dgemv = [gd](bool t, I64 m, I64 n, double al, const double* a, I64 lda, const double* x, I64 ix, double be,
                   double* y, I64 iy) { gd(RowMajor, t ? Trans : NoTrans, Int(m), Int(n), al, a, Int(lda), x, Int(ix), be, y, Int(iy)); };
    auto ds = reinterpret_cast<csdot_t>(sym("cblas_sdot"));
    auto dd = reinterpret_cast<cddot_t>(sym("cblas_ddot"));
    b.sdot  = [ds](I64 n, const float* x, I64 ix, const float* y, I64 iy) { return ds(Int(n), x, Int(ix), y, Int(iy)); };
    b.ddot  = [dd](I64 n, const double* x, I64 ix, const double* y, I64 iy) { return dd(Int(n), x, Int(ix), y, Int(iy)); };
    auto as = reinterpret_cast<csaxpy_t>(sym("cblas_saxpy"));
    auto ad = reinterpret_cast<cdaxpy_t>(sym("cblas_daxpy"));
    b.saxpy = [as](I64 n, float al, const float* x, I64 ix, float* y, I64 iy) { as(Int(n), al, x, Int(ix), y, Int(iy)); };
    b.daxpy = [ad](I64 n, double al, const double* x, I64 ix, double* y, I64 iy) { ad(Int(n), al, x, Int(ix), y, Int(iy)); };
    auto ss = reinterpret_cast<csscal_t>(sym("cblas_sscal"));
    auto sd = reinterpret_cast<cdscal_t>(sym("cblas_dscal"));
    b.sscal = [ss](I64 n, float al, float* x, I64 ix) { ss(Int(n), al, x, Int(ix)); };
    b.dscal = [sd](I64 n, double al, double* x, I64 ix) { sd(Int(n), al, x, Int(ix)); };
    return b;
  }
  // Fortran interface (column-major): row-major C = op(A) op(B) is C^T = op(B)^T op(A)^T.
  using fsgemm_t = void (*)(const char*, const char*, const Int*, const Int*, const Int*, const float*, const float*,
                            const Int*, const float*, const Int*, const float*, float*, const Int*);
  using fdgemm_t = void (*)(const char*, const char*, const Int*, const Int*, const Int*, const double*, const double*,
                            const Int*, const double*, const Int*, const double*, double*, const Int*);
  using fsgemv_t = void (*)(const char*, const Int*, const Int*, const float*, const float*, const Int*, const float*,
                            const Int*, const float*, float*, const Int*);
  using fdgemv_t = void (*)(const char*, const Int*, const Int*, const double*, const double*, const Int*,
                            const double*, const Int*, const double*, double*, const Int*);
  using fsdot_t  = float (*)(const Int*, const float*, const Int*, const float*, const Int*);
  using fddot_t  = double (*)(const Int*, const double*, const Int*, const double*, const Int*);
  using fsaxpy_t = void (*)(const Int*, const float*, const float*, const Int*, float*, const Int*);
  using fdaxpy_t = void (*)(const Int*, const double*, const double*, const Int*, double*, const Int*);
  using fsscal_t = void (*)(const Int*, const float*, float*, const Int*);
  using fdscal_t = void (*)(const Int*, const double*, double*, const Int*);
  auto fs = reinterpret_cast<fsgemm_t>(sym("sgemm_"));
  if (!fs) throw std::runtime_error(name + ": neither cblas_sgemm nor sgemm_ found");
  auto fd = reinterpret_cast<fdgemm_t>(sym("dgemm_"));
  b.sgemm = [fs](bool ta, bool tb, I64 m, I64 n, I64 k, float al, const float* a, I64 lda, const float* bb, I64 ldb,
                 float be, float* c, I64 ldc) {
    const Int M = Int(m), N = Int(n), K = Int(k), LDA = Int(lda), LDB = Int(ldb), LDC = Int(ldc);
    fs(tb ? "T" : "N", ta ? "T" : "N", &N, &M, &K, &al, bb, &LDB, a, &LDA, &be, c, &LDC);
  };
  b.dgemm = [fd](bool ta, bool tb, I64 m, I64 n, I64 k, double al, const double* a, I64 lda, const double* bb, I64 ldb,
                 double be, double* c, I64 ldc) {
    const Int M = Int(m), N = Int(n), K = Int(k), LDA = Int(lda), LDB = Int(ldb), LDC = Int(ldc);
    fd(tb ? "T" : "N", ta ? "T" : "N", &N, &M, &K, &al, bb, &LDB, a, &LDA, &be, c, &LDC);
  };
  // Row-major A [m, n] is column-major A^T [n, m]: op(A) x is op'(A^T) x with the flag flipped.
  auto gs = reinterpret_cast<fsgemv_t>(sym("sgemv_"));
  auto gd = reinterpret_cast<fdgemv_t>(sym("dgemv_"));
  b.sgemv = [gs](bool t, I64 m, I64 n, float al, const float* a, I64 lda, const float* x, I64 ix, float be, float* y,
                 I64 iy) {
    const Int M = Int(m), N = Int(n), LDA = Int(lda), IX = Int(ix), IY = Int(iy);
    gs(t ? "N" : "T", &N, &M, &al, a, &LDA, x, &IX, &be, y, &IY);
  };
  b.dgemv = [gd](bool t, I64 m, I64 n, double al, const double* a, I64 lda, const double* x, I64 ix, double be,
                 double* y, I64 iy) {
    const Int M = Int(m), N = Int(n), LDA = Int(lda), IX = Int(ix), IY = Int(iy);
    gd(t ? "N" : "T", &N, &M, &al, a, &LDA, x, &IX, &be, y, &IY);
  };
  auto ds = reinterpret_cast<fsdot_t>(sym("sdot_"));
  auto dd = reinterpret_cast<fddot_t>(sym("ddot_"));
  b.sdot  = [ds](I64 n, const float* x, I64 ix, const float* y, I64 iy) {
    const Int N = Int(n), IX = Int(ix), IY = Int(iy);
    return ds(&N, x, &IX, y, &IY);
  };
  b.ddot = [dd](I64 n, const double* x, I64 ix, const double* y, I64 iy) {
    const Int N = Int(n), IX = Int(ix), IY = Int(iy);
    return dd(&N, x, &IX, y, &IY);
  };
  auto as = reinterpret_cast<fsaxpy_t>(sym("saxpy_"));
  auto ad = reinterpret_cast<fdaxpy_t>(sym("daxpy_"));
  b.saxpy = [as](I64 n, float al, const float* x, I64 ix, float* y, I64 iy) {
    const Int N = Int(n), IX = Int(ix), IY = Int(iy);
    as(&N, &al, x, &IX, y, &IY);
  };
  b.daxpy = [ad](I64 n, double al, const double* x, I64 ix, double* y, I64 iy) {
    const Int N = Int(n), IX = Int(ix), IY = Int(iy);
    ad(&N, &al, x, &IX, y, &IY);
  };
  auto ss = reinterpret_cast<fsscal_t>(sym("sscal_"));
  auto sd = reinterpret_cast<fdscal_t>(sym("dscal_"));
  b.sscal = [ss](I64 n, float al, float* x, I64 ix) {
    const Int N = Int(n), IX = Int(ix);
    ss(&N, &al, x, &IX);
  };
  b.dscal = [sd](I64 n, double al, double* x, I64 ix) {
    const Int N = Int(n), IX = Int(ix);
    sd(&N, &al, x, &IX);
  };
  return b;
}

Backend load_library(const std::string& spec) {
  // name=path[:ilp64][:fixed-threads]
  const auto eq = spec.find('=');
  if (eq == std::string::npos) throw std::runtime_error("library spec needs name=path: " + spec);
  const std::string        name = spec.substr(0, eq);
  std::vector<std::string> parts;
  std::stringstream        ss(spec.substr(eq + 1));
  for (std::string p; std::getline(ss, p, ':');) parts.push_back(p);
  const std::string path  = parts.at(0);
  bool              ilp64 = false, fixed = false, slow = false;
  for (std::size_t i = 1; i < parts.size(); ++i) {
    if (parts[i] == "ilp64") ilp64 = true;
    else if (parts[i] == "fixed-threads") fixed = true;
    else if (parts[i] == "slow") slow = true;
    else throw std::runtime_error("unknown library flag " + parts[i]);
  }
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) throw std::runtime_error(name + ": " + dlerror());
  // nixpkgs OpenBLAS/BLIS are built with 64-bit integer interfaces; detect OpenBLAS's from its config.
  if (auto cfg = reinterpret_cast<const char* (*)()>(dlsym(handle, "openblas_get_config")))
    if (std::strstr(cfg(), "USE64BITINT")) ilp64 = true;
  Backend b = ilp64 ? library_backend<std::int64_t>(name, handle, fixed) : library_backend<std::int32_t>(name, handle, fixed);
  b.slow    = slow;
  return b;
}

struct Case {
  std::string op;    // gemm, gemv, dot, axpy, scal
  std::string label;
  bool        ta{false}, tb{false};
  I64         m{0}, n{0}, k{0};
};

std::vector<Case> cases(bool quick) {
  std::vector<Case> out;
  for (I64 s : (quick ? std::vector<I64>{256, 1024} : std::vector<I64>{256, 512, 1024, 2048, 4096}))
    out.push_back({"gemm", "square " + std::to_string(s), false, false, s, s, s});
  // SqueezeNet CIFAR-10 (batch 32, 32x32 images) convolution GEMMs: forward (NN),
  // input gradient (NT) and filter gradient (TN) of representative layers.
  out.push_back({"gemm", "conv1 fwd 32768x64x27", false, false, 32768, 64, 27});
  out.push_back({"gemm", "squeeze fwd 8192x16x128", false, false, 8192, 16, 128});
  out.push_back({"gemm", "expand1x1 fwd 8192x64x16", false, false, 8192, 64, 16});
  out.push_back({"gemm", "expand3x3 fwd 8192x64x144", false, false, 8192, 64, 144});
  out.push_back({"gemm", "fire8 fwd 2048x256x576", false, false, 2048, 256, 576});
  out.push_back({"gemm", "expand3x3 dx 8192x144x64 NT", false, true, 8192, 144, 64});
  out.push_back({"gemm", "expand3x3 dw 144x64x8192 TN", true, false, 144, 64, 8192});
  out.push_back({"gemm", "tile 128x64x144", false, false, 128, 64, 144});
  for (I64 s : (quick ? std::vector<I64>{2048} : std::vector<I64>{1024, 4096, 8192})) {
    out.push_back({"gemv", "gemv N " + std::to_string(s), false, false, s, s, 0});
    out.push_back({"gemv", "gemv T " + std::to_string(s), true, false, s, s, 0});
  }
  for (I64 e : (quick ? std::vector<I64>{20} : std::vector<I64>{16, 20, 24})) {
    const I64 n = I64{1} << e;
    out.push_back({"dot", "dot 2^" + std::to_string(e), false, false, n, 0, 0});
    out.push_back({"axpy", "axpy 2^" + std::to_string(e), false, false, n, 0, 0});
    out.push_back({"scal", "scal 2^" + std::to_string(e), false, false, n, 0, 0});
  }
  return out;
}

/// Work (flop) and bytes touched once, for throughput figures.
void cost(const Case& c, int elem, double& flop, double& bytes) {
  if (c.op == "gemm") {
    flop  = 2.0 * double(c.m) * double(c.n) * double(c.k);
    bytes = double(elem) * (double(c.m) * c.k + double(c.k) * c.n + 2.0 * double(c.m) * c.n);
  } else if (c.op == "gemv") {
    flop  = 2.0 * double(c.m) * double(c.n);
    bytes = double(elem) * (double(c.m) * c.n + 2.0 * c.m + c.n);
  } else if (c.op == "dot") {
    flop  = 2.0 * double(c.m);
    bytes = double(elem) * 2.0 * double(c.m);
  } else if (c.op == "axpy") {
    flop  = 2.0 * double(c.m);
    bytes = double(elem) * 3.0 * double(c.m);
  } else {
    flop  = double(c.m);
    bytes = double(elem) * 2.0 * double(c.m);
  }
}

template <typename T>
void fill(std::vector<T>& v, std::mt19937& rng) {
  std::uniform_real_distribution<double> u(-1, 1);
  for (auto& x : v) x = T(u(rng));
}

/// Median seconds per call: one warm-up, then at least `min_reps` calls spanning >= 0.25 s.
template <typename Fn>
double time_call(Fn fn, double budget = 0.25, int min_reps = 5, int max_reps = 10000) {
  fn();
  std::vector<double> t;
  const auto          start = std::chrono::steady_clock::now();
  for (int r = 0; r < max_reps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (r + 1 >= min_reps && std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > budget)
      break;
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

template <typename T>
double run_case(const Backend& b, const Case& c, std::mt19937& rng, double& max_err) {
  constexpr bool is_float = sizeof(T) == 4;
  max_err                 = 0;
  if (c.op == "gemm") {
    std::vector<T> a(static_cast<std::size_t>(c.m * c.k)), bb(static_cast<std::size_t>(c.k * c.n)),
        cc(static_cast<std::size_t>(c.m * c.n), T(0));
    fill(a, rng);
    fill(bb, rng);
    const I64 lda = c.ta ? c.m : c.k, ldb = c.tb ? c.k : c.n;
    auto      call = [&] {
      if constexpr (is_float)
        b.sgemm(c.ta, c.tb, c.m, c.n, c.k, T(1), a.data(), lda, bb.data(), ldb, T(0), cc.data(), c.n);
      else
        b.dgemm(c.ta, c.tb, c.m, c.n, c.k, T(1), a.data(), lda, bb.data(), ldb, T(0), cc.data(), c.n);
    };
    const double seconds = time_call(call);
    // Spot-check a few entries against a double reference.
    for (int s = 0; s < 8; ++s) {
      const I64 i = (c.m * (s * 37 + 1) / 9) % c.m, j = (c.n * (s * 11 + 3) / 7) % c.n;
      double    ref = 0;
      for (I64 l = 0; l < c.k; ++l)
        ref += double(c.ta ? a[l * lda + i] : a[i * lda + l]) * double(c.tb ? bb[j * ldb + l] : bb[l * ldb + j]);
      max_err = std::max(max_err, std::fabs(ref - double(cc[i * c.n + j])) / (1 + std::fabs(ref)));
    }
    return seconds;
  }
  if (c.op == "gemv") {
    std::vector<T> a(static_cast<std::size_t>(c.m * c.n)), x(static_cast<std::size_t>(c.ta ? c.m : c.n)),
        y(static_cast<std::size_t>(c.ta ? c.n : c.m), T(0));
    fill(a, rng);
    fill(x, rng);
    auto call = [&] {
      if constexpr (is_float)
        b.sgemv(c.ta, c.m, c.n, T(1), a.data(), c.n, x.data(), 1, T(0), y.data(), 1);
      else
        b.dgemv(c.ta, c.m, c.n, T(1), a.data(), c.n, x.data(), 1, T(0), y.data(), 1);
    };
    const double seconds = time_call(call);
    for (int s = 0; s < 4; ++s) {
      const I64 i   = (y.size() * (s * 13 + 5) / 17) % I64(y.size());
      double    ref = 0;
      for (I64 j = 0; j < I64(x.size()); ++j) ref += double(c.ta ? a[j * c.n + i] : a[i * c.n + j]) * double(x[j]);
      max_err = std::max(max_err, std::fabs(ref - double(y[i])) / (1 + std::fabs(ref)));
    }
    return seconds;
  }
  std::vector<T> x(static_cast<std::size_t>(c.m)), y(static_cast<std::size_t>(c.m));
  fill(x, rng);
  fill(y, rng);
  if (c.op == "dot") {
    volatile T sink = 0;
    double     ref  = 0;
    for (I64 i = 0; i < c.m; ++i) ref += double(x[i]) * double(y[i]);
    auto call = [&] {
      if constexpr (is_float) sink = b.sdot(c.m, x.data(), 1, y.data(), 1);
      else sink = b.ddot(c.m, x.data(), 1, y.data(), 1);
    };
    const double seconds = time_call(call);
    max_err              = std::fabs(ref - double(sink)) / (1 + std::fabs(ref));
    return seconds;
  }
  if (c.op == "axpy") {
    auto call = [&] {
      if constexpr (is_float) b.saxpy(c.m, T(0.5), x.data(), 1, y.data(), 1);
      else b.daxpy(c.m, T(0.5), x.data(), 1, y.data(), 1);
    };
    return time_call(call);
  }
  auto call = [&] {
    if constexpr (is_float) b.sscal(c.m, T(1.0000001), x.data(), 1);
    else b.dscal(c.m, T(1.0000001), x.data(), 1);
  };
  return time_call(call);
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream        ss(s);
  for (std::string p; std::getline(ss, p, sep);)
    if (!p.empty()) out.push_back(p);
  return out;
}


/// Runs this program once per op, thread count and library: the in-house
/// kernels (every instruction set) in one process, each external library in
/// its own. Libraries size their thread pools from OPENBLAS/GOTO/OMP/
/// BLIS_NUM_THREADS when they are loaded (OpenBLAS's OpenMP build ignores
/// later openblas_set_num_threads calls), so each process gets the count in
/// its environment; and a library measured in a process where another
/// library's pool has run competes with that pool's idle threads (at 16
/// threads OpenBLAS's square 256 SGEMM fell from 540-710 to 7 GFLOP/s, the
/// in-house one from 730-750 to 309). Libraries marked :fixed-threads run at
/// the first count only. The JSON parts are joined into `json`.
int run_isolated(const std::vector<int>& threads, const std::string& libs, const std::vector<std::string>& ops,
                 bool quick, bool inhouse, const std::string& json) {
  std::vector<std::string> groups;  // "" = the in-house kernels, else one library spec
  if (inhouse) groups.emplace_back();
  for (const auto& spec : split(libs, ';'))
    if (!spec.empty()) groups.push_back(spec);
  std::vector<std::string> parts;
  bool                     header = true;
  for (const auto& op : ops)
    for (std::size_t i = 0; i < threads.size(); ++i)
      for (const auto& group : groups) {
        if (i > 0 && group.find(":fixed-threads") != std::string::npos) continue;
        const std::string        t = std::to_string(threads[i]);
        std::vector<std::string> args{"blas_compare", "--child", "--threads", t, "--ops", op, "--libs", group};
        if (!group.empty()) args.emplace_back("--no-inhouse");
        if (quick) args.emplace_back("--quick");
        if (!header) args.emplace_back("--no-header");
        header = false;
        if (!json.empty()) {
          parts.push_back(json + ".part" + std::to_string(parts.size()));
          args.emplace_back("--json");
          args.push_back(parts.back());
        }
        for (const char* var : {"OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS"})
          setenv(var, t.c_str(), 1);
        std::vector<char*> argv;
        for (auto& a : args) argv.push_back(a.data());
        argv.push_back(nullptr);
        std::fflush(stdout);
        pid_t pid = 0;
        if (posix_spawn(&pid, "/proc/self/exe", nullptr, nullptr, argv.data(), environ) != 0) {
          std::perror("posix_spawn");
          return 1;
        }
        int status = 0;
        waitpid(pid, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
      }
  if (json.empty()) return 0;
  std::ofstream out(json);
  out << "[\n";
  bool first = true;
  for (const auto& part : parts) {
    std::ifstream     in(part);
    std::stringstream body;
    body << in.rdbuf();
    std::string rows = body.str();
    const auto  open = rows.find('['), close = rows.rfind(']');
    rows = open == std::string::npos || close == std::string::npos ? "" : rows.substr(open + 1, close - open - 1);
    while (!rows.empty() && (rows.front() == '\n' || rows.front() == ' ')) rows.erase(rows.begin());
    while (!rows.empty() && (rows.back() == '\n' || rows.back() == ' ')) rows.pop_back();
    if (!rows.empty()) {
      out << (first ? "" : ",\n") << rows;
      first = false;
    }
    std::remove(part.c_str());
  }
  out << "\n]\n";
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  std::string      libs = std::getenv("DATAMUNGE_BLAS_LIBS") ? std::getenv("DATAMUNGE_BLAS_LIBS") : "";
  std::vector<int> threads{1, 8, 16};
  std::vector<std::string> ops{"gemm", "gemv", "dot", "axpy", "scal"};
  std::string      json;
  bool             quick = false, inhouse = true, header = true, child = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto              next = [&] {
      if (i + 1 >= argc) throw std::runtime_error(arg + " needs a value");
      return std::string(argv[++i]);
    };
    if (arg == "--libs") libs = next();
    else if (arg == "--threads") {
      threads.clear();
      for (const auto& t : split(next(), ',')) threads.push_back(std::stoi(t));
    } else if (arg == "--ops") ops = split(next(), ',');
    else if (arg == "--json") json = next();
    else if (arg == "--quick") quick = true;
    else if (arg == "--no-inhouse") inhouse = false;
    else if (arg == "--no-header") header = false;
    else if (arg == "--child") child = true;
    else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }
  if (!child && !libs.empty()) return run_isolated(threads, libs, ops, quick, inhouse, json);
  // In-house kernels only, or one child: size the library pools for the first
  // count before any is loaded (a profile exporting OPENBLAS_NUM_THREADS=1
  // would otherwise cap them).
  for (const char* var : {"OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS"})
    setenv(var, std::to_string(threads.front()).c_str(), 1);
  // Place the main thread as the in-house pool would before a library starts
  // threads, which inherit its affinity (placement.hpp).
  bench::place_main_thread(threads.front());
  std::vector<Backend> backends;
  if (inhouse)
    for (const auto& isa : blas::available_isas()) backends.push_back(inhouse_backend(isa));
  for (const auto& spec : split(libs, ';')) {
    try {
      backends.push_back(load_library(spec));
    } catch (const std::exception& e) {
      std::fprintf(stderr, "skipping %s: %s\n", spec.c_str(), e.what());
    }
  }
  if (header) {
    std::printf("# backends:");
    for (const auto& b : backends) std::printf(" %s", b.name.c_str());
    std::printf("\n%-10s %-30s %-4s %-7s %-18s %12s %10s %10s %9s\n", "op", "case", "type", "threads", "backend",
                "us/call", "GFLOP/s", "GB/s", "max_err");
  }
  std::ofstream out;
  if (!json.empty()) {
    out.open(json);
    out << "[\n";
  }
  bool         first = true;
  std::mt19937 rng(42);
  for (const auto& c : cases(quick)) {
    if (std::find(ops.begin(), ops.end(), c.op) == ops.end()) continue;
    for (int dtype = 0; dtype < 2; ++dtype)
      for (int t : threads)
        for (const auto& b : backends) {
          if (b.fixed_threads && t != threads.front()) continue;
          if (b.slow && c.op == "gemm" && 2.0 * double(c.m) * double(c.n) * double(c.k) > 4e9) continue;
          b.set_threads(t);
          double err = 0, flop = 0, bytes = 0;
          const double seconds = dtype == 0 ? run_case<float>(b, c, rng, err) : run_case<double>(b, c, rng, err);
          cost(c, dtype == 0 ? 4 : 8, flop, bytes);
          const double gflops = flop / seconds / 1e9, gbps = bytes / seconds / 1e9;
          std::printf("%-10s %-30s %-4s %-7d %-18s %12.1f %10.1f %10.1f %9.1e\n", c.op.c_str(), c.label.c_str(),
                      dtype == 0 ? "f32" : "f64", t, b.name.c_str(), seconds * 1e6, gflops, gbps, err);
          std::fflush(stdout);
          // Let another library's worker threads stop spinning before the next measurement.
          if (b.name.rfind("inhouse", 0) != 0) std::this_thread::sleep_for(std::chrono::milliseconds(100));
          if (out) {
            out << (first ? "" : ",\n") << "  {\"op\": \"" << c.op << "\", \"case\": \"" << c.label << "\", \"m\": " << c.m
                << ", \"n\": " << c.n << ", \"k\": " << c.k << ", \"ta\": " << (c.ta ? "true" : "false")
                << ", \"tb\": " << (c.tb ? "true" : "false") << ", \"dtype\": \"" << (dtype == 0 ? "f32" : "f64")
                << "\", \"threads\": " << t << ", \"backend\": \"" << b.name << "\", \"seconds\": " << seconds
                << ", \"gflops\": " << gflops << ", \"gbps\": " << gbps << ", \"max_err\": " << err << "}";
            first = false;
          }
        }
  }
  if (out) out << "\n]\n";
  return 0;
}
