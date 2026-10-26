#include <datamunge/nn/einsum.hpp>
#include <datamunge/nn/memory.hpp>

#include "common.hpp"
namespace {
namespace nn = nb::nn;
// Shape arguments: order, batch, dtype (0=f32,1=f64), backend (0=auto,1=builtin,2=LAPACK).
void args(benchmark::internal::Benchmark* b) {
  for (int n : {16, 32, 64, 96, 128, 256})
    for (int dt : {0, 1})
      for (int backend : {0, 1, 2})
        b->Args({n, 1, dt, backend});
  for (int dt : {0, 1})
    b->Args({16, 16, dt, 0});
  b->ArgNames({"n", "batch", "f64", "backend"})->UseRealTime();
}
void BM_TensorLinalg(benchmark::State& s, int op) {
  nb::Scope scope;
  if (s.range(3) == 2 && !nn::linalg_has_lapack()) {
    s.SkipWithError("LAPACK unavailable");
    return;
  }
  nn::set_linalg_backend(static_cast<nn::LinalgBackend>(s.range(3)));
  auto            dt = s.range(2) ? nn::DType::Float64 : nn::DType::Float32;
  const auto      n = s.range(0), batch = s.range(1);
  nn::NoGradGuard guard;
  auto            x    = nb::random({batch, n, n}, dt);
  auto            a    = nn::add(nn::matmul(x, x, false, true), nn::eye(n, dt));
  auto            rhs  = nb::random({batch, n, 4}, dt);
  auto            chol = nn::cholesky(a);
  double          tol  = s.range(2) ? 1e-8 : 2e-3;
  // Check each exact fixture before timing, including reconstruction and orthogonality.
  auto run = [&]() -> nn::Tensor {
    switch (op) {
      case 0:
        return nn::solve(a, rhs);
      case 1:
        return nn::cholesky(a);
      case 2:
        return nn::triangular_solve(chol, rhs);
      case 3: {
        auto r = nn::qr(a);
        nb::consume(r.r);
        return r.q;
      }
      case 4: {
        auto r = nn::svd(a);
        nb::consume(r.u);
        nb::consume(r.v);
        return r.s;
      }
      case 5: {
        auto r = nn::eigh(a);
        nb::consume(r.v);
        return r.e;
      }
      case 6:
        return nn::inverse(a);
      case 7:
        return nn::logdet(a);
      case 8:
        return nn::lstsq(a, rhs);
      default:
        return nn::pinv(a);
    }
  };
  if (op == 0 || op == 8)
    nb::close(nn::matmul(a, run()), rhs, tol);
  else if (op == 1)
    nb::close(nn::matmul(chol, chol, false, true), a, tol);
  else if (op == 2)
    nb::close(nn::matmul(chol, run()), rhs, tol);
  else if (op == 3) {
    auto r = nn::qr(a);
    nb::close(nn::matmul(r.q, r.r), a, tol);
    nb::close(nn::matmul(r.q, r.q, true, false), nn::eye(n, dt), tol);
  } else if (op == 4) {
    auto r = nn::svd(a);
    nb::close(nn::matmul(nn::matmul(r.u, nn::matrix_diag(r.s)), r.v, false, true), a, tol);
  } else if (op == 5) {
    auto r = nn::eigh(a);
    nb::close(nn::matmul(a, r.v), nn::matmul(r.v, nn::matrix_diag(r.e)), tol);
  } else if (op == 6 || op == 9)
    nb::close(nn::matmul(a, run()), nn::eye(n, dt), tol);
  else
    nb::finite(run());
  for (auto _ : s)
    nb::consume(run());
  s.SetItemsProcessed(s.iterations() * batch);
}
#define LA(name, op) BENCHMARK_CAPTURE(BM_TensorLinalg, name, op)->Apply(args)
LA(Solve, 0);
LA(Cholesky, 1);
LA(TriangularSolve, 2);
LA(QR, 3);
LA(SVD, 4);
LA(Eigh, 5);
LA(Inverse, 6);
LA(Logdet, 7);
LA(LeastSquares, 8);
LA(PseudoInverse, 9);
void BM_TensorMatmul(benchmark::State& s) {
  nb::Scope       scope;
  nn::NoGradGuard guard;
  auto            m = s.range(0), k = s.range(1), n = s.range(2), batch = s.range(3);
  auto            dt = s.range(4) ? nn::DType::Float64 : nn::DType::Float32;
  auto            a = nn::ones({batch, m, k}, dt), b = nn::ones({batch, k, n}, dt);
  nb::close(nn::matmul(a, b), nn::full({batch, m, n}, double(k), dt), 1e-5);
  for (auto _ : s)
    nb::consume(nn::matmul(a, b));
  s.counters["FLOP/s"] =
      benchmark::Counter(2.0 * batch * m * k * n, benchmark::Counter::kIsIterationInvariantRate);
  s.SetLabel(nn::gemm_backend());
}
BENCHMARK(BM_TensorMatmul)
    ->ArgsProduct({{16, 64, 256}, {64}, {32, 128}, {1, 16}, {0, 1}})
    ->ArgNames({"m", "k", "n", "batch", "f64"})
    ->UseRealTime();
void BM_TensorRectangular(benchmark::State& s, int op) {
  nb::Scope       scope;
  nn::NoGradGuard guard;
  auto            m = s.range(0), n = s.range(1);
  auto            a   = nb::random({m, n});
  auto            run = [&]() {
    if (op == 0) {
      auto r = nn::qr(a);
      nb::consume(r.q);
      return r.r;
    }
    auto r = nn::svd(a);
    nb::consume(r.u);
    nb::consume(r.v);
    return r.s;
  };
  if (op == 0) {
    auto r = nn::qr(a);
    nb::close(nn::matmul(r.q, r.r), a, 1e-8);
  } else {
    auto r = nn::svd(a);
    nb::close(nn::matmul(nn::matmul(r.u, nn::matrix_diag(r.s)), r.v, false, true), a, 1e-8);
  }
  for (auto _ : s)
    nb::consume(run());
}
BENCHMARK_CAPTURE(BM_TensorRectangular, QR, 0)->Args({128, 32})->Args({32, 128})->UseRealTime();
BENCHMARK_CAPTURE(BM_TensorRectangular, SVD, 1)->Args({128, 32})->Args({32, 128})->UseRealTime();
void BM_TensorThreadScaling(benchmark::State& s) {
  nb::Scope       scope;
  nn::NoGradGuard guard;
  nn::set_num_threads(s.range(0));
  auto x = nn::ones({1048576});
  nb::close(nn::relu(x), x, 1e-6);
  for (auto _ : s)
    nb::consume(nn::relu(x));
  s.SetItemsProcessed(s.iterations() * x.numel());
}
BENCHMARK(BM_TensorThreadScaling)->Arg(1)->Arg(2)->Arg(4)->UseRealTime();

void BM_TensorDetEig(benchmark::State& s, int op) {
  nb::Scope       scope;
  nn::NoGradGuard guard;
  auto            n = s.range(0);
  auto diagonal     = nn::add(nn::mul(nn::arange(0, n, 1, nn::DType::Float64), 1.0 / n), 1.0);
  auto a =
      nn::add(nn::matrix_diag(diagonal), nn::mul(nn::band_part(nb::random({n, n}), 0, -1), 0.01));
  // Clear the random diagonal, keeping a nonsymmetric triangular fixture.
  a        = nn::set_diag(a, diagonal);
  auto run = [&]() {
    if (op == 0)
      return nn::det(a);
    if (op == 1)
      return nn::slogdet(a).log_abs_det;
    auto r = nn::eigvals(a);
    nb::consume(r.imag);
    return r.real;
  };
  if (op == 0)
    nb::check(
        std::abs(run().item() / std::exp(nn::sum(nn::log(diagonal)).item()) - 1) < 1e-8,
        "determinant reference failed");
  else if (op == 1)
    nb::close(run(), nn::sum(nn::log(diagonal)), 1e-8);
  else {
    auto r = nn::eigvals(a);
    nb::close(r.real, diagonal, 1e-8);
    nb::close(r.imag, nn::zeros({n}, nn::DType::Float64), 1e-8);
  }
  for (auto _ : s)
    nb::consume(run());
}
BENCHMARK_CAPTURE(BM_TensorDetEig, Determinant, 0)->Arg(16)->Arg(64)->UseRealTime();
BENCHMARK_CAPTURE(BM_TensorDetEig, Slogdet, 1)->Arg(16)->Arg(64)->UseRealTime();
BENCHMARK_CAPTURE(BM_TensorDetEig, GeneralEigenvalues, 2)->Arg(16)->Arg(64)->UseRealTime();

void BM_TensorEinsum(benchmark::State& s) {
  nb::Scope       scope;
  nn::NoGradGuard guard;
  auto            a = nb::random({s.range(0), 32, 64}), b = nb::random({s.range(0), 64, 16});
  nb::close(nn::einsum("bij,bjk->bik", a, b), nn::matmul(a, b), 1e-8);
  for (auto _ : s)
    nb::consume(nn::einsum("bij,bjk->bik", a, b));
}
BENCHMARK(BM_TensorEinsum)->Arg(1)->Arg(16)->UseRealTime();

// Shared operands exercise batch broadcast packing independently of GEMM size.
void BM_TensorMatmulShared(benchmark::State& s) {
  nb::Scope scope;
  nn::NoGradGuard guard;
  const auto batch = s.range(0), n = s.range(1), mode = s.range(3);
  const auto dt = s.range(2) ? nn::DType::Float64 : nn::DType::Float32;
  const bool ta = mode >= 2, tb = mode == 3;
  const nn::Shape ashape = ta ? nn::Shape{64, 16} : nn::Shape{16, 64};
  const nn::Shape bshape = tb ? nn::Shape{n, 64} : nn::Shape{64, n};
  auto as = ashape, bs = bshape;
  as.insert(as.begin(), mode == 0 ? 1 : batch);
  bs.insert(bs.begin(), mode == 0 ? batch : 1);
  auto a = nn::ones(as, dt), b = nn::ones(bs, dt);
  auto run = [&] { return nn::matmul(a, b, ta, tb); };
  nb::close(run(), nn::full({batch, 16, n}, 64, dt), 1e-5);
  {
    nn::PeakMemoryScope memory;
    auto output = run();
    s.counters["peak_tensor_bytes"] = double(memory.peak_above_baseline());
  }
  for (auto _ : s) nb::consume(run());
  s.counters["FLOP/s"] = benchmark::Counter(2.0 * batch * 16 * 64 * n, benchmark::Counter::kIsIterationInvariantRate);
  s.SetLabel(nn::gemm_backend());
}
BENCHMARK(BM_TensorMatmulShared)
    ->ArgsProduct({{1, 16}, {32, 128}, {0, 1}, {0, 1, 2, 3}})
    ->ArgNames({"batch", "n", "f64", "mode"})->UseRealTime();

void BM_TensorSolveShared(benchmark::State& s) {
  nb::Scope scope;
  nn::NoGradGuard guard;
  const auto n = s.range(0), batch = s.range(1);
  const auto dt = s.range(2) ? nn::DType::Float64 : nn::DType::Float32;
  nn::set_linalg_backend(static_cast<nn::LinalgBackend>(s.range(3)));
  if (s.range(3) == 2 && !nn::linalg_has_lapack()) { s.SkipWithError("LAPACK unavailable"); return; }
  auto x = nb::random({n, n}, dt);
  auto a = nn::add(x, nn::mul(nn::eye(n, dt), 4.0));
  auto rhs = nb::random({batch, n, 4}, dt);
  const bool adjoint = s.range(4);
  auto run = [&] { return nn::solve(a, rhs, adjoint); };
  nb::close(nn::matmul(a, run(), adjoint, false), rhs, s.range(2) ? 1e-8 : 2e-3);
  {
    nn::PeakMemoryScope memory;
    auto output = run();
    s.counters["peak_tensor_bytes"] = double(memory.peak_above_baseline());
  }
  for (auto _ : s) nb::consume(run());
  s.SetItemsProcessed(s.iterations() * batch);
}
BENCHMARK(BM_TensorSolveShared)
    ->ArgsProduct({{16, 64}, {16}, {0, 1}, {0, 1, 2}, {0, 1}})
    ->ArgNames({"n", "batch", "f64", "backend", "adjoint"})->UseRealTime();


// Neither operand spans the full output batch: expanding them would copy each
// stored matrix repeatedly. Check the exact fixture before timing.
void BM_TensorMatmulPartialBatch(benchmark::State& s) {
  nb::Scope scope;
  nn::NoGradGuard guard;
  const auto dt = s.range(1) ? nn::DType::Float64 : nn::DType::Float32;
  const auto width = s.range(0);
  const bool ta = s.range(2), tb = s.range(3);
  auto a = nb::random(ta ? nn::Shape{2, 1, 64, 16} : nn::Shape{2, 1, 16, 64}, dt);
  auto b = nb::random(tb ? nn::Shape{1, 8, width, 64} : nn::Shape{1, 8, 64, width}, dt);
  auto run = [&] { return nn::matmul(a, b, ta, tb); };
  auto sa = a.shape(), sb = b.shape(); sa[1] = 8; sb[0] = 2;
  nb::close(run(), nn::matmul(nn::broadcast_to(a, sa).clone(), nn::broadcast_to(b, sb).clone(), ta, tb), 1e-5);
  {
    nn::PeakMemoryScope memory;
    auto output = run();
    s.counters["peak_tensor_bytes"] = double(memory.peak_above_baseline());
  }
  for (auto _ : s) nb::consume(run());
}
BENCHMARK(BM_TensorMatmulPartialBatch)
    ->ArgsProduct({{16, 128}, {0, 1}, {0, 1}, {0, 1}})
    ->ArgNames({"width", "f64", "transpose_a", "transpose_b"})->UseRealTime();

} // namespace
