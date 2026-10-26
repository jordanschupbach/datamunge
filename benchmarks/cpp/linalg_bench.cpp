#include <benchmark/benchmark.h>

#include <datamunge/linalg/linalg.hpp>

#include <cstdint>
#include <vector>

using namespace datamunge::linalg;

// ============================================================
// Reproducible matrix generators
//
// Using a mixed 64-bit generator so matrix generation is deterministic and
// produces the same matrix for a given (n, seed) pair across runs.
// ============================================================

static uint64_t random_next(uint64_t& s) {
    // SplitMix64: mix low bits before modulo indexing power-of-two dimensions.
    uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// n×n random sparse matrix, ~density fraction non-zeros.
// do_compress controls whether compress() is called before returning.
static SparseCOO<double> make_random_coo(std::size_t        n,
                                          double             density,
                                          bool               do_compress = true,
                                          uint64_t           seed = 0xDEADBEEFULL) {
    const std::size_t target = static_cast<std::size_t>(n * n * density);
    SparseCOO<double> mat(n, n, target);
    uint64_t          s = seed;
    for (std::size_t k = 0; k < target; ++k) {
        const std::size_t r = random_next(s) % n;
        const std::size_t c = random_next(s) % n;
        const double      v = static_cast<double>(random_next(s) % 1000) / 100.0 + 0.1;
        mat.set(r, c, v);
    }
    if (do_compress) mat.compress();
    return mat;
}

// n×n tridiagonal (DIA's natural habitat)
static SparseCOO<double> make_tridiag(std::size_t n) {
    SparseCOO<double> mat(n, n, 3 * n);
    for (std::size_t i = 0; i < n; ++i) {
        mat.set(i, i, 2.0);
        if (i > 0)     mat.set(i, i - 1, -1.0);
        if (i + 1 < n) mat.set(i, i + 1, -1.0);
    }
    mat.compress();
    return mat;
}

// ============================================================
// SpMV — format comparison at the same matrix size and sparsity
// ============================================================

static void BM_CSR_spmv(benchmark::State& state) {
    const auto n   = static_cast<std::size_t>(state.range(0));
    auto coo = make_random_coo(n, 0.01);
    auto mat = to_csr(coo);
    const auto nnz = mat.nnz();
    auto x = std::vector<double>(n, 1.0);
    std::vector<double> y(n);
    for (auto _ : state) {
        mat.spmv(x, y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(nnz));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(nnz) * sizeof(double) * 2);
    state.counters["nnz"] = static_cast<double>(nnz);
}
BENCHMARK(BM_CSR_spmv)->RangeMultiplier(4)->Range(128, 4096)->Unit(benchmark::kMicrosecond);

static void BM_CSC_spmv(benchmark::State& state) {
    const auto n   = static_cast<std::size_t>(state.range(0));
    auto coo = make_random_coo(n, 0.01);
    auto mat = to_csc(coo);
    const auto nnz = mat.nnz();
    auto x = std::vector<double>(n, 1.0);
    std::vector<double> y(n);
    for (auto _ : state) {
        mat.spmv(x, y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(nnz));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(nnz) * sizeof(double) * 2);
    state.counters["nnz"] = static_cast<double>(nnz);
}
BENCHMARK(BM_CSC_spmv)->RangeMultiplier(4)->Range(128, 4096)->Unit(benchmark::kMicrosecond);

static void BM_ELL_spmv(benchmark::State& state) {
    const auto n   = static_cast<std::size_t>(state.range(0));
    auto coo = make_random_coo(n, 0.01);
    auto mat = to_ell(coo);
    const auto nnz = mat.nnz();
    auto x = std::vector<double>(n, 1.0);
    std::vector<double> y(n);
    for (auto _ : state) {
        mat.spmv(x, y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(nnz));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(nnz) * sizeof(double) * 2);
    state.counters["nnz"] = static_cast<double>(nnz);
}
BENCHMARK(BM_ELL_spmv)->RangeMultiplier(4)->Range(128, 4096)->Unit(benchmark::kMicrosecond);

static void BM_DIA_spmv(benchmark::State& state) {
    const auto n   = static_cast<std::size_t>(state.range(0));
    auto coo = make_tridiag(n);
    auto mat = to_dia(coo);
    const auto nnz = mat.nnz();
    auto x = std::vector<double>(n, 1.0);
    std::vector<double> y(n);
    for (auto _ : state) {
        mat.spmv(x, y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(nnz));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(nnz) * sizeof(double) * 2);
    state.counters["nnz"] = static_cast<double>(nnz);
}
BENCHMARK(BM_DIA_spmv)->RangeMultiplier(4)->Range(128, 4096)->Unit(benchmark::kMicrosecond);

// COO SpMV variants share the same setup so we write them by hand

static void BM_COO_spmv(benchmark::State& state) {
    const auto n   = static_cast<std::size_t>(state.range(0));
    auto       A   = make_random_coo(n, 0.01);
    auto       x   = std::vector<double>(n, 1.0);
    std::vector<double> y(n);
    for (auto _ : state) {
        A.spmv(x, y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(A.nnz()));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(A.nnz()) * sizeof(double) * 2);
    state.counters["nnz"] = static_cast<double>(A.nnz());
}
BENCHMARK(BM_COO_spmv)->RangeMultiplier(4)->Range(128, 4096)->Unit(benchmark::kMicrosecond);

static void BM_COO_spmv_tiled_l1(benchmark::State& state) {
    const auto n   = static_cast<std::size_t>(state.range(0));
    auto       A   = make_random_coo(n, 0.01);
    auto       x   = std::vector<double>(n, 1.0);
    std::vector<double> y(n);
    constexpr std::size_t block = 32 * 1024 / sizeof(double);
    for (auto _ : state) {
        A.spmv_tiled(x, y, block);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(A.nnz()));
    state.counters["nnz"] = static_cast<double>(A.nnz());
}
BENCHMARK(BM_COO_spmv_tiled_l1)->RangeMultiplier(4)->Range(128, 4096)->Unit(benchmark::kMicrosecond);

static void BM_COO_spmv_tiled_l2(benchmark::State& state) {
    const auto n   = static_cast<std::size_t>(state.range(0));
    auto       A   = make_random_coo(n, 0.01);
    auto       x   = std::vector<double>(n, 1.0);
    std::vector<double> y(n);
    constexpr std::size_t block = 256 * 1024 / sizeof(double);
    for (auto _ : state) {
        A.spmv_tiled(x, y, block);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(A.nnz()));
    state.counters["nnz"] = static_cast<double>(A.nnz());
}
BENCHMARK(BM_COO_spmv_tiled_l2)->RangeMultiplier(4)->Range(128, 4096)->Unit(benchmark::kMicrosecond);

// DIA vs CSR on tridiagonal — same matrix, shows DIA's sequential-diagonal advantage
static void BM_CSR_spmv_tridiag(benchmark::State& state) {
    const auto n   = static_cast<std::size_t>(state.range(0));
    auto       csr = to_csr(make_tridiag(n));
    auto       x   = std::vector<double>(n, 1.0);
    std::vector<double> y(n);
    for (auto _ : state)
        csr.spmv(x, y);
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(csr.nnz()));
    state.counters["nnz"] = static_cast<double>(csr.nnz());
}
BENCHMARK(BM_CSR_spmv_tridiag)->RangeMultiplier(4)->Range(128, 4096)->Unit(benchmark::kMicrosecond);

// ============================================================
// compress() — excludes matrix generation via PauseTiming
// ============================================================

static void BM_compress(benchmark::State& state) {
    const auto        n      = static_cast<std::size_t>(state.range(0));
    const std::size_t target = static_cast<std::size_t>(n * n * 0.01);

    for (auto _ : state) {
        state.PauseTiming();
        // Build uncompressed COO with 20% extra duplicate insertions
        SparseCOO<double> A(n, n, static_cast<std::size_t>(target * 1.2));
        uint64_t          s = 0xDEADBEEFULL;
        for (std::size_t k = 0; k < target; ++k) {
            const auto row = random_next(s) % n, col = random_next(s) % n;
            A.set(row, col, 1.0);
            if (k % 5 == 0)  // one explicit duplicate per five base entries
                A.set(row, col, 1.0);
        }
        state.ResumeTiming();

        A.compress();
        benchmark::DoNotOptimize(A);
    }
    state.counters["target_nnz"] = static_cast<double>(target);
}
BENCHMARK(BM_compress)->RangeMultiplier(4)->Range(128, 4096)->Unit(benchmark::kMicrosecond);

// ============================================================
// Format conversions (COO already compressed; measures sort + copy)
// ============================================================

#define CONV_BENCH(Name, Expr)                                          \
    static void Name(benchmark::State& state) {                         \
        const auto n   = static_cast<std::size_t>(state.range(0));     \
        auto       coo = make_random_coo(n, 0.01);                     \
        for (auto _ : state) {                                          \
            auto result = (Expr);                                        \
            benchmark::DoNotOptimize(result);                           \
        }                                                                \
        state.counters["nnz"] = static_cast<double>(coo.nnz());        \
    }                                                                    \
    BENCHMARK(Name)->RangeMultiplier(4)->Range(128, 4096)->Unit(benchmark::kMicrosecond)

CONV_BENCH(BM_to_csr, to_csr(coo));
CONV_BENCH(BM_to_csc, to_csc(coo));
CONV_BENCH(BM_to_ell, to_ell(coo));

#undef CONV_BENCH

static void BM_to_dia(benchmark::State& state) {
    const auto n   = static_cast<std::size_t>(state.range(0));
    auto       coo = make_tridiag(n);  // DIA suits structured matrices
    for (auto _ : state) {
        auto result = to_dia(coo);
        benchmark::DoNotOptimize(result);
    }
    state.counters["nnz"] = static_cast<double>(coo.nnz());
}
BENCHMARK(BM_to_dia)->RangeMultiplier(4)->Range(128, 4096)->Unit(benchmark::kMicrosecond);

// ============================================================
// Expression templates — manual vs lazy (should compile to same work)
// ============================================================

// Baseline: manual copy + scale + set_batch + compress
static void BM_expr_manual(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto       A = make_random_coo(n, 0.01);
    auto       B = make_random_coo(n, 0.01, true, 0xCAFEBABEULL);
    for (auto _ : state) {
        SparseCOO<double> C = A;
        C.scale_inplace(2.0);
        C.set_batch(B.row_indices(), B.col_indices(), B.values());
        C.compress();
        benchmark::DoNotOptimize(C);
    }
    state.counters["nnz_a"] = static_cast<double>(A.nnz());
    state.counters["nnz_b"] = static_cast<double>(B.nnz());
}
BENCHMARK(BM_expr_manual)->RangeMultiplier(4)->Range(128, 4096)->Unit(benchmark::kMicrosecond);

// Expression template: 2*A + B (same semantics, lazy evaluation)
static void BM_expr_scale_sum(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto       A = make_random_coo(n, 0.01);
    auto       B = make_random_coo(n, 0.01, true, 0xCAFEBABEULL);
    for (auto _ : state) {
        SparseCOO<double> C = 2.0 * A + B;
        benchmark::DoNotOptimize(C);
    }
    state.counters["nnz_a"] = static_cast<double>(A.nnz());
    state.counters["nnz_b"] = static_cast<double>(B.nnz());
}
BENCHMARK(BM_expr_scale_sum)->RangeMultiplier(4)->Range(128, 4096)->Unit(benchmark::kMicrosecond);

// ============================================================
// SparseMatrix conversion overhead
// ============================================================

static void BM_sparse_matrix_coo_to_csr(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto       A = make_random_coo(n, 0.01);
    for (auto _ : state) {
        SparseMatrix<double> mat(A);
        mat.convert_to(SparseMatrix<double>::Format::CSR);
        benchmark::DoNotOptimize(mat);
    }
}
BENCHMARK(BM_sparse_matrix_coo_to_csr)->RangeMultiplier(4)->Range(128, 4096)->Unit(benchmark::kMicrosecond);

static void BM_CSR_to_COO(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto coo = make_random_coo(n, 0.01);
    auto csr = to_csr(coo);
    auto reference = to_coo(csr);
    if (reference.row_indices() != coo.row_indices() || reference.col_indices() != coo.col_indices() || reference.values() != coo.values()) {
        state.SkipWithError("CSR round-trip reference failed"); return;
    }
    for (auto _ : state) {
        auto result = to_coo(csr);
        benchmark::DoNotOptimize(result);
    }
    state.counters["nnz"] = static_cast<double>(csr.nnz());
}
BENCHMARK(BM_CSR_to_COO)->RangeMultiplier(4)->Range(128, 4096)->Unit(benchmark::kMicrosecond);

static void BM_compress_sorted(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto source = make_random_coo(n, 0.01);
    for (auto _ : state) {
        state.PauseTiming();
        auto work = source;
        state.ResumeTiming();
        work.compress();
        benchmark::DoNotOptimize(work);
    }
    state.counters["nnz"] = static_cast<double>(source.nnz());
}
BENCHMARK(BM_compress_sorted)->RangeMultiplier(4)->Range(128, 4096)->Unit(benchmark::kMicrosecond);
