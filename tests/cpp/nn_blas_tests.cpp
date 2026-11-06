// In-house BLAS (datamunge/nn/blas.hpp): every instruction-set unit this CPU
// can run must agree with a plain reference on awkward shapes, strides,
// transposes and alpha/beta, and the threaded entry points must give the same
// result for any thread count.

#include <datamunge/nn/blas.hpp>
#include <datamunge/nn/parallel.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

namespace blas = datamunge::nn::blas;
using I64      = std::int64_t;

template <typename T>
void reference_gemm(bool ta, bool tb, I64 m, I64 n, I64 k, T alpha, const T* a, I64 lda, const T* b, I64 ldb, T beta,
                    T* c, I64 ldc) {
  for (I64 i = 0; i < m; ++i)
    for (I64 j = 0; j < n; ++j) {
      double s = 0;
      for (I64 l = 0; l < k; ++l)
        s += double(ta ? a[l * lda + i] : a[i * lda + l]) * double(tb ? b[j * ldb + l] : b[l * ldb + j]);
      c[i * ldc + j] = static_cast<T>(double(alpha) * s + double(beta) * double(c[i * ldc + j]));
    }
}

template <typename T>
double tolerance(I64 k) {
  return (sizeof(T) == 4 ? 2e-5 : 1e-12) * std::max<double>(1.0, double(k));
}

/// Runs `body` once per kernel set this CPU supports, restoring auto-detection afterwards.
template <typename Body>
void for_each_isa(Body body) {
  const auto isas = blas::available_isas();
  ASSERT_FALSE(isas.empty());
  for (const auto& name : isas) {
    ASSERT_TRUE(blas::set_isa(name)) << name;
    ASSERT_EQ(blas::isa(), name);
    body(name);
  }
  ASSERT_TRUE(blas::set_isa("auto"));
}

template <typename T>
void check_gemm(const std::string& isa, std::mt19937& rng) {
  std::uniform_real_distribution<double> u(-1, 1);
  const I64 shapes[][3] = {{1, 1, 1},     {1, 1, 7},     {3, 5, 2},     {12, 32, 256}, {13, 33, 257}, {100, 17, 300},
                           {257, 65, 513}, {7, 1000, 3},  {1, 257, 513}, {64, 64, 0},   {40, 16, 27},  {1000, 48, 9}};
  for (const auto& s : shapes)
    for (bool ta : {false, true})
      for (bool tb : {false, true})
        for (int ab = 0; ab < 3; ++ab) {
          const I64 m = s[0], n = s[1], k = s[2];
          const T   alpha = ab == 0 ? T(1) : T(0.7), beta = ab == 0 ? T(0) : ab == 1 ? T(1) : T(-0.3);
          // Leading dimensions wider than the matrices catch stride bugs; the
          // padding column of C must stay untouched.
          const I64      lda = (ta ? m : k) + 3, ldb = (tb ? k : n) + 2, ldc = n + 1;
          std::vector<T> a(static_cast<std::size_t>((ta ? k : m) * lda)),
              b(static_cast<std::size_t>((tb ? n : k) * ldb)), c(static_cast<std::size_t>(m * ldc));
          for (auto& v : a) v = T(u(rng));
          for (auto& v : b) v = T(u(rng));
          for (auto& v : c) v = T(u(rng));
          std::vector<T> ref = c;
          reference_gemm<T>(ta, tb, m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, ref.data(), ldc);
          blas::gemm(ta, tb, m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
          double worst = 0;
          for (I64 i = 0; i < m; ++i)
            for (I64 j = 0; j <= n; ++j) worst = std::max(worst, std::fabs(double(c[i * ldc + j]) - double(ref[i * ldc + j])));
          EXPECT_LE(worst, tolerance<T>(k)) << isa << " m=" << m << " n=" << n << " k=" << k << " ta=" << ta
                                            << " tb=" << tb << " alpha=" << alpha << " beta=" << beta;
        }
}

}  // namespace

TEST(NnBlas, ReportsAnIsaAndRejectsUnknownOnes) {
  EXPECT_FALSE(blas::isa().empty());
  EXPECT_FALSE(blas::set_isa("no-such-isa"));
  EXPECT_TRUE(blas::set_isa("generic"));
  EXPECT_EQ(blas::isa(), "generic");
  EXPECT_TRUE(blas::set_isa("auto"));
  EXPECT_EQ(blas::isa(), blas::available_isas().front());
  const std::string backend = datamunge::nn::gemm_backend();
  EXPECT_TRUE(backend == "inhouse-" + blas::isa() || backend == "openblas" || backend == "cblas" ||
              backend == "accelerate")
      << backend;
}

TEST(NnBlas, GemmMatchesReferenceOnEveryIsa) {
  std::mt19937 rng(7);
  for_each_isa([&](const std::string& isa) {
    check_gemm<float>(isa, rng);
    check_gemm<double>(isa, rng);
  });
}

TEST(NnBlas, GemmIsIndependentOfTheThreadCount) {
  std::mt19937                           rng(11);
  std::uniform_real_distribution<double> u(-1, 1);
  const I64                              m = 301, n = 203, k = 1500;
  std::vector<float>                     a(static_cast<std::size_t>(m * k)), b(static_cast<std::size_t>(k * n));
  for (auto& v : a) v = float(u(rng));
  for (auto& v : b) v = float(u(rng));
  const int saved = datamunge::nn::get_num_threads();
  datamunge::nn::set_num_threads(1);
  std::vector<float> c1(static_cast<std::size_t>(m * n), 0.f);
  blas::gemm(false, false, m, n, k, 1.f, a.data(), k, b.data(), n, 0.f, c1.data(), n);
  for (int threads : {2, 3, 8}) {
    datamunge::nn::set_num_threads(threads);
    std::vector<float> c(static_cast<std::size_t>(m * n), 0.f);
    blas::gemm(false, false, m, n, k, 1.f, a.data(), k, b.data(), n, 0.f, c.data(), n);
    EXPECT_EQ(c, c1) << threads << " threads";
    // The depth split (short M and N, long K: a filter gradient) reduces partials in a fixed order too.
    const I64          dm = 48, dn = 32, dk = 60000;
    std::vector<float> da(static_cast<std::size_t>(dk * dm)), db(static_cast<std::size_t>(dk * dn));
    for (auto& v : da) v = float(u(rng));
    for (auto& v : db) v = float(u(rng));
    std::vector<float> d(static_cast<std::size_t>(dm * dn), 0.f), d1(static_cast<std::size_t>(dm * dn), 0.f);
    datamunge::nn::set_num_threads(1);
    blas::gemm(true, false, dm, dn, dk, 1.f, da.data(), dm, db.data(), dn, 0.f, d1.data(), dn);
    datamunge::nn::set_num_threads(threads);
    blas::gemm(true, false, dm, dn, dk, 1.f, da.data(), dm, db.data(), dn, 0.f, d.data(), dn);
    EXPECT_EQ(d, d1) << threads << " threads (depth split)";
  }
  datamunge::nn::set_num_threads(saved);
}

namespace {

/// Threaded GEMM over several column and depth blocks (the packed-B block is
/// shared by the threads), with edge tiles, padded strides, every transpose
/// and beta != 0: equal to the reference and bit-identical for every thread count.
template <typename T>
void check_threaded_gemm(std::mt19937& rng) {
  std::uniform_real_distribution<double> u(-1, 1);
  // Blocks are NC = 2048 (float) / 1024 (double) columns and at most KC = 512
  // deep with AVX-512 (256 otherwise), k cut into equal blocks. At most one B
  // panel (n <= 32 float, 16 double) reads A in place instead of packing it.
  const I64 shapes[][3] = {{37, 2100, 600}, {700, 130, 1100}, {2000, 33, 64}, {150, 1100, 300},
                           {11, 4100, 64},  {3001, 16, 600},  {1001, 30, 520}};
  const int saved       = datamunge::nn::get_num_threads();
  for (const auto& s : shapes)
    for (bool ta : {false, true})
      for (bool tb : {false, true}) {
        const I64      m = s[0], n = s[1], k = s[2];
        const I64      lda = (ta ? m : k) + 3, ldb = (tb ? k : n) + 2, ldc = n + 1;
        std::vector<T> a(static_cast<std::size_t>((ta ? k : m) * lda)),
            b(static_cast<std::size_t>((tb ? n : k) * ldb)), c0(static_cast<std::size_t>(m * ldc));
        for (auto& v : a) v = T(u(rng));
        for (auto& v : b) v = T(u(rng));
        for (auto& v : c0) v = T(u(rng));
        const T        alpha = T(0.7), beta = T(-0.3);
        std::vector<T> ref = c0;
        reference_gemm<T>(ta, tb, m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, ref.data(), ldc);
        std::vector<T> first;
        for (int threads : {1, 2, 3, 7, 16}) {
          datamunge::nn::set_num_threads(threads);
          std::vector<T> c = c0;
          blas::gemm(ta, tb, m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
          if (first.empty()) {
            double worst = 0;
            for (I64 i = 0; i < m; ++i)
              for (I64 j = 0; j <= n; ++j)
                worst = std::max(worst, std::fabs(double(c[i * ldc + j]) - double(ref[i * ldc + j])));
            EXPECT_LE(worst, tolerance<T>(k)) << "m=" << m << " n=" << n << " k=" << k << " ta=" << ta << " tb=" << tb;
            first = c;
          } else {
            EXPECT_EQ(c, first) << threads << " threads, m=" << m << " n=" << n << " k=" << k << " ta=" << ta
                                << " tb=" << tb;
          }
        }
      }
  datamunge::nn::set_num_threads(saved);
}

}  // namespace

TEST(NnBlas, ThreadedGemmSharesPackedBlocksAndIsIndependentOfTheThreadCount) {
  std::mt19937 rng(13);
  check_threaded_gemm<float>(rng);
  check_threaded_gemm<double>(rng);
}

TEST(NnBlas, LargeGemvMatchesReferenceAndIsIndependentOfTheThreadCount) {
  std::mt19937                           rng(17);
  std::uniform_real_distribution<double> u(-1, 1);
  const I64                              m = 2000, n = 1500, lda = n + 3;
  std::vector<float>                     a(static_cast<std::size_t>(m * lda));
  for (auto& v : a) v = float(u(rng));
  const int saved = datamunge::nn::get_num_threads();
  for (bool trans : {false, true})
    for (I64 inc : {1, 2})
      for (float beta : {0.f, 0.5f}) {
        const I64          xlen = trans ? m : n, ylen = trans ? n : m;
        std::vector<float> x(static_cast<std::size_t>(xlen * inc)), y0(static_cast<std::size_t>(ylen * inc));
        for (auto& v : x) v = float(u(rng));
        for (auto& v : y0) v = float(u(rng));
        // beta = 0 must ignore y (even NaN).
        if (beta == 0.f) y0[0] = std::numeric_limits<float>::quiet_NaN();
        std::vector<float> first;
        for (int threads : {1, 3, 8}) {
          datamunge::nn::set_num_threads(threads);
          std::vector<float> y = y0;
          blas::gemv(trans, m, n, 1.5f, a.data(), lda, x.data(), inc, beta, y.data(), inc);
          if (first.empty()) {
            first = y;
            for (I64 i = 0; i < ylen; ++i) {
              double s = 0;
              for (I64 j = 0; j < xlen; ++j) s += double(trans ? a[j * lda + i] : a[i * lda + j]) * double(x[j * inc]);
              const double ref = 1.5 * s + (beta == 0.f ? 0.0 : double(beta) * double(y0[i * inc]));
              ASSERT_NEAR(y[i * inc], ref, 1e-4 * std::sqrt(double(xlen))) << trans << " " << inc << " " << i;
            }
            if (inc > 1) EXPECT_EQ(y[1], y0[1]) << "strided element touched";
          } else {
            EXPECT_EQ(y, first) << threads << " threads, trans " << trans;
          }
        }
      }
  datamunge::nn::set_num_threads(saved);
}

TEST(NnBlas, LevelOneRoutinesAreIndependentOfTheThreadCount) {
  // DOT adds fixed-chunk partial sums in order on any thread count (one thread included); AXPY
  // and SCAL split their outputs.
  std::mt19937                           rng(23);
  std::uniform_real_distribution<double> u(-1, 1);
  const int                              saved = datamunge::nn::get_num_threads();
  for (I64 n : {I64{9000}, I64{70001}, I64{600000}})
    for (I64 inc : {1, 2}) {
      std::vector<double> x(static_cast<std::size_t>(n * inc)), y(static_cast<std::size_t>(n * inc));
      for (auto& v : x) v = u(rng);
      for (auto& v : y) v = u(rng);
      const std::vector<float> xf(x.begin(), x.end()), yf(y.begin(), y.end());
      double                   dot1 = 0;
      float                    sdot1 = 0;
      std::vector<double>      z1;
      for (int threads : {1, 3, 8}) {
        datamunge::nn::set_num_threads(threads);
        const double d  = blas::dot(n, x.data(), inc, y.data(), inc);
        const float  sd = blas::dot(n, xf.data(), inc, yf.data(), inc);
        std::vector<double> z = y;
        blas::axpy(n, 0.75, x.data(), inc, z.data(), inc);
        blas::scal(n, 1.25, z.data(), inc);
        if (threads == 1) {
          dot1 = d, sdot1 = sd, z1 = z;
          continue;
        }
        EXPECT_EQ(d, dot1) << n << " " << threads << " threads";
        EXPECT_EQ(sd, sdot1) << n << " " << threads << " threads";
        EXPECT_EQ(z, z1) << n << " " << threads << " threads";
      }
    }
  datamunge::nn::set_num_threads(saved);
}

TEST(NnBlas, GemvDotAxpyScalMatchReferenceWithStrides) {
  std::mt19937                           rng(3);
  std::uniform_real_distribution<double> u(-1, 1);
  for_each_isa([&](const std::string& isa) {
    for (I64 m : {1, 7, 300})
      for (I64 n : {1, 5, 1000})
        for (bool trans : {false, true})
          for (I64 inc : {1, 3}) {
            const I64           lda = n + 2, ylen = trans ? n : m, xlen = trans ? m : n;
            std::vector<double> a(static_cast<std::size_t>(m * lda)), x(static_cast<std::size_t>(xlen * inc)),
                y(static_cast<std::size_t>(ylen * inc));
            for (auto& v : a) v = u(rng);
            for (auto& v : x) v = u(rng);
            for (auto& v : y) v = u(rng);
            std::vector<double> ref = y;
            for (I64 i = 0; i < ylen; ++i) {
              double s = 0;
              for (I64 j = 0; j < xlen; ++j) s += (trans ? a[j * lda + i] : a[i * lda + j]) * x[j * inc];
              ref[i * inc] = 0.5 * s + 0.25 * ref[i * inc];
            }
            blas::gemv(trans, m, n, 0.5, a.data(), lda, x.data(), inc, 0.25, y.data(), inc);
            for (I64 i = 0; i < ylen * inc; ++i) EXPECT_NEAR(y[i], ref[i], 1e-10 * xlen) << isa << " gemv";
            // float path too
            std::vector<float> af(a.begin(), a.end()), xf(x.begin(), x.end()), yf(y.begin(), y.end()),
                reff(static_cast<std::size_t>(ylen * inc));
            for (I64 i = 0; i < ylen; ++i) {
              double s = 0;
              for (I64 j = 0; j < xlen; ++j) s += double(trans ? af[j * lda + i] : af[i * lda + j]) * double(xf[j * inc]);
              reff[i * inc] = float(s);
            }
            blas::gemv(trans, m, n, 1.f, af.data(), lda, xf.data(), inc, 0.f, yf.data(), inc);
            for (I64 i = 0; i < ylen; ++i) EXPECT_NEAR(yf[i * inc], reff[i * inc], 1e-4 * xlen) << isa << " sgemv";
          }
    for (I64 n : {0, 1, 17, 1000, 300000})
      for (I64 inc : {1, 2}) {
        std::vector<double> x(static_cast<std::size_t>(n * inc)), y(static_cast<std::size_t>(n * inc));
        for (auto& v : x) v = u(rng);
        for (auto& v : y) v = u(rng);
        double ref = 0;
        for (I64 i = 0; i < n; ++i) ref += x[i * inc] * y[i * inc];
        EXPECT_NEAR(blas::dot(n, x.data(), inc, y.data(), inc), ref, 1e-9 * std::max<double>(1, n)) << isa;
        std::vector<float> xf(x.begin(), x.end()), yf(y.begin(), y.end());
        EXPECT_NEAR(blas::dot(n, xf.data(), inc, yf.data(), inc), ref, 2e-3 * std::max<double>(1, n)) << isa;
        std::vector<double> z = y;
        blas::axpy(n, 1.5, x.data(), inc, z.data(), inc);
        blas::scal(n, -2.0, z.data(), inc);
        for (I64 i = 0; i < n; ++i) {
          // The kernels may fuse the multiply-add (one rounding instead of two).
          EXPECT_NEAR(z[i * inc], -2.0 * (y[i * inc] + 1.5 * x[i * inc]), 1e-15) << isa;
          if (inc > 1) EXPECT_EQ(z[i * inc + 1], y[i * inc + 1]) << isa << " strided element touched";
        }
      }
  });
}
