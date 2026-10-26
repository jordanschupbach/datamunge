// Tests for datamunge::nn FFTs (fft.hpp) and audio features / WAV I/O (audio.hpp).
//
// Reference values marked "TF" were printed by a TensorFlow 2.20 C++ build
// (RFFT/IRFFT kernels and tflow's common/data/audio.h MelWeights).

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <datamunge/nn/audio.hpp>
#include <datamunge/nn/fft.hpp>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <vector>

namespace nn = datamunge::nn;
using nn::ComplexTensor;
using nn::DType;
using nn::Tensor;
using cd = std::complex<double>;

namespace {

using Inputs     = std::vector<Tensor>;
constexpr double kPi = 3.14159265358979323846;

Tensor rnd(const nn::Shape& shape, std::uint64_t seed, DType dt = DType::Float64) {
  nn::Generator gen(seed);
  return nn::normal(shape, 0.0, 1.0, gen, dt);
}

Tensor leaf(const Tensor& t) {
  Tensor l = t.detach();
  l.requires_grad_(true);
  return l;
}

/// Both parts stacked on a new leading axis (a real tensor for gradcheck).
Tensor pack(const ComplexTensor& c) { return nn::stack({c.real, c.imag}, 0); }

/// Naive DFT of each row of x (rows x n), sign -1 forward / +1 inverse, unnormalized.
std::vector<cd> naive_dft(const std::vector<cd>& x, std::int64_t rows, std::int64_t n, double sign) {
  std::vector<cd> y(x.size());
  for (std::int64_t r = 0; r < rows; ++r)
    for (std::int64_t k = 0; k < n; ++k) {
      cd s = 0;
      for (std::int64_t j = 0; j < n; ++j)
        s += x[r * n + j] * std::polar(1.0, sign * 2 * kPi * double((j * k) % n) / double(n));
      y[r * n + k] = s;
    }
  return y;
}

std::vector<cd> to_complex(const ComplexTensor& c) {
  const auto re = c.real.to_vector<double>(), im = c.imag.to_vector<double>();
  std::vector<cd> out(re.size());
  for (std::size_t i = 0; i < re.size(); ++i) out[i] = cd(re[i], im[i]);
  return out;
}

double max_err(const std::vector<cd>& a, const std::vector<cd>& b) {
  double e = 0;
  for (std::size_t i = 0; i < a.size(); ++i) e = std::max(e, std::abs(a[i] - b[i]));
  return e;
}

double max_err(const Tensor& a, const Tensor& b) { return nn::max(nn::abs(a.to(DType::Float64) - b.to(DType::Float64))).item(); }

}  // namespace

// ---- FFT values ------------------------------------------------------------------------------------

TEST(NnFft, MatchesNaiveDftForManyLengths) {
  for (std::int64_t n : {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 31, 64, 97, 100, 128, 243, 1000}) {
    const ComplexTensor x = nn::make_complex(rnd({3, n}, 1 + n), rnd({3, n}, 1000 + n));
    const auto          xv = to_complex(x);
    const auto          f = to_complex(nn::fft(x)), b = to_complex(nn::ifft(x));
    auto                ref_f = naive_dft(xv, 3, n, -1), ref_b = naive_dft(xv, 3, n, +1);
    for (auto& v : ref_b) v /= double(n);
    EXPECT_LT(max_err(f, ref_f), 1e-10 * n) << "fft n=" << n;
    EXPECT_LT(max_err(b, ref_b), 1e-12 * n) << "ifft n=" << n;
  }
}

TEST(NnFft, LargePrimeAndPowerOfTwoAccuracy) {
  for (std::int64_t n : {1009, 1024}) {
    std::vector<cd> x(n);
    for (std::int64_t i = 0; i < n; ++i) x[i] = cd(std::sin(0.3 * i), std::cos(1.7 * i));
    auto y = x;
    nn::fft_inplace(y.data(), n);
    EXPECT_LT(max_err(y, naive_dft(x, 1, n, -1)), 1e-9) << n;
    nn::fft_inplace(y.data(), n, true);
    for (auto& v : y) v /= double(n);
    EXPECT_LT(max_err(y, x), 1e-12) << n;
  }
}

/// max |a - b| / max |b|.
double rel_err(const std::vector<cd>& a, const std::vector<cd>& b) {
  double scale = 0;
  for (const cd& v : b) scale = std::max(scale, std::abs(v));
  return max_err(a, b) / std::max(scale, 1e-300);
}

/// Mixed-radix (2, 3, 4, 5, generic odd) and Bluestein lengths against the naive DFT.
const std::vector<std::int64_t> kMixedLengths = {
    1,   2,    3,    6,    10,   12,   14,   49,   100,  121,  169,  243,  400,  625,  1000,
    1001, 1024, 768, 1536, 3072, 61,   122,  3721, 103,  206,  107,  214,  97,   1009, 2018, 4099,
    8198};

TEST(NnFft, MixedRadixMatchesNaiveDftToRelative1e12) {
  for (const std::int64_t n : kMixedLengths) {
    const std::int64_t  rows = n <= 1024 ? 2 : 1;
    const ComplexTensor x    = nn::make_complex(rnd({rows, n}, 7 + n), rnd({rows, n}, 70000 + n));
    const auto          xv   = to_complex(x);
    const auto          ref_f = naive_dft(xv, rows, n, -1);
    auto                ref_b = naive_dft(xv, rows, n, +1);
    for (auto& v : ref_b) v /= double(n);
    EXPECT_LT(rel_err(to_complex(nn::fft(x)), ref_f), 1e-12) << "fft n=" << n;
    EXPECT_LT(rel_err(to_complex(nn::ifft(x)), ref_b), 1e-12) << "ifft n=" << n;
    // real transforms (even lengths use the half-length packed complex FFT)
    const Tensor     xr = x.real;
    std::vector<cd>  xrv(xv.size());
    for (std::size_t i = 0; i < xv.size(); ++i) xrv[i] = cd(xv[i].real(), 0.0);
    const auto       full = naive_dft(xrv, rows, n, -1);
    const std::int64_t m = n / 2 + 1;
    std::vector<cd>  half;
    for (std::int64_t r = 0; r < rows; ++r)
      for (std::int64_t k = 0; k < m; ++k) half.push_back(full[r * n + k]);
    const ComplexTensor X = nn::rfft(xr);
    EXPECT_LT(rel_err(to_complex(X), half), 1e-12) << "rfft n=" << n;
    EXPECT_LT(max_err(nn::irfft(X, n), xr), 1e-12 * std::sqrt(double(n)) + 1e-13) << "irfft n=" << n;
  }
}

TEST(NnFft, RawKernelMixedRadixAccuracy) {
  for (std::int64_t n : {12, 400, 1000, 3072, 625, 1001, 4099, 8198}) {
    std::vector<cd> x(n);
    for (std::int64_t i = 0; i < n; ++i) x[i] = cd(std::sin(0.3 * i), std::cos(1.7 * i));
    auto y = x;
    nn::fft_inplace(y.data(), n);
    EXPECT_LT(rel_err(y, naive_dft(x, 1, n, -1)), 1e-12) << n;
    nn::fft_inplace(y.data(), n, true);
    for (auto& v : y) v /= double(n);
    EXPECT_LT(max_err(y, x), 1e-13) << n;
  }
}

TEST(NnFft, DeterministicAcrossRunsAndThreadCounts) {
  const int saved = nn::get_num_threads();
  for (std::int64_t n : {400, 1000, 1024, 3072, 4099}) {
    const Tensor x = rnd({64, n}, 99 + n, DType::Float32);
    nn::set_num_threads(1);
    const ComplexTensor a = nn::rfft(x);
    const ComplexTensor c = nn::fft(nn::make_complex(x, x));
    nn::set_num_threads(0);
    const ComplexTensor b = nn::rfft(x);
    const ComplexTensor d = nn::fft(nn::make_complex(x, x));
    EXPECT_EQ(a.real.to_vector<float>(), b.real.to_vector<float>()) << n;
    EXPECT_EQ(a.imag.to_vector<float>(), b.imag.to_vector<float>()) << n;
    EXPECT_EQ(c.real.to_vector<float>(), d.real.to_vector<float>()) << n;
    EXPECT_EQ(c.imag.to_vector<float>(), d.imag.to_vector<float>()) << n;
  }
  nn::set_num_threads(saved);
}

TEST(NnFft, RoundTripsAndParseval) {
  for (std::int64_t n : {8, 13, 30}) {
    const ComplexTensor x = nn::make_complex(rnd({2, 3, n}, 5), rnd({2, 3, n}, 6));
    const ComplexTensor y = nn::fft(x);
    const ComplexTensor z = nn::ifft(y);
    EXPECT_LT(max_err(z.real, x.real), 1e-12);
    EXPECT_LT(max_err(z.imag, x.imag), 1e-12);
    // sum |x|^2 = sum |X|^2 / n per row
    const Tensor ex = nn::sum(nn::complex_abs_squared(x), {-1});
    const Tensor ey = nn::sum(nn::complex_abs_squared(y), {-1}) / double(n);
    EXPECT_LT(max_err(ex, ey), 1e-10);
    // real round trip for even and odd lengths
    const Tensor r = rnd({4, n}, 7);
    EXPECT_LT(max_err(nn::irfft(nn::rfft(r), n), r), 1e-12) << n;
  }
}

TEST(NnFft, RfftMatchesFullFftHalfSpectrum) {
  for (std::int64_t n : {1, 2, 3, 6, 7, 16, 25, 64}) {
    const Tensor        x = rnd({2, n}, 40 + n);
    const ComplexTensor h = nn::rfft(x);
    ASSERT_EQ(h.shape(), (nn::Shape{2, n / 2 + 1}));
    const ComplexTensor f = nn::complex_slice_axis(nn::fft(nn::make_complex(x)), -1, 0, n / 2 + 1);
    EXPECT_LT(max_err(h.real, f.real), 1e-12) << n;
    EXPECT_LT(max_err(h.imag, f.imag), 1e-12) << n;
  }
}

TEST(NnFft, RfftIrfftMatchTensorFlow) {
  // TF: x_i = sin(0.7 i) + 0.1 i, i < 12 (float32)
  std::vector<float> xs(12);
  for (int i = 0; i < 12; ++i) xs[i] = float(std::sin(0.7 * i) + 0.1 * i);
  const Tensor x = nn::from_vector(xs);
  const std::vector<double> r10 = {4.50860023, 0, 0.96597445, -2.83205509, -1.00239813, 1.37357593,
                                   -0.902311385, 0.654199362, -0.878860235, 0.285141349, -0.87340951, 0};
  const std::vector<double> r16 = {8.25375557, 0, -0.619702339, 0.68130368, -1.99603558, -6.68182182,
                                   1.00710952, 1.38399482, -1.66713905, 1.15867662, -1.06178844, -1.13690162,
                                   0.863525927, -0.453797579, -0.00055205822, 1.10035992, -1.30459118, 0};
  const std::vector<double> r7  = {3.70550585, 0, -1.7125206, -1.99022985, -0.124158986, -0.20592396,
                                   -0.016073361, -0.0480614342};
  auto check = [](const ComplexTensor& c, const std::vector<double>& ref, const char* what) {
    ASSERT_EQ(c.real.numel() * 2, std::int64_t(ref.size())) << what;
    const auto re = c.real.to_vector<double>(), im = c.imag.to_vector<double>();
    for (std::size_t k = 0; k < re.size(); ++k) {
      EXPECT_NEAR(re[k], ref[2 * k], 2e-6) << what << " bin " << k;
      EXPECT_NEAR(im[k], ref[2 * k + 1], 2e-6) << what << " bin " << k;
    }
  };
  const ComplexTensor c10 = nn::rfft(x, 10), c16 = nn::rfft(x, 16);
  check(c10, r10, "rfft crop to 10");
  check(c16, r16, "rfft pad to 16");
  check(nn::rfft(nn::slice_axis(x, 0, 0, 7)), r7, "rfft odd 7");
  const std::vector<double> i10 = {0.0640432164, 1.70683253, 1.77686906, 0.379956454, -0.449994564,
                                   0.121438026,  2.50618744, 2.32552028, -0.301121682, 0.124024771};
  const std::vector<double> i7  = {0.375875771, 1.28108299, 1.78525543, 0.749907315, -0.201458514, -0.425313562,
                                   0.943251014};
  const auto a = nn::irfft(c16, 10).to_vector<double>(), b = nn::irfft(c10, 7).to_vector<double>();
  ASSERT_EQ(a.size(), 10u);
  ASSERT_EQ(b.size(), 7u);
  for (int i = 0; i < 10; ++i) EXPECT_NEAR(a[i], i10[i], 2e-6) << i;
  for (int i = 0; i < 7; ++i) EXPECT_NEAR(b[i], i7[i], 2e-6) << i;
  EXPECT_EQ(nn::irfft(c16).shape(), (nn::Shape{16}));  // default 2 (bins - 1)
}

TEST(NnFft, IrfftIgnoresImaginaryDcAndNyquist) {
  const Tensor        re = rnd({5}, 3), im = rnd({5}, 4);
  const ComplexTensor a  = nn::make_complex(re, im);
  Tensor              im2 = im.clone();
  {
    nn::NoGradGuard g;
    im2.data_ptr<double>()[0] = 7;
    im2.data_ptr<double>()[4] = -3;
  }
  EXPECT_LT(max_err(nn::irfft(a, 8), nn::irfft(nn::make_complex(re, im2), 8)), 1e-14);
}

TEST(NnFft, TwoDimensional) {
  const std::int64_t  h = 5, w = 6;
  const ComplexTensor x = nn::make_complex(rnd({2, h, w}, 8), rnd({2, h, w}, 9));
  const auto          xv = to_complex(x);
  std::vector<cd>     ref(xv.size());
  for (int b = 0; b < 2; ++b)
    for (int u = 0; u < h; ++u)
      for (int v = 0; v < w; ++v) {
        cd s = 0;
        for (int i = 0; i < h; ++i)
          for (int j = 0; j < w; ++j)
            s += xv[(b * h + i) * w + j] * std::polar(1.0, -2 * kPi * (double(u * i) / h + double(v * j) / w));
        ref[(b * h + u) * w + v] = s;
      }
  EXPECT_LT(max_err(to_complex(nn::fft2d(x)), ref), 1e-10);
  const ComplexTensor back = nn::ifft2d(nn::fft2d(x));
  EXPECT_LT(max_err(back.real, x.real), 1e-12);
  const Tensor        r  = rnd({3, h, w}, 10);
  const ComplexTensor rf = nn::rfft2d(r);
  ASSERT_EQ(rf.shape(), (nn::Shape{3, h, w / 2 + 1}));
  const ComplexTensor full = nn::complex_slice_axis(nn::fft2d(nn::make_complex(r)), -1, 0, w / 2 + 1);
  EXPECT_LT(max_err(rf.real, full.real), 1e-10);
  EXPECT_LT(max_err(rf.imag, full.imag), 1e-10);
  EXPECT_LT(max_err(nn::irfft2d(rf, {h, w}), r), 1e-12);
}

TEST(NnFft, Float32UsesDoubleInternally) {
  const Tensor        x = rnd({4096}, 11, DType::Float32);
  const ComplexTensor y = nn::rfft(x);
  EXPECT_EQ(y.dtype(), DType::Float32);
  const ComplexTensor y64 = nn::rfft(x.to(DType::Float64));
  EXPECT_LT(max_err(y.real, y64.real), 1e-4);
  EXPECT_LT(max_err(nn::irfft(y, 4096), x), 1e-6);
}

TEST(NnFft, FreqHelpersAndShift) {
  EXPECT_EQ(nn::fftfreq(5).to_vector<double>(), (std::vector<double>{0, 0.2, 0.4, -0.4, -0.2}));
  EXPECT_EQ(nn::fftfreq(4, 0.5).to_vector<double>(), (std::vector<double>{0, 0.5, -1, -0.5}));
  EXPECT_EQ(nn::rfftfreq(4).to_vector<double>(), (std::vector<double>{0, 0.25, 0.5}));
  EXPECT_EQ(nn::fftshift(nn::arange(0, 5)).to_vector<double>(), (std::vector<double>{3, 4, 0, 1, 2}));
  EXPECT_EQ(nn::fftshift(nn::arange(0, 4)).to_vector<double>(), (std::vector<double>{2, 3, 0, 1}));
  EXPECT_EQ(nn::next_power_of_two(1), 1);
  EXPECT_EQ(nn::next_power_of_two(129), 256);
}

TEST(NnFft, ComplexHelpers) {
  const ComplexTensor a = nn::make_complex(nn::tensor({3, 0, -1}, {}, DType::Float64), nn::tensor({4, 0, 1}, {}, DType::Float64));
  const ComplexTensor b = nn::make_complex(nn::tensor({1, 2, 0.5}, {}, DType::Float64), nn::tensor({-1, 1, 2}, {}, DType::Float64));
  EXPECT_EQ(nn::complex_abs(a).to_vector<double>()[0], 5);
  EXPECT_EQ(nn::complex_abs(a).to_vector<double>()[1], 0);
  EXPECT_NEAR(nn::complex_angle(a).to_vector<double>()[2], 3 * kPi / 4, 1e-15);
  const auto p = to_complex(nn::complex_mul(a, b)), q = to_complex(nn::complex_div(a, b));
  const auto av = to_complex(a), bv = to_complex(b);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(std::abs(p[i] - av[i] * bv[i]), 0, 1e-14);
    EXPECT_NEAR(std::abs(q[i] - av[i] / bv[i]), 0, 1e-14);
  }
  const auto e = to_complex(nn::complex_exp(b));
  for (int i = 0; i < 3; ++i) EXPECT_NEAR(std::abs(e[i] - std::exp(bv[i])), 0, 1e-13);
  const auto pol = to_complex(nn::complex_from_polar(nn::complex_abs(b), nn::complex_angle(b)));
  for (int i = 0; i < 3; ++i) EXPECT_NEAR(std::abs(pol[i] - bv[i]), 0, 1e-14);
  // gradients of |z| and arg(z) at z = 0 are finite (0)
  const Tensor re = leaf(nn::zeros({2}, DType::Float64)), im = leaf(nn::zeros({2}, DType::Float64));
  const auto   g  = nn::grad(nn::sum(nn::complex_abs(nn::make_complex(re, im)) + nn::complex_angle(nn::make_complex(re, im))), {re, im});
  EXPECT_EQ(g[0].to_vector<double>(), (std::vector<double>{0, 0}));
  EXPECT_EQ(g[1].to_vector<double>(), (std::vector<double>{0, 0}));
}

// ---- convolution --------------------------------------------------------------------------------

TEST(NnFft, ConvolutionsMatchDirectSums) {
  const Tensor a = rnd({2, 3, 37}, 12), b = rnd({3, 11}, 13);
  const Tensor full = nn::fft_convolve(a, b);
  ASSERT_EQ(full.shape(), (nn::Shape{2, 3, 47}));
  const Tensor causal = nn::fft_causal_conv(a, b);
  ASSERT_EQ(causal.shape(), (nn::Shape{2, 3, 37}));
  const auto av = a.to_vector<double>(), bv = b.to_vector<double>(), fv = full.to_vector<double>(),
             cv = causal.to_vector<double>();
  double worst = 0;
  for (int n = 0; n < 2; ++n)
    for (int c = 0; c < 3; ++c)
      for (int t = 0; t < 47; ++t) {
        double s = 0;
        for (int k = 0; k < 11; ++k)
          if (t - k >= 0 && t - k < 37) s += bv[c * 11 + k] * av[(n * 3 + c) * 37 + t - k];
        worst = std::max(worst, std::abs(s - fv[(n * 3 + c) * 47 + t]));
        if (t < 37) worst = std::max(worst, std::abs(s - cv[(n * 3 + c) * 37 + t]));
      }
  EXPECT_LT(worst, 1e-12);
  // kernel longer than the signal: taps beyond L are ignored
  const Tensor u = rnd({16}, 14), k = rnd({40}, 15);
  EXPECT_LT(max_err(nn::fft_causal_conv(u, k), nn::fft_causal_conv(u, nn::slice_axis(k, 0, 0, 16))), 1e-13);
  // circular convolution
  const Tensor x = rnd({9}, 16), y = rnd({9}, 17);
  const auto   xc = x.to_vector<double>(), yc = y.to_vector<double>(), zc = nn::fft_circular_convolve(x, y).to_vector<double>();
  for (int t = 0; t < 9; ++t) {
    double s = 0;
    for (int j = 0; j < 9; ++j) s += xc[j] * yc[(t - j + 9) % 9];
    EXPECT_NEAR(zc[t], s, 1e-12);
  }
}

// ---- gradients ----------------------------------------------------------------------------------

TEST(NnFftGrad, ComplexTransforms) {
  for (std::int64_t n : {8, 7}) {
    Inputs in{leaf(rnd({2, n}, 20)), leaf(rnd({2, n}, 21))};
    auto   r = nn::gradcheck([](const Inputs& v) { return pack(nn::fft(nn::make_complex(v[0], v[1]))); }, in);
    EXPECT_TRUE(r.ok) << "fft " << n << ": " << r.message;
    r = nn::gradcheck([](const Inputs& v) { return pack(nn::ifft(nn::make_complex(v[0], v[1]))); }, in);
    EXPECT_TRUE(r.ok) << "ifft " << n << ": " << r.message;
  }
  Inputs in2{leaf(rnd({3, 4}, 22)), leaf(rnd({3, 4}, 23))};
  const auto r = nn::gradcheck([](const Inputs& v) { return pack(nn::fft2d(nn::make_complex(v[0], v[1]))); }, in2);
  EXPECT_TRUE(r.ok) << "fft2d: " << r.message;
}

TEST(NnFftGrad, RealTransforms) {
  struct Case { std::int64_t n, fft_length; };
  for (const Case c : {Case{8, -1}, Case{7, -1}, Case{6, 10}, Case{9, 4}, Case{1, -1}, Case{5, 5}}) {
    auto r = nn::gradcheck([&](const Inputs& v) { return pack(nn::rfft(v[0], c.fft_length)); }, {leaf(rnd({2, c.n}, 30 + c.n))});
    EXPECT_TRUE(r.ok) << "rfft " << c.n << "/" << c.fft_length << ": " << r.message;
  }
  for (const Case c : {Case{5, -1}, Case{5, 9}, Case{5, 6}, Case{4, 12}, Case{6, 4}}) {
    auto r = nn::gradcheck([&](const Inputs& v) { return nn::irfft(nn::make_complex(v[0], v[1]), c.fft_length); },
                           {leaf(rnd({2, c.n}, 40)), leaf(rnd({2, c.n}, 41))});
    EXPECT_TRUE(r.ok) << "irfft " << c.n << "/" << c.fft_length << ": " << r.message;
  }
  auto r = nn::gradcheck([](const Inputs& v) { return pack(nn::rfft2d(v[0], {4, 6})); }, {leaf(rnd({2, 3, 5}, 42))});
  EXPECT_TRUE(r.ok) << "rfft2d: " << r.message;
  r = nn::gradcheck([](const Inputs& v) { return nn::irfft2d(nn::make_complex(v[0], v[1]), {3, 6}); },
                    {leaf(rnd({3, 4}, 43)), leaf(rnd({3, 4}, 44))});
  EXPECT_TRUE(r.ok) << "irfft2d: " << r.message;
}

TEST(NnFftGrad, ConvolutionAndComplexHelpers) {
  auto r = nn::gradcheck([](const Inputs& v) { return nn::fft_convolve(v[0], v[1]); }, {leaf(rnd({2, 7}, 50)), leaf(rnd({5}, 51))});
  EXPECT_TRUE(r.ok) << "fft_convolve: " << r.message;
  r = nn::gradcheck([](const Inputs& v) { return nn::fft_causal_conv(v[0], v[1]); }, {leaf(rnd({2, 3, 9}, 52)), leaf(rnd({3, 9}, 53))});
  EXPECT_TRUE(r.ok) << "fft_causal_conv: " << r.message;
  r = nn::gradcheck(
      [](const Inputs& v) {
        const ComplexTensor z = nn::make_complex(v[0], v[1]);
        return nn::concat({nn::complex_abs(z), nn::complex_angle(z), nn::flatten(pack(nn::complex_div(z, nn::complex_exp(z))))}, 0);
      },
      {leaf(rnd({4}, 54)), leaf(rnd({4}, 55))});
  EXPECT_TRUE(r.ok) << "complex helpers: " << r.message;
}

TEST(NnFftGrad, SecondOrder) {
  // nonlinear in x so the second derivative is not trivially zero
  auto r = nn::gradgradcheck([](const Inputs& v) { return nn::complex_abs_squared(nn::rfft(v[0], 10)); }, {leaf(rnd({2, 7}, 60))});
  EXPECT_TRUE(r.ok) << "rfft: " << r.message;
  r = nn::gradgradcheck([](const Inputs& v) { return nn::square(nn::irfft(nn::make_complex(v[0], v[1]), 7)); },
                        {leaf(rnd({5}, 61)), leaf(rnd({5}, 62))});
  EXPECT_TRUE(r.ok) << "irfft: " << r.message;
  r = nn::gradgradcheck([](const Inputs& v) { return nn::complex_abs_squared(nn::fft(nn::make_complex(v[0], v[1]))); },
                        {leaf(rnd({6}, 63)), leaf(rnd({6}, 64))});
  EXPECT_TRUE(r.ok) << "fft: " << r.message;
  r = nn::gradgradcheck([](const Inputs& v) { return nn::fft_causal_conv(v[0], v[1]); }, {leaf(rnd({6}, 65)), leaf(rnd({4}, 66))});
  EXPECT_TRUE(r.ok) << "causal conv: " << r.message;
}

// ---- audio ------------------------------------------------------------------------------------------

TEST(NnAudio, Windows) {
  const auto hp = nn::hann_window(4, true, DType::Float64).to_vector<double>();
  const auto hs = nn::hann_window(5, false, DType::Float64).to_vector<double>();
  EXPECT_NEAR(hp[0], 0, 1e-15);
  EXPECT_NEAR(hp[1], 0.5, 1e-15);
  EXPECT_NEAR(hp[2], 1, 1e-15);
  EXPECT_NEAR(hp[3], 0.5, 1e-15);
  EXPECT_NEAR(hs[2], 1, 1e-15);
  EXPECT_NEAR(hs[4], 0, 1e-15);
  const auto mp = nn::hamming_window(4, true, DType::Float64).to_vector<double>();
  EXPECT_NEAR(mp[0], 0.08, 1e-15);
  EXPECT_NEAR(mp[2], 1.0, 1e-15);
  EXPECT_EQ(nn::hann_window(1).to_vector<double>(), std::vector<double>{1});
}

TEST(NnAudio, FrameAndOverlapAdd) {
  const Tensor x = nn::arange(0, 10, 1, DType::Float64);
  Tensor       f = nn::frame(x, 4, 3);
  ASSERT_EQ(f.shape(), (nn::Shape{3, 4}));
  EXPECT_EQ(f.to_vector<double>(), (std::vector<double>{0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9}));
  f = nn::frame(x, 4, 3, true, -1);
  ASSERT_EQ(f.shape(), (nn::Shape{4, 4}));
  EXPECT_EQ(f.at({3, 0}), 9);
  EXPECT_EQ(f.at({3, 1}), -1);
  EXPECT_EQ(nn::frame(x, 11, 3).shape(), (nn::Shape{0, 11}));
  // a framed 2-D signal along axis 0
  EXPECT_EQ(nn::frame(nn::reshape(nn::arange(0, 12, 1, DType::Float64), {6, 2}), 3, 2, false, 0, 0).shape(),
            (nn::Shape{2, 3, 2}));
  const Tensor o = nn::overlap_and_add(nn::ones({3, 4}, DType::Float64), 2);
  EXPECT_EQ(o.to_vector<double>(), (std::vector<double>{1, 1, 2, 2, 2, 2, 1, 1}));
  auto r = nn::gradcheck([](const Inputs& v) { return nn::overlap_and_add(nn::frame(v[0], 5, 2, true), 2); }, {leaf(rnd({2, 11}, 70))});
  EXPECT_TRUE(r.ok) << r.message;
}

TEST(NnAudio, StftMatchesDirectComputationAndInverts) {
  const Tensor        x = rnd({2, 300}, 71);
  const ComplexTensor s = nn::stft(x, 64, 16);  // fft_length 64, periodic Hann
  ASSERT_EQ(s.shape(), (nn::Shape{2, 1 + (300 - 64) / 16, 33}));
  const auto xv = x.to_vector<double>();
  const auto sv = to_complex(s);
  double     worst = 0;
  for (int b = 0; b < 2; ++b)
    for (int fr = 0; fr < 15; ++fr)
      for (int k = 0; k < 33; ++k) {
        cd ref = 0;
        for (int i = 0; i < 64; ++i)
          ref += xv[b * 300 + fr * 16 + i] * (0.5 - 0.5 * std::cos(2 * kPi * i / 64)) * std::polar(1.0, -2 * kPi * i * k / 64);
        worst = std::max(worst, std::abs(ref - sv[(b * 15 + fr) * 33 + k]));
      }
  EXPECT_LT(worst, 1e-11);
  EXPECT_EQ(nn::stft(x, 50, 16).size(-1), 33);  // default fft_length: next power of two
  EXPECT_EQ(nn::stft(x, 64, 16, 64, Tensor(), true).size(-2), 19);  // ceil(300/16)
  // inverse with the dual window reconstructs samples covered by full overlap
  const Tensor y = nn::inverse_stft(s, 64, 16, 64, nn::inverse_stft_window(64, 16, nn::hann_window(64, true, DType::Float64)));
  ASSERT_EQ(y.shape(), (nn::Shape{2, 14 * 16 + 64}));
  EXPECT_LT(max_err(nn::slice_axis(y, -1, 64, 240), nn::slice_axis(x, -1, 64, 240)), 1e-12);
}

TEST(NnAudio, MelMatrixMatchesTensorFlow) {
  // TF (tflow MelWeights == tf.signal.linear_to_mel_weight_matrix formula):
  // sample_rate 8000, fft 256 (129 bins), 20 bands, 0..4000 Hz; nonzero entries.
  struct E { int r, c; double v; };
  const std::vector<E> ref = {
    {1, 0, 0.481650203}, {2, 0, 0.943141639}, {3, 0, 0.613905787}, {3, 1, 0.386094213},
    {4, 0, 0.188059971}, {4, 1, 0.811940014}, {5, 1, 0.778048575}, {5, 2, 0.22195144},
    {6, 1, 0.382736117}, {6, 2, 0.617263913}, {7, 1, 0.00110507896}, {7, 2, 0.99889493},
    {8, 2, 0.632240057}, {8, 3, 0.367759943}, {9, 2, 0.27531451}, {9, 3, 0.72468549},
    {10, 3, 0.929579735}, {10, 4, 0.0704202801}, {11, 3, 0.594355226}, {11, 4, 0.405644774},
    {12, 3, 0.269020826}, {12, 4, 0.730979145}, {13, 4, 0.953009605}, {13, 5, 0.0469903722},
    {14, 4, 0.64580214}, {14, 5, 0.35419789}, {15, 4, 0.346921057}, {15, 5, 0.653078973},
    {16, 4, 0.0559269525}, {16, 5, 0.944073021}, {17, 5, 0.772414267}, {17, 6, 0.227585733},
    {18, 5, 0.49600789}, {18, 6, 0.503992081}, {19, 5, 0.226360321}, {19, 6, 0.773639679},
    {20, 6, 0.963148892}, {20, 7, 0.0368511304}, {21, 6, 0.706073463}, {21, 7, 0.293926567},
    {22, 6, 0.454854459}, {22, 7, 0.545145512}, {23, 6, 0.209231019}, {23, 7, 0.790768981},
    {24, 7, 0.968959272}, {24, 8, 0.0310407113}, {25, 7, 0.733811021}, {25, 8, 0.266188979},
    {26, 7, 0.503572226}, {26, 8, 0.496427745}, {27, 7, 0.278042167}, {27, 8, 0.721957862},
    {28, 7, 0.0570320338}, {28, 8, 0.942967951}, {29, 8, 0.840364218}, {29, 9, 0.159635767},
    {30, 8, 0.627871394}, {30, 9, 0.372128606}, {31, 8, 0.419395685}, {31, 9, 0.580604315},
    {32, 8, 0.214788035}, {32, 9, 0.78521198}, {33, 8, 0.0139075061}, {33, 9, 0.986092508},
    {34, 9, 0.816620708}, {34, 10, 0.183379263}, {35, 9, 0.622801423}, {35, 10, 0.377198607},
    {36, 9, 0.432329714}, {36, 10, 0.567670286}, {37, 9, 0.245092005}, {37, 10, 0.754908025},
    {38, 9, 0.0609802678}, {38, 10, 0.93901974}, {39, 10, 0.879891872}, {39, 11, 0.120108142},
    {40, 10, 0.701729059}, {40, 11, 0.298270911}, {41, 10, 0.526398897}, {41, 11, 0.473601103},
    {42, 10, 0.353812635}, {42, 11, 0.646187365}, {43, 10, 0.183885753}, {43, 11, 0.816114247},
    {44, 10, 0.0165375285}, {44, 11, 0.983462453}, {45, 11, 0.851690888}, {45, 12, 0.148309141},
    {46, 11, 0.689272106}, {46, 12, 0.310727924}, {47, 11, 0.529210687}, {47, 12, 0.470789313},
    {48, 11, 0.371439219}, {48, 12, 0.628560781}, {49, 11, 0.21589312}, {49, 12, 0.784106851},
    {50, 11, 0.0625104457}, {50, 12, 0.937489569}, {51, 12, 0.911231875}, {51, 13, 0.0887681469},
    {52, 12, 0.762000382}, {52, 13, 0.237999603}, {53, 12, 0.614761412}, {53, 13, 0.385238588},
    {54, 12, 0.469462395}, {54, 13, 0.530537605}, {55, 12, 0.326052874}, {55, 13, 0.673947155},
    {56, 12, 0.184484348}, {56, 13, 0.815515637}, {57, 12, 0.0447101481}, {57, 13, 0.955289841},
    {58, 13, 0.906685352}, {58, 14, 0.0933146402}, {59, 13, 0.770366728}, {59, 14, 0.229633257},
    {60, 13, 0.635712624}, {60, 14, 0.364287406}, {61, 13, 0.502682805}, {61, 14, 0.497317195},
    {62, 13, 0.371238619}, {62, 14, 0.628761351}, {63, 13, 0.241342694}, {63, 14, 0.758657277},
    {64, 13, 0.11295899}, {64, 14, 0.887041032}, {65, 14, 0.986052692}, {65, 15, 0.0139473202},
    {66, 14, 0.86059016}, {66, 15, 0.13940984}, {67, 14, 0.736538947}, {67, 15, 0.263461053},
    {68, 14, 0.61386764}, {68, 15, 0.38613236}, {69, 14, 0.492545903}, {69, 15, 0.507454097},
    {70, 14, 0.372544318}, {70, 15, 0.627455711}, {71, 14, 0.253834486}, {71, 15, 0.746165514},
    {72, 14, 0.136388898}, {72, 15, 0.863611102}, {73, 14, 0.0201809034}, {73, 15, 0.979819119},
    {74, 15, 0.905184686}, {74, 16, 0.0948153064}, {75, 15, 0.791375279}, {75, 16, 0.208624735},
    {76, 15, 0.678728342}, {76, 16, 0.321271658}, {77, 15, 0.567220449}, {77, 16, 0.432779551},
    {78, 15, 0.456828773}, {78, 16, 0.543171227}, {79, 15, 0.347531199}, {79, 16, 0.652468801},
    {80, 15, 0.239306226}, {80, 16, 0.760693789}, {81, 15, 0.132133007}, {81, 16, 0.867866993},
    {82, 15, 0.0259913206}, {82, 16, 0.974008679}, {83, 16, 0.920861483}, {83, 17, 0.0791385174},
    {84, 16, 0.81672436}, {84, 17, 0.18327561}, {85, 16, 0.713561475}, {85, 17, 0.286438525},
    {86, 16, 0.611354649}, {86, 17, 0.388645351}, {87, 16, 0.510086358}, {87, 17, 0.489913613},
    {88, 16, 0.409739584}, {88, 17, 0.590260386}, {89, 16, 0.310297638}, {89, 17, 0.689702332},
    {90, 16, 0.211744383}, {90, 17, 0.788255632}, {91, 16, 0.114064068}, {91, 17, 0.885935962},
    {92, 16, 0.0172413513}, {92, 17, 0.982758641}, {93, 17, 0.921261311}, {93, 18, 0.078738682},
    {94, 17, 0.826109409}, {94, 18, 0.173890576}, {95, 17, 0.731771469}, {95, 18, 0.268228501},
    {96, 17, 0.638233721}, {96, 18, 0.361766279}, {97, 17, 0.545482635}, {97, 18, 0.454517335},
    {98, 17, 0.453505188}, {98, 18, 0.546494842}, {99, 17, 0.362288475}, {99, 18, 0.637711525},
    {100, 17, 0.271820068}, {100, 18, 0.728179932}, {101, 17, 0.182087794}, {101, 18, 0.817912221},
    {102, 17, 0.0930797532}, {102, 18, 0.906920254}, {103, 17, 0.00478435308}, {103, 18, 0.995215654},
    {104, 18, 0.917190254}, {104, 19, 0.0828097239}, {105, 18, 0.830286443}, {105, 19, 0.169713542},
    {106, 18, 0.744062126}, {106, 19, 0.255937874}, {107, 18, 0.658506751}, {107, 19, 0.341493279},
    {108, 18, 0.573609948}, {108, 19, 0.426390052}, {109, 18, 0.489361733}, {109, 19, 0.510638237},
    {110, 18, 0.405752271}, {110, 19, 0.594247699}, {111, 18, 0.322771937}, {111, 19, 0.677228093},
    {112, 18, 0.240411296}, {112, 19, 0.759588718}, {113, 18, 0.158661216}, {113, 19, 0.841338813},
    {114, 18, 0.077512674}, {114, 19, 0.922487319}, {115, 19, 0.996956885}, {116, 19, 0.916985273},
    {117, 19, 0.837589443}, {118, 19, 0.758761108}, {119, 19, 0.680492222}, {120, 19, 0.602774978},
    {121, 19, 0.525601566}, {122, 19, 0.448964447}, {123, 19, 0.37285623}, {124, 19, 0.297269642},
    {125, 19, 0.222197637}, {126, 19, 0.14763321}, {127, 19, 0.0735695586},
  };
  const Tensor w = nn::linear_to_mel_weight_matrix(20, 129, 8000, 0, 4000);
  ASSERT_EQ(w.shape(), (nn::Shape{129, 20}));
  const auto wv = w.to_vector<double>();
  std::vector<double> dense(wv.size(), 0.0);
  for (const auto& e : ref) dense[e.r * 20 + e.c] = e.v;
  for (std::size_t i = 0; i < wv.size(); ++i) EXPECT_NEAR(wv[i], dense[i], 2e-7) << i / 20 << "," << i % 20;
  // sample_rate 16000, fft 64 (33 bins), 6 bands, 125..7600 Hz
  const std::vector<double> ref2 = {
    0, 0, 0, 0, 0, 0,
    0.427765071, 0, 0, 0, 0, 0,
    0.863890231, 0.136109769, 0, 0, 0, 0,
    0.290088683, 0.709911346, 0, 0, 0, 0,
    0, 0.807787895, 0.192212135, 0, 0, 0,
    0, 0.391779691, 0.608220279, 0, 0, 0,
    0, 0.0260230172, 0.97397697, 0, 0, 0,
    0, 0, 0.69967556, 0.30032441, 0, 0,
    0, 0, 0.405064911, 0.59493506, 0, 0,
    0, 0, 0.136561751, 0.863438249, 0, 0,
    0, 0, 0, 0.889913261, 0.110086747, 0,
    0, 0, 0, 0.661827981, 0.338172048, 0,
    0, 0, 0, 0.449706197, 0.550293803, 0,
    0, 0, 0, 0.251458824, 0.748541176, 0,
    0, 0, 0, 0.0653817579, 0.934618235, 0,
    0, 0, 0, 0, 0.890066743, 0.109933235,
    0, 0, 0, 0, 0.724336743, 0.275663257,
    0, 0, 0, 0, 0.56719768, 0.43280229,
    0, 0, 0, 0, 0.417802721, 0.582197249,
    0, 0, 0, 0, 0.275424242, 0.724575758,
    0, 0, 0, 0, 0.139432624, 0.860567391,
    0, 0, 0, 0, 0.00927939266, 0.99072063,
    0, 0, 0, 0, 0, 0.884483814,
    0, 0, 0, 0, 0, 0.764622152,
    0, 0, 0, 0, 0, 0.649319112,
    0, 0, 0, 0, 0, 0.538240612,
    0, 0, 0, 0, 0, 0.431087941,
    0, 0, 0, 0, 0, 0.327593088,
    0, 0, 0, 0, 0, 0.227514535,
    0, 0, 0, 0, 0, 0.13063395,
    0, 0, 0, 0, 0, 0.0367532335,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
  };
  const auto w2 = nn::linear_to_mel_weight_matrix(6, 33, 16000, 125, 7600).to_vector<double>();
  ASSERT_EQ(w2.size(), ref2.size());
  for (std::size_t i = 0; i < w2.size(); ++i) EXPECT_NEAR(w2[i], ref2[i], 2e-7) << i / 6 << "," << i % 6;
  // tf.signal defaults and tflow's analytic one-band check
  EXPECT_EQ(nn::linear_to_mel_weight_matrix().shape(), (nn::Shape{129, 20}));
  const auto a = nn::linear_to_mel_weight_matrix(1, 4, 4200, 0, 2100, DType::Float64).to_vector<double>();
  EXPECT_NEAR(a[1], 1, 1e-12);
  EXPECT_EQ(a[0], 0);
  EXPECT_NEAR(a[3], 0, 1e-12);
  EXPECT_NEAR(nn::mel_to_hertz(nn::hertz_to_mel(1234.5)), 1234.5, 1e-9);
  EXPECT_NEAR(nn::hertz_to_mel(700), 1127 * std::log(2.0), 1e-12);
  EXPECT_THROW(nn::linear_to_mel_weight_matrix(20, 129, 8000, 125, 4001), std::invalid_argument);
}

TEST(NnAudio, DctAndMfcc) {
  const Tensor x  = rnd({3, 8}, 80);
  const auto   xv = x.to_vector<double>();
  const auto   d2 = nn::dct(x).to_vector<double>();
  const auto   mf = nn::mfccs_from_log_mel_spectrograms(x).to_vector<double>();
  for (int r = 0; r < 3; ++r)
    for (int k = 0; k < 8; ++k) {
      double s = 0;
      for (int n = 0; n < 8; ++n) s += 2 * xv[r * 8 + n] * std::cos(kPi * k * (2 * n + 1) / 16.0);
      EXPECT_NEAR(d2[r * 8 + k], s, 1e-12);
      EXPECT_NEAR(mf[r * 8 + k], s / std::sqrt(16.0), 1e-12);
    }
  // ortho type 2 and type 3 are inverse orthonormal transforms
  EXPECT_LT(max_err(nn::dct(nn::dct(x, 2, true), 3, true), x), 1e-12);
  EXPECT_NEAR(nn::sum(nn::square(nn::dct(x, 2, true))).item(), nn::sum(nn::square(x)).item(), 1e-10);
  // unnormalized type 3 inverts type 2 up to 2N
  EXPECT_LT(max_err(nn::dct(nn::dct(x), 3) / 16.0, x), 1e-12);
}

TEST(NnAudio, SpectrogramGradchecks) {
  const Tensor mel = nn::linear_to_mel_weight_matrix(6, 17, 8000, 100, 3900, DType::Float64);
  auto r = nn::gradcheck([](const Inputs& v) { return pack(nn::stft(v[0], 12, 5, 16)); }, {leaf(rnd({2, 30}, 90))});
  EXPECT_TRUE(r.ok) << "stft: " << r.message;
  r = nn::gradcheck([&](const Inputs& v) { return nn::log_mel_spectrogram(nn::power_spectrogram(nn::stft(v[0], 32, 8)), mel); },
                    {leaf(rnd({40}, 91))});
  EXPECT_TRUE(r.ok) << "log-mel: " << r.message;
  r = nn::gradcheck(
      [&](const Inputs& v) {
        return nn::mfccs_from_log_mel_spectrograms(nn::log_mel_spectrogram(nn::magnitude_spectrogram(nn::stft(v[0], 32, 8)), mel));
      },
      {leaf(rnd({40}, 92))});
  EXPECT_TRUE(r.ok) << "mfcc: " << r.message;
  r = nn::gradcheck([](const Inputs& v) { return nn::inverse_stft(nn::make_complex(v[0], v[1]), 12, 4, 16); },
                    {leaf(rnd({3, 9}, 93)), leaf(rnd({3, 9}, 94))});
  EXPECT_TRUE(r.ok) << "inverse_stft: " << r.message;
  r = nn::gradgradcheck([&](const Inputs& v) { return nn::mel_spectrogram(nn::power_spectrogram(nn::stft(v[0], 32, 8)), mel); },
                        {leaf(rnd({40}, 95))});
  EXPECT_TRUE(r.ok) << "mel second order: " << r.message;
}

TEST(NnAudio, WavRoundTrips) {
  const auto dir = std::filesystem::temp_directory_path() / "datamunge_nn_fft_tests";
  std::filesystem::create_directories(dir);
  std::vector<float> v(2 * 500);
  for (int i = 0; i < 500; ++i) {
    v[2 * i]     = float(0.9 * std::sin(0.05 * i));
    v[2 * i + 1] = float(-0.5 * std::cos(0.11 * i));
  }
  v[0] = 1.0f;
  v[1] = -1.0f;
  const Tensor x = nn::from_vector(v, {500, 2});
  struct C { nn::WavFormat f; double tol; };
  for (const C c : {C{nn::WavFormat::Pcm16, 1.0 / 32768}, C{nn::WavFormat::Pcm24, 1.0 / 8388608}, C{nn::WavFormat::Pcm32, 1e-7},
                    C{nn::WavFormat::Pcm8, 1.0 / 128}, C{nn::WavFormat::Float32, 0}}) {
    const auto path = (dir / "tone.wav").string();
    nn::write_wav(path, x, 22050, c.f);
    const nn::Wav w = nn::read_wav(path);
    EXPECT_EQ(w.sample_rate, 22050);
    ASSERT_EQ(w.samples.shape(), (nn::Shape{500, 2}));
    EXPECT_LE(max_err(w.samples, x), c.tol) << int(c.f);
  }
  // TF DecodeWav conversion: int16 / 32768, and EncodeWav clamps +1 to 32767
  const auto bytes = nn::encode_wav(x, 8000);
  EXPECT_EQ(bytes.size(), 44u + 2000u);
  EXPECT_EQ(bytes[44] | bytes[45] << 8, 32767);
  EXPECT_EQ(nn::decode_wav(bytes).samples.at({0, 1}), -1.0);
  // channel / sample selection like TF DecodeWav
  const auto mono = nn::encode_wav(nn::slice_axis(x, 1, 0, 1), 8000);
  EXPECT_EQ(nn::decode_wav(mono, 2).samples.shape(), (nn::Shape{500, 2}));
  EXPECT_EQ(nn::decode_wav(bytes, 1, 600).samples.shape(), (nn::Shape{600, 1}));
  EXPECT_EQ(nn::decode_wav(bytes, -1, 10).samples.shape(), (nn::Shape{10, 2}));
  // malformed input
  EXPECT_THROW(nn::decode_wav({'b', 'a', 'd'}), std::runtime_error);
  auto broken = bytes;
  broken[8]   = 'X';
  EXPECT_THROW(nn::decode_wav(broken), std::runtime_error);
  EXPECT_THROW(nn::decode_wav(std::vector<unsigned char>(bytes.begin(), bytes.begin() + 30)), std::runtime_error);
  std::filesystem::remove_all(dir);
}

TEST(NnFft, PlanCacheInsertionOrderGrowthAndEvictionPreserveTransforms) {
  nn::NoGradGuard guard;
  // More distinct lengths than the bounded plan cache, in both directions.
  for (int pass = 0; pass < 2; ++pass) {
    for (std::int64_t i = 1; i <= 72; ++i) {
      const std::int64_t n = pass == 0 ? i : 73 - i;
      auto x = nn::arange(0, static_cast<double>(n), 1, DType::Float64);
      auto reconstructed = nn::irfft(nn::rfft(x), n);
      EXPECT_LT(nn::max(nn::abs(nn::sub(x, reconstructed))).item(), 1e-10) << n;
    }
  }
}
