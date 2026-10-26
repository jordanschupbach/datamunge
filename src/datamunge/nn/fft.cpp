// Differentiable FFTs: a mixed-radix Stockham kernel (radix 2, 3, 4, 5 and
// generic odd primes), Bluestein for lengths with a large prime factor,
// real-input packing (any even length), and the three recorded linear primitives
// (dft, rdft, rdft_adjoint) the public transforms are composed from.

#include "datamunge/nn/fft.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>

#include "detail.hpp"

namespace datamunge::nn {

using detail::dispatch_float;
using detail::nograd;
using Grads = std::vector<Tensor>;
using I64   = std::int64_t;

namespace {

using cd             = std::complex<double>;
constexpr double kPi = 3.14159265358979323846;

// std::complex operator* may call __muldc3 (NaN handling); this is the plain product.
inline cd cmul(cd a, cd b) {
  return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
}

/// -i z.
inline cd mul_neg_i(cd z) { return {z.imag(), -z.real()}; }

/// Lengths whose largest prime factor exceeds this use Bluestein; smaller
/// prime factors use the generic odd-radix butterfly. Chosen by measurement
/// (single thread, Apple M-series): the O(p) generic butterfly beats
/// Bluestein's two padded smooth FFTs up to p ~ 103 for a prime length and
/// up to p ~ 113 as a factor of a larger length.
constexpr I64 kMaxRadix = 103;

/// Prime factors of n in the order the stages use them (4s first, then 2,
/// 3, 5, larger primes ascending); empty when some factor exceeds kMaxRadix.
std::vector<I64> radices(I64 n) {
  std::vector<I64> r;
  while (n % 4 == 0) { r.push_back(4); n /= 4; }
  while (n % 2 == 0) { r.push_back(2); n /= 2; }
  for (I64 p = 3; p <= kMaxRadix && n > 1; p += 2)
    while (n % p == 0) { r.push_back(p); n /= p; }
  if (n > 1) r.clear();
  return r;
}

/// Smallest 2^a 3^b 5^c >= n (Bluestein's convolution length).
I64 next_smooth(I64 n) {
  I64 best = 1;
  while (best < n) best <<= 1;
  for (I64 p5 = 1; p5 < best; p5 *= 5)
    for (I64 p35 = p5; p35 < best; p35 *= 3) {
      I64 v = p35;
      while (v < n) v <<= 1;
      best = std::min(best, v);
    }
  return best;
}

/// One Stockham stage: the current length p * m is split by radix p with
/// input stride s (= product of the earlier radices).
struct Stage {
  I64         p, m, s;
  std::size_t tw;   // offset of this stage's twiddles w^{qk} (q < m, 1 <= k < p) in tw_
  std::size_t gen;  // offset of the generic butterfly's cos/sin tables in gen_ (p > 5)
};

/// Precomputed tables for one transform length: a mixed-radix Stockham
/// (autosort, no bit reversal) plan when every prime factor is <= kMaxRadix,
/// otherwise Bluestein's chirp-z algorithm over a smooth padded length.
class Plan {
 public:
  explicit Plan(I64 n) : n_(n) {
    const std::vector<I64> rs = n <= 1 ? std::vector<I64>{} : radices(n);
    if (n <= 1 || !rs.empty()) {
      I64 cur = n, s = 1;
      for (const I64 p : rs) {
        const I64 m = cur / p;
        Stage     st{p, m, s, tw_.size(), gen_.size()};
        for (I64 q = 0; q < m; ++q)
          for (I64 k = 1; k < p; ++k) tw_.push_back(root(q * k % cur, cur));
        if (p > 5) {
          // cos / sin(2 pi jk / p) for 1 <= j, k <= (p-1)/2, row k
          const I64 h = (p - 1) / 2;
          for (I64 k = 1; k <= h; ++k)
            for (I64 j = 1; j <= h; ++j) {
              const double ang = 2.0 * kPi * double(j * k % p) / double(p);
              gen_.push_back(std::cos(ang));
              gen_.push_back(std::sin(ang));
            }
        }
        stages_.push_back(st);
        cur = m;
        s *= p;
      }
      return;
    }
    // Bluestein: X_k = c_k sum_j (x_j c_j) conj(c_{k-j}), c_k = exp(-i pi k^2 / n),
    // a linear convolution done with FFTs of a smooth length m >= 2n - 1.
    bluestein_ = true;
    m_         = next_smooth(2 * n - 1);
    sub_       = std::make_unique<Plan>(m_);
    chirp_.resize(static_cast<std::size_t>(n));
    I64 k2 = 0;  // k^2 mod 2n, updated incrementally (no overflow)
    for (I64 k = 0; k < n; ++k) {
      chirp_[k] = std::polar(1.0, -kPi * double(k2) / double(n));
      k2        = (k2 + 2 * k + 1) % (2 * n);
    }
    filter_.assign(static_cast<std::size_t>(m_), cd(0, 0));
    filter_[0] = std::conj(chirp_[0]);
    for (I64 k = 1; k < n; ++k) filter_[k] = filter_[m_ - k] = std::conj(chirp_[k]);
    std::vector<cd> work(static_cast<std::size_t>(sub_->scratch()));
    sub_->forward(filter_.data(), work.data());
    const double scale = 1.0 / double(m_);  // folds the inverse FFT normalization in
    for (auto& f : filter_) f *= scale;
  }

  /// Scratch elements `forward` needs.
  [[nodiscard]] I64 scratch() const { return bluestein_ ? m_ + sub_->scratch() : n_; }

  /// In-place unnormalized forward DFT.
  void forward(cd* a, cd* work) const {
    if (!bluestein_) {
      stockham(a, work);
      return;
    }
    cd* sw = work + m_;
    for (I64 k = 0; k < n_; ++k) work[k] = cmul(a[k], chirp_[k]);
    for (I64 k = n_; k < m_; ++k) work[k] = cd(0, 0);
    sub_->forward(work, sw);
    // multiply by the filter spectrum, then inverse FFT via conj(FFT(conj(.)))
    for (I64 k = 0; k < m_; ++k) work[k] = std::conj(cmul(work[k], filter_[k]));
    sub_->forward(work, sw);
    for (I64 k = 0; k < n_; ++k) a[k] = cmul(std::conj(work[k]), chirp_[k]);
  }

  /// In-place unnormalized inverse DFT (positive exponent).
  void inverse(cd* a, cd* work) const {
    for (I64 k = 0; k < n_; ++k) a[k] = std::conj(a[k]);
    forward(a, work);
    for (I64 k = 0; k < n_; ++k) a[k] = std::conj(a[k]);
  }

  void run(cd* a, cd* work, bool inv) const { inv ? inverse(a, work) : forward(a, work); }

 private:
  /// exp(-2 pi i k / n), with k already reduced mod n.
  static cd root(I64 k, I64 n) { return std::polar(1.0, -2.0 * kPi * double(k) / double(n)); }

  /// Runs every stage, ping-ponging between a and work; the result ends in a.
  void stockham(cd* a, cd* work) const {
    cd* x = a;
    cd* y = work;
    for (const Stage& st : stages_) {
      const cd* w = tw_.data() + st.tw;
      switch (st.p) {
        case 2: stage(st, x, y, [&](const cd* in, I64 is, cd* out, I64 os, I64 q) { radix2(in, is, out, os, w + q); }); break;
        case 3: stage(st, x, y, [&](const cd* in, I64 is, cd* out, I64 os, I64 q) { radix3(in, is, out, os, w + 2 * q); }); break;
        case 4: stage(st, x, y, [&](const cd* in, I64 is, cd* out, I64 os, I64 q) { radix4(in, is, out, os, w + 3 * q); }); break;
        case 5: stage(st, x, y, [&](const cd* in, I64 is, cd* out, I64 os, I64 q) { radix5(in, is, out, os, w + 4 * q); }); break;
        default: {
          const double* g = gen_.data() + st.gen;
          const I64     p = st.p;
          stage(st, x, y, [&](const cd* in, I64 is, cd* out, I64 os, I64 q) { radix_odd(p, g, in, is, out, os, w + (p - 1) * q); });
        }
      }
      std::swap(x, y);
    }
    if (x != a) std::copy(x, x + n_, a);
  }

  /// Applies `bf` for every q < m and offset s0 < s: inputs x[s0 + s (q + m j)],
  /// outputs y[s0 + s (p q + k)] (k-th output multiplied by w^{qk}).
  template <typename F>
  static void stage(const Stage& st, const cd* x, cd* y, F&& bf) {
    const I64 p = st.p, m = st.m, s = st.s;
    if (s == 1) {
      for (I64 q = 0; q < m; ++q) bf(x + q, m, y + p * q, 1, q);
      return;
    }
    for (I64 q = 0; q < m; ++q)
      for (I64 s0 = 0; s0 < s; ++s0) bf(x + s0 + s * q, s * m, y + s0 + s * p * q, s, q);
  }

  static void radix2(const cd* in, I64 is, cd* out, I64 os, const cd* w) {
    const cd a0 = in[0], a1 = in[is];
    out[0]  = a0 + a1;
    out[os] = cmul(a0 - a1, w[0]);
  }

  static void radix3(const cd* in, I64 is, cd* out, I64 os, const cd* w) {
    constexpr double h = 0.86602540378443864676;  // sin(2 pi / 3)
    const cd a0 = in[0], a1 = in[is], a2 = in[2 * is];
    const cd t = a1 + a2, u = mul_neg_i((a1 - a2) * h), v = a0 - 0.5 * t;
    out[0]      = a0 + t;
    out[os]     = cmul(v + u, w[0]);
    out[2 * os] = cmul(v - u, w[1]);
  }

  static void radix4(const cd* in, I64 is, cd* out, I64 os, const cd* w) {
    const cd a0 = in[0], a1 = in[is], a2 = in[2 * is], a3 = in[3 * is];
    const cd t0 = a0 + a2, t1 = a0 - a2, t2 = a1 + a3, t3 = mul_neg_i(a1 - a3);
    out[0]      = t0 + t2;
    out[os]     = cmul(t1 + t3, w[0]);
    out[2 * os] = cmul(t0 - t2, w[1]);
    out[3 * os] = cmul(t1 - t3, w[2]);
  }

  static void radix5(const cd* in, I64 is, cd* out, I64 os, const cd* w) {
    constexpr double c1 = 0.30901699437494742410, c2 = -0.80901699437494742410;  // cos(2 pi / 5), cos(4 pi / 5)
    constexpr double s1 = 0.95105651629515357212, s2 = 0.58778525229247312917;   // sin(2 pi / 5), sin(4 pi / 5)
    const cd a0 = in[0], a1 = in[is], a2 = in[2 * is], a3 = in[3 * is], a4 = in[4 * is];
    const cd t1 = a1 + a4, t2 = a2 + a3, d1 = a1 - a4, d2 = a2 - a3;
    const cd m1 = a0 + c1 * t1 + c2 * t2, m2 = a0 + c2 * t1 + c1 * t2;
    const cd n1 = mul_neg_i(s1 * d1 + s2 * d2), n2 = mul_neg_i(s2 * d1 - s1 * d2);
    out[0]      = a0 + t1 + t2;
    out[os]     = cmul(m1 + n1, w[0]);
    out[2 * os] = cmul(m2 + n2, w[1]);
    out[3 * os] = cmul(m2 - n2, w[2]);
    out[4 * os] = cmul(m1 - n1, w[3]);
  }

  /// Odd prime radix p: pairs inputs j and p - j so each output pair costs
  /// (p - 1) / 2 real-by-complex products per term.
  static void radix_odd(I64 p, const double* g, const cd* in, I64 is, cd* out, I64 os, const cd* w) {
    const I64 h = (p - 1) / 2;
    cd        sp[kMaxRadix / 2 + 1], dm[kMaxRadix / 2 + 1];
    const cd  a0  = in[0];
    cd        sum = a0;
    for (I64 j = 1; j <= h; ++j) {
      const cd aj = in[j * is], ak = in[(p - j) * is];
      sp[j]       = aj + ak;
      dm[j]       = aj - ak;
      sum += sp[j];
    }
    out[0] = sum;
    for (I64 k = 1; k <= h; ++k) {
      const double* row = g + 2 * h * (k - 1);
      double        rr = a0.real(), ri = a0.imag(), ir = 0, ii = 0;
      for (I64 j = 1; j <= h; ++j) {
        const double c = row[2 * (j - 1)], s = row[2 * (j - 1) + 1];
        rr += c * sp[j].real();
        ri += c * sp[j].imag();
        ir += s * dm[j].real();
        ii += s * dm[j].imag();
      }
      // b_k = re - i im, b_{p-k} = re + i im
      out[k * os]       = cmul(cd(rr + ii, ri - ir), w[k - 1]);
      out[(p - k) * os] = cmul(cd(rr - ii, ri + ir), w[p - k - 1]);
    }
  }

  I64                   n_;
  bool                  bluestein_{false};
  std::vector<Stage>    stages_;
  std::vector<cd>       tw_;
  std::vector<double>   gen_;
  I64                   m_{0};
  std::unique_ptr<Plan> sub_;
  std::vector<cd>       chirp_, filter_;
};

/// Plans are cached per length (bounded; cleared when it grows large).
std::shared_ptr<const Plan> get_plan(I64 n) {
  static std::mutex                                           mu;
  static std::vector<std::pair<I64, std::shared_ptr<const Plan>>> cache;
  std::lock_guard<std::mutex>                                 lock(mu);
  auto it = std::lower_bound(cache.begin(), cache.end(), n,
      [](const auto& entry, I64 length) { return entry.first < length; });
  if (it != cache.end() && it->first == n) return it->second;
  if (cache.size() > 64) {
    cache.clear();
    it = cache.begin();
  }
  auto plan = std::make_shared<const Plan>(n);
  cache.emplace(it, n, plan);
  return plan;
}

/// Rows per parallel task so each task does a reasonable amount of work.
I64 row_grain(I64 n) { return std::max<I64>(1, 16384 / std::max<I64>(n, 1)); }

/// e^{sign 2 pi i k / n} for k = 0 .. count-1.
std::vector<cd> twiddles(I64 n, I64 count, double sign) {
  std::vector<cd> t(static_cast<std::size_t>(count));
  for (I64 k = 0; k < count; ++k) t[k] = std::polar(1.0, sign * 2.0 * kPi * double(k) / double(n));
  return t;
}

// ---- batched kernels over rows of the last axis ---------------------------------------------

template <typename T>
void c2c_rows(const T* re, const T* im, T* ore, T* oim, I64 rows, I64 n, bool inverse) {
  if (rows == 0 || n == 0) return;
  const auto plan = get_plan(n);
  parallel_for(0, rows, row_grain(n), [&](I64 lo, I64 hi) {
    std::vector<cd> buf(static_cast<std::size_t>(n)), work(static_cast<std::size_t>(plan->scratch()));
    for (I64 r = lo; r < hi; ++r) {
      const T* pr = re + r * n;
      const T* pi = im + r * n;
      for (I64 k = 0; k < n; ++k) buf[k] = cd(double(pr[k]), double(pi[k]));
      plan->run(buf.data(), work.data(), inverse);
      T* qr = ore + r * n;
      T* qi = oim + r * n;
      for (I64 k = 0; k < n; ++k) {
        qr[k] = static_cast<T>(buf[k].real());
        qi[k] = static_cast<T>(buf[k].imag());
      }
    }
  });
}

/// Real rows of length n -> bins 0..n/2 (unnormalized forward DFT).
template <typename T>
void r2c_rows(const T* x, T* ore, T* oim, I64 rows, I64 n) {
  if (rows == 0) return;
  const I64 m = n / 2 + 1;
  if (n % 2 == 1 || n < 2) {
    const auto plan = get_plan(n);
    parallel_for(0, rows, row_grain(n), [&](I64 lo, I64 hi) {
      std::vector<cd> buf(static_cast<std::size_t>(n)), work(static_cast<std::size_t>(plan->scratch()));
      for (I64 r = lo; r < hi; ++r) {
        for (I64 k = 0; k < n; ++k) buf[k] = cd(double(x[r * n + k]), 0.0);
        plan->forward(buf.data(), work.data());
        for (I64 k = 0; k < m; ++k) {
          ore[r * m + k] = static_cast<T>(buf[k].real());
          oim[r * m + k] = static_cast<T>(buf[k].imag());
        }
      }
    });
    return;
  }
  // Even n: pack z_j = x_{2j} + i x_{2j+1} into one half-length complex FFT.
  const I64             h    = n / 2;
  const auto            plan = get_plan(h);
  const std::vector<cd> tw   = twiddles(n, h + 1, -1.0);
  parallel_for(0, rows, row_grain(n), [&](I64 lo, I64 hi) {
    std::vector<cd> z(static_cast<std::size_t>(h)), work(static_cast<std::size_t>(plan->scratch()));
    for (I64 r = lo; r < hi; ++r) {
      const T* xr = x + r * n;
      for (I64 j = 0; j < h; ++j) z[j] = cd(double(xr[2 * j]), double(xr[2 * j + 1]));
      plan->forward(z.data(), work.data());
      T* qr = ore + r * m;
      T* qi = oim + r * m;
      for (I64 k = 0; k <= h; ++k) {
        const cd zk = z[k % h], zc = std::conj(z[(h - k) % h]);
        const cd e  = 0.5 * (zk + zc);
        const cd d  = zk - zc;                  // 2 i O_k
        const cd o  = cd(0.5 * d.imag(), -0.5 * d.real());  // d / (2i)
        const cd X  = e + cmul(tw[k], o);
        qr[k]       = static_cast<T>(X.real());
        qi[k]       = static_cast<T>(X.imag());
      }
    }
  });
}

/// The adjoint of r2c_rows: bins 0..n/2 -> real rows of length n,
/// y_j = Re sum_k g_k e^{2 pi i j k / n}.
template <typename T>
void adjoint_rows(const T* re, const T* im, T* y, I64 rows, I64 n) {
  if (rows == 0) return;
  const I64 m = n / 2 + 1;
  // Hermitian half spectrum X with sum_{k<n} X~_k e^{+} equal to y.
  auto half = [&](const T* gr, const T* gi, I64 k) -> cd {
    if (k == 0 || (n % 2 == 0 && k == n / 2)) return cd(double(gr[k]), 0.0);
    return cd(0.5 * double(gr[k]), 0.5 * double(gi[k]));
  };
  if (n % 2 == 1 || n < 2) {
    const auto plan = get_plan(n);
    parallel_for(0, rows, row_grain(n), [&](I64 lo, I64 hi) {
      std::vector<cd> buf(static_cast<std::size_t>(n)), work(static_cast<std::size_t>(plan->scratch()));
      for (I64 r = lo; r < hi; ++r) {
        const T* gr = re + r * m;
        const T* gi = im + r * m;
        buf[0]      = half(gr, gi, 0);
        for (I64 k = 1; k < m; ++k) {
          buf[k]     = half(gr, gi, k);
          buf[n - k] = std::conj(buf[k]);
        }
        plan->inverse(buf.data(), work.data());
        for (I64 j = 0; j < n; ++j) y[r * n + j] = static_cast<T>(buf[j].real());
      }
    });
    return;
  }
  const I64             h    = n / 2;
  const auto            plan = get_plan(h);
  const std::vector<cd> tw   = twiddles(n, h, +1.0);
  parallel_for(0, rows, row_grain(n), [&](I64 lo, I64 hi) {
    std::vector<cd> z(static_cast<std::size_t>(h)), work(static_cast<std::size_t>(plan->scratch()));
    for (I64 r = lo; r < hi; ++r) {
      const T* gr = re + r * m;
      const T* gi = im + r * m;
      for (I64 k = 0; k < h; ++k) {
        const cd xk = half(gr, gi, k), xc = std::conj(half(gr, gi, h - k));
        const cd s = xk + xc, d = cmul(tw[k], xk - xc);
        z[k]       = s + cd(-d.imag(), d.real());  // s + i d
      }
      plan->inverse(z.data(), work.data());
      T* yr = y + r * n;
      for (I64 j = 0; j < h; ++j) {
        yr[2 * j]     = static_cast<T>(z[j].real());
        yr[2 * j + 1] = static_cast<T>(z[j].imag());
      }
    }
  });
}

DType float_dtype(const Tensor& t) { return is_floating(t.dtype()) ? t.dtype() : DType::Float32; }

Tensor as_float(const Tensor& t) { return is_floating(t.dtype()) ? t : cast(t, DType::Float32); }

void check_complex(const ComplexTensor& x, const char* what) {
  if (!x.real.defined() || !x.imag.defined())
    throw std::invalid_argument(std::string(what) + ": both parts of the complex tensor must be defined");
  if (x.real.shape() != x.imag.shape() || x.real.dtype() != x.imag.dtype())
    throw std::invalid_argument(std::string(what) + ": real and imaginary parts differ in shape or dtype (" +
                                shape_str(x.real.shape()) + " vs " + shape_str(x.imag.shape()) + ")");
  if (x.real.dim() < 1) throw std::invalid_argument(std::string(what) + ": needs rank >= 1");
}

/// Crops or zero-pads `t` along `axis` to `length`.
Tensor fit_axis(const Tensor& t, I64 axis, I64 length) {
  const I64 a   = detail::normalize_axis(axis, t.dim(), "fft");
  const I64 cur = t.size(a);
  if (cur == length) return t;
  if (cur > length) return slice_axis(t, a, 0, length);
  std::vector<std::pair<I64, I64>> p(static_cast<std::size_t>(t.dim()), {0, 0});
  p[static_cast<std::size_t>(a)] = {0, length - cur};
  return pad(t, p);
}

ComplexTensor fit_axis(const ComplexTensor& t, I64 axis, I64 length) {
  return {fit_axis(t.real, axis, length), fit_axis(t.imag, axis, length)};
}

}  // namespace

void fft_inplace(std::complex<double>* data, std::int64_t n, bool inverse) {
  if (n < 1) return;
  const auto      plan = get_plan(n);
  std::vector<cd> work(static_cast<std::size_t>(plan->scratch()));
  plan->run(data, work.data(), inverse);
}

// ---- complex helpers -------------------------------------------------------------------------

ComplexTensor make_complex(const Tensor& real, const Tensor& imag) {
  if (!real.defined()) throw std::invalid_argument("make_complex: undefined real part");
  if (!imag.defined()) {
    const Tensor re = as_float(real);
    return {re, zeros(re.shape(), re.dtype())};
  }
  const DType dt = (float_dtype(real) == DType::Float64 || float_dtype(imag) == DType::Float64) ? DType::Float64
                                                                                                 : DType::Float32;
  Tensor re = detail::as_dtype(real, dt), im = detail::as_dtype(imag, dt);
  if (re.shape() != im.shape()) {
    const Shape s = broadcast_shapes(re.shape(), im.shape());
    re            = broadcast_to(re, s);
    im            = broadcast_to(im, s);
  }
  return {re, im};
}

ComplexTensor complex_add(const ComplexTensor& a, const ComplexTensor& b) { return {a.real + b.real, a.imag + b.imag}; }
ComplexTensor complex_sub(const ComplexTensor& a, const ComplexTensor& b) { return {a.real - b.real, a.imag - b.imag}; }

ComplexTensor complex_mul(const ComplexTensor& a, const ComplexTensor& b) {
  return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}

ComplexTensor complex_div(const ComplexTensor& a, const ComplexTensor& b) {
  const Tensor den = complex_abs_squared(b);
  return {(a.real * b.real + a.imag * b.imag) / den, (a.imag * b.real - a.real * b.imag) / den};
}

ComplexTensor complex_scale(const ComplexTensor& a, const Tensor& factor) { return {a.real * factor, a.imag * factor}; }
ComplexTensor complex_scale(const ComplexTensor& a, double factor) { return {a.real * factor, a.imag * factor}; }
ComplexTensor complex_conj(const ComplexTensor& a) { return {a.real, neg(a.imag)}; }
ComplexTensor complex_neg(const ComplexTensor& a) { return {neg(a.real), neg(a.imag)}; }

Tensor complex_abs_squared(const ComplexTensor& a) { return square(a.real) + square(a.imag); }

Tensor complex_abs(const ComplexTensor& a) {
  const Tensor s    = complex_abs_squared(a);
  const Tensor pos  = greater(s, 0.0);
  const Tensor safe = where(pos, s, 1.0);  // keeps sqrt's gradient finite at 0
  return where(pos, sqrt(safe), 0.0);
}

Tensor complex_angle(const ComplexTensor& a) {
  const Tensor pos = greater(complex_abs_squared(a), 0.0);
  return atan2(where(pos, a.imag, 0.0), where(pos, a.real, 1.0));
}

ComplexTensor complex_from_polar(const Tensor& magnitude, const Tensor& phase) {
  return make_complex(magnitude * cos(phase), magnitude * sin(phase));
}

ComplexTensor complex_exp(const ComplexTensor& a) { return complex_from_polar(exp(a.real), a.imag); }

ComplexTensor complex_sum(const ComplexTensor& a, const std::vector<std::int64_t>& axes, bool keepdims) {
  return {sum(a.real, axes, keepdims), sum(a.imag, axes, keepdims)};
}

ComplexTensor complex_slice_axis(const ComplexTensor& a, std::int64_t axis, std::int64_t start, std::int64_t stop,
                                 std::int64_t step) {
  return {slice_axis(a.real, axis, start, stop, step), slice_axis(a.imag, axis, start, stop, step)};
}

ComplexTensor complex_reshape(const ComplexTensor& a, const Shape& shape) {
  return {reshape(a.real, shape), reshape(a.imag, shape)};
}

ComplexTensor complex_swapaxes(const ComplexTensor& a, std::int64_t axis0, std::int64_t axis1) {
  return {swapaxes(a.real, axis0, axis1), swapaxes(a.imag, axis0, axis1)};
}

ComplexTensor complex_cast(const ComplexTensor& a, DType dtype) { return {cast(a.real, dtype), cast(a.imag, dtype)}; }

// ---- primitives --------------------------------------------------------------------------------

ComplexTensor dft(const ComplexTensor& x_in, bool inverse) {
  const ComplexTensor x = x_in.imag.defined() ? x_in : make_complex(x_in.real);
  check_complex(x, "dft");
  const DType dt   = x.real.dtype();
  const I64   n    = x.real.size(-1);
  const I64   rows = n == 0 ? 0 : x.real.numel() / n;
  Tensor      ore = empty(x.real.shape(), dt), oim = empty(x.real.shape(), dt);
  {
    NoGradGuard  guard;
    const Tensor rk = nograd(x.real).contiguous(), ik = nograd(x.imag).contiguous();
    dispatch_float(dt, "fft", [&](auto tag) {
      using T = typename decltype(tag)::type;
      c2c_rows<T>(rk.data_ptr<T>(), ik.data_ptr<T>(), ore.data_ptr<T>(), oim.data_ptr<T>(), rows, n, inverse);
    });
  }
  record(inverse ? "IDFT" : "DFT", {x.real, x.imag}, {ore, oim}, [inverse](const Grads& gr, const Grads&) {
    const ComplexTensor g = dft({gr[0], gr[1]}, !inverse);
    return Grads{g.real, g.imag};
  });
  return {ore, oim};
}

ComplexTensor rdft(const Tensor& x_in) {
  const Tensor x = as_float(x_in);
  if (x.dim() < 1) throw std::invalid_argument("rfft: needs rank >= 1");
  const DType dt   = x.dtype();
  const I64   n    = x.size(-1);
  if (n < 1) throw std::invalid_argument("rfft: length must be >= 1");
  const I64   m    = n / 2 + 1;
  const I64   rows = x.numel() / n;
  Shape       shape = x.shape();
  shape.back()      = m;
  Tensor ore = empty(shape, dt), oim = empty(shape, dt);
  {
    NoGradGuard  guard;
    const Tensor xk = nograd(x).contiguous();
    dispatch_float(dt, "rfft", [&](auto tag) {
      using T = typename decltype(tag)::type;
      r2c_rows<T>(xk.data_ptr<T>(), ore.data_ptr<T>(), oim.data_ptr<T>(), rows, n);
    });
  }
  record("RDFT", {x}, {ore, oim}, [n](const Grads& gr, const Grads&) {
    return Grads{rdft_adjoint({gr[0], gr[1]}, n)};
  });
  return {ore, oim};
}

Tensor rdft_adjoint(const ComplexTensor& x_in, std::int64_t n) {
  const ComplexTensor x = x_in.imag.defined() ? x_in : make_complex(x_in.real);
  check_complex(x, "rdft_adjoint");
  if (n < 1) throw std::invalid_argument("rdft_adjoint: length must be >= 1");
  const I64 m = n / 2 + 1;
  if (x.real.size(-1) != m)
    throw std::invalid_argument("rdft_adjoint: expected " + std::to_string(m) + " bins for length " +
                                std::to_string(n) + ", got " + std::to_string(x.real.size(-1)));
  const DType dt    = x.real.dtype();
  const I64   rows  = x.real.numel() / m;
  Shape       shape = x.real.shape();
  shape.back()      = n;
  Tensor y          = empty(shape, dt);
  {
    NoGradGuard  guard;
    const Tensor rk = nograd(x.real).contiguous(), ik = nograd(x.imag).contiguous();
    dispatch_float(dt, "irfft", [&](auto tag) {
      using T = typename decltype(tag)::type;
      adjoint_rows<T>(rk.data_ptr<T>(), ik.data_ptr<T>(), y.data_ptr<T>(), rows, n);
    });
  }
  record("RDFTAdjoint", {x.real, x.imag}, {y}, [](const Grads& gr, const Grads&) {
    const ComplexTensor g = rdft(gr[0]);
    return Grads{g.real, g.imag};
  });
  return y;
}

// ---- transforms --------------------------------------------------------------------------------

ComplexTensor fft(const ComplexTensor& x) { return dft(x, false); }

ComplexTensor ifft(const ComplexTensor& x) {
  const ComplexTensor y = dft(x, true);
  const I64           n = y.size(-1);
  return n == 0 ? y : complex_scale(y, 1.0 / double(n));
}

ComplexTensor fft2d(const ComplexTensor& x) {
  if (x.dim() < 2) throw std::invalid_argument("fft2d: needs rank >= 2");
  return complex_swapaxes(fft(complex_swapaxes(fft(x), -1, -2)), -1, -2);
}

ComplexTensor ifft2d(const ComplexTensor& x) {
  if (x.dim() < 2) throw std::invalid_argument("ifft2d: needs rank >= 2");
  return complex_swapaxes(ifft(complex_swapaxes(ifft(x), -1, -2)), -1, -2);
}

ComplexTensor rfft(const Tensor& x, std::int64_t fft_length) {
  if (x.dim() < 1) throw std::invalid_argument("rfft: needs rank >= 1");
  const I64 n = fft_length < 0 ? x.size(-1) : fft_length;
  if (n < 1) throw std::invalid_argument("rfft: fft_length must be >= 1");
  return rdft(fit_axis(as_float(x), -1, n));
}

Tensor irfft(const ComplexTensor& x_in, std::int64_t fft_length) {
  const ComplexTensor x = x_in.imag.defined() ? x_in : make_complex(x_in.real);
  check_complex(x, "irfft");
  const I64 bins = x.size(-1);
  const I64 n    = fft_length < 0 ? 2 * (bins - 1) : fft_length;
  if (n < 1) throw std::invalid_argument("irfft: fft_length must be >= 1");
  const I64 m = n / 2 + 1;
  // irfft(X) = rdft_adjoint(w * X) with w = 1/n at DC and Nyquist, 2/n elsewhere.
  std::vector<double> w(static_cast<std::size_t>(m), 2.0 / double(n));
  w[0] = 1.0 / double(n);
  if (n % 2 == 0) w[static_cast<std::size_t>(m - 1)] = 1.0 / double(n);
  const Tensor weights = from_vector(w).to(x.dtype());
  return rdft_adjoint(complex_scale(fit_axis(x, -1, m), weights), n);
}

ComplexTensor rfft2d(const Tensor& x, std::array<std::int64_t, 2> fft_length) {
  if (x.dim() < 2) throw std::invalid_argument("rfft2d: needs rank >= 2");
  const I64 rows = fft_length[0] < 0 ? x.size(-2) : fft_length[0];
  if (rows < 1) throw std::invalid_argument("rfft2d: fft_length must be >= 1");
  const ComplexTensor inner = rfft(fit_axis(as_float(x), -2, rows), fft_length[1]);
  return complex_swapaxes(fft(complex_swapaxes(inner, -1, -2)), -1, -2);
}

Tensor irfft2d(const ComplexTensor& x, std::array<std::int64_t, 2> fft_length) {
  if (x.dim() < 2) throw std::invalid_argument("irfft2d: needs rank >= 2");
  const I64 rows = fft_length[0] < 0 ? x.size(-2) : fft_length[0];
  if (rows < 1) throw std::invalid_argument("irfft2d: fft_length must be >= 1");
  const ComplexTensor cols = complex_swapaxes(ifft(complex_swapaxes(fit_axis(x, -2, rows), -1, -2)), -1, -2);
  return irfft(cols, fft_length[1]);
}

Tensor fftshift(const Tensor& x) {
  if (x.dim() < 1) throw std::invalid_argument("fftshift: needs rank >= 1");
  const I64 n = x.size(-1), s = n / 2;
  if (s == 0) return x;
  return concat({slice_axis(x, -1, n - s, n), slice_axis(x, -1, 0, n - s)}, -1);
}

Tensor fftfreq(std::int64_t n, double d, DType dtype) {
  if (n < 1 || d == 0) throw std::invalid_argument("fftfreq: need n >= 1 and d != 0");
  std::vector<double> f(static_cast<std::size_t>(n));
  for (I64 i = 0; i < n; ++i) f[i] = double(i < (n + 1) / 2 ? i : i - n) / (d * double(n));
  return from_vector(f).to(dtype);
}

Tensor rfftfreq(std::int64_t n, double d, DType dtype) {
  if (n < 1 || d == 0) throw std::invalid_argument("rfftfreq: need n >= 1 and d != 0");
  std::vector<double> f(static_cast<std::size_t>(n / 2 + 1));
  for (I64 i = 0; i <= n / 2; ++i) f[i] = double(i) / (d * double(n));
  return from_vector(f).to(dtype);
}

// ---- convolution -------------------------------------------------------------------------------

std::int64_t next_power_of_two(std::int64_t n) {
  I64 p = 1;
  while (p < n) p <<= 1;
  return p;
}

Tensor fft_convolve(const Tensor& a, const Tensor& b) {
  if (a.dim() < 1 || b.dim() < 1) throw std::invalid_argument("fft_convolve: needs rank >= 1 inputs");
  const I64 la = a.size(-1), lb = b.size(-1);
  if (la < 1 || lb < 1) throw std::invalid_argument("fft_convolve: empty input");
  const I64 out = la + lb - 1, n = next_power_of_two(out);
  const Tensor y = irfft(complex_mul(rfft(a, n), rfft(b, n)), n);
  return slice_axis(y, -1, 0, out);
}

Tensor fft_causal_conv(const Tensor& u, const Tensor& kernel) {
  if (u.dim() < 1 || kernel.dim() < 1) throw std::invalid_argument("fft_causal_conv: needs rank >= 1 inputs");
  const I64 l = u.size(-1);
  if (l < 1 || kernel.size(-1) < 1) throw std::invalid_argument("fft_causal_conv: empty input");
  const Tensor k  = kernel.size(-1) > l ? slice_axis(kernel, -1, 0, l) : kernel;
  const I64    n  = next_power_of_two(l + k.size(-1) - 1);
  const Tensor y  = irfft(complex_mul(rfft(u, n), rfft(k, n)), n);
  return slice_axis(y, -1, 0, l);
}

Tensor fft_circular_convolve(const Tensor& a, const Tensor& b) {
  if (a.dim() < 1 || b.dim() < 1 || a.size(-1) != b.size(-1))
    throw std::invalid_argument("fft_circular_convolve: inputs need the same last-axis length");
  const I64 n = a.size(-1);
  return irfft(complex_mul(rfft(a), rfft(b)), n);
}

}  // namespace datamunge::nn
