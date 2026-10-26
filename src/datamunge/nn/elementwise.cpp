// Elementwise ops: arithmetic, unary math, activations, comparisons, where.

#include <cmath>
#include <limits>

#include "detail.hpp"
#include "vector_math.hpp"

namespace datamunge::nn {

using detail::as_dtype;
using detail::dispatch;
using detail::dispatch_float;
using detail::nograd;
using detail::Tag;

namespace {

using Grads = std::vector<Tensor>;

constexpr double kPi = 3.14159265358979323846;

/// Arithmetic on bool promotes to int64.
DType arithmetic_dtype(DType a, DType b) {
  DType t = promote_types(a, b);
  return t == DType::Bool ? DType::Int64 : t;
}

/// Runs `f(Tag<T>)` with recording off.
template <typename F>
Tensor compute(DType dtype, F&& f) {
  NoGradGuard guard;
  return dispatch(dtype, std::forward<F>(f));
}

template <typename F>
Tensor compute_float(DType dtype, const char* what, F&& f) {
  NoGradGuard guard;
  return dispatch_float(dtype, what, std::forward<F>(f));
}

/// Elementwise binary map of two tensors already converted to dtype T.
/// map_* only reads input storage and creates a fresh output under compute's
/// NoGradGuard; detached TensorImpl copies serve no purpose in these kernels.
template <typename Op>
Tensor binary_kernel(const Tensor& a, const Tensor& b, DType dtype, Op op) {
  return compute(dtype, [&](auto tag) -> Tensor {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same<T, bool>::value) {
      throw std::logic_error("nn: bool arithmetic kernel");
    } else {
      return detail::map_binary<T, T>(a, b, dtype, [&](T x, T y) { return static_cast<T>(op(x, y)); });
    }
  });
}

/// Elementwise unary map; x already has dtype `dtype`.
template <typename Op>
Tensor unary_kernel(const Tensor& x, DType dtype, Op op) {
  return compute(dtype, [&](auto tag) -> Tensor {
    using T = typename decltype(tag)::type;
    return detail::map_unary<T, T>(x, dtype, [&](T v) { return static_cast<T>(op(v)); });
  });
}

/// Float-only unary map: converts x to float first (recorded).
template <typename Op>
std::pair<Tensor, Tensor> float_unary(const Tensor& x, const char* what, Op op) {
  Tensor xf  = as_dtype(x, to_floating(x.dtype()));
  Tensor out = compute_float(xf.dtype(), what, [&](auto tag) -> Tensor {
    using T = typename decltype(tag)::type;
    return detail::map_unary<T, T>(xf, xf.dtype(), [&](T v) { return static_cast<T>(op(v)); });
  });
  return {xf, out};
}

// Keep generic strided/tiny inputs on the scalar map. Packed larger inputs
// can use a platform vector function without allocating an extra tensor.
template<typename Scalar, typename Vector>
std::pair<Tensor, Tensor>
vector_float_unary(const Tensor& x, const char* what, Scalar scalar_fn, Vector vector_fn) {
  Tensor xf  = as_dtype(x, to_floating(x.dtype()));
  Tensor out = compute_float(xf.dtype(), what, [&](auto tag) -> Tensor {
    using T = typename decltype(tag)::type;
    if (!xf.is_contiguous() || xf.numel() < 32)
      return detail::map_unary<T, T>(xf, xf.dtype(), scalar_fn);
    Tensor   result = empty(xf.shape(), xf.dtype());
    const T* source = xf.data_ptr<T>();
    T*       dest   = result.data_ptr<T>();
    parallel_for(0, xf.numel(), detail::kParallelGrain, [&](std::int64_t lo, std::int64_t hi) {
      vector_fn(source + lo, dest + lo, hi - lo);
    });
    return result;
  });
  return {xf, out};
}

// ---- special functions -------------------------------------------------------------

double digamma_scalar(double x) {
  if (std::isnan(x)) return x;
  if (x <= 0 && std::floor(x) == x) return std::numeric_limits<double>::quiet_NaN();
  if (x < 0) return digamma_scalar(1 - x) - kPi / std::tan(kPi * x);
  double result = 0;
  while (x < 6) {
    result -= 1 / x;
    x += 1;
  }
  const double f = 1 / (x * x);
  return result + std::log(x) - 0.5 / x -
         f * (1.0 / 12 - f * (1.0 / 120 - f * (1.0 / 252 - f * (1.0 / 240 - f * (1.0 / 132)))));
}

double trigamma_scalar(double x) {
  if (std::isnan(x)) return x;
  if (x <= 0 && std::floor(x) == x) return std::numeric_limits<double>::infinity();
  if (x < 0) {
    const double s = std::sin(kPi * x);
    return kPi * kPi / (s * s) - trigamma_scalar(1 - x);
  }
  double result = 0;
  while (x < 6) {
    result += 1 / (x * x);
    x += 1;
  }
  const double f = 1 / (x * x);
  return result + 1 / x + f / 2 + (f / x) * (1.0 / 6 - f * (1.0 / 30 - f * (1.0 / 42 - f * (1.0 / 30))));
}

template <typename T>
T round_half_even(T v) {
  if constexpr (std::is_floating_point<T>::value)
    return std::nearbyint(v);  // the default rounding mode is to-nearest-even
  else
    return v;
}

template <typename T>
T stable_sigmoid(T v) {
  if (v >= 0) {
    const T e = std::exp(-v);
    return T(1) / (T(1) + e);
  }
  const T e = std::exp(v);
  return e / (T(1) + e);
}

template <typename T>
T stable_softplus(T v) {
  return std::max(v, T(0)) + std::log1p(std::exp(-std::abs(v)));
}

}  // namespace

// ---- scalars -------------------------------------------------------------------------

Tensor scalar_like(double value, const Tensor& like) {
  DType dt = like.dtype();
  if (!is_floating(dt)) {
    if (std::floor(value) == value && std::isfinite(value))
      dt = dt == DType::Bool ? DType::Int64 : dt;
    else
      dt = DType::Float32;
  }
  return scalar(value, dt);
}

// ---- binary arithmetic -----------------------------------------------------------------

Tensor add(const Tensor& a, const Tensor& b) {
  const DType dt  = arithmetic_dtype(a.dtype(), b.dtype());
  Tensor      ac  = as_dtype(a, dt), bc = as_dtype(b, dt);
  Tensor      out = binary_kernel(ac, bc, dt, [](auto x, auto y) { return x + y; });
  const Shape sa = a.shape(), sb = b.shape();
  record("Add", {ac, bc}, {out}, [sa, sb](const Grads& g, const Grads&) {
    return Grads{sum_to(g[0], sa), sum_to(g[0], sb)};
  });
  return out;
}

Tensor sub(const Tensor& a, const Tensor& b) {
  const DType dt  = arithmetic_dtype(a.dtype(), b.dtype());
  Tensor      ac  = as_dtype(a, dt), bc = as_dtype(b, dt);
  Tensor      out = binary_kernel(ac, bc, dt, [](auto x, auto y) { return x - y; });
  const Shape sa = a.shape(), sb = b.shape();
  record("Sub", {ac, bc}, {out}, [sa, sb](const Grads& g, const Grads&) {
    return Grads{sum_to(g[0], sa), sum_to(neg(g[0]), sb)};
  });
  return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
  const DType dt  = arithmetic_dtype(a.dtype(), b.dtype());
  Tensor      ac = as_dtype(a, dt), bc = as_dtype(b, dt);
  Tensor      out = binary_kernel(ac, bc, dt, [](auto x, auto y) { return x * y; });
  if (ac.is_same(bc)) {
    SavedTensor sx(ac);
    // One edge and one contribution instead of two identical gradient buffers
    // followed by accumulation. Scale after g*x to retain overflow behaviour.
    record("Mul", {ac}, {out}, [sx](const Grads& g, const Grads&) {
      const Tensor& value = sx.get();
      if (!is_grad_enabled() && g[0].dtype() == value.dtype())
        return Grads{binary_kernel(g[0], value, value.dtype(), [](auto seed, auto v) {
          const auto product = seed * v;
          return product + product;
        })};
      return Grads{mul(mul(g[0], sx.get()), 2.0)};
    });
    return out;
  }
  SavedTensor sa(ac), sb(bc);
  const bool  need_a = ac.requires_grad(), need_b = bc.requires_grad();
  record("Mul", {ac, bc}, {out}, [sa, sb, need_a, need_b](const Grads& g, const Grads&) {
    const Tensor& x = sa.get();
    const Tensor& y = sb.get();
    return Grads{need_a ? sum_to(mul(g[0], y), x.shape()) : Tensor(),
                 need_b ? sum_to(mul(g[0], x), y.shape()) : Tensor()};
  });
  return out;
}

Tensor div(const Tensor& a, const Tensor& b) {
  const DType dt  = to_floating(promote_types(a.dtype(), b.dtype()));
  Tensor      ac  = as_dtype(a, dt), bc = as_dtype(b, dt);
  Tensor      out = binary_kernel(ac, bc, dt, [](auto x, auto y) { return x / y; });
  SavedTensor sb(bc);
  const Shape sa     = a.shape();
  const bool  need_b = bc.requires_grad();
  record(
      "Div", {ac, bc}, {out},
      [sb, sa, need_b](const Grads& g, const Grads& outs) {
        const Tensor& y = sb.get();
        return Grads{sum_to(div(g[0], y), sa), need_b ? sum_to(neg(div(mul(g[0], outs[0]), y)), y.shape()) : Tensor()};
      },
      need_b);
  return out;
}

Tensor pow(const Tensor& a, const Tensor& b) {
  const DType dt  = to_floating(promote_types(a.dtype(), b.dtype()));
  Tensor      ac  = as_dtype(a, dt), bc = as_dtype(b, dt);
  Tensor      out = binary_kernel(ac, bc, dt, [](auto x, auto y) { return std::pow(x, y); });
  SavedTensor sa(ac), sb(bc);
  const bool  need_a = ac.requires_grad(), need_b = bc.requires_grad();
  record(
      "Pow", {ac, bc}, {out},
      [sa, sb, need_a, need_b](const Grads& g, const Grads& outs) {
        const Tensor& x = sa.get();
        const Tensor& y = sb.get();
        Tensor        ga, gb;
        if (need_a) ga = sum_to(mul(g[0], mul(y, pow(x, sub(y, scalar_like(1.0, y))))), x.shape());
        if (need_b) {
          const Tensor positive = greater(x, 0.0);
          const Tensor safe_log = log(where(positive, x, 1.0));
          gb = sum_to(where(positive, mul(mul(g[0], outs[0]), safe_log), 0.0), y.shape());
        }
        return Grads{ga, gb};
      },
      true);
  return out;
}

Tensor pow(const Tensor& a, double exponent) {
  Tensor      ac  = as_dtype(a, to_floating(a.dtype()));
  Tensor      out = unary_kernel(ac, ac.dtype(), [exponent](auto v) { return std::pow(v, static_cast<decltype(v)>(exponent)); });
  SavedTensor sa(ac);
  record("PowScalar", {ac}, {out}, [sa, exponent](const Grads& g, const Grads&) {
    const Tensor& x = sa.get();
    if (exponent == 0) return Grads{zeros_like(x)};
    if (exponent == 1) return Grads{g[0]};
    if (exponent == 2) return Grads{mul(g[0], mul(x, 2.0))};
    return Grads{mul(g[0], mul(pow(x, exponent - 1), exponent))};
  });
  return out;
}

Tensor maximum(const Tensor& a, const Tensor& b) {
  const DType dt  = arithmetic_dtype(a.dtype(), b.dtype());
  Tensor      ac  = as_dtype(a, dt), bc = as_dtype(b, dt);
  Tensor      out = binary_kernel(ac, bc, dt, [](auto x, auto y) { return x >= y ? x : y; });
  SavedTensor sa(ac), sb(bc);
  record("Maximum", {ac, bc}, {out}, [sa, sb](const Grads& g, const Grads&) {
    const Tensor mask = greater_equal(sa.get(), sb.get());  // ties go to a, as in TF
    const Tensor gb   = where(mask, 0.0, g[0]);
    return Grads{sum_to(where(mask, g[0], 0.0), sa.get().shape()), sum_to(gb, sb.get().shape())};
  });
  return out;
}

Tensor minimum(const Tensor& a, const Tensor& b) {
  const DType dt  = arithmetic_dtype(a.dtype(), b.dtype());
  Tensor      ac  = as_dtype(a, dt), bc = as_dtype(b, dt);
  Tensor      out = binary_kernel(ac, bc, dt, [](auto x, auto y) { return x <= y ? x : y; });
  SavedTensor sa(ac), sb(bc);
  record("Minimum", {ac, bc}, {out}, [sa, sb](const Grads& g, const Grads&) {
    const Tensor mask = less_equal(sa.get(), sb.get());
    return Grads{sum_to(where(mask, g[0], 0.0), sa.get().shape()), sum_to(where(mask, 0.0, g[0]), sb.get().shape())};
  });
  return out;
}

Tensor squared_difference(const Tensor& a, const Tensor& b) {
  const DType dt  = arithmetic_dtype(a.dtype(), b.dtype());
  Tensor      ac  = as_dtype(a, dt), bc = as_dtype(b, dt);
  Tensor      out = binary_kernel(ac, bc, dt, [](auto x, auto y) { return (x - y) * (x - y); });
  SavedTensor sa(ac), sb(bc);
  record("SquaredDifference", {ac, bc}, {out}, [sa, sb](const Grads& g, const Grads&) {
    const Tensor d = mul(mul(g[0], sub(sa.get(), sb.get())), 2.0);
    return Grads{sum_to(d, sa.get().shape()), sum_to(neg(d), sb.get().shape())};
  });
  return out;
}

Tensor atan2(const Tensor& a, const Tensor& b) {
  const DType dt  = to_floating(promote_types(a.dtype(), b.dtype()));
  Tensor      ac  = as_dtype(a, dt), bc = as_dtype(b, dt);
  Tensor      out = binary_kernel(ac, bc, dt, [](auto x, auto y) { return std::atan2(x, y); });
  SavedTensor sa(ac), sb(bc);
  record("Atan2", {ac, bc}, {out}, [sa, sb](const Grads& g, const Grads&) {
    const Tensor& x  = sa.get();
    const Tensor& y  = sb.get();
    const Tensor  r2 = add(square(x), square(y));
    return Grads{sum_to(div(mul(g[0], y), r2), x.shape()), sum_to(neg(div(mul(g[0], x), r2)), y.shape())};
  });
  return out;
}

Tensor add(const Tensor& a, double b) { return add(a, scalar_like(b, a)); }
Tensor sub(const Tensor& a, double b) { return sub(a, scalar_like(b, a)); }
Tensor sub(double a, const Tensor& b) { return sub(scalar_like(a, b), b); }
Tensor mul(const Tensor& a, double b) { return mul(a, scalar_like(b, a)); }
Tensor div(const Tensor& a, double b) { return div(a, scalar_like(b, a)); }
Tensor div(double a, const Tensor& b) { return div(scalar_like(a, b), b); }

Tensor operator+(const Tensor& a, const Tensor& b) { return add(a, b); }
Tensor operator-(const Tensor& a, const Tensor& b) { return sub(a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return mul(a, b); }
Tensor operator/(const Tensor& a, const Tensor& b) { return div(a, b); }
Tensor operator+(const Tensor& a, double b) { return add(a, scalar_like(b, a)); }
Tensor operator-(const Tensor& a, double b) { return sub(a, scalar_like(b, a)); }
Tensor operator*(const Tensor& a, double b) { return mul(a, scalar_like(b, a)); }
Tensor operator/(const Tensor& a, double b) { return div(a, scalar_like(b, a)); }
Tensor operator+(double a, const Tensor& b) { return add(scalar_like(a, b), b); }
Tensor operator-(double a, const Tensor& b) { return sub(scalar_like(a, b), b); }
Tensor operator*(double a, const Tensor& b) { return mul(scalar_like(a, b), b); }
Tensor operator/(double a, const Tensor& b) { return div(scalar_like(a, b), b); }
Tensor operator-(const Tensor& x) { return neg(x); }

// ---- unary math ---------------------------------------------------------------------------

Tensor neg(const Tensor& x) {
  Tensor xc  = as_dtype(x, x.dtype() == DType::Bool ? DType::Int64 : x.dtype());
  Tensor out = unary_kernel(xc, xc.dtype(), [](auto v) { return -v; });
  record("Neg", {xc}, {out}, [](const Grads& g, const Grads&) { return Grads{neg(g[0])}; });
  return out;
}

Tensor square(const Tensor& x) {
  Tensor      xc  = as_dtype(x, x.dtype() == DType::Bool ? DType::Int64 : x.dtype());
  Tensor      out = unary_kernel(xc, xc.dtype(), [](auto v) { return v * v; });
  SavedTensor sx(xc);
  record("Square", {xc}, {out}, [sx](const Grads& g, const Grads&) {
    const Tensor& value = sx.get();
    if (!is_grad_enabled() && g[0].dtype() == value.dtype())
      return Grads{binary_kernel(g[0], value, value.dtype(), [](auto seed, auto v) {
        // Retain Square's existing order: scale x before multiplying by g.
        return seed * (v + v);
      })};
    return Grads{mul(g[0], mul(value, 2.0))};
  });
  return out;
}

Tensor sqrt(const Tensor& x) {
  auto [xf, out] = float_unary(x, "sqrt", [](auto v) { return std::sqrt(v); });
  record(
      "Sqrt", {xf}, {out}, [](const Grads& g, const Grads& o) { return Grads{div(mul(g[0], 0.5), o[0])}; }, true);
  return out;
}

Tensor rsqrt(const Tensor& x) {
  auto [xf, out] = float_unary(x, "rsqrt", [](auto v) { return decltype(v)(1) / std::sqrt(v); });
  record(
      "Rsqrt", {xf}, {out},
      [](const Grads& g, const Grads& o) { return Grads{mul(g[0], mul(pow(o[0], 3.0), -0.5))}; }, true);
  return out;
}

Tensor reciprocal(const Tensor& x) {
  auto [xf, out] = float_unary(x, "reciprocal", [](auto v) { return decltype(v)(1) / v; });
  record(
      "Reciprocal", {xf}, {out}, [](const Grads& g, const Grads& o) { return Grads{neg(mul(g[0], square(o[0])))}; },
      true);
  return out;
}

Tensor exp(const Tensor& x) {
  Tensor xf  = as_dtype(x, to_floating(x.dtype()));
  Tensor out = compute_float(xf.dtype(), "exp", [&](auto tag) -> Tensor {
    using T = typename decltype(tag)::type;
    if (!xf.is_contiguous())
      return detail::map_unary<T, T>(xf, xf.dtype(), [](T v) { return std::exp(v); });
    Tensor   result = empty(xf.shape(), xf.dtype());
    const T* source = xf.data_ptr<T>();
    T*       dest   = result.data_ptr<T>();
    parallel_for(0, xf.numel(), detail::kParallelGrain, [&](std::int64_t lo, std::int64_t hi) {
      detail::exp_values(source + lo, dest + lo, hi - lo);
    });
    return result;
  });
  record(
      "Exp",
      {xf},
      {out},
      [](const Grads& g, const Grads& o) { return Grads{mul(g[0], o[0])}; },
      true);
  return out;
}

Tensor log(const Tensor& x) {
  auto [xf, out] = float_unary(x, "log", [](auto v) { return std::log(v); });
  SavedTensor sx(xf);
  record("Log", {xf}, {out}, [sx](const Grads& g, const Grads&) { return Grads{div(g[0], sx.get())}; });
  return out;
}

Tensor log1p(const Tensor& x) {
  auto [xf, out] = float_unary(x, "log1p", [](auto v) { return std::log1p(v); });
  SavedTensor sx(xf);
  record("Log1p", {xf}, {out}, [sx](const Grads& g, const Grads&) { return Grads{div(g[0], add(sx.get(), 1.0))}; });
  return out;
}

Tensor expm1(const Tensor& x) {
  auto [xf, out] = float_unary(x, "expm1", [](auto v) { return std::expm1(v); });
  record(
      "Expm1", {xf}, {out}, [](const Grads& g, const Grads& o) { return Grads{mul(g[0], add(o[0], 1.0))}; }, true);
  return out;
}

Tensor abs(const Tensor& x) {
  if (x.dtype() == DType::Bool) return x;
  Tensor      out = unary_kernel(x, x.dtype(), [](auto v) { return v < 0 ? -v : v; });
  SavedTensor sx(x);
  record("Abs", {x}, {out}, [sx](const Grads& g, const Grads&) { return Grads{mul(g[0], sign(sx.get()))}; });
  return out;
}

Tensor sign(const Tensor& x) {
  if (x.dtype() == DType::Bool) return x;
  return unary_kernel(x, x.dtype(), [](auto v) { return (v > 0) - (v < 0); });
}

Tensor floor(const Tensor& x) {
  if (!is_floating(x.dtype())) return x;
  return unary_kernel(x, x.dtype(), [](auto v) { return std::floor(v); });
}

Tensor ceil(const Tensor& x) {
  if (!is_floating(x.dtype())) return x;
  return unary_kernel(x, x.dtype(), [](auto v) { return std::ceil(v); });
}

Tensor round(const Tensor& x) {
  if (!is_floating(x.dtype())) return x;
  return unary_kernel(x, x.dtype(), [](auto v) { return round_half_even(v); });
}

Tensor sin(const Tensor& x) {
  auto [xf, out] = float_unary(x, "sin", [](auto v) { return std::sin(v); });
  SavedTensor sx(xf);
  record("Sin", {xf}, {out}, [sx](const Grads& g, const Grads&) { return Grads{mul(g[0], cos(sx.get()))}; });
  return out;
}

Tensor cos(const Tensor& x) {
  auto [xf, out] = float_unary(x, "cos", [](auto v) { return std::cos(v); });
  SavedTensor sx(xf);
  record("Cos", {xf}, {out}, [sx](const Grads& g, const Grads&) { return Grads{neg(mul(g[0], sin(sx.get())))}; });
  return out;
}

Tensor tan(const Tensor& x) {
  auto [xf, out] = float_unary(x, "tan", [](auto v) { return std::tan(v); });
  record(
      "Tan", {xf}, {out}, [](const Grads& g, const Grads& o) { return Grads{mul(g[0], add(square(o[0]), 1.0))}; },
      true);
  return out;
}

Tensor asin(const Tensor& x) {
  auto [xf, out] = float_unary(x, "asin", [](auto v) { return std::asin(v); });
  SavedTensor sx(xf);
  record("Asin", {xf}, {out}, [sx](const Grads& g, const Grads&) {
    return Grads{mul(g[0], rsqrt(sub(1.0, square(sx.get()))))};
  });
  return out;
}

Tensor acos(const Tensor& x) {
  auto [xf, out] = float_unary(x, "acos", [](auto v) { return std::acos(v); });
  SavedTensor sx(xf);
  record("Acos", {xf}, {out}, [sx](const Grads& g, const Grads&) {
    return Grads{neg(mul(g[0], rsqrt(sub(1.0, square(sx.get())))))};
  });
  return out;
}

Tensor atan(const Tensor& x) {
  auto [xf, out] = float_unary(x, "atan", [](auto v) { return std::atan(v); });
  SavedTensor sx(xf);
  record("Atan", {xf}, {out}, [sx](const Grads& g, const Grads&) {
    return Grads{div(g[0], add(square(sx.get()), 1.0))};
  });
  return out;
}

Tensor sinh(const Tensor& x) {
  auto [xf, out] = float_unary(x, "sinh", [](auto v) { return std::sinh(v); });
  SavedTensor sx(xf);
  record("Sinh", {xf}, {out}, [sx](const Grads& g, const Grads&) { return Grads{mul(g[0], cosh(sx.get()))}; });
  return out;
}

Tensor cosh(const Tensor& x) {
  auto [xf, out] = float_unary(x, "cosh", [](auto v) { return std::cosh(v); });
  SavedTensor sx(xf);
  record("Cosh", {xf}, {out}, [sx](const Grads& g, const Grads&) { return Grads{mul(g[0], sinh(sx.get()))}; });
  return out;
}

Tensor erf(const Tensor& x) {
  auto [xf, out] = float_unary(x, "erf", [](auto v) { return std::erf(v); });
  SavedTensor sx(xf);
  record("Erf", {xf}, {out}, [sx](const Grads& g, const Grads&) {
    return Grads{mul(g[0], mul(exp(neg(square(sx.get()))), 2.0 / std::sqrt(kPi)))};
  });
  return out;
}

Tensor erfc(const Tensor& x) {
  auto [xf, out] = float_unary(x, "erfc", [](auto v) { return std::erfc(v); });
  SavedTensor sx(xf);
  record("Erfc", {xf}, {out}, [sx](const Grads& g, const Grads&) {
    return Grads{mul(g[0], mul(exp(neg(square(sx.get()))), -2.0 / std::sqrt(kPi)))};
  });
  return out;
}

Tensor lgamma(const Tensor& x) {
  auto [xf, out] = float_unary(x, "lgamma", [](auto v) { return std::lgamma(v); });
  SavedTensor sx(xf);
  record("Lgamma", {xf}, {out}, [sx](const Grads& g, const Grads&) { return Grads{mul(g[0], digamma(sx.get()))}; });
  return out;
}

Tensor digamma(const Tensor& x) {
  auto [xf, out] = float_unary(x, "digamma", [](auto v) { return digamma_scalar(static_cast<double>(v)); });
  SavedTensor sx(xf);
  record("Digamma", {xf}, {out}, [sx](const Grads& g, const Grads&) { return Grads{mul(g[0], trigamma(sx.get()))}; });
  return out;
}

Tensor trigamma(const Tensor& x) {
  NoGradGuard guard;
  return float_unary(x, "trigamma", [](auto v) { return trigamma_scalar(static_cast<double>(v)); }).second;
}

Tensor clip(const Tensor& x, double low, double high) {
  if (low > high) throw std::invalid_argument("clip: low > high");
  Tensor out = unary_kernel(x, x.dtype(), [low, high](auto v) {
    using T = decltype(v);
    return v < static_cast<T>(low) ? static_cast<T>(low) : (v > static_cast<T>(high) ? static_cast<T>(high) : v);
  });
  SavedTensor sx(x);
  record("ClipByValue", {x}, {out}, [sx, low, high](const Grads& g, const Grads&) {
    const Tensor inside = logical_and(greater_equal(sx.get(), low), less_equal(sx.get(), high));
    return Grads{where(inside, g[0], 0.0)};
  });
  return out;
}

Tensor clip(const Tensor& x, const Tensor& low, const Tensor& high) { return minimum(maximum(x, low), high); }

// ---- activations ----------------------------------------------------------------------------

/// g where x > 0, alpha * g elsewhere, in one pass. Linear in g (and its
/// own gradient), so second-order gradients still flow; constant in x a.e.
Tensor leaky_relu_backward(const Tensor& x, const Tensor& g, double alpha) {
  if (x.dtype() != g.dtype() || g.shape() != x.shape())
    return where(greater(x, 0.0), g, alpha == 0.0 ? zeros({}, g.dtype()) : mul(g, alpha));
  Tensor out = compute(g.dtype(), [&](auto tag) -> Tensor {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same<T, bool>::value) {
      throw std::logic_error("leaky_relu_backward: bool");
    } else {
      const T a = static_cast<T>(alpha);
      return detail::map_binary<T, T>(x, g, g.dtype(), [a](T xv, T gv) { return xv > T(0) ? gv : a * gv; });
    }
  });
  SavedTensor sx(x);
  record("LeakyReluGrad", {g}, {out}, [sx, alpha](const Grads& gg, const Grads&) {
    return Grads{leaky_relu_backward(sx.get(), gg[0], alpha)};
  });
  return out;
}

Tensor relu(const Tensor& x) {
  Tensor      out = unary_kernel(x, x.dtype(), [](auto v) { return v > 0 ? v : decltype(v)(0); });
  // The mask out > 0 equals x > 0, so keep the output (often kept alive by
  // the next op anyway) instead of the input.
  record(
      "Relu", {x}, {out}, [](const Grads& g, const Grads& outs) { return Grads{leaky_relu_backward(outs[0].detach(), g[0], 0.0)}; },
      /*save_outputs=*/true);
  return out;
}

Tensor relu6(const Tensor& x) {
  Tensor out = unary_kernel(x, x.dtype(), [](auto v) {
    using T = decltype(v);
    return v < T(0) ? T(0) : (v > T(6) ? T(6) : v);
  });
  // 0 < out < 6 exactly where 0 < x < 6, so the output is enough.
  record(
      "Relu6", {x}, {out},
      [](const Grads& g, const Grads& outs) {
        const Tensor y = outs[0].detach();
        return Grads{where(logical_and(greater(y, 0.0), less(y, 6.0)), g[0], 0.0)};
      },
      /*save_outputs=*/true);
  return out;
}

Tensor leaky_relu(const Tensor& x, double alpha) {
  Tensor xf  = as_dtype(x, to_floating(x.dtype()));
  Tensor out = unary_kernel(xf, xf.dtype(), [alpha](auto v) { return v > 0 ? v : static_cast<decltype(v)>(alpha) * v; });
  SavedTensor sx(xf);
  record("LeakyRelu", {xf}, {out}, [sx, alpha](const Grads& g, const Grads&) {
    return Grads{leaky_relu_backward(sx.get(), g[0], alpha)};
  });
  return out;
}

Tensor elu(const Tensor& x, double alpha) {
  Tensor xf  = as_dtype(x, to_floating(x.dtype()));
  Tensor out = unary_kernel(xf, xf.dtype(), [alpha](auto v) {
    using T = decltype(v);
    return v > 0 ? v : static_cast<T>(alpha) * std::expm1(v);
  });
  SavedTensor sx(xf);
  record(
      "Elu", {xf}, {out},
      [sx, alpha](const Grads& g, const Grads& o) {
        // x >= 0 takes the linear branch, as TF EluGrad does at x = 0.
        return Grads{where(greater_equal(sx.get(), 0.0), g[0], mul(g[0], add(o[0], alpha)))};
      },
      true);
  return out;
}

Tensor selu(const Tensor& x) {
  constexpr double scale = 1.0507009873554804934193349852946;
  constexpr double alpha = 1.6732632423543772848170429916717;
  Tensor           xf    = as_dtype(x, to_floating(x.dtype()));
  Tensor           out   = unary_kernel(xf, xf.dtype(), [](auto v) {
    using T = decltype(v);
    return static_cast<T>(scale) * (v > 0 ? v : static_cast<T>(alpha) * std::expm1(v));
  });
  SavedTensor sx(xf);
  record(
      "Selu", {xf}, {out},
      [sx](const Grads& g, const Grads& o) {
        // x >= 0 takes the linear branch, as TF SeluGrad does at x = 0.
        return Grads{where(greater_equal(sx.get(), 0.0), mul(g[0], scale), mul(g[0], add(o[0], scale * alpha)))};
      },
      true);
  return out;
}

Tensor sigmoid(const Tensor& x) {
  auto [xf, out] = vector_float_unary(
      x,
      "sigmoid",
      [](auto v) { return stable_sigmoid(v); },
      [](const auto* source, auto* dest, std::int64_t count) {
        detail::sigmoid_values(source, dest, count);
      });
  record(
      "Sigmoid",
      {xf},
      {out},
      [](const Grads& g, const Grads& o) { return Grads{mul(g[0], mul(o[0], sub(1.0, o[0])))}; },
      true);
  return out;
}

Tensor tanh(const Tensor& x) {
  auto [xf, out] = vector_float_unary(
      x,
      "tanh",
      [](auto v) { return std::tanh(v); },
      [](const auto* source, auto* dest, std::int64_t count) {
        detail::tanh_values(source, dest, count);
      });
  record(
      "Tanh",
      {xf},
      {out},
      [](const Grads& g, const Grads& o) { return Grads{mul(g[0], sub(1.0, square(o[0])))}; },
      true);
  return out;
}

Tensor softplus(const Tensor& x) {
  auto [xf, out] = float_unary(x, "softplus", [](auto v) { return stable_softplus(v); });
  SavedTensor sx(xf);
  record("Softplus", {xf}, {out}, [sx](const Grads& g, const Grads&) { return Grads{mul(g[0], sigmoid(sx.get()))}; });
  return out;
}

Tensor softsign(const Tensor& x) {
  auto [xf, out] = float_unary(x, "softsign", [](auto v) { return v / (decltype(v)(1) + std::abs(v)); });
  SavedTensor sx(xf);
  record("Softsign", {xf}, {out}, [sx](const Grads& g, const Grads&) {
    return Grads{div(g[0], square(add(abs(sx.get()), 1.0)))};
  });
  return out;
}

Tensor silu(const Tensor& x) { return mul(x, sigmoid(x)); }

Tensor gelu(const Tensor& x, bool approximate) {
  Tensor xf = as_dtype(x, to_floating(x.dtype()));
  if (approximate) {
    const double c     = std::sqrt(2.0 / kPi);
    Tensor       inner = mul(add(xf, mul(pow(xf, 3.0), 0.044715)), c);
    return mul(mul(xf, 0.5), add(tanh(inner), 1.0));
  }
  if (!is_grad_enabled() || !xf.requires_grad()) {
    return compute_float(xf.dtype(), "gelu", [&](auto tag) {
      using T       = typename decltype(tag)::type;
      const T scale = static_cast<T>(1.0 / std::sqrt(2.0));
      return detail::map_unary<T, T>(xf, xf.dtype(), [scale](T v) {
        const T left = v * T(0.5), scaled = v * scale;
        const T factor = T(1) + std::erf(scaled);
        return left * factor;
      });
    });
  }
  return mul(mul(xf, 0.5), add(erf(mul(xf, 1.0 / std::sqrt(2.0))), 1.0));
}

Tensor mish(const Tensor& x) { return mul(x, tanh(softplus(x))); }

Tensor log_sigmoid(const Tensor& x) { return neg(softplus(neg(x))); }

// ---- comparisons -----------------------------------------------------------------------------

namespace {

template <typename Op>
Tensor compare(const Tensor& a, const Tensor& b, Op op) {
  const DType dt = promote_types(a.dtype(), b.dtype());
  NoGradGuard guard;
  Tensor      ac = detail::cast_raw(nograd(a), dt), bc = detail::cast_raw(nograd(b), dt);
  return dispatch(dt, [&](auto tag) {
    using T = typename decltype(tag)::type;
    return detail::map_binary<T, bool>(ac, bc, DType::Bool, [&](T x, T y) { return op(x, y); });
  });
}

Tensor as_bool(const Tensor& x) {
  NoGradGuard guard;
  return detail::cast_raw(nograd(x), DType::Bool);
}

}  // namespace

Tensor equal(const Tensor& a, const Tensor& b) { return compare(a, b, [](auto x, auto y) { return x == y; }); }
Tensor not_equal(const Tensor& a, const Tensor& b) { return compare(a, b, [](auto x, auto y) { return x != y; }); }
Tensor less(const Tensor& a, const Tensor& b) { return compare(a, b, [](auto x, auto y) { return x < y; }); }
Tensor less_equal(const Tensor& a, const Tensor& b) { return compare(a, b, [](auto x, auto y) { return x <= y; }); }
Tensor greater(const Tensor& a, const Tensor& b) { return compare(a, b, [](auto x, auto y) { return x > y; }); }
Tensor greater_equal(const Tensor& a, const Tensor& b) { return compare(a, b, [](auto x, auto y) { return x >= y; }); }

namespace {
/// A scalar for comparing against `like` without losing precision: a float
/// dtype for non-integral values so `int_tensor < 2.5` behaves.
Tensor compare_scalar(double v, const Tensor& like) {
  if (is_floating(like.dtype())) return scalar(v, like.dtype());
  if (std::floor(v) == v) return scalar(v, like.dtype() == DType::Bool ? DType::Int64 : like.dtype());
  return scalar(v, DType::Float64);
}
}  // namespace

Tensor equal(const Tensor& a, double b) { return equal(a, compare_scalar(b, a)); }
Tensor not_equal(const Tensor& a, double b) { return not_equal(a, compare_scalar(b, a)); }
Tensor less(const Tensor& a, double b) { return less(a, compare_scalar(b, a)); }
Tensor less_equal(const Tensor& a, double b) { return less_equal(a, compare_scalar(b, a)); }
Tensor greater(const Tensor& a, double b) { return greater(a, compare_scalar(b, a)); }
Tensor greater_equal(const Tensor& a, double b) { return greater_equal(a, compare_scalar(b, a)); }

Tensor logical_and(const Tensor& a, const Tensor& b) {
  return compare(as_bool(a), as_bool(b), [](auto x, auto y) { return x && y; });
}
Tensor logical_or(const Tensor& a, const Tensor& b) {
  return compare(as_bool(a), as_bool(b), [](auto x, auto y) { return x || y; });
}
Tensor logical_xor(const Tensor& a, const Tensor& b) {
  return compare(as_bool(a), as_bool(b), [](auto x, auto y) { return x != y; });
}
Tensor logical_not(const Tensor& x) {
  NoGradGuard guard;
  return detail::map_unary<bool, bool>(as_bool(x), DType::Bool, [](bool v) { return !v; });
}

Tensor is_finite(const Tensor& x) {
  if (!is_floating(x.dtype())) return full(x.shape(), 1.0, DType::Bool);
  NoGradGuard guard;
  return dispatch_float(x.dtype(), "is_finite", [&](auto tag) {
    using T = typename decltype(tag)::type;
    return detail::map_unary<T, bool>(x, DType::Bool, [](T v) { return std::isfinite(v); });
  });
}

Tensor is_nan(const Tensor& x) {
  if (!is_floating(x.dtype())) return zeros(x.shape(), DType::Bool);
  NoGradGuard guard;
  return dispatch_float(x.dtype(), "is_nan", [&](auto tag) {
    using T = typename decltype(tag)::type;
    return detail::map_unary<T, bool>(x, DType::Bool, [](T v) { return std::isnan(v); });
  });
}

// ---- where ---------------------------------------------------------------------------------------

Tensor where(const Tensor& condition, const Tensor& a, const Tensor& b) {
  const DType dt   = promote_types(a.dtype(), b.dtype());
  Tensor      ac   = as_dtype(a, dt), bc = as_dtype(b, dt);
  Tensor      cond = as_bool(condition);
  const Shape shape = broadcast_shapes(broadcast_shapes(cond.shape(), ac.shape()), bc.shape());
  Tensor      out;
  {
    NoGradGuard guard;
    out = dispatch(dt, [&](auto tag) {
      using T          = typename decltype(tag)::type;
      Tensor     o     = empty(shape, dt);
      T*         dst   = o.data_ptr<T>();
      const bool* pc   = detail::raw_data<bool>(cond);
      const T*   pa    = detail::raw_data<T>(ac);
      const T*   pb    = detail::raw_data<T>(bc);
      const Shape so   = contiguous_strides(shape);
      const Shape sc   = detail::broadcast_strides(cond, shape);
      const Shape sa   = detail::broadcast_strides(ac, shape);
      const Shape sb   = detail::broadcast_strides(bc, shape);
      const std::array<const Shape*, 4> st{&so, &sc, &sa, &sb};
      detail::parallel_strided_rows<4>(shape, st, {0, cond.offset(), ac.offset(), bc.offset()},
                              [&](const auto& off, std::int64_t n, const auto& step) {
                                for (std::int64_t j = 0; j < n; ++j)
                                  dst[off[0] + j] = pc[off[1] + j * step[1]] ? pa[off[2] + j * step[2]]
                                                                             : pb[off[3] + j * step[3]];
                              });
      return o;
    });
  }
  const Shape sa = a.shape(), sb = b.shape();
  record("SelectV2", {ac, bc}, {out}, [cond, sa, sb](const Grads& g, const Grads&) {
    const Tensor zero = zeros({}, g[0].dtype());
    return Grads{sum_to(where(cond, g[0], zero), sa), sum_to(where(cond, zero, g[0]), sb)};
  });
  return out;
}

Tensor where(const Tensor& condition, const Tensor& a, double b) { return where(condition, a, scalar_like(b, a)); }
Tensor where(const Tensor& condition, double a, const Tensor& b) { return where(condition, scalar_like(a, b), b); }

}  // namespace datamunge::nn
