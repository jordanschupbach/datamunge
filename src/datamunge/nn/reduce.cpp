// Reductions, softmax, cumsum.

#include <cmath>
#include <limits>

#include "detail.hpp"

namespace datamunge::nn {

using detail::as_dtype;
using detail::dispatch;
using detail::nograd;

namespace {

using Grads = std::vector<Tensor>;

/// x viewed as a packed [outer, inner] block: kept axes first, reduced axes last.
struct Packed {
  Tensor       data;
  std::int64_t outer{1};
  std::int64_t inner{1};
  Shape        out_shape;   // result shape honouring keepdims
  Shape        keep_shape;  // result shape with keepdims = true
};

Packed pack_for_reduce(const Tensor& x, const std::vector<std::int64_t>& axes, bool keepdims) {
  const std::int64_t rank = x.dim();
  std::vector<bool>  reduced(static_cast<std::size_t>(rank), false);
  for (auto a : axes) reduced[static_cast<std::size_t>(a)] = true;
  Shape   perm_shape, perm_strides;
  Packed  p;
  for (std::int64_t i = 0; i < rank; ++i) {
    const auto k = static_cast<std::size_t>(i);
    if (reduced[k]) continue;
    perm_shape.push_back(x.shape()[k]);
    perm_strides.push_back(x.strides()[k]);
    p.outer *= x.shape()[k];
    p.out_shape.push_back(x.shape()[k]);
  }
  for (std::int64_t i = 0; i < rank; ++i) {
    const auto k = static_cast<std::size_t>(i);
    if (!reduced[k]) continue;
    perm_shape.push_back(x.shape()[k]);
    perm_strides.push_back(x.strides()[k]);
    p.inner *= x.shape()[k];
  }
  p.keep_shape = detail::keepdims_shape(x.shape(), axes);
  if (keepdims) p.out_shape = p.keep_shape;
  NoGradGuard guard;
  p.data = detail::make_view(nograd(x), perm_shape, perm_strides, x.offset()).contiguous();
  return p;
}

/// Folds each row of the packed block with `step(acc, value)` starting from `init`.
template <typename T, typename Acc, typename Out, typename Step, typename Finish>
Tensor fold_rows(const Packed& p, DType out_dtype, Acc init, Step step, Finish finish) {
  Tensor   out = empty(p.out_shape, out_dtype);
  Out*     dst = out.data_ptr<Out>();
  const T* src = p.data.data_ptr<T>();
  for (std::int64_t o = 0; o < p.outer; ++o) {
    Acc      acc = init;
    const T* row = src + o * p.inner;
    for (std::int64_t i = 0; i < p.inner; ++i) acc = step(acc, row[i]);
    dst[o] = static_cast<Out>(finish(acc, p.inner));
  }
  return out;
}

/// Broadcasts a reduced gradient back to the input shape.
Tensor expand_reduced(const Tensor& g, const Shape& keep_shape, const Shape& input_shape) {
  return broadcast_to(reshape(g, keep_shape), input_shape);
}

}  // namespace

Tensor sum(const Tensor& x, const Axes& axes_in, bool keepdims) {
  const auto  axes = detail::normalize_axes(axes_in, x.dim(), "sum");
  const DType dt   = x.dtype() == DType::Bool ? DType::Int64 : x.dtype();
  Tensor      xc   = as_dtype(x, dt);
  Packed      p    = pack_for_reduce(xc, axes, keepdims);
  Tensor      out  = dispatch(dt, [&](auto tag) -> Tensor {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same<T, bool>::value) {
      throw std::logic_error("sum: bool");
    } else {
      using Acc = typename std::conditional<std::is_floating_point<T>::value, double, std::int64_t>::type;
      return fold_rows<T, Acc, T>(p, dt, Acc(0), [](Acc a, T v) { return a + static_cast<Acc>(v); },
                                  [](Acc a, std::int64_t) { return a; });
    }
  });
  const Shape keep = p.keep_shape, in_shape = x.shape();
  record("Sum", {xc}, {out}, [keep, in_shape](const Grads& g, const Grads&) {
    return Grads{expand_reduced(g[0], keep, in_shape)};
  });
  return out;
}

Tensor mean(const Tensor& x, const Axes& axes_in, bool keepdims) {
  const auto   axes  = detail::normalize_axes(axes_in, x.dim(), "mean");
  std::int64_t count = 1;
  for (auto a : axes) count *= x.shape()[static_cast<std::size_t>(a)];
  Tensor xf = as_dtype(x, to_floating(x.dtype()));
  return div(sum(xf, axes, keepdims), scalar(static_cast<double>(count), xf.dtype()));
}

namespace {

template <bool IsMax>
Tensor extremum(const Tensor& x, const Axes& axes_in, bool keepdims, const char* name) {
  const auto axes = detail::normalize_axes(axes_in, x.dim(), name);
  for (auto a : axes)
    if (x.shape()[static_cast<std::size_t>(a)] == 0) throw std::invalid_argument(std::string(name) + ": empty axis");
  Packed p   = pack_for_reduce(x, axes, keepdims);
  Tensor out = dispatch(x.dtype(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    return fold_rows<T, T, T>(
        p, x.dtype(), IsMax ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max(),
        [](T a, T v) {
          if (v != v) return v;  // NaN propagates
          return IsMax ? (v > a ? v : a) : (v < a ? v : a);
        },
        [](T a, std::int64_t) { return a; });
  });
  SavedTensor sx(x);
  const Shape keep = p.keep_shape;
  record(
      name, {x}, {out},
      [sx, keep, axes](const Grads& g, const Grads& o) {
        // Share the gradient equally between tied extrema (TF Max/Min).
        const Tensor& in    = sx.get();
        const Tensor  mask  = equal(in, reshape(o[0], keep));
        const Tensor  ties  = sum(cast(mask, in.dtype()), axes, true);
        const Tensor  share = div(reshape(g[0], keep), ties);
        return Grads{where(mask, broadcast_to(share, in.shape()), 0.0)};
      },
      true);
  return out;
}

}  // namespace

Tensor max(const Tensor& x, const Axes& axes, bool keepdims) { return extremum<true>(x, axes, keepdims, "Max"); }
Tensor min(const Tensor& x, const Axes& axes, bool keepdims) { return extremum<false>(x, axes, keepdims, "Min"); }

Tensor prod(const Tensor& x, const Axes& axes_in, bool keepdims) {
  const auto  axes = detail::normalize_axes(axes_in, x.dim(), "prod");
  const DType dt   = x.dtype() == DType::Bool ? DType::Int64 : x.dtype();
  Tensor      xc   = as_dtype(x, dt);
  Packed      p    = pack_for_reduce(xc, axes, keepdims);
  Tensor      out  = dispatch(dt, [&](auto tag) -> Tensor {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same<T, bool>::value) {
      throw std::logic_error("prod: bool");
    } else {
      using Acc = typename std::conditional<std::is_floating_point<T>::value, double, std::int64_t>::type;
      return fold_rows<T, Acc, T>(p, dt, Acc(1), [](Acc a, T v) { return a * static_cast<Acc>(v); },
                                  [](Acc a, std::int64_t) { return a; });
    }
  });
  SavedTensor sx(xc);
  const Shape keep = p.keep_shape;
  record(
      "Prod", {xc}, {out},
      [sx, keep, axes](const Grads& g, const Grads& o) {
        // d prod / d x_i is the product of the other elements: prod / x_i
        // when nothing in the group is zero, else computed from the zero count.
        const Tensor& in      = sx.get();
        const Tensor  is_zero = equal(in, 0.0);
        const Tensor  zeros_n = sum(cast(is_zero, DType::Int64), axes, true);
        Tensor        nonzero_prod;
        {
          NoGradGuard guard;
          nonzero_prod = prod(where(is_zero, 1.0, nograd(in)), axes, true);
        }
        const Tensor safe   = where(is_zero, 1.0, in);
        const Tensor no_zero = equal(zeros_n, 0.0);
        const Tensor one_zero = equal(zeros_n, 1.0);
        const Tensor at_nonzero = where(no_zero, div(reshape(o[0], keep), safe), 0.0);
        const Tensor at_zero    = where(one_zero, broadcast_to(nonzero_prod, in.shape()), 0.0);
        const Tensor local      = where(is_zero, at_zero, at_nonzero);
        return Grads{mul(broadcast_to(reshape(g[0], keep), in.shape()), local)};
      },
      true);
  return out;
}

Tensor variance(const Tensor& x, const Axes& axes, bool keepdims) {
  Tensor xf = as_dtype(x, to_floating(x.dtype()));
  return mean(square(sub(xf, mean(xf, axes, true))), axes, keepdims);
}

Tensor std_dev(const Tensor& x, const Axes& axes, bool keepdims) { return sqrt(variance(x, axes, keepdims)); }

Tensor logsumexp(const Tensor& x, const Axes& axes_in, bool keepdims) {
  const auto axes = detail::normalize_axes(axes_in, x.dim(), "logsumexp");
  Tensor     xf   = as_dtype(x, to_floating(x.dtype()));
  Tensor     m;
  {
    NoGradGuard guard;
    m = max(nograd(xf), axes, true);
    m = where(is_finite(m), m, 0.0);
  }
  Tensor out = add(log(sum(exp(sub(xf, m)), axes, true)), m);
  return keepdims ? out : squeeze(out, axes);
}

namespace {

template <bool IsMax>
Tensor arg_extremum(const Tensor& x, std::int64_t axis_in, bool keepdims, const char* name) {
  const std::int64_t axis = detail::normalize_axis(axis_in, x.dim(), name);
  if (x.shape()[static_cast<std::size_t>(axis)] == 0) throw std::invalid_argument(std::string(name) + ": empty axis");
  NoGradGuard guard;
  Packed      p = pack_for_reduce(x, {axis}, keepdims);
  return dispatch(x.dtype(), [&](auto tag) {
    using T             = typename decltype(tag)::type;
    Tensor       out    = empty(p.out_shape, DType::Int64);
    std::int64_t* dst   = out.data_ptr<std::int64_t>();
    const T*     src    = p.data.template data_ptr<T>();
    for (std::int64_t o = 0; o < p.outer; ++o) {
      const T*     row  = src + o * p.inner;
      std::int64_t best = 0;
      for (std::int64_t i = 1; i < p.inner; ++i) {
        if (row[best] != row[best]) break;  // NaN wins
        if (row[i] != row[i] || (IsMax ? row[i] > row[best] : row[i] < row[best])) best = i;
      }
      dst[o] = best;
    }
    return out;
  });
}

template <bool IsAll>
Tensor bool_reduce(const Tensor& x, const Axes& axes_in, bool keepdims) {
  const auto  axes = detail::normalize_axes(axes_in, x.dim(), IsAll ? "all" : "any");
  NoGradGuard guard;
  Packed      p = pack_for_reduce(detail::cast_raw(nograd(x), DType::Bool), axes, keepdims);
  return fold_rows<bool, bool, bool>(
      p, DType::Bool, IsAll, [](bool a, bool v) { return IsAll ? (a && v) : (a || v); },
      [](bool a, std::int64_t) { return a; });
}

}  // namespace

Tensor argmax(const Tensor& x, std::int64_t axis, bool keepdims) { return arg_extremum<true>(x, axis, keepdims, "argmax"); }
Tensor argmin(const Tensor& x, std::int64_t axis, bool keepdims) {
  return arg_extremum<false>(x, axis, keepdims, "argmin");
}
Tensor any(const Tensor& x, const Axes& axes, bool keepdims) { return bool_reduce<false>(x, axes, keepdims); }
Tensor all(const Tensor& x, const Axes& axes, bool keepdims) { return bool_reduce<true>(x, axes, keepdims); }

Tensor count_nonzero(const Tensor& x, const Axes& axes, bool keepdims) {
  NoGradGuard guard;
  return sum(cast(not_equal(nograd(x), 0.0), DType::Int64), axes, keepdims);
}

Tensor softmax(const Tensor& x, std::int64_t axis_in) {
  const std::int64_t axis = detail::normalize_axis(axis_in, x.dim(), "softmax");
  Tensor             xf   = as_dtype(x, to_floating(x.dtype()));
  Tensor             out;
  {
    NoGradGuard guard;
    const Tensor xd = nograd(xf);
    const Tensor e  = exp(sub(xd, max(xd, {axis}, true)));
    out             = div(e, sum(e, {axis}, true));
  }
  record(
      "Softmax", {xf}, {out},
      [axis](const Grads& g, const Grads& o) {
        const Tensor& s = o[0];
        return Grads{mul(s, sub(g[0], sum(mul(g[0], s), {axis}, true)))};
      },
      true);
  return out;
}

Tensor log_softmax(const Tensor& x, std::int64_t axis_in) {
  const std::int64_t axis = detail::normalize_axis(axis_in, x.dim(), "log_softmax");
  Tensor             xf   = as_dtype(x, to_floating(x.dtype()));
  Tensor             out;
  {
    NoGradGuard guard;
    const Tensor xd      = nograd(xf);
    const Tensor shifted = sub(xd, max(xd, {axis}, true));
    out                  = sub(shifted, log(sum(exp(shifted), {axis}, true)));
  }
  record(
      "LogSoftmax", {xf}, {out},
      [axis](const Grads& g, const Grads& o) {
        return Grads{sub(g[0], mul(exp(o[0]), sum(g[0], {axis}, true)))};
      },
      true);
  return out;
}

Tensor cumsum(const Tensor& x, std::int64_t axis_in, bool exclusive, bool reverse) {
  const std::int64_t axis = detail::normalize_axis(axis_in, x.dim(), "cumsum");
  const DType        dt   = x.dtype() == DType::Bool ? DType::Int64 : x.dtype();
  Tensor             xc   = as_dtype(x, dt);
  Tensor             out;
  {
    NoGradGuard guard;
    // Move the axis last, scan rows, move it back.
    Axes perm;
    for (std::int64_t i = 0; i < x.dim(); ++i)
      if (i != axis) perm.push_back(i);
    perm.push_back(axis);
    const Tensor       moved = transpose(nograd(xc), perm).contiguous();
    const std::int64_t n     = x.shape()[static_cast<std::size_t>(axis)];
    const std::int64_t rows  = n ? moved.numel() / n : 0;
    Tensor             scanned = empty(moved.shape(), dt);
    dispatch(dt, [&](auto tag) {
      using T = typename decltype(tag)::type;
      if constexpr (!std::is_same<T, bool>::value) {
        using Acc  = typename std::conditional<std::is_floating_point<T>::value, double, std::int64_t>::type;
        const T* s = moved.data_ptr<T>();
        T*       d = scanned.data_ptr<T>();
        for (std::int64_t r = 0; r < rows; ++r) {
          Acc acc = 0;
          for (std::int64_t k = 0; k < n; ++k) {
            const std::int64_t i = r * n + (reverse ? n - 1 - k : k);
            if (exclusive) {
              d[i] = static_cast<T>(acc);
              acc += s[i];
            } else {
              acc += s[i];
              d[i] = static_cast<T>(acc);
            }
          }
        }
      }
    });
    Axes inverse(perm.size());
    for (std::size_t i = 0; i < perm.size(); ++i) inverse[static_cast<std::size_t>(perm[i])] = static_cast<std::int64_t>(i);
    out = transpose(scanned, inverse).contiguous();
    if (out.is_same(scanned)) out = scanned;
  }
  record("Cumsum", {xc}, {out}, [axis, exclusive, reverse](const Grads& g, const Grads&) {
    return Grads{cumsum(g[0], axis, exclusive, !reverse)};
  });
  return out;
}

Tensor norm(const Tensor& x, const Axes& axes, bool keepdims) {
  return sqrt(sum(square(as_dtype(x, to_floating(x.dtype()))), axes, keepdims));
}

}  // namespace datamunge::nn
