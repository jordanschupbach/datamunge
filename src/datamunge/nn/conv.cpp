// Convolution (im2col + GEMM), pooling and batch normalization.
//
// conv2d, conv2d_backprop_input and conv2d_backprop_filter are three
// primitives whose backward passes are written in terms of each other, so
// gradients of any order flow through convolutions.

#include "datamunge/nn/conv.hpp"

#include <cstring>
#include <limits>
#include <map>
#include <mutex>

#include "datamunge/nn/parallel.hpp"
#include "detail.hpp"
#include "gemm.hpp"

namespace datamunge::nn {

using detail::as_dtype;
using detail::dispatch_float;
using detail::gemm;
using detail::nograd;

namespace {

using I64   = std::int64_t;
using Grads = std::vector<Tensor>;

/// Upper bound on the im2col buffer; larger batches are processed in chunks of images.
constexpr I64 kColBudgetBytes = I64{64} << 20;

struct Axis {
  I64 out{0}, pad_before{0};
};

Axis axis_geometry(I64 in, I64 k, I64 s, I64 d, Padding p, I64 explicit_before, I64 explicit_after) {
  const I64 eff = d * (k - 1) + 1;
  Axis      a;
  switch (p) {
    case Padding::Same: {
      a.out          = (in + s - 1) / s;
      const I64 total = std::max<I64>((a.out - 1) * s + eff - in, 0);
      a.pad_before   = total / 2;
      break;
    }
    case Padding::Valid:
      a.out = in >= eff ? (in - eff) / s + 1 : 0;
      break;
    case Padding::Explicit: {
      const I64 padded = in + explicit_before + explicit_after;
      a.out            = padded >= eff ? (padded - eff) / s + 1 : 0;
      a.pad_before     = explicit_before;
      break;
    }
  }
  return a;
}

struct Geometry {
  I64 n, h, w, c;       // input
  I64 kh, kw, o;        // filter
  I64 oh, ow;           // output
  I64 sh, sw, dh, dw;   // strides, dilations
  I64 pt, pl;           // top / left padding
  I64 groups, cg, og;   // channels per group
  [[nodiscard]] I64  kc() const { return kh * kw * cg; }
  [[nodiscard]] I64  rows_per_image() const { return oh * ow; }
  /// A 1x1, stride-1, unpadded, ungrouped convolution is a plain matmul over pixels.
  [[nodiscard]] bool pointwise() const {
    return kh == 1 && kw == 1 && sh == 1 && sw == 1 && pt == 0 && pl == 0 && oh == h && ow == w && groups == 1;
  }
  [[nodiscard]] Shape out_shape() const { return {n, oh, ow, o}; }
};

void check_options(const Conv2DOptions& o, const char* what) {
  for (auto s : o.strides)
    if (s < 1) throw std::invalid_argument(std::string(what) + ": strides must be >= 1");
  for (auto d : o.dilations)
    if (d < 1) throw std::invalid_argument(std::string(what) + ": dilations must be >= 1");
  if (o.groups < 1) throw std::invalid_argument(std::string(what) + ": groups must be >= 1");
}

Geometry geometry(const Shape& xs, const Shape& fs, const Conv2DOptions& o, const char* what) {
  check_options(o, what);
  if (xs.size() != 4) throw std::invalid_argument(std::string(what) + ": input must be [N, H, W, C], got " + shape_str(xs));
  if (fs.size() != 4)
    throw std::invalid_argument(std::string(what) + ": filter must be [KH, KW, C_in, C_out], got " + shape_str(fs));
  Geometry g{};
  g.n = xs[0], g.h = xs[1], g.w = xs[2], g.c = xs[3];
  g.kh = fs[0], g.kw = fs[1], g.o = fs[3];
  g.groups = o.groups;
  if (fs[2] * o.groups != g.c || g.o % o.groups != 0)
    throw std::invalid_argument(std::string(what) + ": filter " + shape_str(fs) + " does not fit input " + shape_str(xs) +
                                " with groups=" + std::to_string(o.groups));
  g.cg = fs[2], g.og = g.o / o.groups;
  g.sh = o.strides[0], g.sw = o.strides[1], g.dh = o.dilations[0], g.dw = o.dilations[1];
  const Axis ay = axis_geometry(g.h, g.kh, g.sh, g.dh, o.padding, o.explicit_paddings[0], o.explicit_paddings[1]);
  const Axis ax = axis_geometry(g.w, g.kw, g.sw, g.dw, o.padding, o.explicit_paddings[2], o.explicit_paddings[3]);
  g.oh = ay.out, g.ow = ax.out, g.pt = ay.pad_before, g.pl = ax.pad_before;
  return g;
}

DType float_type(const Tensor& a, const Tensor& b) {
  const DType dt = promote_types(a.dtype(), b.dtype());
  return is_floating(dt) ? dt : DType::Float32;
}

I64 images_per_chunk(const Geometry& g, std::size_t elem) {
  const I64 per_image = std::max<I64>(1, g.rows_per_image() * g.kc() * static_cast<I64>(elem));
  return std::max<I64>(1, std::min<I64>(g.n, kColBudgetBytes / per_image));
}

/// cols[r, (i KW + j) Cg + c] = x[n, oy s - pt + i d, ox s - pl + j d, group Cg + c] (0 outside),
/// for the images [n0, n1) of the batch.
template <typename T>
void im2col(const Geometry& g, const T* x, I64 n0, I64 n1, I64 group, T* cols) {
  const I64 kc = g.kc();
  parallel_for(0, (n1 - n0) * g.oh, std::max<I64>(1, 4096 / std::max<I64>(1, g.ow * kc)), [&](I64 lo, I64 hi) {
    for (I64 r = lo; r < hi; ++r) {
      const I64 n = n0 + r / g.oh, oy = r % g.oh;
      for (I64 ox = 0; ox < g.ow; ++ox) {
        T* dst = cols + (r * g.ow + ox) * kc;
        for (I64 i = 0; i < g.kh; ++i) {
          const I64 iy = oy * g.sh - g.pt + i * g.dh;
          if (iy < 0 || iy >= g.h) {
            std::fill(dst, dst + g.kw * g.cg, T(0));
            dst += g.kw * g.cg;
            continue;
          }
          const T* row = x + ((n * g.h + iy) * g.w) * g.c + group * g.cg;
          for (I64 j = 0; j < g.kw; ++j, dst += g.cg) {
            const I64 ix = ox * g.sw - g.pl + j * g.dw;
            if (ix < 0 || ix >= g.w)
              std::fill(dst, dst + g.cg, T(0));
            else
              std::memcpy(dst, row + ix * g.c, sizeof(T) * static_cast<std::size_t>(g.cg));
          }
        }
      }
    }
  });
}

/// The adjoint of im2col: adds cols back into dx for the images [n0, n1).
template <typename T>
void col2im_add(const Geometry& g, const T* cols, I64 n0, I64 n1, I64 group, T* dx) {
  const I64 kc = g.kc();
  parallel_for(n0, n1, 1, [&](I64 lo, I64 hi) {
    for (I64 n = lo; n < hi; ++n)
      for (I64 oy = 0; oy < g.oh; ++oy)
        for (I64 ox = 0; ox < g.ow; ++ox) {
          const T* src = cols + (((n - n0) * g.oh + oy) * g.ow + ox) * kc;
          for (I64 i = 0; i < g.kh; ++i) {
            const I64 iy = oy * g.sh - g.pt + i * g.dh;
            if (iy < 0 || iy >= g.h) {
              src += g.kw * g.cg;
              continue;
            }
            T* row = dx + ((n * g.h + iy) * g.w) * g.c + group * g.cg;
            for (I64 j = 0; j < g.kw; ++j, src += g.cg) {
              const I64 ix = ox * g.sw - g.pl + j * g.dw;
              if (ix < 0 || ix >= g.w) continue;
              T* d = row + ix * g.c;
              for (I64 c = 0; c < g.cg; ++c) d[c] += src[c];
            }
          }
        }
  });
}

template <typename T>
void conv_forward(const Geometry& g, const T* x, const T* f, T* y) {
  if (g.pointwise()) {
    gemm(false, false, g.n * g.h * g.w, g.o, g.c, T(1), x, g.c, f, g.o, T(0), y, g.o);
    return;
  }
  const I64      kc = g.kc(), rpi = g.rows_per_image(), chunk = images_per_chunk(g, sizeof(T));
  std::vector<T> cols(static_cast<std::size_t>(chunk * rpi * kc));
  for (I64 n0 = 0; n0 < g.n; n0 += chunk) {
    const I64 n1 = std::min(g.n, n0 + chunk), rows = (n1 - n0) * rpi;
    for (I64 grp = 0; grp < g.groups; ++grp) {
      im2col(g, x, n0, n1, grp, cols.data());
      gemm(false, false, rows, g.og, kc, T(1), cols.data(), kc, f + grp * g.og, g.o, T(0),
           y + n0 * rpi * g.o + grp * g.og, g.o);
    }
  }
}

template <typename T>
void conv_backward_input(const Geometry& g, const T* f, const T* dy, T* dx) {
  if (g.pointwise()) {
    gemm(false, true, g.n * g.h * g.w, g.c, g.o, T(1), dy, g.o, f, g.o, T(0), dx, g.c);
    return;
  }
  std::fill(dx, dx + g.n * g.h * g.w * g.c, T(0));
  const I64      kc = g.kc(), rpi = g.rows_per_image(), chunk = images_per_chunk(g, sizeof(T));
  std::vector<T> cols(static_cast<std::size_t>(chunk * rpi * kc));
  for (I64 n0 = 0; n0 < g.n; n0 += chunk) {
    const I64 n1 = std::min(g.n, n0 + chunk), rows = (n1 - n0) * rpi;
    for (I64 grp = 0; grp < g.groups; ++grp) {
      gemm(false, true, rows, kc, g.og, T(1), dy + n0 * rpi * g.o + grp * g.og, g.o, f + grp * g.og, g.o, T(0),
           cols.data(), kc);
      col2im_add(g, cols.data(), n0, n1, grp, dx);
    }
  }
}

template <typename T>
void conv_backward_filter(const Geometry& g, const T* x, const T* dy, T* df) {
  if (g.pointwise()) {
    gemm(true, false, g.c, g.o, g.n * g.h * g.w, T(1), x, g.c, dy, g.o, T(0), df, g.o);
    return;
  }
  const I64 kc = g.kc(), rpi = g.rows_per_image(), chunk = images_per_chunk(g, sizeof(T));
  std::fill(df, df + kc * g.o, T(0));
  std::vector<T> cols(static_cast<std::size_t>(chunk * rpi * kc));
  for (I64 n0 = 0; n0 < g.n; n0 += chunk) {
    const I64 n1 = std::min(g.n, n0 + chunk), rows = (n1 - n0) * rpi;
    for (I64 grp = 0; grp < g.groups; ++grp) {
      im2col(g, x, n0, n1, grp, cols.data());
      gemm(true, false, kc, g.og, rows, T(1), cols.data(), kc, dy + n0 * rpi * g.o + grp * g.og, g.o, T(1),
           df + grp * g.og, g.o);
    }
  }
}

void check_grad_shape(const Geometry& g, const Tensor& grad, const char* what) {
  if (grad.shape() != g.out_shape())
    throw std::invalid_argument(std::string(what) + ": gradient shape " + shape_str(grad.shape()) +
                                " does not match the convolution output " + shape_str(g.out_shape()));
}

}  // namespace

Conv2DOptions conv_options(std::int64_t stride, Padding padding, std::int64_t dilation) {
  Conv2DOptions o;
  o.strides   = {stride, stride};
  o.dilations = {dilation, dilation};
  o.padding   = padding;
  return o;
}

std::int64_t conv_output_size(std::int64_t in, std::int64_t kernel, std::int64_t stride, std::int64_t dilation,
                              Padding padding, std::int64_t pad_before, std::int64_t pad_after) {
  return axis_geometry(in, kernel, stride, dilation, padding, pad_before, pad_after).out;
}

// ---- convolution primitives ---------------------------------------------------------------

Tensor conv2d(const Tensor& x, const Tensor& filter, const Conv2DOptions& options) {
  const DType    dt = float_type(x, filter);
  const Tensor   xc = as_dtype(x, dt), fc = as_dtype(filter, dt);
  const Geometry g  = geometry(xc.shape(), fc.shape(), options, "conv2d");
  Tensor         out = empty(g.out_shape(), dt);
  {
    NoGradGuard  guard;
    const Tensor xk = nograd(xc).contiguous(), fk = nograd(fc).contiguous();
    dispatch_float(dt, "conv2d", [&](auto tag) {
      using T = typename decltype(tag)::type;
      conv_forward<T>(g, xk.data_ptr<T>(), fk.data_ptr<T>(), out.data_ptr<T>());
    });
  }
  const SavedTensor sx(xc), sf(fc);
  const bool        need_x = xc.requires_grad(), need_f = fc.requires_grad();
  record("Conv2D", {xc, fc}, {out}, [sx, sf, options, need_x, need_f](const Grads& gr, const Grads&) {
    const Tensor& X = sx.get();
    const Tensor& F = sf.get();
    return Grads{need_x ? conv2d_backprop_input(X.shape(), F, gr[0], options) : Tensor(),
                 need_f ? conv2d_backprop_filter(X, F.shape(), gr[0], options) : Tensor()};
  });
  return out;
}

Tensor conv2d_backprop_input(const Shape& input_shape, const Tensor& filter, const Tensor& grad,
                             const Conv2DOptions& options) {
  const DType    dt = float_type(filter, grad);
  const Tensor   fc = as_dtype(filter, dt), gc = as_dtype(grad, dt);
  const Geometry g  = geometry(input_shape, fc.shape(), options, "conv2d_backprop_input");
  check_grad_shape(g, gc, "conv2d_backprop_input");
  Tensor out = empty(input_shape, dt);
  {
    NoGradGuard  guard;
    const Tensor fk = nograd(fc).contiguous(), gk = nograd(gc).contiguous();
    dispatch_float(dt, "conv2d_backprop_input", [&](auto tag) {
      using T = typename decltype(tag)::type;
      conv_backward_input<T>(g, fk.data_ptr<T>(), gk.data_ptr<T>(), out.data_ptr<T>());
    });
  }
  const SavedTensor sf(fc), sg(gc);
  const bool        need_f = fc.requires_grad(), need_g = gc.requires_grad();
  record("Conv2DBackpropInput", {fc, gc}, {out}, [sf, sg, options, need_f, need_g](const Grads& gr, const Grads&) {
    const Tensor& F = sf.get();
    const Tensor& G = sg.get();
    return Grads{need_f ? conv2d_backprop_filter(gr[0], F.shape(), G, options) : Tensor(),
                 need_g ? conv2d(gr[0], F, options) : Tensor()};
  });
  return out;
}

Tensor conv2d_backprop_filter(const Tensor& x, const Shape& filter_shape, const Tensor& grad,
                              const Conv2DOptions& options) {
  const DType    dt = float_type(x, grad);
  const Tensor   xc = as_dtype(x, dt), gc = as_dtype(grad, dt);
  const Geometry g  = geometry(xc.shape(), filter_shape, options, "conv2d_backprop_filter");
  check_grad_shape(g, gc, "conv2d_backprop_filter");
  Tensor out = empty(filter_shape, dt);
  {
    NoGradGuard  guard;
    const Tensor xk = nograd(xc).contiguous(), gk = nograd(gc).contiguous();
    dispatch_float(dt, "conv2d_backprop_filter", [&](auto tag) {
      using T = typename decltype(tag)::type;
      conv_backward_filter<T>(g, xk.data_ptr<T>(), gk.data_ptr<T>(), out.data_ptr<T>());
    });
  }
  const SavedTensor sx(xc), sg(gc);
  const bool        need_x = xc.requires_grad(), need_g = gc.requires_grad();
  record("Conv2DBackpropFilter", {xc, gc}, {out}, [sx, sg, options, need_x, need_g](const Grads& gr, const Grads&) {
    const Tensor& X = sx.get();
    const Tensor& G = sg.get();
    return Grads{need_x ? conv2d_backprop_input(X.shape(), gr[0], G, options) : Tensor(),
                 need_g ? conv2d(X, gr[0], options) : Tensor()};
  });
  return out;
}

// ---- convolution compositions -----------------------------------------------------------------

Tensor conv2d_transpose(const Tensor& x, const Tensor& filter, const Shape& output_shape,
                        const Conv2DOptions& options) {
  return conv2d_backprop_input(output_shape, filter, x, options);
}

Tensor conv2d_transpose(const Tensor& x, const Tensor& filter, const Conv2DOptions& options) {
  if (x.dim() != 4 || filter.dim() != 4)
    throw std::invalid_argument("conv2d_transpose: need x [N, H, W, C] and filter [KH, KW, O, C]");
  const auto out_len = [&](I64 in, I64 k, I64 s, I64 d, I64 pb, I64 pa) {
    const I64 eff = d * (k - 1) + 1;
    switch (options.padding) {
      case Padding::Same: return in * s;
      case Padding::Valid: return (in - 1) * s + eff;
      case Padding::Explicit: return (in - 1) * s + eff - pb - pa;
    }
    return in;
  };
  const auto& xs = x.shape();
  const auto& fs = filter.shape();
  const auto& ep = options.explicit_paddings;
  const Shape out{xs[0], out_len(xs[1], fs[0], options.strides[0], options.dilations[0], ep[0], ep[1]),
                  out_len(xs[2], fs[1], options.strides[1], options.dilations[1], ep[2], ep[3]), fs[2] * options.groups};
  return conv2d_transpose(x, filter, out, options);
}

Tensor depthwise_conv2d(const Tensor& x, const Tensor& filter, const Conv2DOptions& options) {
  if (x.dim() != 4 || filter.dim() != 4 || filter.shape()[2] != x.shape()[3])
    throw std::invalid_argument("depthwise_conv2d: need x [N, H, W, C] and filter [KH, KW, C, M], got " +
                                shape_str(x.shape()) + " and " + shape_str(filter.shape()));
  const auto& fs = filter.shape();
  Conv2DOptions o = options;
  o.groups        = fs[2];
  return conv2d(x, reshape(filter, {fs[0], fs[1], 1, fs[2] * fs[3]}), o);
}

Tensor separable_conv2d(const Tensor& x, const Tensor& depthwise_filter, const Tensor& pointwise_filter,
                        const Conv2DOptions& options) {
  const Tensor h = depthwise_conv2d(x, depthwise_filter, options);
  return conv2d(h, pointwise_filter, conv_options(1, Padding::Valid));
}

Tensor conv1d(const Tensor& x, const Tensor& filter, std::int64_t stride, Padding padding, std::int64_t dilation) {
  if (x.dim() != 3 || filter.dim() != 3)
    throw std::invalid_argument("conv1d: need x [N, L, C] and filter [K, C, O], got " + shape_str(x.shape()) + " and " +
                                shape_str(filter.shape()));
  Conv2DOptions o;
  o.strides   = {1, stride};
  o.dilations = {1, dilation};
  o.padding   = padding;
  const auto& fs = filter.shape();
  return squeeze(conv2d(expand_dims(x, 1), reshape(filter, {1, fs[0], fs[1], fs[2]}), o), {1});
}

// ---- pooling --------------------------------------------------------------------------------

namespace {

struct PoolGeometry {
  I64 n, h, w, c, kh, kw, sh, sw, oh, ow, pt, pl;
  [[nodiscard]] Shape out_shape() const { return {n, oh, ow, c}; }
};

PoolGeometry pool_geometry(const Shape& xs, std::array<I64, 2> window, std::array<I64, 2> strides, Padding padding,
                           const char* what) {
  if (xs.size() != 4) throw std::invalid_argument(std::string(what) + ": input must be [N, H, W, C], got " + shape_str(xs));
  if (window[0] < 1 || window[1] < 1 || strides[0] < 1 || strides[1] < 1)
    throw std::invalid_argument(std::string(what) + ": window and strides must be >= 1");
  if (padding == Padding::Explicit) throw std::invalid_argument(std::string(what) + ": use SAME or VALID padding");
  PoolGeometry g{};
  g.n = xs[0], g.h = xs[1], g.w = xs[2], g.c = xs[3];
  g.kh = window[0], g.kw = window[1], g.sh = strides[0], g.sw = strides[1];
  const Axis ay = axis_geometry(g.h, g.kh, g.sh, 1, padding, 0, 0);
  const Axis ax = axis_geometry(g.w, g.kw, g.sw, 1, padding, 0, 0);
  g.oh = ay.out, g.ow = ax.out, g.pt = ay.pad_before, g.pl = ax.pad_before;
  return g;
}

/// In-bounds window rows [y0, y1) and columns [x0, x1) for one output.
struct Window {
  I64 y0, y1, x0, x1;
};
Window window_at(const PoolGeometry& g, I64 oy, I64 ox) {
  const I64 ys = oy * g.sh - g.pt, xs = ox * g.sw - g.pl;
  return {std::max<I64>(ys, 0), std::min(ys + g.kh, g.h), std::max<I64>(xs, 0), std::min(xs + g.kw, g.w)};
}

template <typename T>
void avg_pool_forward(const PoolGeometry& g, const T* x, T* y) {
  parallel_for(0, g.n * g.oh, 1, [&](I64 lo, I64 hi) {
    for (I64 r = lo; r < hi; ++r) {
      const I64 n = r / g.oh, oy = r % g.oh;
      for (I64 ox = 0; ox < g.ow; ++ox) {
        const Window wd  = window_at(g, oy, ox);
        T*           out = y + ((n * g.oh + oy) * g.ow + ox) * g.c;
        std::fill(out, out + g.c, T(0));
        for (I64 iy = wd.y0; iy < wd.y1; ++iy)
          for (I64 ix = wd.x0; ix < wd.x1; ++ix) {
            const T* src = x + ((n * g.h + iy) * g.w + ix) * g.c;
            for (I64 c = 0; c < g.c; ++c) out[c] += src[c];
          }
        const T inv = T(1) / static_cast<T>(std::max<I64>(1, (wd.y1 - wd.y0) * (wd.x1 - wd.x0)));
        for (I64 c = 0; c < g.c; ++c) out[c] *= inv;
      }
    }
  });
}

template <typename T>
void avg_pool_backward(const PoolGeometry& g, const T* dy, T* dx) {
  std::fill(dx, dx + g.n * g.h * g.w * g.c, T(0));
  parallel_for(0, g.n, 1, [&](I64 lo, I64 hi) {
    for (I64 n = lo; n < hi; ++n)
      for (I64 oy = 0; oy < g.oh; ++oy)
        for (I64 ox = 0; ox < g.ow; ++ox) {
          const Window wd  = window_at(g, oy, ox);
          const T*     src = dy + ((n * g.oh + oy) * g.ow + ox) * g.c;
          const T      inv = T(1) / static_cast<T>(std::max<I64>(1, (wd.y1 - wd.y0) * (wd.x1 - wd.x0)));
          for (I64 iy = wd.y0; iy < wd.y1; ++iy)
            for (I64 ix = wd.x0; ix < wd.x1; ++ix) {
              T* d = dx + ((n * g.h + iy) * g.w + ix) * g.c;
              for (I64 c = 0; c < g.c; ++c) d[c] += src[c] * inv;
            }
        }
  });
}

Tensor avg_pool_grad(const Tensor& grad, const Shape& input_shape, std::array<I64, 2> window,
                     std::array<I64, 2> strides, Padding padding);

Tensor avg_pool_impl(const Tensor& x_in, std::array<I64, 2> window, std::array<I64, 2> strides, Padding padding) {
  const Tensor       x = is_floating(x_in.dtype()) ? x_in : cast(x_in, DType::Float32);
  const PoolGeometry g = pool_geometry(x.shape(), window, strides, padding, "avg_pool2d");
  Tensor             out = empty(g.out_shape(), x.dtype());
  {
    NoGradGuard  guard;
    const Tensor xk = nograd(x).contiguous();
    dispatch_float(x.dtype(), "avg_pool2d", [&](auto tag) {
      using T = typename decltype(tag)::type;
      avg_pool_forward<T>(g, xk.data_ptr<T>(), out.data_ptr<T>());
    });
  }
  const Shape xs = x.shape();
  record("AvgPool", {x}, {out}, [xs, window, strides, padding](const Grads& gr, const Grads&) {
    return Grads{avg_pool_grad(gr[0], xs, window, strides, padding)};
  });
  return out;
}

Tensor avg_pool_grad(const Tensor& grad, const Shape& input_shape, std::array<I64, 2> window,
                     std::array<I64, 2> strides, Padding padding) {
  const PoolGeometry g = pool_geometry(input_shape, window, strides, padding, "avg_pool2d_grad");
  if (grad.shape() != g.out_shape()) throw std::invalid_argument("avg_pool2d_grad: gradient shape mismatch");
  Tensor out = empty(input_shape, grad.dtype());
  {
    NoGradGuard  guard;
    const Tensor gk = nograd(grad).contiguous();
    dispatch_float(grad.dtype(), "avg_pool2d_grad", [&](auto tag) {
      using T = typename decltype(tag)::type;
      avg_pool_backward<T>(g, gk.data_ptr<T>(), out.data_ptr<T>());
    });
  }
  record("AvgPoolGrad", {grad}, {out}, [window, strides, padding](const Grads& gr, const Grads&) {
    return Grads{avg_pool_impl(gr[0], window, strides, padding)};
  });
  return out;
}

// Max pooling's gradient routes each output gradient to its argmax (a flat
// index into the input). The scatter and the gather at the same indices are
// adjoint, so each is the other's backward and higher orders work.
Tensor max_pool_gather(const Tensor& grad_in, const Tensor& argmax);

Tensor max_pool_scatter(const Tensor& grad_out, const Tensor& argmax, const Shape& input_shape) {
  Tensor out = zeros(input_shape, grad_out.dtype());
  {
    NoGradGuard  guard;
    const Tensor gk = nograd(grad_out).contiguous();
    const I64    n = input_shape[0], per_out = n ? argmax.numel() / n : 0;
    const I64*   idx = argmax.data_ptr<I64>();
    dispatch_float(grad_out.dtype(), "max_pool2d backward", [&](auto tag) {
      using T      = typename decltype(tag)::type;
      const T* src = gk.data_ptr<T>();
      T*       dst = out.data_ptr<T>();
      // Each image's argmaxes stay inside that image, so images never collide.
      parallel_for(0, n, 1, [&](I64 lo, I64 hi) {
        for (I64 i = lo * per_out; i < hi * per_out; ++i) dst[idx[i]] += src[i];
      });
    });
  }
  record("MaxPoolGrad", {grad_out}, {out}, [argmax](const Grads& gr, const Grads&) {
    return Grads{max_pool_gather(gr[0], argmax)};
  });
  return out;
}

Tensor max_pool_gather(const Tensor& grad_in, const Tensor& argmax) {
  Tensor out = empty(argmax.shape(), grad_in.dtype());
  {
    NoGradGuard  guard;
    const Tensor gk  = nograd(grad_in).contiguous();
    const I64*   idx = argmax.data_ptr<I64>();
    dispatch_float(grad_in.dtype(), "max_pool2d double backward", [&](auto tag) {
      using T      = typename decltype(tag)::type;
      const T* src = gk.data_ptr<T>();
      T*       dst = out.data_ptr<T>();
      parallel_for(0, argmax.numel(), 32768, [&](I64 lo, I64 hi) {
        for (I64 i = lo; i < hi; ++i) dst[i] = src[idx[i]];
      });
    });
  }
  const Shape in_shape = grad_in.shape();
  record("MaxPoolGradGrad", {grad_in}, {out}, [argmax, in_shape](const Grads& gr, const Grads&) {
    return Grads{max_pool_scatter(gr[0], argmax, in_shape)};
  });
  return out;
}

}  // namespace

Tensor max_pool2d(const Tensor& x_in, std::array<std::int64_t, 2> window, std::array<std::int64_t, 2> strides,
                  Padding padding) {
  const Tensor       x = is_floating(x_in.dtype()) ? x_in : cast(x_in, DType::Float32);
  const PoolGeometry g = pool_geometry(x.shape(), window, strides, padding, "max_pool2d");
  Tensor             out    = empty(g.out_shape(), x.dtype());
  Tensor             argmax = empty(g.out_shape(), DType::Int64);
  {
    NoGradGuard  guard;
    const Tensor xk = nograd(x).contiguous();
    dispatch_float(x.dtype(), "max_pool2d", [&](auto tag) {
      using T      = typename decltype(tag)::type;
      const T* src = xk.data_ptr<T>();
      T*       dst = out.data_ptr<T>();
      I64*     idx = argmax.data_ptr<I64>();
      parallel_for(0, g.n * g.oh, 1, [&](I64 lo, I64 hi) {
        for (I64 r = lo; r < hi; ++r) {
          const I64 n = r / g.oh, oy = r % g.oh;
          for (I64 ox = 0; ox < g.ow; ++ox) {
            const Window wd   = window_at(g, oy, ox);
            const I64    o    = ((n * g.oh + oy) * g.ow + ox) * g.c;
            T*           best = dst + o;
            I64*         at   = idx + o;
            std::fill(best, best + g.c, -std::numeric_limits<T>::infinity());
            std::fill(at, at + g.c, I64{-1});
            for (I64 iy = wd.y0; iy < wd.y1; ++iy)
              for (I64 ix = wd.x0; ix < wd.x1; ++ix) {
                const I64 base = ((n * g.h + iy) * g.w + ix) * g.c;
                for (I64 c = 0; c < g.c; ++c)
                  if (src[base + c] > best[c] || at[c] < 0) {
                    best[c] = src[base + c];
                    at[c]   = base + c;
                  }
              }
          }
        }
      });
    });
  }
  const Shape xs = x.shape();
  record("MaxPool", {x}, {out}, [argmax, xs](const Grads& gr, const Grads&) {
    return Grads{max_pool_scatter(gr[0], argmax, xs)};
  });
  return out;
}

Tensor max_pool2d(const Tensor& x, std::int64_t window, std::int64_t stride, Padding padding) {
  return max_pool2d(x, {window, window}, {stride, stride}, padding);
}

Tensor avg_pool2d(const Tensor& x, std::array<std::int64_t, 2> window, std::array<std::int64_t, 2> strides,
                  Padding padding) {
  return avg_pool_impl(x, window, strides, padding);
}

Tensor avg_pool2d(const Tensor& x, std::int64_t window, std::int64_t stride, Padding padding) {
  return avg_pool2d(x, {window, window}, {stride, stride}, padding);
}

Tensor max_pool1d(const Tensor& x, std::int64_t window, std::int64_t stride, Padding padding) {
  if (x.dim() != 3) throw std::invalid_argument("max_pool1d: input must be [N, L, C]");
  return squeeze(max_pool2d(expand_dims(x, 1), {1, window}, {1, stride}, padding), {1});
}

Tensor avg_pool1d(const Tensor& x, std::int64_t window, std::int64_t stride, Padding padding) {
  if (x.dim() != 3) throw std::invalid_argument("avg_pool1d: input must be [N, L, C]");
  return squeeze(avg_pool2d(expand_dims(x, 1), {1, window}, {1, stride}, padding), {1});
}

Tensor global_avg_pool2d(const Tensor& x, bool keepdims) {
  if (x.dim() != 4) throw std::invalid_argument("global_avg_pool2d: input must be [N, H, W, C]");
  return mean(x, {1, 2}, keepdims);
}

Tensor global_max_pool2d(const Tensor& x, bool keepdims) {
  if (x.dim() != 4) throw std::invalid_argument("global_max_pool2d: input must be [N, H, W, C]");
  return max(x, {1, 2}, keepdims);
}

// ---- batch normalization ------------------------------------------------------------------------

namespace {

/// Per-channel sums over the rows of a [rows, C] buffer, computed in double.
template <typename T, typename F>
std::vector<double> channel_sums(I64 rows, I64 c, F&& value) {
  // Partial sums are kept per chunk and added in chunk order, so results do
  // not depend on which thread finishes first.
  std::map<I64, std::vector<double>> partials;
  std::mutex                         merge;
  parallel_for(0, rows, std::max<I64>(1, 8192 / std::max<I64>(c, 1)), [&](I64 lo, I64 hi) {
    std::vector<double> local(static_cast<std::size_t>(c), 0.0);
    for (I64 r = lo; r < hi; ++r)
      for (I64 k = 0; k < c; ++k) local[static_cast<std::size_t>(k)] += value(r, k);
    std::lock_guard<std::mutex> lock(merge);
    partials.emplace(lo, std::move(local));
  });
  std::vector<double> total(static_cast<std::size_t>(c), 0.0);
  for (const auto& entry : partials)
    for (I64 k = 0; k < c; ++k) total[static_cast<std::size_t>(k)] += entry.second[static_cast<std::size_t>(k)];
  return total;
}

Axes leading_axes(const Tensor& x) {
  Axes axes;
  for (I64 i = 0; i + 1 < x.dim(); ++i) axes.push_back(i);
  return axes;
}

}  // namespace

BatchNormResult batch_norm_train(const Tensor& x_in, const Tensor& gamma_in, const Tensor& beta_in, double eps) {
  if (x_in.dim() < 1) throw std::invalid_argument("batch_norm_train: input needs rank >= 1");
  const Tensor x     = is_floating(x_in.dtype()) ? x_in : cast(x_in, DType::Float32);
  const DType  dt    = x.dtype();
  const I64    c     = x.shape().back();
  const I64    rows  = c == 0 ? 0 : x.numel() / c;
  const Tensor gamma = as_dtype(gamma_in, dt), beta = as_dtype(beta_in, dt);
  if (gamma.shape() != Shape{c} || beta.shape() != Shape{c})
    throw std::invalid_argument("batch_norm_train: gamma and beta must be [" + std::to_string(c) + "]");

  Tensor y = empty(x.shape(), dt), mean_t = empty({c}, dt), var_t = empty({c}, dt), inv_t = empty({c}, dt);
  {
    NoGradGuard  guard;
    const Tensor xk = nograd(x).contiguous(), gk = nograd(gamma).contiguous(), bk = nograd(beta).contiguous();
    dispatch_float(dt, "batch_norm_train", [&](auto tag) {
      using T       = typename decltype(tag)::type;
      const T* px   = xk.data_ptr<T>();
      const auto s1 = channel_sums<T>(rows, c, [&](I64 r, I64 k) { return static_cast<double>(px[r * c + k]); });
      std::vector<double> mu(static_cast<std::size_t>(c));
      for (I64 k = 0; k < c; ++k) mu[static_cast<std::size_t>(k)] = rows > 0 ? s1[static_cast<std::size_t>(k)] / rows : 0.0;
      const auto s2 = channel_sums<T>(rows, c, [&](I64 r, I64 k) {
        const double d = static_cast<double>(px[r * c + k]) - mu[static_cast<std::size_t>(k)];
        return d * d;
      });
      std::vector<T> scale(static_cast<std::size_t>(c)), shift(static_cast<std::size_t>(c));
      const T*       pg = gk.data_ptr<T>();
      const T*       pb = bk.data_ptr<T>();
      for (I64 k = 0; k < c; ++k) {
        const auto   kk  = static_cast<std::size_t>(k);
        const double var = rows > 0 ? s2[kk] / rows : 0.0;
        const double inv = 1.0 / std::sqrt(var + eps);
        mean_t.data_ptr<T>()[k] = static_cast<T>(mu[kk]);
        var_t.data_ptr<T>()[k]  = static_cast<T>(rows > 1 ? s2[kk] / (rows - 1) : var);
        inv_t.data_ptr<T>()[k]  = static_cast<T>(inv);
        scale[kk]               = static_cast<T>(pg[k] * inv);
        shift[kk]               = static_cast<T>(pb[k] - mu[kk] * pg[k] * inv);
      }
      T* py = y.data_ptr<T>();
      parallel_for(0, rows, std::max<I64>(1, 16384 / std::max<I64>(c, 1)), [&](I64 lo, I64 hi) {
        for (I64 r = lo; r < hi; ++r)
          for (I64 k = 0; k < c; ++k)
            py[r * c + k] = px[r * c + k] * scale[static_cast<std::size_t>(k)] + shift[static_cast<std::size_t>(k)];
      });
    });
  }

  const SavedTensor sx(x), sg(gamma);
  const Tensor      saved_mean = mean_t, saved_inv = inv_t;
  const bool        need_x = x.requires_grad(), need_g = gamma.requires_grad(), need_b = beta.requires_grad();
  record("FusedBatchNormV3", {x, gamma, beta}, {y},
         [sx, sg, saved_mean, saved_inv, eps, need_x, need_g, need_b](const Grads& gr, const Grads&) {
           const Tensor& X  = sx.get();
           const Tensor& G  = sg.get();
           const Tensor& dy = gr[0];
           const I64     c  = X.shape().back();
           const I64     rows = c == 0 ? 0 : X.numel() / c;
           if (!is_grad_enabled()) {
             // First-order pass: one fused kernel.
             Tensor       dx = empty(X.shape(), X.dtype()), dg = empty({c}, X.dtype()), db = empty({c}, X.dtype());
             const Tensor xk = X.contiguous(), gyk = dy.contiguous(), gk = G.contiguous();
             dispatch_float(X.dtype(), "batch_norm_train backward", [&](auto tag) {
               using T        = typename decltype(tag)::type;
               const T* px    = xk.data_ptr<T>();
               const T* pdy   = gyk.data_ptr<T>();
               const T* pmu   = saved_mean.data_ptr<T>();
               const T* pinv  = saved_inv.data_ptr<T>();
               const T* pgam  = gk.data_ptr<T>();
               const auto sdy = channel_sums<T>(rows, c, [&](I64 r, I64 k) { return static_cast<double>(pdy[r * c + k]); });
               const auto sdx = channel_sums<T>(rows, c, [&](I64 r, I64 k) {
                 return static_cast<double>(pdy[r * c + k]) * (static_cast<double>(px[r * c + k]) - pmu[k]) * pinv[k];
               });
               T* pdx = dx.data_ptr<T>();
               parallel_for(0, rows, std::max<I64>(1, 16384 / std::max<I64>(c, 1)), [&](I64 lo, I64 hi) {
                 for (I64 r = lo; r < hi; ++r)
                   for (I64 k = 0; k < c; ++k) {
                     const auto   kk   = static_cast<std::size_t>(k);
                     const double xhat = (static_cast<double>(px[r * c + k]) - pmu[k]) * pinv[k];
                     pdx[r * c + k]    = static_cast<T>(pgam[k] * pinv[k] / rows *
                                                        (rows * static_cast<double>(pdy[r * c + k]) - sdy[kk] - xhat * sdx[kk]));
                   }
               });
               for (I64 k = 0; k < c; ++k) {
                 dg.data_ptr<T>()[k] = static_cast<T>(sdx[static_cast<std::size_t>(k)]);
                 db.data_ptr<T>()[k] = static_cast<T>(sdy[static_cast<std::size_t>(k)]);
               }
             });
             return Grads{need_x ? dx : Tensor(), need_g ? dg : Tensor(), need_b ? db : Tensor()};
           }
           // Differentiable pass (create_graph): the same formula with recorded
           // ops, recomputing the statistics from X so they carry history.
           const Axes   axes = leading_axes(X);
           const Tensor mu   = mean(X, axes, true);
           const Tensor xc   = X - mu;
           const Tensor inv  = rsqrt(mean(square(xc), axes, true) + eps);
           const Tensor xhat = xc * inv;
           const Tensor db   = sum(dy, axes);
           const Tensor dg   = sum(dy * xhat, axes);
           Tensor       dx;
           if (need_x) {
             const double m = static_cast<double>(rows);
             dx             = G * inv / m * (dy * m - db - xhat * dg);
           }
           return Grads{dx, need_g ? dg : Tensor(), need_b ? db : Tensor()};
         });
  return {y, mean_t, var_t};
}

Tensor batch_norm_infer(const Tensor& x, const Tensor& gamma, const Tensor& beta, const Tensor& mean_in,
                        const Tensor& variance, double eps) {
  const DType  dt    = is_floating(x.dtype()) ? x.dtype() : DType::Float32;
  const Tensor scale = as_dtype(gamma, dt) * rsqrt(as_dtype(variance, dt) + eps);
  return as_dtype(x, dt) * scale + (as_dtype(beta, dt) - as_dtype(mean_in, dt) * scale);
}

}  // namespace datamunge::nn
