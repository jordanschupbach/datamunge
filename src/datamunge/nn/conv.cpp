// Convolution (im2col / vol2col + GEMM), pooling and batch normalization,
// in 2-D (NHWC) and 3-D (NDHWC).
//
// conv{2,3}d, conv{2,3}d_backprop_input and conv{2,3}d_backprop_filter are three
// primitives whose backward passes are written in terms of each other, so
// gradients of any order flow through convolutions.

#include "datamunge/nn/conv.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/parallel.hpp"
#include "detail.hpp"
#include "device_ops.hpp"
#include "gemm.hpp"

namespace datamunge::nn {

using detail::as_dtype;
using detail::dispatch_float;
using detail::gemm;
using detail::gemm_local;
using detail::nograd;

namespace {

using I64   = std::int64_t;
using Grads = std::vector<Tensor>;

/// Upper bound on the 3-D vol2col buffer; larger batches are processed in chunks of units.
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
  /// GEMM work is chunked in units of whole images (rows_per_unit output rows each).
  [[nodiscard]] I64  units() const { return n; }
  [[nodiscard]] I64  rows_per_unit() const { return oh * ow; }
  [[nodiscard]] I64  pixels() const { return h * w; }
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

template <typename G>
I64 units_per_chunk(const G& g, std::size_t elem) {
  const I64 per_unit = std::max<I64>(1, g.rows_per_unit() * g.kc() * static_cast<I64>(elem));
  return std::max<I64>(1, std::min<I64>(g.units(), kColBudgetBytes / per_unit));
}

// ---- 2-D patch tiles -----------------------------------------------------------------
//
// A 2-D convolution is computed one tile at a time: the patch rows of a block
// of output rows of one image, sized to stay in a core's L2 cache, so that
// im2col (or the input gradient's col2im) and the GEMM on that tile run back
// to back on one thread and the patch matrix never leaves the cache. The
// tiles are fixed by the geometry, so every sum runs in the same order
// whatever the thread count.

/// Patch bytes per tile.
constexpr I64 kTileBytes = I64{384} << 10;
/// Partial filter gradients kept by the backward-filter reduction (fixed, for determinism).
constexpr I64 kFilterPartials = 16;

struct RowBlocks {
  I64 rows{1};    ///< output rows (oy) per block
  I64 blocks{1};  ///< blocks per image
};

RowBlocks row_blocks(const Geometry& g, std::size_t elem) {
  RowBlocks b;
  const I64 per_row = std::max<I64>(1, g.ow * g.kc() * static_cast<I64>(elem));
  b.rows            = std::max<I64>(1, std::min<I64>(g.oh, kTileBytes / per_row));
  b.blocks          = (g.oh + b.rows - 1) / b.rows;
  return b;
}

/// cols[r, (i KW + j) Cg + c] = x[n, oy s - pt + i d, ox s - pl + j d, group Cg + c] (0 outside)
/// for the output rows oy in [oy0, oy1) of image n, r = (oy - oy0) OW + ox.
template <typename T>
void im2col_rows(const Geometry& g, const T* x, I64 n, I64 oy0, I64 oy1, I64 group, T* cols) {
  const I64 kc = g.kc(), line = g.kw * g.cg;
  for (I64 oy = oy0; oy < oy1; ++oy)
    for (I64 ox = 0; ox < g.ow; ++ox) {
      T* dst = cols + ((oy - oy0) * g.ow + ox) * kc;
      for (I64 i = 0; i < g.kh; ++i) {
        const I64 iy = oy * g.sh - g.pt + i * g.dh;
        if (iy < 0 || iy >= g.h) {
          std::fill(dst, dst + line, T(0));
          dst += line;
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

/// The adjoint of im2col_rows: adds cols back into dx for the output rows [oy0, oy1) of image n.
template <typename T>
void col2im_add_rows(const Geometry& g, const T* cols, I64 n, I64 oy0, I64 oy1, I64 group, T* dx) {
  const I64 kc = g.kc(), line = g.kw * g.cg;
  for (I64 oy = oy0; oy < oy1; ++oy)
    for (I64 ox = 0; ox < g.ow; ++ox) {
      const T* src = cols + ((oy - oy0) * g.ow + ox) * kc;
      for (I64 i = 0; i < g.kh; ++i) {
        const I64 iy = oy * g.sh - g.pt + i * g.dh;
        if (iy < 0 || iy >= g.h) {
          src += line;
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
}

// ---- 3-D geometry and vol2col -------------------------------------------------------

struct Geometry3 {
  I64 n, d, h, w, c;          // input
  I64 kd, kh, kw, o;          // filter
  I64 od, oh, ow;             // output
  I64 sd, sh, sw, dd, dh, dw; // strides, dilations
  I64 pf, pt, pl;             // front / top / left padding
  I64 groups, cg, og;         // channels per group
  [[nodiscard]] I64  kc() const { return kd * kh * kw * cg; }
  /// GEMM work is chunked in units of one output depth slice of one volume,
  /// so a single large volume still fits the column budget.
  [[nodiscard]] I64  units() const { return n * od; }
  [[nodiscard]] I64  rows_per_unit() const { return oh * ow; }
  [[nodiscard]] I64  pixels() const { return d * h * w; }
  [[nodiscard]] bool pointwise() const {
    return kd == 1 && kh == 1 && kw == 1 && sd == 1 && sh == 1 && sw == 1 && pf == 0 && pt == 0 && pl == 0 &&
           od == d && oh == h && ow == w && groups == 1;
  }
  [[nodiscard]] Shape out_shape() const { return {n, od, oh, ow, o}; }
};

void check_options3(const Conv3DOptions& o, const char* what) {
  for (auto s : o.strides)
    if (s < 1) throw std::invalid_argument(std::string(what) + ": strides must be >= 1");
  for (auto d : o.dilations)
    if (d < 1) throw std::invalid_argument(std::string(what) + ": dilations must be >= 1");
  if (o.groups < 1) throw std::invalid_argument(std::string(what) + ": groups must be >= 1");
}

Geometry3 geometry3(const Shape& xs, const Shape& fs, const Conv3DOptions& o, const char* what) {
  check_options3(o, what);
  if (xs.size() != 5)
    throw std::invalid_argument(std::string(what) + ": input must be [N, D, H, W, C], got " + shape_str(xs));
  if (fs.size() != 5)
    throw std::invalid_argument(std::string(what) + ": filter must be [KD, KH, KW, C_in, C_out], got " + shape_str(fs));
  Geometry3 g{};
  g.n = xs[0], g.d = xs[1], g.h = xs[2], g.w = xs[3], g.c = xs[4];
  g.kd = fs[0], g.kh = fs[1], g.kw = fs[2], g.o = fs[4];
  g.groups = o.groups;
  if (fs[3] * o.groups != g.c || g.o % o.groups != 0)
    throw std::invalid_argument(std::string(what) + ": filter " + shape_str(fs) + " does not fit input " + shape_str(xs) +
                                " with groups=" + std::to_string(o.groups));
  g.cg = fs[3], g.og = g.o / o.groups;
  g.sd = o.strides[0], g.sh = o.strides[1], g.sw = o.strides[2];
  g.dd = o.dilations[0], g.dh = o.dilations[1], g.dw = o.dilations[2];
  const auto& ep = o.explicit_paddings;
  const Axis  az = axis_geometry(g.d, g.kd, g.sd, g.dd, o.padding, ep[0], ep[1]);
  const Axis  ay = axis_geometry(g.h, g.kh, g.sh, g.dh, o.padding, ep[2], ep[3]);
  const Axis  ax = axis_geometry(g.w, g.kw, g.sw, g.dw, o.padding, ep[4], ep[5]);
  g.od = az.out, g.oh = ay.out, g.ow = ax.out, g.pf = az.pad_before, g.pt = ay.pad_before, g.pl = ax.pad_before;
  return g;
}

/// cols[r, ((a KH + i) KW + j) Cg + c] = x[n, oz sd - pf + a dd, oy sh - pt + i dh, ox sw - pl + j dw, group Cg + c]
/// (0 outside), for the units (image n, output slice oz) = u0 .. u1 - 1, u = n OD + oz.
template <typename T>
void im2col(const Geometry3& g, const T* x, I64 u0, I64 u1, I64 group, T* cols) {
  const I64 kc = g.kc(), plane = g.kh * g.kw * g.cg, line = g.kw * g.cg;
  parallel_for(0, (u1 - u0) * g.oh, std::max<I64>(1, 4096 / std::max<I64>(1, g.ow * kc)), [&](I64 lo, I64 hi) {
    for (I64 r = lo; r < hi; ++r) {
      const I64 u = u0 + r / g.oh, n = u / g.od, oz = u % g.od, oy = r % g.oh;
      for (I64 ox = 0; ox < g.ow; ++ox) {
        T* dst = cols + (r * g.ow + ox) * kc;
        for (I64 a = 0; a < g.kd; ++a) {
          const I64 iz = oz * g.sd - g.pf + a * g.dd;
          if (iz < 0 || iz >= g.d) {
            std::fill(dst, dst + plane, T(0));
            dst += plane;
            continue;
          }
          for (I64 i = 0; i < g.kh; ++i) {
            const I64 iy = oy * g.sh - g.pt + i * g.dh;
            if (iy < 0 || iy >= g.h) {
              std::fill(dst, dst + line, T(0));
              dst += line;
              continue;
            }
            const T* row = x + (((n * g.d + iz) * g.h + iy) * g.w) * g.c + group * g.cg;
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
    }
  });
}

/// Runs fn(n, oz) for the units u = n out_depth + oz in [u0, u1), in phases
/// such that within a phase no two calls touch the same input depth slices
/// (windows of `eff` slices at stride `stride` whose output depths differ by
/// a multiple of the phase count are disjoint). Each input element then
/// receives its contributions in a fixed order, whatever the thread count,
/// and a batch of one volume still runs in parallel.
template <typename F>
void depth_phased_for(I64 u0, I64 u1, I64 out_depth, I64 eff, I64 stride, F&& fn) {
  const I64        phases = std::max<I64>(1, (eff + stride - 1) / stride);
  std::vector<I64> units;
  for (I64 phase = 0; phase < phases; ++phase) {
    units.clear();
    for (I64 u = u0; u < u1; ++u)
      if ((u % out_depth) % phases == phase) units.push_back(u);
    parallel_for(0, static_cast<I64>(units.size()), 1, [&](I64 lo, I64 hi) {
      for (I64 t = lo; t < hi; ++t) {
        const I64 u = units[static_cast<std::size_t>(t)];
        fn(u / out_depth, u % out_depth);
      }
    });
  }
}

/// The adjoint of the 3-D im2col: adds cols back into dx for the units [u0, u1).
template <typename T>
void col2im_add(const Geometry3& g, const T* cols, I64 u0, I64 u1, I64 group, T* dx) {
  const I64 kc = g.kc(), plane = g.kh * g.kw * g.cg, line = g.kw * g.cg;
  depth_phased_for(u0, u1, g.od, g.dd * (g.kd - 1) + 1, g.sd, [&](I64 n, I64 oz) {
    for (I64 oy = 0; oy < g.oh; ++oy)
      for (I64 ox = 0; ox < g.ow; ++ox) {
        const T* src = cols + (((n * g.od + oz - u0) * g.oh + oy) * g.ow + ox) * kc;
        for (I64 a = 0; a < g.kd; ++a) {
          const I64 iz = oz * g.sd - g.pf + a * g.dd;
          if (iz < 0 || iz >= g.d) {
            src += plane;
            continue;
          }
          for (I64 i = 0; i < g.kh; ++i) {
            const I64 iy = oy * g.sh - g.pt + i * g.dh;
            if (iy < 0 || iy >= g.h) {
              src += line;
              continue;
            }
            T* row = dx + (((n * g.d + iz) * g.h + iy) * g.w) * g.c + group * g.cg;
            for (I64 j = 0; j < g.kw; ++j, src += g.cg) {
              const I64 ix = ox * g.sw - g.pl + j * g.dw;
              if (ix < 0 || ix >= g.w) continue;
              T* dst = row + ix * g.c;
              for (I64 c = 0; c < g.cg; ++c) dst[c] += src[c];
            }
          }
        }
      }
  });
}

// ---- depthwise 2-D convolution (groups == C, one input channel per group) ----
//
// im2col + one GEMM per channel is very inefficient here (C tiny GEMMs), so
// these kernels work directly on NHWC data. Output channel c * M + m reads
// input channel c. Work is split so that no two threads write the same
// value, and every sum runs in a fixed order (deterministic results).

bool is_depthwise(const Geometry& g) { return g.groups > 1 && g.cg == 1; }

template <typename T>
void depthwise_forward(const Geometry& g, const T* x, const T* f, T* y) {
  const I64 m = g.og;  // channel multiplier
  parallel_for(0, g.n * g.oh, 1, [&](I64 lo, I64 hi) {
    for (I64 r = lo; r < hi; ++r) {
      const I64 n = r / g.oh, oy = r % g.oh;
      for (I64 ox = 0; ox < g.ow; ++ox) {
        T* out = y + ((n * g.oh + oy) * g.ow + ox) * g.o;
        std::fill(out, out + g.o, T(0));
        for (I64 i = 0; i < g.kh; ++i) {
          const I64 iy = oy * g.sh - g.pt + i * g.dh;
          if (iy < 0 || iy >= g.h) continue;
          for (I64 j = 0; j < g.kw; ++j) {
            const I64 ix = ox * g.sw - g.pl + j * g.dw;
            if (ix < 0 || ix >= g.w) continue;
            const T* in = x + ((n * g.h + iy) * g.w + ix) * g.c;
            const T* fk = f + (i * g.kw + j) * g.o;
            if (m == 1) {
              for (I64 c = 0; c < g.c; ++c) out[c] += in[c] * fk[c];
            } else {
              for (I64 c = 0; c < g.c; ++c)
                for (I64 k = 0; k < m; ++k) out[c * m + k] += in[c] * fk[c * m + k];
            }
          }
        }
      }
    }
  });
}

template <typename T>
void depthwise_backward_input(const Geometry& g, const T* f, const T* dy, T* dx) {
  const I64 m = g.og;
  // Gather form: each input pixel sums the outputs whose window reads it,
  // over all channels at once. Rows of dx are independent.
  parallel_for(0, g.n * g.h, 1, [&](I64 lo, I64 hi) {
    for (I64 r = lo; r < hi; ++r) {
      const I64 n = r / g.h, iy = r % g.h;
      for (I64 ix = 0; ix < g.w; ++ix) {
        T* d = dx + ((n * g.h + iy) * g.w + ix) * g.c;
        std::fill(d, d + g.c, T(0));
        for (I64 i = 0; i < g.kh; ++i) {
          const I64 ty = iy + g.pt - i * g.dh;  // = oy * sh
          if (ty < 0 || ty % g.sh != 0 || ty / g.sh >= g.oh) continue;
          const I64 oy = ty / g.sh;
          for (I64 j = 0; j < g.kw; ++j) {
            const I64 tx = ix + g.pl - j * g.dw;
            if (tx < 0 || tx % g.sw != 0 || tx / g.sw >= g.ow) continue;
            const I64 ox   = tx / g.sw;
            const T*  grad = dy + ((n * g.oh + oy) * g.ow + ox) * g.o;
            const T*  fk   = f + (i * g.kw + j) * g.o;
            if (m == 1) {
              for (I64 c = 0; c < g.c; ++c) d[c] += grad[c] * fk[c];
            } else {
              for (I64 c = 0; c < g.c; ++c)
                for (I64 k = 0; k < m; ++k) d[c] += grad[c * m + k] * fk[c * m + k];
            }
          }
        }
      }
    }
  });
}

template <typename T>
void depthwise_backward_filter(const Geometry& g, const T* x, const T* dy, T* df) {
  const I64 taps = g.kh * g.kw, m = g.og;
  // One partial sum per image (in double), added in image order, so the
  // result does not depend on how images are spread over threads.
  std::vector<double> partial(static_cast<std::size_t>(g.n * taps * g.o), 0.0);
  parallel_for(0, g.n, 1, [&](I64 lo, I64 hi) {
    std::vector<T> row(static_cast<std::size_t>(g.o));
    for (I64 n = lo; n < hi; ++n) {
      double* acc = partial.data() + n * taps * g.o;
      for (I64 oy = 0; oy < g.oh; ++oy)
        for (I64 ox = 0; ox < g.ow; ++ox) {
          const T* grad = dy + ((n * g.oh + oy) * g.ow + ox) * g.o;
          for (I64 i = 0; i < g.kh; ++i) {
            const I64 iy = oy * g.sh - g.pt + i * g.dh;
            if (iy < 0 || iy >= g.h) continue;
            for (I64 j = 0; j < g.kw; ++j) {
              const I64 ix = ox * g.sw - g.pl + j * g.dw;
              if (ix < 0 || ix >= g.w) continue;
              const T* in = x + ((n * g.h + iy) * g.w + ix) * g.c;
              double*  a  = acc + (i * g.kw + j) * g.o;
              if (m == 1) {
                for (I64 c = 0; c < g.c; ++c) a[c] += static_cast<double>(in[c] * grad[c]);
              } else {
                for (I64 c = 0; c < g.c; ++c)
                  for (I64 k = 0; k < m; ++k) a[c * m + k] += static_cast<double>(in[c] * grad[c * m + k]);
              }
            }
          }
        }
    }
  });
  for (I64 q = 0; q < taps * g.o; ++q) {
    double total = 0.0;
    for (I64 n = 0; n < g.n; ++n) total += partial[static_cast<std::size_t>(n * taps * g.o + q)];
    df[q] = static_cast<T>(total);
  }
}

/// Reusable, uninitialised workspace for the patch matrix of one thread, grown
/// on demand and kept (at most kColBudgetBytes). A fresh std::vector per call
/// would be value-initialised and page-faulted again every time, although
/// im2col and the input-gradient GEMM write every element before it is read.
template <typename T, int Slot = 0>
T* patch_scratch(I64 n) {
  static thread_local std::unique_ptr<T[]> buffer;
  static thread_local I64                  capacity = 0;
  if (n > capacity) {
    buffer.reset(new T[static_cast<std::size_t>(n)]);
    capacity = n;
  }
  return buffer.get();
}

/// Reusable workspace for the partial filter gradients of one call (same policy as patch_scratch).
template <typename T>
T* partial_scratch(I64 n) {
  static thread_local std::unique_ptr<T[]> buffer;
  static thread_local I64                  capacity = 0;
  if (n > capacity) {
    buffer.reset(new T[static_cast<std::size_t>(n)]);
    capacity = n;
  }
  return buffer.get();
}

/// out [rows, cols] = the sum of `parts` partials laid out [parts, rows, cols], in part order.
template <typename T>
void reduce_partials(const T* partial, I64 parts, I64 rows, I64 cols, T* out) {
  const I64 plane = rows * cols;
  parallel_for(0, rows, std::max<I64>(1, 16384 / std::max<I64>(cols, 1)), [&](I64 lo, I64 hi) {
    for (I64 r = lo; r < hi; ++r) {
      T* dst = out + r * cols;
      std::copy(partial + r * cols, partial + (r + 1) * cols, dst);
      for (I64 p = 1; p < parts; ++p) {
        const T* src = partial + p * plane + r * cols;
        for (I64 c = 0; c < cols; ++c) dst[c] += src[c];
      }
    }
  });
}

/// Pixel rows per task of a pointwise convolution (a plain GEMM over pixels):
/// enough rows for an efficient GEMM, few enough to keep the tile in cache.
I64 pointwise_rows(const Geometry& g, std::size_t elem) {
  const I64 widest = std::max<I64>(1, std::max(g.c, g.o) * static_cast<I64>(elem));
  return std::max<I64>(64, std::min<I64>(256, kTileBytes / widest));
}

template <typename T>
void parallel_zero(T* p, I64 n) {
  parallel_for(0, n, I64{1} << 18, [&](I64 lo, I64 hi) { std::fill(p + lo, p + hi, T(0)); });
}

// The GEMM drivers below serve 2-D (Geometry, im2col) and 3-D (Geometry3,
// vol2col) convolutions alike; the overloads of im2col / col2im_add pick the
// patch layout.
template <typename T, typename G>
void conv_forward(const G& g, const T* x, const T* f, T* y) {
  if constexpr (std::is_same<G, Geometry>::value) {
    if (is_depthwise(g)) return depthwise_forward(g, x, f, y);
  }
  if (g.pointwise()) {
    gemm(false, false, g.n * g.pixels(), g.o, g.c, T(1), x, g.c, f, g.o, T(0), y, g.o);
    return;
  }
  if constexpr (std::is_same<G, Geometry>::value) {
    // One task per tile: im2col into this thread's scratch, then the GEMM on it.
    const I64       kc = g.kc();
    const RowBlocks rb = row_blocks(g, sizeof(T));
    parallel_for(0, g.n * rb.blocks, 1, [&](I64 lo, I64 hi) {
      T* cols = patch_scratch<T>(rb.rows * g.ow * kc);
      for (I64 t = lo; t < hi; ++t) {
        const I64 n = t / rb.blocks, oy0 = (t % rb.blocks) * rb.rows, oy1 = std::min(g.oh, oy0 + rb.rows);
        const I64 rows = (oy1 - oy0) * g.ow;
        T*        out  = y + ((n * g.oh + oy0) * g.ow) * g.o;
        for (I64 grp = 0; grp < g.groups; ++grp) {
          im2col_rows(g, x, n, oy0, oy1, grp, cols);
          gemm_local(false, false, rows, g.og, kc, T(1), cols, kc, f + grp * g.og, g.o, T(0), out + grp * g.og, g.o);
        }
      }
    });
    return;
  } else {
    const I64 kc = g.kc(), rpu = g.rows_per_unit(), chunk = units_per_chunk(g, sizeof(T));
    T*        cols = patch_scratch<T>(chunk * rpu * kc);
    for (I64 u0 = 0; u0 < g.units(); u0 += chunk) {
      const I64 u1 = std::min(g.units(), u0 + chunk), rows = (u1 - u0) * rpu;
      for (I64 grp = 0; grp < g.groups; ++grp) {
        im2col(g, x, u0, u1, grp, cols);
        gemm(false, false, rows, g.og, kc, T(1), cols, kc, f + grp * g.og, g.o, T(0),
             y + u0 * rpu * g.o + grp * g.og, g.o);
      }
    }
  }
}

template <typename T, typename G>
void conv_backward_input(const G& g, const T* f, const T* dy, T* dx) {
  if constexpr (std::is_same<G, Geometry>::value) {
    if (is_depthwise(g)) return depthwise_backward_input(g, f, dy, dx);
  }
  if (g.pointwise()) {
    gemm(false, true, g.n * g.pixels(), g.c, g.o, T(1), dy, g.o, f, g.o, T(0), dx, g.c);
    return;
  }
  if constexpr (std::is_same<G, Geometry>::value) {
    // One task per image (its tiles overlap in dx, so one thread owns the whole
    // image): zero its dx, then per tile the GEMM into scratch and col2im.
    const I64       kc = g.kc(), image = g.pixels() * g.c;
    const RowBlocks rb = row_blocks(g, sizeof(T));
    parallel_for(0, g.n, 1, [&](I64 lo, I64 hi) {
      T* cols = patch_scratch<T>(rb.rows * g.ow * kc);
      for (I64 n = lo; n < hi; ++n) {
        std::fill(dx + n * image, dx + (n + 1) * image, T(0));
        for (I64 b = 0; b < rb.blocks; ++b) {
          const I64 oy0 = b * rb.rows, oy1 = std::min(g.oh, oy0 + rb.rows), rows = (oy1 - oy0) * g.ow;
          const T*  grad = dy + ((n * g.oh + oy0) * g.ow) * g.o;
          for (I64 grp = 0; grp < g.groups; ++grp) {
            gemm_local(false, true, rows, kc, g.og, T(1), grad + grp * g.og, g.o, f + grp * g.og, g.o, T(0), cols, kc);
            col2im_add_rows(g, cols, n, oy0, oy1, grp, dx);
          }
        }
      }
    });
    return;
  } else {
    parallel_zero(dx, g.n * g.pixels() * g.c);
    const I64 kc = g.kc(), rpu = g.rows_per_unit(), chunk = units_per_chunk(g, sizeof(T));
    T*        cols = patch_scratch<T>(chunk * rpu * kc);
    for (I64 u0 = 0; u0 < g.units(); u0 += chunk) {
      const I64 u1 = std::min(g.units(), u0 + chunk), rows = (u1 - u0) * rpu;
      for (I64 grp = 0; grp < g.groups; ++grp) {
        gemm(false, true, rows, kc, g.og, T(1), dy + u0 * rpu * g.o + grp * g.og, g.o, f + grp * g.og, g.o, T(0),
             cols, kc);
        col2im_add(g, cols, u0, u1, grp, dx);
      }
    }
  }
}

template <typename T, typename G>
void conv_backward_filter(const G& g, const T* x, const T* dy, T* df) {
  if constexpr (std::is_same<G, Geometry>::value) {
    if (is_depthwise(g)) return depthwise_backward_filter(g, x, dy, df);
  }
  if (g.pointwise()) {
    gemm(true, false, g.c, g.o, g.n * g.pixels(), T(1), x, g.c, dy, g.o, T(0), df, g.o);
    return;
  }
  if constexpr (std::is_same<G, Geometry>::value) {
    // Tile t accumulates into partial t % parts (a fixed assignment); each
    // partial is owned by one thread and stays in its cache across its tiles.
    const I64       kc = g.kc(), plane = kc * g.o;
    const RowBlocks rb = row_blocks(g, sizeof(T));
    const I64       tasks = g.n * rb.blocks, parts = std::max<I64>(1, std::min(tasks, kFilterPartials));
    T*              partial = partial_scratch<T>(parts * plane);
    parallel_for(0, parts, 1, [&](I64 lo, I64 hi) {
      T* cols = patch_scratch<T>(rb.rows * g.ow * kc);
      for (I64 p = lo; p < hi; ++p) {
        T* acc = partial + p * plane;
        for (I64 t = p; t < tasks; t += parts) {
          const I64 n = t / rb.blocks, oy0 = (t % rb.blocks) * rb.rows, oy1 = std::min(g.oh, oy0 + rb.rows);
          const I64 rows = (oy1 - oy0) * g.ow;
          const T*  grad = dy + ((n * g.oh + oy0) * g.ow) * g.o;
          for (I64 grp = 0; grp < g.groups; ++grp) {
            im2col_rows(g, x, n, oy0, oy1, grp, cols);
            // The partial's first tile overwrites it (no zero fill).
            gemm_local(true, false, kc, g.og, rows, T(1), cols, kc, grad + grp * g.og, g.o, t == p ? T(0) : T(1),
                       acc + grp * g.og, g.o);
          }
        }
      }
    });
    reduce_partials(partial, parts, kc, g.o, df);
    return;
  } else {
    const I64 kc = g.kc(), rpu = g.rows_per_unit(), chunk = units_per_chunk(g, sizeof(T));
    std::fill(df, df + kc * g.o, T(0));
    T* cols = patch_scratch<T>(chunk * rpu * kc);
    for (I64 u0 = 0; u0 < g.units(); u0 += chunk) {
      const I64 u1 = std::min(g.units(), u0 + chunk), rows = (u1 - u0) * rpu;
      for (I64 grp = 0; grp < g.groups; ++grp) {
        im2col(g, x, u0, u1, grp, cols);
        gemm(true, false, kc, g.og, rows, T(1), cols, kc, dy + u0 * rpu * g.o + grp * g.og, g.o, T(1),
             df + grp * g.og, g.o);
      }
    }
  }
}

// ---- fused convolution + bias + ReLU ---------------------------------------------------
//
// The conv layer's bias and ReLU are applied to each output tile of
// conv_forward while it is in cache, and the backward pass forms the masked
// gradient dz = grad * [y > 0] once per tile and consumes it in cache for the
// bias reduction, the filter-gradient GEMM and the input-gradient GEMM +
// col2im: one pass over the activations instead of the five of the
// composition (mask, bias reduction, two GEMM inputs). Ungrouped,
// non-depthwise 2-D geometries only.

/// y = relu(conv(x, f) + b), with output pixel rows y_ld apart (y_ld >= g.o; a
/// wider stride writes one channel band of a concatenated output).
template <typename T>
void conv_bias_relu_forward(const Geometry& g, const T* x, const T* f, const T* b, T* y, I64 y_ld) {
  const I64  o        = g.o;
  const auto epilogue = [b, o, y_ld](T* out, I64 rows) {
    for (I64 r = 0; r < rows; ++r) {
      T* row = out + r * y_ld;
      for (I64 c = 0; c < o; ++c) {
        const T v = row[c] + b[c];
        row[c]    = v > T(0) ? v : T(0);
      }
    }
  };
  if (g.pointwise()) {
    const I64 total = g.n * g.pixels(), block = pointwise_rows(g, sizeof(T)), tasks = (total + block - 1) / block;
    parallel_for(0, tasks, 1, [&](I64 lo, I64 hi) {
      for (I64 t = lo; t < hi; ++t) {
        const I64 r0 = t * block, rows = std::min(total, r0 + block) - r0;
        gemm_local(false, false, rows, o, g.c, T(1), x + r0 * g.c, g.c, f, o, T(0), y + r0 * y_ld, y_ld);
        epilogue(y + r0 * y_ld, rows);
      }
    });
    return;
  }
  const I64       kc = g.kc();
  const RowBlocks rb = row_blocks(g, sizeof(T));
  parallel_for(0, g.n * rb.blocks, 1, [&](I64 lo, I64 hi) {
    T* cols = patch_scratch<T>(rb.rows * g.ow * kc);
    for (I64 t = lo; t < hi; ++t) {
      const I64 n = t / rb.blocks, oy0 = (t % rb.blocks) * rb.rows, oy1 = std::min(g.oh, oy0 + rb.rows);
      const I64 rows = (oy1 - oy0) * g.ow;
      T*        out  = y + ((n * g.oh + oy0) * g.ow) * y_ld;
      im2col_rows(g, x, n, oy0, oy1, 0, cols);
      gemm_local(false, false, rows, o, kc, T(1), cols, kc, f, o, T(0), out, y_ld);
      epilogue(out, rows);
    }
  });
}

/// (dx, df, db) of y = relu(conv(x, f) + b) from y (pixel rows y_ld apart)
/// and its gradient (rows grad_ld apart). Tile t accumulates df and db into
/// partial t % parts (a fixed assignment), so the sums are deterministic. With
/// dx_accumulate the input gradient is added to dx instead of overwriting it
/// (several convolutions of one input). A null output is skipped.
template <typename T>
void conv_bias_relu_backward(const Geometry& g, const T* x, const T* y, I64 y_ld, const T* grad, I64 grad_ld,
                             const T* f, T* dx, bool dx_accumulate, T* df, T* db) {
  const I64 o = g.o;
  // dz for the pixel rows [r0, r0 + rows) of the output.
  const auto mask = [y, y_ld, grad, grad_ld, o](I64 r0, I64 rows, T* dz) {
    for (I64 r = 0; r < rows; ++r) {
      const T* yr = y + (r0 + r) * y_ld;
      const T* gr = grad + (r0 + r) * grad_ld;
      T*       d  = dz + r * o;
      for (I64 c = 0; c < o; ++c) {
        const T gv = gr[c];
        d[c]       = yr[c] > T(0) ? gv : T(0);
      }
    }
  };
  const auto bias_sum = [o](const T* dz, I64 rows, double* acc) {
    for (I64 r = 0; r < rows; ++r)
      for (I64 c = 0; c < o; ++c) acc[c] += static_cast<double>(dz[r * o + c]);
  };
  const auto finish_bias = [db, o](const std::vector<double>& partial, I64 parts) {
    if (!db) return;
    for (I64 c = 0; c < o; ++c) {
      double total = 0;
      for (I64 p = 0; p < parts; ++p) total += partial[static_cast<std::size_t>(p * o + c)];
      db[c] = static_cast<T>(total);
    }
  };

  if (g.pointwise()) {
    const I64 total = g.n * g.pixels(), block = pointwise_rows(g, sizeof(T)), tasks = (total + block - 1) / block;
    const I64 parts = std::max<I64>(1, std::min(tasks, kFilterPartials)), plane = g.c * o;
    T*        partial = df ? partial_scratch<T>(parts * plane) : nullptr;
    std::vector<double> bias_partial(static_cast<std::size_t>(db ? parts * o : 0), 0.0);
    parallel_for(0, parts, 1, [&](I64 lo, I64 hi) {
      T* dz = patch_scratch<T, 2>(block * o);
      for (I64 p = lo; p < hi; ++p) {
        T*      acc  = partial ? partial + p * plane : nullptr;
        double* bacc = db ? bias_partial.data() + p * o : nullptr;
        for (I64 t = p; t < tasks; t += parts) {
          const I64 r0 = t * block, rows = std::min(total, r0 + block) - r0;
          mask(r0, rows, dz);
          if (bacc) bias_sum(dz, rows, bacc);
          // The partial's first tile overwrites it (no zero fill).
          if (acc) gemm_local(true, false, g.c, o, rows, T(1), x + r0 * g.c, g.c, dz, o, t == p ? T(0) : T(1), acc, o);
          if (dx)
            gemm_local(false, true, rows, g.c, o, T(1), dz, o, f, o, dx_accumulate ? T(1) : T(0), dx + r0 * g.c, g.c);
        }
      }
    });
    if (df) reduce_partials(partial, parts, g.c, o, df);
    finish_bias(bias_partial, parts);
    return;
  }

  // One task per image (its tiles overlap in dx); image n feeds partial n % parts.
  const I64       kc = g.kc(), image = g.pixels() * g.c, plane = kc * o;
  const RowBlocks rb = row_blocks(g, sizeof(T));
  const I64       parts   = std::max<I64>(1, std::min(g.n, kFilterPartials));
  T*              partial = df ? partial_scratch<T>(parts * plane) : nullptr;
  std::vector<double> bias_partial(static_cast<std::size_t>(db ? parts * o : 0), 0.0);
  parallel_for(0, parts, 1, [&](I64 lo, I64 hi) {
    T* cols  = df ? patch_scratch<T, 0>(rb.rows * g.ow * kc) : nullptr;
    T* cols2 = dx ? patch_scratch<T, 1>(rb.rows * g.ow * kc) : nullptr;
    T* dz    = patch_scratch<T, 2>(rb.rows * g.ow * o);
    for (I64 p = lo; p < hi; ++p) {
      T*      acc  = partial ? partial + p * plane : nullptr;
      double* bacc = db ? bias_partial.data() + p * o : nullptr;
      for (I64 n = p; n < g.n; n += parts) {
        if (dx && !dx_accumulate) std::fill(dx + n * image, dx + (n + 1) * image, T(0));
        for (I64 b = 0; b < rb.blocks; ++b) {
          const I64 oy0 = b * rb.rows, oy1 = std::min(g.oh, oy0 + rb.rows), rows = (oy1 - oy0) * g.ow;
          mask((n * g.oh + oy0) * g.ow, rows, dz);
          if (bacc) bias_sum(dz, rows, bacc);
          if (acc) {
            // The partial's first tile overwrites it (no zero fill).
            im2col_rows(g, x, n, oy0, oy1, 0, cols);
            gemm_local(true, false, kc, o, rows, T(1), cols, kc, dz, o, n == p && b == 0 ? T(0) : T(1), acc, o);
          }
          if (dx) {
            gemm_local(false, true, rows, kc, o, T(1), dz, o, f, o, T(0), cols2, kc);
            col2im_add_rows(g, cols2, n, oy0, oy1, 0, dx);
          }
        }
      }
    }
  });
  if (df) reduce_partials(partial, parts, kc, o, df);
  finish_bias(bias_partial, parts);
}

template <typename G>
void check_grad_shape(const G& g, const Tensor& grad, const char* what) {
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

namespace {

// conv{2,3}d, their input gradients and their filter gradients share one
// implementation, picked by the options type.
struct ConvNames {
  const char *forward, *forward_op, *input, *input_op, *filter, *filter_op;
};
const ConvNames kNames2{"conv2d", "Conv2D", "conv2d_backprop_input", "Conv2DBackpropInput", "conv2d_backprop_filter",
                        "Conv2DBackpropFilter"};
const ConvNames kNames3{"conv3d", "Conv3D", "conv3d_backprop_input", "Conv3DBackpropInputV2", "conv3d_backprop_filter",
                        "Conv3DBackpropFilterV2"};
const ConvNames& names_for(const Conv2DOptions&) { return kNames2; }
const ConvNames& names_for(const Conv3DOptions&) { return kNames3; }
Geometry  conv_geometry(const Shape& xs, const Shape& fs, const Conv2DOptions& o, const char* what) { return geometry(xs, fs, o, what); }
Geometry3 conv_geometry(const Shape& xs, const Shape& fs, const Conv3DOptions& o, const char* what) { return geometry3(xs, fs, o, what); }
Tensor any_conv(const Tensor& x, const Tensor& f, const Conv2DOptions& o) { return conv2d(x, f, o); }
Tensor any_conv(const Tensor& x, const Tensor& f, const Conv3DOptions& o) { return conv3d(x, f, o); }
Tensor any_backprop_input(const Shape& s, const Tensor& f, const Tensor& g, const Conv2DOptions& o) {
  return conv2d_backprop_input(s, f, g, o);
}
Tensor any_backprop_input(const Shape& s, const Tensor& f, const Tensor& g, const Conv3DOptions& o) {
  return conv3d_backprop_input(s, f, g, o);
}
Tensor any_backprop_filter(const Tensor& x, const Shape& s, const Tensor& g, const Conv2DOptions& o) {
  return conv2d_backprop_filter(x, s, g, o);
}
Tensor any_backprop_filter(const Tensor& x, const Shape& s, const Tensor& g, const Conv3DOptions& o) {
  return conv3d_backprop_filter(x, s, g, o);
}

template <typename Opt>
Tensor conv_impl(const Tensor& x, const Tensor& filter, const Opt& options) {
  const ConvNames& nm = names_for(options);
  const DType      dt = float_type(x, filter);
  const Tensor     xc = as_dtype(x, dt), fc = as_dtype(filter, dt);
  const auto       g  = conv_geometry(xc.shape(), fc.shape(), options, nm.forward);
  Tensor           out = empty(g.out_shape(), dt);
  {
    NoGradGuard  guard;
    const Tensor xk = nograd(xc).contiguous(), fk = nograd(fc).contiguous();
    dispatch_float(dt, nm.forward, [&](auto tag) {
      using T = typename decltype(tag)::type;
      conv_forward<T>(g, xk.data_ptr<T>(), fk.data_ptr<T>(), out.data_ptr<T>());
    });
  }
  const SavedTensor sx(xc), sf(fc);
  const bool        need_x = xc.requires_grad(), need_f = fc.requires_grad();
  record(nm.forward_op, {xc, fc}, {out}, [sx, sf, options, need_x, need_f](const Grads& gr, const Grads&) {
    const Tensor& X = sx.get();
    const Tensor& F = sf.get();
    return Grads{need_x ? any_backprop_input(X.shape(), F, gr[0], options) : Tensor(),
                 need_f ? any_backprop_filter(X, F.shape(), gr[0], options) : Tensor()};
  });
  return out;
}

template <typename Opt>
Tensor backprop_input_impl(const Shape& input_shape, const Tensor& filter, const Tensor& grad, const Opt& options) {
  const ConvNames& nm = names_for(options);
  const DType      dt = float_type(filter, grad);
  const Tensor     fc = as_dtype(filter, dt), gc = as_dtype(grad, dt);
  const auto       g  = conv_geometry(input_shape, fc.shape(), options, nm.input);
  check_grad_shape(g, gc, nm.input);
  Tensor out = empty(input_shape, dt);
  {
    NoGradGuard  guard;
    const Tensor fk = nograd(fc).contiguous(), gk = nograd(gc).contiguous();
    dispatch_float(dt, nm.input, [&](auto tag) {
      using T = typename decltype(tag)::type;
      conv_backward_input<T>(g, fk.data_ptr<T>(), gk.data_ptr<T>(), out.data_ptr<T>());
    });
  }
  const SavedTensor sf(fc), sg(gc);
  const bool        need_f = fc.requires_grad(), need_g = gc.requires_grad();
  record(nm.input_op, {fc, gc}, {out}, [sf, sg, options, need_f, need_g](const Grads& gr, const Grads&) {
    const Tensor& F = sf.get();
    const Tensor& G = sg.get();
    return Grads{need_f ? any_backprop_filter(gr[0], F.shape(), G, options) : Tensor(),
                 need_g ? any_conv(gr[0], F, options) : Tensor()};
  });
  return out;
}

template <typename Opt>
Tensor backprop_filter_impl(const Tensor& x, const Shape& filter_shape, const Tensor& grad, const Opt& options) {
  const ConvNames& nm = names_for(options);
  const DType      dt = float_type(x, grad);
  const Tensor     xc = as_dtype(x, dt), gc = as_dtype(grad, dt);
  const auto       g  = conv_geometry(xc.shape(), filter_shape, options, nm.filter);
  check_grad_shape(g, gc, nm.filter);
  Tensor out = empty(filter_shape, dt);
  {
    NoGradGuard  guard;
    const Tensor xk = nograd(xc).contiguous(), gk = nograd(gc).contiguous();
    dispatch_float(dt, nm.filter, [&](auto tag) {
      using T = typename decltype(tag)::type;
      conv_backward_filter<T>(g, xk.data_ptr<T>(), gk.data_ptr<T>(), out.data_ptr<T>());
    });
  }
  const SavedTensor sx(xc), sg(gc);
  const bool        need_x = xc.requires_grad(), need_g = gc.requires_grad();
  record(nm.filter_op, {xc, gc}, {out}, [sx, sg, options, need_x, need_g](const Grads& gr, const Grads&) {
    const Tensor& X = sx.get();
    const Tensor& G = sg.get();
    return Grads{need_x ? any_backprop_input(X.shape(), gr[0], G, options) : Tensor(),
                 need_g ? any_conv(X, gr[0], options) : Tensor()};
  });
  return out;
}

/// Conventional transposed-convolution output length: in s (SAME), (in - 1) s + k' (VALID),
/// (in - 1) s + k' - pads (explicit).
I64 transpose_length(I64 in, I64 k, I64 s, I64 d, Padding padding, I64 pb, I64 pa) {
  const I64 eff = d * (k - 1) + 1;
  switch (padding) {
    case Padding::Same: return in * s;
    case Padding::Valid: return (in - 1) * s + eff;
    case Padding::Explicit: return (in - 1) * s + eff - pb - pa;
  }
  return in;
}

}  // namespace

Tensor conv2d(const Tensor& x, const Tensor& filter, const Conv2DOptions& options) {
  if (detail::is_native(x) || detail::is_native(filter)) {
    const auto g = conv_geometry(x.shape(), filter.shape(), options, "conv2d");
    return detail::device_conv2d(
        x, filter,
        {g.n, g.h, g.w, g.c, g.kh, g.kw, g.o, g.oh, g.ow, g.sh, g.sw, g.dh, g.dw, g.pt, g.pl,
         g.groups});
  }
  return conv_impl(x, filter, options);
}

Tensor conv2d_bias_relu(const Tensor& x, const Tensor& filter, const Tensor& bias, const Conv2DOptions& options) {
  if (filter.dim() != 4 || bias.dim() != 1 || bias.shape()[0] != filter.shape()[3])
    throw std::invalid_argument("conv2d_bias_relu: bias " + shape_str(bias.shape()) + " must be [C_out] for filter " +
                                shape_str(filter.shape()));
  if (detail::is_native(x) || detail::is_native(filter) || detail::is_native(bias) || options.groups != 1)
    return bias_add_relu(conv2d(x, filter, options), bias);
  DType dt = promote_types(promote_types(x.dtype(), filter.dtype()), bias.dtype());
  if (!is_floating(dt)) dt = DType::Float32;
  const Tensor xc = as_dtype(x, dt), fc = as_dtype(filter, dt), bc = as_dtype(bias, dt);
  const auto   g  = conv_geometry(xc.shape(), fc.shape(), options, "conv2d_bias_relu");
  if (is_depthwise(g)) return bias_add_relu(conv2d(xc, fc, options), bc);
  Tensor out = empty(g.out_shape(), dt);
  {
    NoGradGuard  guard;
    const Tensor xk = nograd(xc).contiguous(), fk = nograd(fc).contiguous(), bk = nograd(bc).contiguous();
    dispatch_float(dt, "conv2d_bias_relu", [&](auto tag) {
      using T = typename decltype(tag)::type;
      conv_bias_relu_forward<T>(g, xk.data_ptr<T>(), fk.data_ptr<T>(), bk.data_ptr<T>(), out.data_ptr<T>(), g.o);
    });
  }
  const SavedTensor sx(xc), sf(fc);
  const bool        need_x = xc.requires_grad(), need_f = fc.requires_grad(), need_b = bc.requires_grad();
  record(
      "Conv2DBiasRelu", {xc, fc, bc}, {out},
      [sx, sf, options, need_x, need_f, need_b](const Grads& gr, const Grads& outs) {
        const Tensor& X = sx.get();
        const Tensor& F = sf.get();
        const Tensor  Y = outs[0].detach();
        if (is_grad_enabled()) {
          // Recording: the differentiable composition, for higher-order gradients.
          const Tensor dz = where(greater(Y, 0.0), gr[0], 0.0);
          Axes         axes;
          for (std::int64_t i = 0; i + 1 < Y.dim(); ++i) axes.push_back(i);
          return Grads{need_x ? conv2d_backprop_input(X.shape(), F, dz, options) : Tensor(),
                       need_f ? conv2d_backprop_filter(X, F.shape(), dz, options) : Tensor(),
                       need_b ? sum(dz, axes) : Tensor()};
        }
        const DType dt = Y.dtype();
        const auto  g  = conv_geometry(X.shape(), F.shape(), options, "conv2d_bias_relu");
        check_grad_shape(g, gr[0], "conv2d_bias_relu");
        Tensor dx = need_x ? empty(X.shape(), dt) : Tensor(), df = need_f ? empty(F.shape(), dt) : Tensor(),
               db = need_b ? empty({g.o}, dt) : Tensor();
        const Tensor xk = nograd(X).contiguous(), fk = nograd(F).contiguous();
        Tensor       gk = detail::cast_raw(nograd(gr[0]), dt);
        I64          ld = detail::uniform_row_stride(gk);
        if (ld < 0) {
          gk = gk.contiguous();
          ld = g.o;
        }
        dispatch_float(dt, "conv2d_bias_relu backward", [&](auto tag) {
          using T = typename decltype(tag)::type;
          conv_bias_relu_backward<T>(g, xk.data_ptr<T>(), Y.data_ptr<T>(), g.o, gk.data_ptr<T>(), ld,
                                     fk.data_ptr<T>(), need_x ? dx.data_ptr<T>() : nullptr, false,
                                     need_f ? df.data_ptr<T>() : nullptr, need_b ? db.data_ptr<T>() : nullptr);
        });
        return Grads{dx, df, db};
      },
      /*save_outputs=*/true);
  return out;
}

Tensor conv2d_bias_relu_concat(const Tensor& x, const std::vector<Tensor>& filters, const std::vector<Tensor>& biases,
                               const std::vector<Conv2DOptions>& options) {
  const std::size_t branches = filters.size();
  if (branches == 0 || biases.size() != branches || options.size() != branches)
    throw std::invalid_argument("conv2d_bias_relu_concat: need matching, nonempty filter, bias and option lists");
  for (std::size_t i = 0; i < branches; ++i)
    if (filters[i].dim() != 4 || biases[i].dim() != 1 || biases[i].shape()[0] != filters[i].shape()[3])
      throw std::invalid_argument("conv2d_bias_relu_concat: bias " + shape_str(biases[i].shape()) +
                                  " must be [C_out] for filter " + shape_str(filters[i].shape()));
  // The composition: every case the fused kernels do not cover.
  const auto composed = [&] {
    std::vector<Tensor> outs;
    for (std::size_t i = 0; i < branches; ++i) outs.push_back(conv2d_bias_relu(x, filters[i], biases[i], options[i]));
    return branches == 1 ? outs[0] : concat(outs, -1);
  };
  DType dt = x.dtype();
  bool  fused = !detail::is_native(x);
  for (std::size_t i = 0; i < branches; ++i) {
    dt    = promote_types(promote_types(dt, filters[i].dtype()), biases[i].dtype());
    fused = fused && !detail::is_native(filters[i]) && !detail::is_native(biases[i]) && options[i].groups == 1;
  }
  if (!is_floating(dt)) dt = DType::Float32;
  if (!fused) return composed();
  const Tensor          xc = as_dtype(x, dt);
  std::vector<Tensor>   fc, bc;
  std::vector<Geometry> geos;
  std::vector<I64>      offsets;
  I64                   width = 0;
  for (std::size_t i = 0; i < branches; ++i) {
    fc.push_back(as_dtype(filters[i], dt));
    bc.push_back(as_dtype(biases[i], dt));
    geos.push_back(conv_geometry(xc.shape(), fc.back().shape(), options[i], "conv2d_bias_relu_concat"));
    const Geometry& g = geos.back();
    if (is_depthwise(g) || g.oh != geos[0].oh || g.ow != geos[0].ow) return composed();
    offsets.push_back(width);
    width += g.o;
  }
  const Geometry& g0  = geos[0];
  Tensor          out = empty({g0.n, g0.oh, g0.ow, width}, dt);
  {
    NoGradGuard  guard;
    const Tensor xk = nograd(xc).contiguous();
    for (std::size_t i = 0; i < branches; ++i) {
      const Tensor fk = nograd(fc[i]).contiguous(), bk = nograd(bc[i]).contiguous();
      dispatch_float(dt, "conv2d_bias_relu_concat", [&](auto tag) {
        using T = typename decltype(tag)::type;
        conv_bias_relu_forward<T>(geos[i], xk.data_ptr<T>(), fk.data_ptr<T>(), bk.data_ptr<T>(),
                                  out.data_ptr<T>() + offsets[i], width);
      });
    }
  }
  std::vector<Tensor> inputs{xc};
  inputs.insert(inputs.end(), fc.begin(), fc.end());
  inputs.insert(inputs.end(), bc.begin(), bc.end());
  std::vector<bool> need;
  for (const Tensor& t : inputs) need.push_back(t.requires_grad());
  const SavedTensor        sx(xc);
  std::vector<SavedTensor> sf;
  for (const Tensor& f : fc) sf.emplace_back(f);
  record(
      "Conv2DBiasReluConcat", inputs, {out},
      [sx, sf, options, need, offsets, width](const Grads& gr, const Grads& outs) {
        const std::size_t branches = sf.size();
        const Tensor&     X        = sx.get();
        const Tensor      Y        = outs[0].detach();
        Grads             result(1 + 2 * branches);
        if (is_grad_enabled()) {
          // Recording: the differentiable composition, for higher-order gradients.
          const Tensor dz = where(greater(Y, 0.0), gr[0], 0.0);
          for (std::size_t i = 0; i < branches; ++i) {
            const Tensor& F   = sf[i].get();
            const Tensor  dzi = slice_axis(dz, -1, offsets[i], offsets[i] + F.shape()[3]);
            if (need[0]) {
              const Tensor dxi = conv2d_backprop_input(X.shape(), F, dzi, options[i]);
              result[0]        = result[0].defined() ? add(result[0], dxi) : dxi;
            }
            if (need[1 + i]) result[1 + i] = conv2d_backprop_filter(X, F.shape(), dzi, options[i]);
            if (need[1 + branches + i]) result[1 + branches + i] = sum(dzi, Axes{0, 1, 2});
          }
          return result;
        }
        const DType dt = Y.dtype();
        if (gr[0].shape() != Y.shape())
          throw std::invalid_argument("conv2d_bias_relu_concat: gradient shape " + shape_str(gr[0].shape()) +
                                      " does not match the output " + shape_str(Y.shape()));
        const Tensor xk = nograd(X).contiguous();
        Tensor       gk = detail::cast_raw(nograd(gr[0]), dt);
        I64          ld = detail::uniform_row_stride(gk);
        if (ld < 0) {
          gk = gk.contiguous();
          ld = width;
        }
        if (need[0]) result[0] = empty(X.shape(), dt);
        for (std::size_t i = 0; i < branches; ++i) {
          const Tensor& F  = sf[i].get();
          const auto    g  = conv_geometry(X.shape(), F.shape(), options[i], "conv2d_bias_relu_concat");
          const Tensor  fk = nograd(F).contiguous();
          if (need[1 + i]) result[1 + i] = empty(F.shape(), dt);
          if (need[1 + branches + i]) result[1 + branches + i] = empty({g.o}, dt);
          dispatch_float(dt, "conv2d_bias_relu_concat backward", [&](auto tag) {
            using T = typename decltype(tag)::type;
            conv_bias_relu_backward<T>(g, xk.data_ptr<T>(), Y.data_ptr<T>() + offsets[i], width,
                                       gk.data_ptr<T>() + offsets[i], ld, fk.data_ptr<T>(),
                                       need[0] ? result[0].data_ptr<T>() : nullptr, /*dx_accumulate=*/i > 0,
                                       need[1 + i] ? result[1 + i].data_ptr<T>() : nullptr,
                                       need[1 + branches + i] ? result[1 + branches + i].data_ptr<T>() : nullptr);
          });
        }
        return result;
      },
      /*save_outputs=*/true);
  return out;
}

Tensor conv2d_backprop_input(const Shape& input_shape, const Tensor& filter, const Tensor& grad,
                             const Conv2DOptions& options) {
  return backprop_input_impl(input_shape, filter, grad, options);
}

Tensor conv2d_backprop_filter(const Tensor& x, const Shape& filter_shape, const Tensor& grad,
                              const Conv2DOptions& options) {
  return backprop_filter_impl(x, filter_shape, grad, options);
}

// ---- convolution compositions -----------------------------------------------------------------

Tensor conv2d_transpose(const Tensor& x, const Tensor& filter, const Shape& output_shape,
                        const Conv2DOptions& options) {
  return conv2d_backprop_input(output_shape, filter, x, options);
}

Tensor conv2d_transpose(const Tensor& x, const Tensor& filter, const Conv2DOptions& options) {
  if (x.dim() != 4 || filter.dim() != 4)
    throw std::invalid_argument("conv2d_transpose: need x [N, H, W, C] and filter [KH, KW, O, C]");
  const auto& xs = x.shape();
  const auto& fs = filter.shape();
  const auto& ep = options.explicit_paddings;
  const auto& st = options.strides;
  const auto& dl = options.dilations;
  const Shape out{xs[0], transpose_length(xs[1], fs[0], st[0], dl[0], options.padding, ep[0], ep[1]),
                  transpose_length(xs[2], fs[1], st[1], dl[1], options.padding, ep[2], ep[3]), fs[2] * options.groups};
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

// ---- 3-D convolution -------------------------------------------------------------------------

Conv3DOptions conv3d_options(std::int64_t stride, Padding padding, std::int64_t dilation) {
  Conv3DOptions o;
  o.strides   = {stride, stride, stride};
  o.dilations = {dilation, dilation, dilation};
  o.padding   = padding;
  return o;
}

Tensor conv3d(const Tensor& x, const Tensor& filter, const Conv3DOptions& options) {
  return conv_impl(x, filter, options);
}

Tensor conv3d_backprop_input(const Shape& input_shape, const Tensor& filter, const Tensor& grad,
                             const Conv3DOptions& options) {
  return backprop_input_impl(input_shape, filter, grad, options);
}

Tensor conv3d_backprop_filter(const Tensor& x, const Shape& filter_shape, const Tensor& grad,
                              const Conv3DOptions& options) {
  return backprop_filter_impl(x, filter_shape, grad, options);
}

Tensor conv3d_transpose(const Tensor& x, const Tensor& filter, const Shape& output_shape,
                        const Conv3DOptions& options) {
  return conv3d_backprop_input(output_shape, filter, x, options);
}

Tensor conv3d_transpose(const Tensor& x, const Tensor& filter, const Conv3DOptions& options) {
  if (x.dim() != 5 || filter.dim() != 5)
    throw std::invalid_argument("conv3d_transpose: need x [N, D, H, W, C] and filter [KD, KH, KW, O, C]");
  const auto& xs = x.shape();
  const auto& fs = filter.shape();
  const auto& ep = options.explicit_paddings;
  const auto& st = options.strides;
  const auto& dl = options.dilations;
  const Shape out{xs[0], transpose_length(xs[1], fs[0], st[0], dl[0], options.padding, ep[0], ep[1]),
                  transpose_length(xs[2], fs[1], st[1], dl[1], options.padding, ep[2], ep[3]),
                  transpose_length(xs[3], fs[2], st[2], dl[2], options.padding, ep[4], ep[5]), fs[3] * options.groups};
  return conv3d_transpose(x, filter, out, options);
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
  Tensor out = empty(input_shape, grad_out.dtype());
  {
    NoGradGuard  guard;
    const Tensor gk = nograd(grad_out).contiguous();
    const I64    n = input_shape[0], per_out = n ? argmax.numel() / n : 0, per_in = n ? out.numel() / n : 0;
    const I64*   idx = argmax.data_ptr<I64>();
    dispatch_float(grad_out.dtype(), "max_pool2d backward", [&](auto tag) {
      using T      = typename decltype(tag)::type;
      const T* src = gk.data_ptr<T>();
      T*       dst = out.data_ptr<T>();
      // Each image's argmaxes stay inside that image, so images never collide;
      // zeroing the image right before its scatter keeps it in cache.
      parallel_for(0, n, 1, [&](I64 lo, I64 hi) {
        for (I64 image = lo; image < hi; ++image) {
          std::fill(dst + image * per_in, dst + (image + 1) * per_in, T(0));
          for (I64 i = image * per_out; i < (image + 1) * per_out; ++i) dst[idx[i]] += src[i];
        }
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

/// Max pooling forward: the pooled values (with the MaxPool history) and the
/// global flat argmax of every output (int64, indexing x including the batch).
std::pair<Tensor, Tensor> max_pool_with_global_argmax(const Tensor& x_in, std::array<I64, 2> window,
                                                      std::array<I64, 2> strides, Padding padding, const char* what) {
  const Tensor       x = is_floating(x_in.dtype()) ? x_in : cast(x_in, DType::Float32);
  const PoolGeometry g = pool_geometry(x.shape(), window, strides, padding, what);
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
      // Each window is pooled in blocks of kBlock channels held in local arrays,
      // so the per-tap compare/select over a block vectorises and never bounces
      // through memory. The first tap of a window is taken as is and every
      // later tap replaces it only when strictly greater (so a NaN first tap
      // stays and a later NaN is ignored).
      constexpr I64 kBlock = 16;
      parallel_for(0, g.n * g.oh, 1, [&](I64 lo, I64 hi) {
        for (I64 r = lo; r < hi; ++r) {
          const I64 n = r / g.oh, oy = r % g.oh;
          for (I64 ox = 0; ox < g.ow; ++ox) {
            const Window wd = window_at(g, oy, ox);
            const I64    o  = ((n * g.oh + oy) * g.ow + ox) * g.c;
            if (wd.y0 >= wd.y1 || wd.x0 >= wd.x1) {
              std::fill(dst + o, dst + o + g.c, -std::numeric_limits<T>::infinity());
              std::fill(idx + o, idx + o + g.c, I64{-1});
              continue;
            }
            for (I64 c0 = 0; c0 < g.c; c0 += kBlock) {
              const I64 cn = std::min(kBlock, g.c - c0);
              T         best[kBlock];
              I64       at[kBlock];
              bool      first = true;
              for (I64 iy = wd.y0; iy < wd.y1; ++iy)
                for (I64 ix = wd.x0; ix < wd.x1; ++ix) {
                  const I64 base = ((n * g.h + iy) * g.w + ix) * g.c + c0;
                  const T*  v    = src + base;
                  if (first) {
                    for (I64 c = 0; c < cn; ++c) {
                      best[c] = v[c];
                      at[c]   = base + c;
                    }
                    first = false;
                  } else {
                    for (I64 c = 0; c < cn; ++c) {
                      const bool better = v[c] > best[c];
                      best[c]           = better ? v[c] : best[c];
                      at[c]             = better ? base + c : at[c];
                    }
                  }
                }
              std::copy(best, best + cn, dst + o + c0);
              std::copy(at, at + cn, idx + o + c0);
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
  return {out, argmax};
}

/// Tap value of an output whose window holds no input (it gets no gradient).
constexpr std::uint8_t kNoTap = 255;

/// Max pooling that keeps each output's winning tap (its row-major position
/// in the window, one byte) instead of a global input index: 8x less index
/// memory, and both passes run over channels in vector-friendly blocks. Same
/// values and switches as max_pool_with_global_argmax: the first in-bounds tap
/// is taken as is and a later one wins only when strictly greater.
template <typename T>
void max_pool_taps_forward(const PoolGeometry& g, const T* x, T* y, std::uint8_t* taps) {
  constexpr I64 kBlock = 64;
  parallel_for(0, g.n * g.oh, 1, [&](I64 lo, I64 hi) {
    for (I64 r = lo; r < hi; ++r) {
      const I64 n = r / g.oh, oy = r % g.oh, ys = oy * g.sh - g.pt;
      for (I64 ox = 0; ox < g.ow; ++ox) {
        const I64    xs = ox * g.sw - g.pl, o = ((n * g.oh + oy) * g.ow + ox) * g.c;
        const Window wd = window_at(g, oy, ox);
        if (wd.y0 >= wd.y1 || wd.x0 >= wd.x1) {
          std::fill(y + o, y + o + g.c, -std::numeric_limits<T>::infinity());
          std::fill(taps + o, taps + o + g.c, kNoTap);
          continue;
        }
        for (I64 c0 = 0; c0 < g.c; c0 += kBlock) {
          const I64    cn = std::min(kBlock, g.c - c0);
          T            best[kBlock];
          std::uint8_t at[kBlock];
          bool         first = true;
          for (I64 iy = wd.y0; iy < wd.y1; ++iy)
            for (I64 ix = wd.x0; ix < wd.x1; ++ix) {
              const auto t = static_cast<std::uint8_t>((iy - ys) * g.kw + (ix - xs));
              const T*   v = x + ((n * g.h + iy) * g.w + ix) * g.c + c0;
              if (first) {
                for (I64 c = 0; c < cn; ++c) {
                  best[c] = v[c];
                  at[c]   = t;
                }
                first = false;
              } else {
                for (I64 c = 0; c < cn; ++c) {
                  const bool better = v[c] > best[c];
                  best[c]           = better ? v[c] : best[c];
                  at[c]             = better ? t : at[c];
                }
              }
            }
          for (I64 c = 0; c < cn; ++c) {
            y[o + c0 + c]    = best[c];
            taps[o + c0 + c] = at[c];
          }
        }
      }
    }
  });
}

/// dx of max_pool_taps_forward. Each image is zeroed right before it is
/// filled (so it stays in cache), and every input element receives its
/// windows' gradients in output order, the summation order of
/// max_pool_scatter, so the result is the same.
template <typename T>
void max_pool_taps_backward(const PoolGeometry& g, const T* grad, const std::uint8_t* taps, T* dx) {
  const I64 per_in = g.h * g.w * g.c;
  parallel_for(0, g.n, 1, [&](I64 lo, I64 hi) {
    for (I64 n = lo; n < hi; ++n) {
      T* d = dx + n * per_in;
      std::fill(d, d + per_in, T(0));
      for (I64 oy = 0; oy < g.oh; ++oy)
        for (I64 ox = 0; ox < g.ow; ++ox) {
          const I64           ys = oy * g.sh - g.pt, xs = ox * g.sw - g.pl, o = ((n * g.oh + oy) * g.ow + ox) * g.c;
          const Window        wd = window_at(g, oy, ox);
          const T* __restrict            gv = grad + o;
          const std::uint8_t* __restrict tv = taps + o;
          for (I64 iy = wd.y0; iy < wd.y1; ++iy)
            for (I64 ix = wd.x0; ix < wd.x1; ++ix) {
              const auto     t   = static_cast<std::uint8_t>((iy - ys) * g.kw + (ix - xs));
              T* __restrict dst = d + (iy * g.w + ix) * g.c;
              // Both values are loaded unconditionally so the select vectorises.
              for (I64 c = 0; c < g.c; ++c) {
                const T v = gv[c];
                dst[c] += tv[c] == t ? v : T(0);
              }
            }
        }
    }
  });
}

/// The global input index of every output (the max_pool_scatter form; -1 for
/// an empty window), for higher-order gradients of the tap form.
Tensor taps_to_argmax(const PoolGeometry& g, const std::vector<std::uint8_t>& taps) {
  Tensor argmax = empty(g.out_shape(), DType::Int64);
  I64*   idx    = argmax.data_ptr<I64>();
  parallel_for(0, g.n * g.oh, 1, [&](I64 lo, I64 hi) {
    for (I64 r = lo; r < hi; ++r) {
      const I64 n = r / g.oh, oy = r % g.oh, ys = oy * g.sh - g.pt;
      for (I64 ox = 0; ox < g.ow; ++ox) {
        const I64 xs = ox * g.sw - g.pl, o = ((n * g.oh + oy) * g.ow + ox) * g.c;
        for (I64 c = 0; c < g.c; ++c) {
          const std::uint8_t t = taps[static_cast<std::size_t>(o + c)];
          idx[o + c] = t == kNoTap ? I64{-1}
                                   : ((n * g.h + ys + t / g.kw) * g.w + xs + t % g.kw) * g.c + c;
        }
      }
    }
  });
  return argmax;
}

Tensor max_pool_with_taps(const Tensor& x_in, std::array<I64, 2> window, std::array<I64, 2> strides,
                          Padding padding) {
  const Tensor       x = is_floating(x_in.dtype()) ? x_in : cast(x_in, DType::Float32);
  const PoolGeometry g = pool_geometry(x.shape(), window, strides, padding, "max_pool2d");
  Tensor             out  = empty(g.out_shape(), x.dtype());
  auto               taps = std::make_shared<std::vector<std::uint8_t>>(static_cast<std::size_t>(out.numel()));
  {
    NoGradGuard  guard;
    const Tensor xk = nograd(x).contiguous();
    dispatch_float(x.dtype(), "max_pool2d", [&](auto tag) {
      using T = typename decltype(tag)::type;
      max_pool_taps_forward<T>(g, xk.data_ptr<T>(), out.data_ptr<T>(), taps->data());
    });
  }
  record("MaxPool", {x}, {out}, [g, taps](const Grads& gr, const Grads&) {
    if (is_grad_enabled()) return Grads{max_pool_scatter(gr[0], taps_to_argmax(g, *taps), {g.n, g.h, g.w, g.c})};
    const DType  dt = gr[0].dtype();
    const Tensor gk = nograd(gr[0]).contiguous();
    Tensor       dx = empty({g.n, g.h, g.w, g.c}, dt);
    dispatch_float(dt, "max_pool2d backward", [&](auto tag) {
      using T = typename decltype(tag)::type;
      max_pool_taps_backward<T>(g, gk.data_ptr<T>(), taps->data(), dx.data_ptr<T>());
    });
    return Grads{dx};
  });
  return out;
}

}  // namespace

Tensor max_pool2d(const Tensor& x, std::array<std::int64_t, 2> window, std::array<std::int64_t, 2> strides,
                  Padding padding) {
  if (detail::is_native(x)) {
    const PoolGeometry g = pool_geometry(x.shape(), window, strides, padding, "max_pool2d");
    return detail::device_max_pool2d(
        x, {g.n, g.h, g.w, g.c, g.kh, g.kw, g.sh, g.sw, g.oh, g.ow, g.pt, g.pl});
  }
  if (window[0] * window[1] < kNoTap) return max_pool_with_taps(x, window, strides, padding);
  return max_pool_with_global_argmax(x, window, strides, padding, "max_pool2d").first;
}

PoolWithArgmax max_pool2d_with_argmax(const Tensor& x, std::array<std::int64_t, 2> window,
                                      std::array<std::int64_t, 2> strides, Padding padding,
                                      bool include_batch_in_index) {
  auto [out, argmax] = max_pool_with_global_argmax(x, window, strides, padding, "max_pool2d_with_argmax");
  if (!include_batch_in_index && argmax.numel() > 0) {
    argmax = argmax.clone();  // the MaxPool backward keeps the global indices
    const I64 n = x.size(0), per_image = x.numel() / n, per_out = argmax.numel() / n;
    I64*      idx = argmax.data_ptr<I64>();
    for (I64 i = 0; i < argmax.numel(); ++i) idx[i] -= (i / per_out) * per_image;
  }
  return {out, argmax};
}

PoolWithArgmax max_pool2d_with_argmax(const Tensor& x, std::int64_t window, std::int64_t stride, Padding padding,
                                      bool include_batch_in_index) {
  return max_pool2d_with_argmax(x, {window, window}, {stride, stride}, padding, include_batch_in_index);
}

Tensor max_unpool2d(const Tensor& values, const Tensor& argmax, const Shape& output_shape,
                    bool include_batch_in_index) {
  if (output_shape.size() != 4)
    throw std::invalid_argument("max_unpool2d: output_shape must be [N, H, W, C], got " + shape_str(output_shape));
  if (values.dim() < 1 || values.shape() != argmax.shape() || values.size(0) != output_shape[0])
    throw std::invalid_argument("max_unpool2d: values " + shape_str(values.shape()) + " and argmax " +
                                shape_str(argmax.shape()) + " must have the same shape, with batch " +
                                std::to_string(output_shape[0]));
  if (is_floating(argmax.dtype()) || argmax.dtype() == DType::Bool)
    throw std::invalid_argument("max_unpool2d: argmax must be an integer tensor");
  const Tensor v = is_floating(values.dtype()) ? values : cast(values, DType::Float32);
  Tensor       global = cast(argmax, DType::Int64).contiguous();
  if (global.is_same(argmax)) global = global.clone();
  const I64 n = output_shape[0], per_image = shape_numel(output_shape) / std::max<I64>(n, 1),
            per_out = n ? global.numel() / n : 0;
  I64* idx = global.data_ptr<I64>();
  for (I64 i = 0; i < global.numel(); ++i) {
    const I64 image = i / per_out, lo = image * per_image;
    if (!include_batch_in_index) idx[i] += lo;
    // Each index must stay inside its own image (the scatter runs one image per thread).
    if (idx[i] < lo || idx[i] >= lo + per_image)
      throw std::invalid_argument("max_unpool2d: argmax entry " + std::to_string(i) + " is outside image " +
                                  std::to_string(image) + " of " + shape_str(output_shape));
  }
  return max_pool_scatter(v, global, output_shape);
}

Tensor local_response_normalization(const Tensor& x, std::int64_t depth_radius, double bias, double alpha,
                                    double beta) {
  if (x.dim() < 1) throw std::invalid_argument("local_response_normalization: x needs a channel (last) axis");
  if (depth_radius < 0) throw std::invalid_argument("local_response_normalization: depth_radius must be >= 0");
  const Tensor xf = is_floating(x.dtype()) ? x : cast(x, DType::Float32);
  const I64    C  = xf.size(-1), M = C ? xf.numel() / C : 0, width = 2 * depth_radius + 1;
  // Windowed channel sum of squares: a VALID average pool of width 2r + 1 over
  // the zero-padded channel axis, times the width (a single differentiable op).
  const Tensor sq     = reshape(square(xf), {M, 1, C, 1});
  const Tensor padded = pad(sq, {{0, 0}, {0, 0}, {depth_radius, depth_radius}, {0, 0}});
  const Tensor sums   = reshape(mul(avg_pool2d(padded, {1, width}, {1, 1}, Padding::Valid), static_cast<double>(width)),
                                xf.shape());
  return mul(xf, pow(add(mul(sums, alpha), bias), -beta));
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
  if (detail::is_native(x)) return detail::device_global_avg_pool2d(x, keepdims);
  return mean(x, {1, 2}, keepdims);
}

Tensor global_max_pool2d(const Tensor& x, bool keepdims) {
  if (x.dim() != 4) throw std::invalid_argument("global_max_pool2d: input must be [N, H, W, C]");
  return max(x, {1, 2}, keepdims);
}

// ---- 3-D pooling ------------------------------------------------------------------------------

namespace {

struct PoolGeometry3 {
  I64 n, d, h, w, c, kd, kh, kw, sd, sh, sw, od, oh, ow, pf, pt, pl;
  [[nodiscard]] Shape out_shape() const { return {n, od, oh, ow, c}; }
};

PoolGeometry3 pool_geometry3(const Shape& xs, std::array<I64, 3> window, std::array<I64, 3> strides, Padding padding,
                             const char* what) {
  if (xs.size() != 5)
    throw std::invalid_argument(std::string(what) + ": input must be [N, D, H, W, C], got " + shape_str(xs));
  for (int i = 0; i < 3; ++i)
    if (window[i] < 1 || strides[i] < 1)
      throw std::invalid_argument(std::string(what) + ": window and strides must be >= 1");
  if (padding == Padding::Explicit) throw std::invalid_argument(std::string(what) + ": use SAME or VALID padding");
  PoolGeometry3 g{};
  g.n = xs[0], g.d = xs[1], g.h = xs[2], g.w = xs[3], g.c = xs[4];
  g.kd = window[0], g.kh = window[1], g.kw = window[2], g.sd = strides[0], g.sh = strides[1], g.sw = strides[2];
  const Axis az = axis_geometry(g.d, g.kd, g.sd, 1, padding, 0, 0);
  const Axis ay = axis_geometry(g.h, g.kh, g.sh, 1, padding, 0, 0);
  const Axis ax = axis_geometry(g.w, g.kw, g.sw, 1, padding, 0, 0);
  g.od = az.out, g.oh = ay.out, g.ow = ax.out, g.pf = az.pad_before, g.pt = ay.pad_before, g.pl = ax.pad_before;
  return g;
}

/// In-bounds window [z0, z1) x [y0, y1) x [x0, x1) for one output.
struct Window3 {
  I64 z0, z1, y0, y1, x0, x1;
  [[nodiscard]] I64 count() const { return std::max<I64>(1, (z1 - z0) * (y1 - y0) * (x1 - x0)); }
};
Window3 window_at(const PoolGeometry3& g, I64 oz, I64 oy, I64 ox) {
  const I64 zs = oz * g.sd - g.pf, ys = oy * g.sh - g.pt, xs = ox * g.sw - g.pl;
  return {std::max<I64>(zs, 0), std::min(zs + g.kd, g.d), std::max<I64>(ys, 0),
          std::min(ys + g.kh, g.h), std::max<I64>(xs, 0), std::min(xs + g.kw, g.w)};
}

/// Calls fn(out_offset, window) for every output position of the rows [lo, hi) of (n, oz, oy).
template <typename F>
void for_pool_rows(const PoolGeometry3& g, I64 lo, I64 hi, F&& fn) {
  for (I64 r = lo; r < hi; ++r) {
    const I64 n = r / (g.od * g.oh), oz = (r / g.oh) % g.od, oy = r % g.oh;
    for (I64 ox = 0; ox < g.ow; ++ox) fn(n, (r * g.ow + ox) * g.c, window_at(g, oz, oy, ox));
  }
}

template <typename T>
void avg_pool3d_forward(const PoolGeometry3& g, const T* x, T* y) {
  parallel_for(0, g.n * g.od * g.oh, 1, [&](I64 lo, I64 hi) {
    for_pool_rows(g, lo, hi, [&](I64 n, I64 o, const Window3& wd) {
      T* out = y + o;
      std::fill(out, out + g.c, T(0));
      for (I64 iz = wd.z0; iz < wd.z1; ++iz)
        for (I64 iy = wd.y0; iy < wd.y1; ++iy)
          for (I64 ix = wd.x0; ix < wd.x1; ++ix) {
            const T* src = x + (((n * g.d + iz) * g.h + iy) * g.w + ix) * g.c;
            for (I64 c = 0; c < g.c; ++c) out[c] += src[c];
          }
      const T inv = T(1) / static_cast<T>(wd.count());
      for (I64 c = 0; c < g.c; ++c) out[c] *= inv;
    });
  });
}

template <typename T>
void avg_pool3d_backward(const PoolGeometry3& g, const T* dy, T* dx) {
  std::fill(dx, dx + g.n * g.d * g.h * g.w * g.c, T(0));
  depth_phased_for(0, g.n * g.od, g.od, g.kd, g.sd, [&](I64 n, I64 oz) {
    for (I64 oy = 0; oy < g.oh; ++oy)
      for (I64 ox = 0; ox < g.ow; ++ox) {
        const Window3 wd  = window_at(g, oz, oy, ox);
        const T*      src = dy + (((n * g.od + oz) * g.oh + oy) * g.ow + ox) * g.c;
        const T       inv = T(1) / static_cast<T>(wd.count());
        for (I64 iz = wd.z0; iz < wd.z1; ++iz)
          for (I64 iy = wd.y0; iy < wd.y1; ++iy)
            for (I64 ix = wd.x0; ix < wd.x1; ++ix) {
              T* d = dx + (((n * g.d + iz) * g.h + iy) * g.w + ix) * g.c;
              for (I64 c = 0; c < g.c; ++c) d[c] += src[c] * inv;
            }
      }
  });
}

Tensor avg_pool3d_grad(const Tensor& grad, const Shape& input_shape, std::array<I64, 3> window,
                       std::array<I64, 3> strides, Padding padding);

Tensor avg_pool3d_impl(const Tensor& x_in, std::array<I64, 3> window, std::array<I64, 3> strides, Padding padding) {
  const Tensor        x = is_floating(x_in.dtype()) ? x_in : cast(x_in, DType::Float32);
  const PoolGeometry3 g = pool_geometry3(x.shape(), window, strides, padding, "avg_pool3d");
  Tensor              out = empty(g.out_shape(), x.dtype());
  {
    NoGradGuard  guard;
    const Tensor xk = nograd(x).contiguous();
    dispatch_float(x.dtype(), "avg_pool3d", [&](auto tag) {
      using T = typename decltype(tag)::type;
      avg_pool3d_forward<T>(g, xk.data_ptr<T>(), out.data_ptr<T>());
    });
  }
  const Shape xs = x.shape();
  record("AvgPool3D", {x}, {out}, [xs, window, strides, padding](const Grads& gr, const Grads&) {
    return Grads{avg_pool3d_grad(gr[0], xs, window, strides, padding)};
  });
  return out;
}

Tensor avg_pool3d_grad(const Tensor& grad, const Shape& input_shape, std::array<I64, 3> window,
                       std::array<I64, 3> strides, Padding padding) {
  const PoolGeometry3 g = pool_geometry3(input_shape, window, strides, padding, "avg_pool3d_grad");
  if (grad.shape() != g.out_shape()) throw std::invalid_argument("avg_pool3d_grad: gradient shape mismatch");
  Tensor out = empty(input_shape, grad.dtype());
  {
    NoGradGuard  guard;
    const Tensor gk = nograd(grad).contiguous();
    dispatch_float(grad.dtype(), "avg_pool3d_grad", [&](auto tag) {
      using T = typename decltype(tag)::type;
      avg_pool3d_backward<T>(g, gk.data_ptr<T>(), out.data_ptr<T>());
    });
  }
  // The adjoint of the adjoint is the pooling itself.
  record("AvgPool3DGrad", {grad}, {out}, [window, strides, padding](const Grads& gr, const Grads&) {
    return Grads{avg_pool3d_impl(gr[0], window, strides, padding)};
  });
  return out;
}

}  // namespace

Tensor max_pool3d(const Tensor& x_in, std::array<std::int64_t, 3> window, std::array<std::int64_t, 3> strides,
                  Padding padding) {
  const Tensor        x = is_floating(x_in.dtype()) ? x_in : cast(x_in, DType::Float32);
  const PoolGeometry3 g = pool_geometry3(x.shape(), window, strides, padding, "max_pool3d");
  Tensor              out    = empty(g.out_shape(), x.dtype());
  Tensor              argmax = empty(g.out_shape(), DType::Int64);
  {
    NoGradGuard  guard;
    const Tensor xk = nograd(x).contiguous();
    dispatch_float(x.dtype(), "max_pool3d", [&](auto tag) {
      using T      = typename decltype(tag)::type;
      const T* src = xk.data_ptr<T>();
      T*       dst = out.data_ptr<T>();
      I64*     idx = argmax.data_ptr<I64>();
      parallel_for(0, g.n * g.od * g.oh, 1, [&](I64 lo, I64 hi) {
        for_pool_rows(g, lo, hi, [&](I64 n, I64 o, const Window3& wd) {
          T*   best = dst + o;
          I64* at   = idx + o;
          std::fill(best, best + g.c, -std::numeric_limits<T>::infinity());
          std::fill(at, at + g.c, I64{-1});
          // Scanned in input order with a strict '>', so the first maximum wins.
          for (I64 iz = wd.z0; iz < wd.z1; ++iz)
            for (I64 iy = wd.y0; iy < wd.y1; ++iy)
              for (I64 ix = wd.x0; ix < wd.x1; ++ix) {
                const I64 base = (((n * g.d + iz) * g.h + iy) * g.w + ix) * g.c;
                for (I64 c = 0; c < g.c; ++c)
                  if (src[base + c] > best[c] || at[c] < 0) {
                    best[c] = src[base + c];
                    at[c]   = base + c;
                  }
              }
        });
      });
    });
  }
  const Shape xs = x.shape();
  record("MaxPool3D", {x}, {out}, [argmax, xs](const Grads& gr, const Grads&) {
    return Grads{max_pool_scatter(gr[0], argmax, xs)};
  });
  return out;
}

Tensor max_pool3d(const Tensor& x, std::int64_t window, std::int64_t stride, Padding padding) {
  return max_pool3d(x, {window, window, window}, {stride, stride, stride}, padding);
}

Tensor avg_pool3d(const Tensor& x, std::array<std::int64_t, 3> window, std::array<std::int64_t, 3> strides,
                  Padding padding) {
  return avg_pool3d_impl(x, window, strides, padding);
}

Tensor avg_pool3d(const Tensor& x, std::int64_t window, std::int64_t stride, Padding padding) {
  return avg_pool3d(x, {window, window, window}, {stride, stride, stride}, padding);
}

Tensor global_avg_pool3d(const Tensor& x, bool keepdims) {
  if (x.dim() != 5) throw std::invalid_argument("global_avg_pool3d: input must be [N, D, H, W, C]");
  return mean(x, {1, 2, 3}, keepdims);
}

Tensor global_max_pool3d(const Tensor& x, bool keepdims) {
  if (x.dim() != 5) throw std::invalid_argument("global_max_pool3d: input must be [N, D, H, W, C]");
  return max(x, {1, 2, 3}, keepdims);
}

// ---- batch normalization ------------------------------------------------------------------------

namespace {

/// Per-channel sums over the rows of a [rows, C] buffer, computed in double.
template <typename T, typename F>
std::vector<double> channel_sums(I64 rows, I64 c, F&& value) {
  std::vector<double> total(static_cast<std::size_t>(c), 0.0);
  if (rows <= 0 || c <= 0) return total;
  const I64 grain = std::max<I64>(1, 8192 / c);
  if (get_num_threads() <= 1 || rows <= grain) {
    for (I64 r = 0; r < rows; ++r)
      for (I64 k = 0; k < c; ++k) total[static_cast<std::size_t>(k)] += value(r, k);
    return total;
  }
  const I64 chunks = std::min<I64>((rows + grain - 1) / grain,
                                 static_cast<I64>(get_num_threads()) * 4);
  const I64 step = (rows + chunks - 1) / chunks;
  // Align and pad worker regions for 128-byte lines (also covering 64-byte
  // hosts). Merely padding an unaligned vector would still share boundary lines.
  // Storage is bounded by four chunks per configured thread, not by row count.
  constexpr I64 line_values = 128 / sizeof(double);
  const I64 stride = ((c + line_values - 1) / line_values) * line_values;
  std::vector<double> storage(static_cast<std::size_t>(chunks * stride + line_values - 1), 0.0);
  const auto misalignment = reinterpret_cast<std::uintptr_t>(storage.data()) % 128;
  double* partials = storage.data() + ((128 - misalignment) % 128) / sizeof(double);
  parallel_for(0, chunks, 1, [&](I64 lo, I64 hi) {
    for (I64 chunk = lo; chunk < hi; ++chunk) {
      double* local = partials + chunk * stride;
      const I64 end = std::min(rows, (chunk + 1) * step);
      for (I64 r = chunk * step; r < end; ++r)
        for (I64 k = 0; k < c; ++k) local[k] += value(r, k);
    }
  });
  // Merge in input chunk order regardless of worker completion order.
  for (I64 chunk = 0; chunk < chunks; ++chunk)
    for (I64 k = 0; k < c; ++k)
      total[static_cast<std::size_t>(k)] += partials[static_cast<std::size_t>(chunk * stride + k)];
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
  if (detail::is_native(x))
    return detail::device_batch_norm_infer(x, gamma, beta, mean_in, variance, eps);
  const DType  dt    = is_floating(x.dtype()) ? x.dtype() : DType::Float32;
  const Tensor scale = as_dtype(gamma, dt) * rsqrt(as_dtype(variance, dt) + eps);
  return as_dtype(x, dt) * scale + (as_dtype(beta, dt) - as_dtype(mean_in, dt) * scale);
}

}  // namespace datamunge::nn
