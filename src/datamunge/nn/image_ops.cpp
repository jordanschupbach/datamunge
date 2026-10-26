// Image ops: resizes, crop_and_resize, space/depth rearrangements.
//
// Resizes and crop_and_resize share one separable sampling kernel driven by
// a precomputed plan (per output row and column: two source taps and a lerp
// weight). The forward map and its adjoint are recorded as each other's
// gradient, so every order of derivative is available.

#include "datamunge/nn/image_ops.hpp"

#include <memory>

#include "detail.hpp"

namespace datamunge::nn {

using detail::nograd;

namespace {

using Grads = std::vector<Tensor>;

/// One output coordinate along an axis: value = s[lo] + (s[hi] - s[lo]) * lerp.
struct Tap {
  std::int64_t lo{0};
  std::int64_t hi{0};
  float        lerp{0.0f};
  bool         valid{true};  // false: the sample lies outside the image
};

/// Where every output pixel of a [out_n, out_h, out_w, C] result samples a
/// [in_n, in_h, in_w, C] input.
struct SamplePlan {
  std::int64_t              in_n{0}, in_h{0}, in_w{0};
  std::int64_t              out_n{0}, out_h{0}, out_w{0};
  std::vector<std::int64_t> batch;  // source image of each output image
  std::vector<Tap>          ys;     // out_n * out_h
  std::vector<Tap>          xs;     // out_n * out_w
};
using PlanPtr = std::shared_ptr<const SamplePlan>;

Tensor sample_adjoint(const Tensor& grad, const PlanPtr& plan);

/// out[b, y, x, c] = bilinear sample of images[plan.batch[b]] (or `fill` outside).
Tensor sample_forward(const Tensor& images, const PlanPtr& plan, double fill, const char* name) {
  const SamplePlan& p = *plan;
  const std::int64_t C = images.shape().back();
  Tensor             out = empty({p.out_n, p.out_h, p.out_w, C}, images.dtype());
  {
    NoGradGuard  guard;
    const Tensor xc = nograd(images).contiguous();
    detail::dispatch_float(images.dtype(), name, [&](auto tag) {
      using T      = typename decltype(tag)::type;
      const T* src = xc.data_ptr<T>();
      T*       dst = out.data_ptr<T>();
      const T  fv  = static_cast<T>(fill);
      for (std::int64_t b = 0; b < p.out_n; ++b) {
        const T* img = src + p.batch[static_cast<std::size_t>(b)] * p.in_h * p.in_w * C;
        for (std::int64_t y = 0; y < p.out_h; ++y) {
          const Tap& ty  = p.ys[static_cast<std::size_t>(b * p.out_h + y)];
          T*         row = dst + ((b * p.out_h + y) * p.out_w) * C;
          for (std::int64_t x = 0; x < p.out_w; ++x) {
            const Tap& tx = p.xs[static_cast<std::size_t>(b * p.out_w + x)];
            T*         o  = row + x * C;
            if (!ty.valid || !tx.valid) {
              for (std::int64_t c = 0; c < C; ++c) o[c] = fv;
              continue;
            }
            const T* tl = img + (ty.lo * p.in_w + tx.lo) * C;
            const T* tr = img + (ty.lo * p.in_w + tx.hi) * C;
            const T* bl = img + (ty.hi * p.in_w + tx.lo) * C;
            const T* br = img + (ty.hi * p.in_w + tx.hi) * C;
            const T  xl = static_cast<T>(tx.lerp), yl = static_cast<T>(ty.lerp);
            for (std::int64_t c = 0; c < C; ++c) {
              const T top    = tl[c] + (tr[c] - tl[c]) * xl;
              const T bottom = bl[c] + (br[c] - bl[c]) * xl;
              o[c]           = top + (bottom - top) * yl;
            }
          }
        }
      }
    });
  }
  record(name, {images}, {out}, [plan](const Grads& g, const Grads&) { return Grads{sample_adjoint(g[0], plan)}; });
  return out;
}

/// The adjoint of sample_forward (with fill 0): scatters gradients back to the taps.
Tensor sample_adjoint(const Tensor& grad, const PlanPtr& plan) {
  const SamplePlan& p = *plan;
  const std::int64_t C = grad.shape().back();
  Tensor             out = zeros({p.in_n, p.in_h, p.in_w, C}, grad.dtype());
  {
    NoGradGuard  guard;
    const Tensor gc = nograd(grad).contiguous();
    detail::dispatch_float(grad.dtype(), "resize gradient", [&](auto tag) {
      using T      = typename decltype(tag)::type;
      const T* src = gc.data_ptr<T>();
      T*       dst = out.data_ptr<T>();
      for (std::int64_t b = 0; b < p.out_n; ++b) {
        T* img = dst + p.batch[static_cast<std::size_t>(b)] * p.in_h * p.in_w * C;
        for (std::int64_t y = 0; y < p.out_h; ++y) {
          const Tap& ty = p.ys[static_cast<std::size_t>(b * p.out_h + y)];
          if (!ty.valid) continue;
          const T* row = src + ((b * p.out_h + y) * p.out_w) * C;
          const T  yl  = static_cast<T>(ty.lerp);
          for (std::int64_t x = 0; x < p.out_w; ++x) {
            const Tap& tx = p.xs[static_cast<std::size_t>(b * p.out_w + x)];
            if (!tx.valid) continue;
            const T  xl = static_cast<T>(tx.lerp);
            const T  w_tl = (1 - yl) * (1 - xl), w_tr = (1 - yl) * xl, w_bl = yl * (1 - xl), w_br = yl * xl;
            const T* gv = row + x * C;
            T*       tl = img + (ty.lo * p.in_w + tx.lo) * C;
            T*       tr = img + (ty.lo * p.in_w + tx.hi) * C;
            T*       bl = img + (ty.hi * p.in_w + tx.lo) * C;
            T*       br = img + (ty.hi * p.in_w + tx.hi) * C;
            for (std::int64_t c = 0; c < C; ++c) {
              tl[c] += gv[c] * w_tl;
              tr[c] += gv[c] * w_tr;
              bl[c] += gv[c] * w_bl;
              br[c] += gv[c] * w_br;
            }
          }
        }
      }
    });
  }
  record("ResizeGrad", {grad}, {out}, [plan](const Grads& g, const Grads&) {
    return Grads{sample_forward(g[0], plan, 0.0, "ResizeGradGrad")};
  });
  return out;
}

/// Converts to a float dtype and adds a batch axis to a single [H, W, C] image.
Tensor as_float_batch(const Tensor& x, bool& unbatched, const char* what) {
  if (x.dim() != 3 && x.dim() != 4)
    throw std::invalid_argument(std::string(what) + ": expected [N, H, W, C] or [H, W, C], got " +
                                shape_str(x.shape()));
  Tensor f  = is_floating(x.dtype()) ? x : cast(x, DType::Float32);
  unbatched = x.dim() == 3;
  return unbatched ? expand_dims(f, 0) : f;
}

/// TF CalculateResizeScale.
float resize_scale(std::int64_t in, std::int64_t out, bool align_corners) {
  return (align_corners && out > 1) ? static_cast<float>(in - 1) / static_cast<float>(out - 1)
                                    : static_cast<float>(in) / static_cast<float>(out);
}

std::vector<Tap> bilinear_taps(std::int64_t in, std::int64_t out, bool align_corners, bool half_pixel) {
  const float      scale = resize_scale(in, out, align_corners);
  std::vector<Tap> taps(static_cast<std::size_t>(out));
  for (std::int64_t i = 0; i < out; ++i) {
    const float fi  = static_cast<float>(i);
    const float pos = half_pixel ? (fi + 0.5f) * scale - 0.5f : fi * scale;
    const float pf  = std::floor(pos);
    Tap&        t   = taps[static_cast<std::size_t>(i)];
    t.lo            = std::max<std::int64_t>(static_cast<std::int64_t>(pf), 0);
    t.hi            = std::min<std::int64_t>(static_cast<std::int64_t>(std::ceil(pos)), in - 1);
    t.lerp          = pos - pf;
  }
  return taps;
}

std::vector<Tap> nearest_taps(std::int64_t in, std::int64_t out, bool align_corners, bool half_pixel) {
  const float      scale = resize_scale(in, out, align_corners);
  std::vector<Tap> taps(static_cast<std::size_t>(out));
  for (std::int64_t i = 0; i < out; ++i) {
    const float  fi  = static_cast<float>(i);
    const float  pos = half_pixel ? (fi + 0.5f) * scale : fi * scale;  // TF HalfPixelScalerForNN has no -0.5
    std::int64_t j   = static_cast<std::int64_t>(align_corners ? std::round(pos) : std::floor(pos));
    j                = std::min(j, in - 1);
    if (half_pixel) j = std::max<std::int64_t>(j, 0);
    taps[static_cast<std::size_t>(i)] = Tap{j, j, 0.0f, true};
  }
  return taps;
}

Tensor resize_impl(const Tensor& images, ImageSize size, bool align_corners, bool half_pixel, bool nearest,
                   const char* name) {
  if (align_corners && half_pixel)
    throw std::invalid_argument(std::string(name) + ": align_corners and half_pixel_centers are mutually exclusive");
  if (size.height <= 0 || size.width <= 0)
    throw std::invalid_argument(std::string(name) + ": output size must be positive");
  bool         unbatched = false;
  const Tensor x         = as_float_batch(images, unbatched, name);
  auto         plan      = std::make_shared<SamplePlan>();
  plan->in_n = plan->out_n = x.shape()[0];
  plan->in_h               = x.shape()[1];
  plan->in_w               = x.shape()[2];
  if (plan->in_h <= 0 || plan->in_w <= 0) throw std::invalid_argument(std::string(name) + ": empty input image");
  plan->out_h = size.height;
  plan->out_w = size.width;
  const auto ty = nearest ? nearest_taps(plan->in_h, size.height, align_corners, half_pixel)
                          : bilinear_taps(plan->in_h, size.height, align_corners, half_pixel);
  const auto tx = nearest ? nearest_taps(plan->in_w, size.width, align_corners, half_pixel)
                          : bilinear_taps(plan->in_w, size.width, align_corners, half_pixel);
  for (std::int64_t b = 0; b < plan->out_n; ++b) {
    plan->batch.push_back(b);
    plan->ys.insert(plan->ys.end(), ty.begin(), ty.end());
    plan->xs.insert(plan->xs.end(), tx.begin(), tx.end());
  }
  const Tensor out = sample_forward(x, plan, 0.0, name);
  return unbatched ? squeeze(out, {0}) : out;
}

/// TF CropAndResize sample position along one axis; `valid` is false outside.
Tap crop_tap(float a1, float a2, std::int64_t in, std::int64_t out, std::int64_t i, bool nearest) {
  const float scale = out > 1 ? (a2 - a1) * static_cast<float>(in - 1) / static_cast<float>(out - 1) : 0.0f;
  const float pos   = out > 1 ? a1 * static_cast<float>(in - 1) + static_cast<float>(i) * scale
                              : static_cast<float>(0.5 * (a1 + a2) * static_cast<double>(in - 1));
  Tap t;
  if (pos < 0 || pos > static_cast<float>(in - 1)) {
    t.valid = false;
    return t;
  }
  if (nearest) {
    t.lo = t.hi = static_cast<std::int64_t>(std::round(pos));
    return t;
  }
  t.lo   = static_cast<std::int64_t>(std::floor(pos));
  t.hi   = static_cast<std::int64_t>(std::ceil(pos));
  t.lerp = pos - static_cast<float>(t.lo);
  return t;
}

}  // namespace

Tensor resize_bilinear(const Tensor& images, ImageSize size, bool align_corners, bool half_pixel_centers) {
  return resize_impl(images, size, align_corners, half_pixel_centers, false, "ResizeBilinear");
}

Tensor resize_nearest(const Tensor& images, ImageSize size, bool align_corners, bool half_pixel_centers) {
  return resize_impl(images, size, align_corners, half_pixel_centers, true, "ResizeNearestNeighbor");
}

Tensor crop_and_resize(const Tensor& images, const Tensor& boxes, const Tensor& box_indices, ImageSize crop_size,
                       ResizeMethod method, double extrapolation_value) {
  if (images.dim() != 4)
    throw std::invalid_argument("crop_and_resize: images must be [N, H, W, C], got " + shape_str(images.shape()));
  if (boxes.dim() != 2 || boxes.shape()[1] != 4)
    throw std::invalid_argument("crop_and_resize: boxes must be [B, 4], got " + shape_str(boxes.shape()));
  if (box_indices.dim() != 1 || box_indices.shape()[0] != boxes.shape()[0] || !is_integral(box_indices.dtype()))
    throw std::invalid_argument("crop_and_resize: box_indices must be an integer [B] tensor");
  if (crop_size.height <= 0 || crop_size.width <= 0)
    throw std::invalid_argument("crop_and_resize: crop size must be positive");
  bool         unbatched = false;
  const Tensor x         = as_float_batch(images, unbatched, "crop_and_resize");
  const auto   bx        = boxes.to_vector<float>();
  const auto   bi        = box_indices.to_vector<std::int64_t>();
  auto         plan      = std::make_shared<SamplePlan>();
  plan->in_n             = x.shape()[0];
  plan->in_h             = x.shape()[1];
  plan->in_w             = x.shape()[2];
  plan->out_n            = boxes.shape()[0];
  plan->out_h            = crop_size.height;
  plan->out_w            = crop_size.width;
  const bool nearest     = method == ResizeMethod::Nearest;
  for (std::int64_t b = 0; b < plan->out_n; ++b) {
    const std::int64_t img = bi[static_cast<std::size_t>(b)];
    if (img < 0 || img >= plan->in_n)
      throw std::out_of_range("crop_and_resize: box index " + std::to_string(img) + " out of range [0, " +
                              std::to_string(plan->in_n) + ")");
    plan->batch.push_back(img);
    const float* box = bx.data() + 4 * b;
    for (std::int64_t y = 0; y < plan->out_h; ++y)
      plan->ys.push_back(crop_tap(box[0], box[2], plan->in_h, plan->out_h, y, nearest));
    for (std::int64_t xx = 0; xx < plan->out_w; ++xx)
      plan->xs.push_back(crop_tap(box[1], box[3], plan->in_w, plan->out_w, xx, nearest));
  }
  return sample_forward(x, plan, extrapolation_value, "CropAndResize");
}

Tensor space_to_depth(const Tensor& x, std::int64_t b) { return space_to_depth(x, b, DepthToSpaceMode::DCR); }

Tensor depth_to_space(const Tensor& x, std::int64_t b) { return depth_to_space(x, b, DepthToSpaceMode::DCR); }

Tensor space_to_depth(const Tensor& x, std::int64_t b, DepthToSpaceMode mode) {
  if (x.dim() != 4) throw std::invalid_argument("space_to_depth: expected [N, H, W, C], got " + shape_str(x.shape()));
  const auto& s = x.shape();
  if (b < 2 || s[1] % b != 0 || s[2] % b != 0)
    throw std::invalid_argument("space_to_depth: block size " + std::to_string(b) + " must be >= 2 and divide H and W of " +
                                shape_str(s));
  const Tensor r    = reshape(x, {s[0], s[1] / b, b, s[2] / b, b, s[3]});
  const Axes   perm = mode == DepthToSpaceMode::CRD ? Axes{0, 1, 3, 5, 2, 4} : Axes{0, 1, 3, 2, 4, 5};
  return reshape(transpose(r, perm), {s[0], s[1] / b, s[2] / b, b * b * s[3]});
}

Tensor depth_to_space(const Tensor& x, std::int64_t b, DepthToSpaceMode mode) {
  if (x.dim() != 4) throw std::invalid_argument("depth_to_space: expected [N, H, W, C], got " + shape_str(x.shape()));
  const auto& s = x.shape();
  if (b < 2 || s[3] % (b * b) != 0)
    throw std::invalid_argument("depth_to_space: block size " + std::to_string(b) + " must be >= 2 and b*b must divide C of " +
                                shape_str(s));
  const std::int64_t c = s[3] / (b * b);
  if (mode == DepthToSpaceMode::CRD) {  // channel c * b * b + dy * b + dx
    const Tensor r = reshape(x, {s[0], s[1], s[2], c, b, b});
    return reshape(transpose(r, {0, 1, 4, 2, 5, 3}), {s[0], s[1] * b, s[2] * b, c});
  }
  const Tensor r = reshape(x, {s[0], s[1], s[2], b, b, c});  // channel (dy * b + dx) * C + c
  return reshape(transpose(r, {0, 1, 3, 2, 4, 5}), {s[0], s[1] * b, s[2] * b, c});
}

Tensor upsample_nearest(const Tensor& x, std::int64_t factor) {
  if (x.dim() != 3 && x.dim() != 4)
    throw std::invalid_argument("upsample_nearest: expected [N, H, W, C] or [H, W, C], got " + shape_str(x.shape()));
  if (factor < 1) throw std::invalid_argument("upsample_nearest: factor must be >= 1");
  if (factor == 1) return x;
  const std::int64_t h = x.dim() - 3;  // H axis
  return repeat(repeat(x, factor, h), factor, h + 1);
}

Tensor mirror_pad(const Tensor& x, std::int64_t pad_h, std::int64_t pad_w, PadMode mode) {
  if (x.dim() != 3 && x.dim() != 4)
    throw std::invalid_argument("mirror_pad: expected [N, H, W, C] or [H, W, C], got " + shape_str(x.shape()));
  std::vector<std::pair<std::int64_t, std::int64_t>> p(static_cast<std::size_t>(x.dim()), {0, 0});
  const auto h = static_cast<std::size_t>(x.dim() - 3);
  p[h]         = {pad_h, pad_h};
  p[h + 1]     = {pad_w, pad_w};
  return pad(x, p, mode);
}

// ---- arbitrary-position sampling ----------------------------------------------------
//
// One kernel samples images [N, H, W, C] at points grid [N, P, 2] (x, y in
// grid units, mapped affinely to pixels) with the corner weights
// differentiated `ox` times w.r.t. the point's x and `oy` times w.r.t. its y
// (bilinear weights are linear in each coordinate, so orders above one vanish).
// Its adjoint scatters into an image. Gradients:
//   d<g, S_ab(img, grid)>/dimg  = A_ab(g, grid)
//   d<g, S_ab(img, grid)>/dgrid = (sum_c g * S_(a+1)b(img), sum_c g * S_a(b+1)(img))
//   d<u, A_ab(g, grid)>/dg      = S_ab(u, grid)
//   d<u, A_ab(g, grid)>/dgrid   = d<g, S_ab(u, grid)>/dgrid
// all built from these two primitives, so every order is available.

namespace {

/// Pixel x = ax * gx + bx, y = ay * gy + by, plus the sampling rules.
struct PointSampler {
  GridSampleMode    mode{GridSampleMode::Bilinear};
  GridSamplePadding padding{GridSamplePadding::Zeros};
  double            ax{1}, bx{0}, ay{1}, by{0};
  std::int64_t      in_h{0}, in_w{0};
};
using SamplerPtr = std::shared_ptr<const PointSampler>;

/// Up to four taps (pixel index y * W + x and weight) of one point.
template <typename T>
struct PointTaps {
  std::int64_t pixel[4];
  T            weight[4];
  int          count{0};
};

/// Taps of the point (gx, gy) with weights differentiated (ox, oy) times
/// w.r.t. gx and gy. Taps outside the image are dropped (they read zero).
template <typename T>
void point_taps(const PointSampler& s, T gx, T gy, int ox, int oy, PointTaps<T>& t) {
  t.count     = 0;
  const T  W = static_cast<T>(s.in_w - 1), H = static_cast<T>(s.in_h - 1);
  T        x = static_cast<T>(s.ax) * gx + static_cast<T>(s.bx), y = static_cast<T>(s.ay) * gy + static_cast<T>(s.by);
  T        dx = static_cast<T>(s.ax), dy = static_cast<T>(s.ay);  // d pixel / d grid
  if (!std::isfinite(x) || !std::isfinite(y)) return;
  if (s.padding == GridSamplePadding::Border) {
    if (x <= 0) x = 0, dx = 0;
    else if (x >= W) x = W, dx = 0;
    if (y <= 0) y = 0, dy = 0;
    else if (y >= H) y = H, dy = 0;
  }
  if (s.mode == GridSampleMode::Nearest) {
    if (ox || oy) return;
    const T rx = std::nearbyint(x), ry = std::nearbyint(y);
    if (rx < 0 || rx > W || ry < 0 || ry > H) return;
    t.pixel[0]  = static_cast<std::int64_t>(ry) * s.in_w + static_cast<std::int64_t>(rx);
    t.weight[0] = 1;
    t.count     = 1;
    return;
  }
  if (ox > 1 || oy > 1) return;
  const T fx = std::floor(x), fy = std::floor(y);
  if (fx < -2 || fx > W + 1 || fy < -2 || fy > H + 1) return;  // no tap inside the image
  // Per axis: two taps (offset from the floor, weight). The derivative of the
  // piecewise-linear interpolant at an exact integer coordinate (a kink) is
  // the mean of the one-sided slopes, (v[x+1] - v[x-1]) / 2, which also equals
  // the central finite difference there.
  struct Axis {
    std::int64_t off[2];
    T            w[2];
  };
  const auto axis = [](T l, int order, T d) {
    if (!order) return Axis{{0, 1}, {1 - l, l}};
    if (l == 0) return Axis{{-1, 1}, {-d / 2, d / 2}};
    return Axis{{0, 1}, {-d, d}};
  };
  const Axis         ax = axis(x - fx, ox, dx), ay = axis(y - fy, oy, dy);
  const std::int64_t x0 = static_cast<std::int64_t>(fx), y0 = static_cast<std::int64_t>(fy);
  for (int a = 0; a < 2; ++a) {
    const std::int64_t yy = y0 + ay.off[a];
    if (yy < 0 || yy >= s.in_h) continue;
    for (int b = 0; b < 2; ++b) {
      const std::int64_t xx = x0 + ax.off[b];
      if (xx < 0 || xx >= s.in_w) continue;
      t.pixel[t.count]  = yy * s.in_w + xx;
      t.weight[t.count] = ay.w[a] * ax.w[b];
      ++t.count;
    }
  }
}

/// True when sampling with derivative orders (ox, oy) can be nonzero.
bool order_exists(const PointSampler& s, int ox, int oy) {
  return ox <= 1 && oy <= 1 && (s.mode == GridSampleMode::Bilinear || (ox == 0 && oy == 0));
}

Tensor sample_points(const Tensor& images, const Tensor& grid, const SamplerPtr& sampler, int ox, int oy);
Tensor sample_points_adjoint(const Tensor& grad, const Tensor& grid, const SamplerPtr& sampler, int ox, int oy);

/// d<g, S_(ox,oy)(images, grid)>/dgrid as [N, P, 2]. Fused when no history
/// is needed, else composed of recorded primitives.
Tensor grid_gradient(const Tensor& images, const Tensor& grid, const Tensor& g, const SamplerPtr& sampler, int ox,
                     int oy) {
  const PointSampler& s = *sampler;
  const std::int64_t  N = grid.shape()[0], P = grid.shape()[1], C = images.shape()[3];
  if (any_requires_grad({images, grid, g})) {
    std::vector<Tensor> parts;
    for (int axis = 0; axis < 2; ++axis) {
      const int ax = ox + (axis == 0), ay = oy + (axis == 1);
      parts.push_back(order_exists(s, ax, ay) ? sum(g * sample_points(images, grid, sampler, ax, ay), {-1})
                                              : zeros({N, P}, grid.dtype()));
    }
    return stack(parts, -1);
  }
  Tensor out = zeros({N, P, 2}, grid.dtype());
  NoGradGuard  guard;
  const Tensor ic = nograd(images).contiguous(), gc = nograd(grid).contiguous(), dc = nograd(g).contiguous();
  detail::dispatch_float(grid.dtype(), "grid_sample gradient", [&](auto tag) {
    using T        = typename decltype(tag)::type;
    const T* img   = ic.data_ptr<T>();
    const T* pts   = gc.data_ptr<T>();
    const T* gr    = dc.data_ptr<T>();
    T*       dst   = out.data_ptr<T>();
    const std::int64_t hw = s.in_h * s.in_w;
    detail::parallel_elements(N * P, [&](std::int64_t i) {
      const std::int64_t n = i / P;
      const T*           base = img + n * hw * C;
      const T*           gv   = gr + i * C;
      PointTaps<T>       taps;
      for (int axis = 0; axis < 2; ++axis) {
        const int ax = ox + (axis == 0), ay = oy + (axis == 1);
        if (!order_exists(s, ax, ay)) continue;
        point_taps(s, pts[2 * i], pts[2 * i + 1], ax, ay, taps);
        T acc = 0;
        for (int k = 0; k < taps.count; ++k) {
          const T* px  = base + taps.pixel[k] * C;
          T        dot = 0;
          for (std::int64_t c = 0; c < C; ++c) dot += gv[c] * px[c];
          acc += taps.weight[k] * dot;
        }
        dst[2 * i + axis] = acc;
      }
    });
  });
  return out;
}

/// S_(ox,oy): images [N, H, W, C] sampled at grid [N, P, 2] -> [N, P, C].
Tensor sample_points(const Tensor& images, const Tensor& grid, const SamplerPtr& sampler, int ox, int oy) {
  const PointSampler& s = *sampler;
  const std::int64_t  N = grid.shape()[0], P = grid.shape()[1], C = images.shape()[3];
  Tensor              out = empty({N, P, C}, images.dtype());
  {
    NoGradGuard  guard;
    const Tensor ic = nograd(images).contiguous(), gc = nograd(grid).contiguous();
    detail::dispatch_float(images.dtype(), "grid_sample", [&](auto tag) {
      using T      = typename decltype(tag)::type;
      const T* img = ic.data_ptr<T>();
      const T* pts = gc.data_ptr<T>();
      T*       dst = out.data_ptr<T>();
      const std::int64_t hw = s.in_h * s.in_w;
      detail::parallel_elements(N * P, [&](std::int64_t i) {
        const T*     base = img + (i / P) * hw * C;
        T*           o    = dst + i * C;
        PointTaps<T> taps;
        point_taps(s, pts[2 * i], pts[2 * i + 1], ox, oy, taps);
        for (std::int64_t c = 0; c < C; ++c) o[c] = 0;
        for (int k = 0; k < taps.count; ++k) {
          const T* px = base + taps.pixel[k] * C;
          const T  w  = taps.weight[k];
          for (std::int64_t c = 0; c < C; ++c) o[c] += w * px[c];
        }
      });
    });
  }
  const SavedTensor saved_images(images), saved_grid(grid);
  record("GridSample", {images, grid}, {out}, [saved_images, saved_grid, sampler, ox, oy](const Grads& g, const Grads&) {
    const Tensor& img = saved_images.get();
    const Tensor& pts = saved_grid.get();
    Grads         r(2);
    if (img.requires_grad()) r[0] = sample_points_adjoint(g[0], pts, sampler, ox, oy);
    if (pts.requires_grad()) r[1] = grid_gradient(img, pts, g[0], sampler, ox, oy);
    return r;
  });
  return out;
}

/// A_(ox,oy): the adjoint of S_(ox,oy), grad [N, P, C] -> [N, H, W, C].
Tensor sample_points_adjoint(const Tensor& grad, const Tensor& grid, const SamplerPtr& sampler, int ox, int oy) {
  const PointSampler& s = *sampler;
  const std::int64_t  N = grid.shape()[0], P = grid.shape()[1], C = grad.shape()[2];
  Tensor              out = zeros({N, s.in_h, s.in_w, C}, grad.dtype());
  {
    NoGradGuard  guard;
    const Tensor dc = nograd(grad).contiguous(), gc = nograd(grid).contiguous();
    detail::dispatch_float(grad.dtype(), "grid_sample gradient", [&](auto tag) {
      using T      = typename decltype(tag)::type;
      const T* gr  = dc.data_ptr<T>();
      const T* pts = gc.data_ptr<T>();
      T*       dst = out.data_ptr<T>();
      const std::int64_t hw = s.in_h * s.in_w;
      // One image per task: scatters never race and the order is fixed.
      parallel_for(0, N, 1, [&](std::int64_t lo, std::int64_t hi) {
        PointTaps<T> taps;
        for (std::int64_t n = lo; n < hi; ++n) {
          T* base = dst + n * hw * C;
          for (std::int64_t p = 0; p < P; ++p) {
            const std::int64_t i = n * P + p;
            point_taps(s, pts[2 * i], pts[2 * i + 1], ox, oy, taps);
            const T* gv = gr + i * C;
            for (int k = 0; k < taps.count; ++k) {
              T*      px = base + taps.pixel[k] * C;
              const T w  = taps.weight[k];
              for (std::int64_t c = 0; c < C; ++c) px[c] += w * gv[c];
            }
          }
        }
      });
    });
  }
  const SavedTensor saved_grad(grad), saved_grid(grid);
  record("GridSampleGrad", {grad, grid}, {out}, [saved_grad, saved_grid, sampler, ox, oy](const Grads& g, const Grads&) {
    const Tensor& gr  = saved_grad.get();
    const Tensor& pts = saved_grid.get();
    Grads         r(2);
    if (gr.requires_grad()) r[0] = sample_points(g[0], pts, sampler, ox, oy);
    if (pts.requires_grad()) r[1] = grid_gradient(g[0], pts, gr, sampler, ox, oy);
    return r;
  });
  return out;
}

/// Checks images [N, H, W, C] (converted to float) and returns them.
Tensor sampling_images(const Tensor& images, const char* what) {
  if (images.dim() != 4)
    throw std::invalid_argument(std::string(what) + ": images must be [N, H, W, C], got " + shape_str(images.shape()));
  if (images.shape()[1] <= 0 || images.shape()[2] <= 0)
    throw std::invalid_argument(std::string(what) + ": empty images");
  return is_floating(images.dtype()) ? images : cast(images, DType::Float32);
}

/// Samples images at points [N, ..., 2]; returns [N, ..., C].
Tensor sample_at(const Tensor& images, const Tensor& points, PointSampler sampler, const char* what) {
  const std::int64_t N = images.shape()[0], C = images.shape()[3];
  if (points.dim() < 2 || points.shape().back() != 2 || points.shape()[0] != N)
    throw std::invalid_argument(std::string(what) + ": positions must be [N, ..., 2] with N = " + std::to_string(N) +
                                ", got " + shape_str(points.shape()));
  sampler.in_h          = images.shape()[1];
  sampler.in_w          = images.shape()[2];
  const Tensor  pts     = detail::as_dtype(points, images.dtype());
  const std::int64_t P  = N ? points.numel() / (2 * N) : 0;
  Shape         out_shape(points.shape().begin(), points.shape().end() - 1);
  out_shape.push_back(C);
  const Tensor out = sample_points(images, reshape(pts, {N, P, 2}), std::make_shared<PointSampler>(sampler), 0, 0);
  return reshape(out, out_shape);
}

}  // namespace

Tensor grid_sample(const Tensor& images, const Tensor& grid, const GridSampleOptions& options) {
  const Tensor x = sampling_images(images, "grid_sample");
  if (grid.dim() != 4)
    throw std::invalid_argument("grid_sample: grid must be [N, Ho, Wo, 2], got " + shape_str(grid.shape()));
  PointSampler       s;
  s.mode             = options.mode;
  s.padding          = options.padding;
  const double H = static_cast<double>(x.shape()[1]), W = static_cast<double>(x.shape()[2]);
  s.ax = options.align_corners ? (W - 1) / 2 : W / 2;
  s.ay = options.align_corners ? (H - 1) / 2 : H / 2;
  s.bx = (W - 1) / 2;
  s.by = (H - 1) / 2;
  return sample_at(x, grid, s, "grid_sample");
}

Tensor sample_pixels(const Tensor& images, const Tensor& coords, GridSampleMode mode, GridSamplePadding padding) {
  PointSampler s;
  s.mode    = mode;
  s.padding = padding;
  return sample_at(sampling_images(images, "sample_pixels"), coords, s, "sample_pixels");
}

Tensor affine_grid(const Tensor& theta, ImageSize size, bool align_corners) {
  if (size.height <= 0 || size.width <= 0) throw std::invalid_argument("affine_grid: output size must be positive");
  if (!((theta.dim() == 3 && theta.shape()[1] == 2 && theta.shape()[2] == 3) || (theta.dim() == 2 && theta.shape()[1] == 6)))
    throw std::invalid_argument("affine_grid: theta must be [N, 2, 3] or [N, 6], got " + shape_str(theta.shape()));
  const Tensor t = reshape(is_floating(theta.dtype()) ? theta : cast(theta, DType::Float32), {theta.shape()[0], 2, 3});
  auto coordinate = [align_corners](std::int64_t i, std::int64_t n) {
    if (align_corners) return n > 1 ? -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(n - 1) : 0.0;
    return (2.0 * static_cast<double>(i) + 1.0) / static_cast<double>(n) - 1.0;
  };
  std::vector<double> base;
  base.reserve(static_cast<std::size_t>(size.height * size.width * 3));
  for (std::int64_t i = 0; i < size.height; ++i)
    for (std::int64_t j = 0; j < size.width; ++j) {
      base.push_back(coordinate(j, size.width));
      base.push_back(coordinate(i, size.height));
      base.push_back(1.0);
    }
  const Tensor target = from_vector(base, {1, size.height * size.width, 3}).to(t.dtype());
  return reshape(matmul(target, t, false, true), {theta.shape()[0], size.height, size.width, 2});
}

ImageSize deformable_output_size(std::int64_t height, std::int64_t width, const DeformableOptions& o) {
  if (o.kernel_h < 1 || o.kernel_w < 1 || o.stride < 1 || o.dilation < 1 || o.pad_h < 0 || o.pad_w < 0)
    throw std::invalid_argument("deformable convolution: kernel, stride and dilation must be positive, padding >= 0");
  const std::int64_t h = height + 2 * o.pad_h - o.dilation * (o.kernel_h - 1) - 1;
  const std::int64_t w = width + 2 * o.pad_w - o.dilation * (o.kernel_w - 1) - 1;
  if (h < 0 || w < 0) throw std::invalid_argument("deformable convolution: kernel larger than the padded input");
  return {h / o.stride + 1, w / o.stride + 1};
}

Tensor deformable_sample(const Tensor& images, const Tensor& offsets, const DeformableOptions& o) {
  const Tensor       x   = sampling_images(images, "deformable_sample");
  const std::int64_t N   = x.shape()[0];
  const ImageSize    out = deformable_output_size(x.shape()[1], x.shape()[2], o);
  const std::int64_t K   = o.kernel_h * o.kernel_w;
  if (offsets.shape() != Shape{N, out.height, out.width, 2 * K})
    throw std::invalid_argument("deformable_sample: offsets must be " + shape_str({N, out.height, out.width, 2 * K}) +
                                ", got " + shape_str(offsets.shape()));
  std::vector<double> by, bx;
  for (std::int64_t i = 0; i < out.height; ++i)
    for (std::int64_t j = 0; j < out.width; ++j)
      for (std::int64_t k = 0; k < K; ++k) {
        by.push_back(static_cast<double>(i * o.stride - o.pad_h + o.dilation * (k / o.kernel_w)));
        bx.push_back(static_cast<double>(j * o.stride - o.pad_w + o.dilation * (k % o.kernel_w)));
      }
  const Shape  base_shape{1, out.height, out.width, K};
  const Tensor pairs = reshape(detail::as_dtype(offsets, x.dtype()), {N, out.height, out.width, K, 2});
  const Tensor py    = from_vector(by, base_shape).to(x.dtype()) + select(pairs, 4, 0);
  const Tensor px    = from_vector(bx, base_shape).to(x.dtype()) + select(pairs, 4, 1);
  return sample_pixels(x, stack({px, py}, -1));
}

Tensor deformable_conv2d(const Tensor& images, const Tensor& offsets, const Tensor& filter, const DeformableOptions& o,
                         const Tensor& mask) {
  const std::int64_t C = images.dim() == 4 ? images.shape()[3] : -1;
  if (filter.dim() != 4 || filter.shape()[0] != o.kernel_h || filter.shape()[1] != o.kernel_w || filter.shape()[2] != C)
    throw std::invalid_argument("deformable_conv2d: filter must be [kernel_h, kernel_w, C, Cout], got " +
                                shape_str(filter.shape()));
  Tensor             samples = deformable_sample(images, offsets, o);  // [N, Ho, Wo, K, C]
  // Applying a mask replaces samples. In inference there is no autograd
  // node retaining the old tensor, so a reference to its shape would dangle.
  const Shape        s       = samples.shape();
  const std::int64_t K       = s[3];
  if (mask.defined()) {
    if (mask.shape() != Shape{s[0], s[1], s[2], K})
      throw std::invalid_argument("deformable_conv2d: mask must be " + shape_str({s[0], s[1], s[2], K}) + ", got " +
                                  shape_str(mask.shape()));
    samples = samples * expand_dims(detail::as_dtype(mask, samples.dtype()), 4);
  }
  const Tensor w = reshape(detail::as_dtype(filter, samples.dtype()), {K * C, filter.shape()[3]});
  return reshape(matmul(reshape(samples, {-1, K * C}), w), {s[0], s[1], s[2], filter.shape()[3]});
}

}  // namespace datamunge::nn
