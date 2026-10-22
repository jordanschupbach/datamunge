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

Tensor space_to_depth(const Tensor& x, std::int64_t b) {
  if (x.dim() != 4) throw std::invalid_argument("space_to_depth: expected [N, H, W, C], got " + shape_str(x.shape()));
  const auto& s = x.shape();
  if (b < 2 || s[1] % b != 0 || s[2] % b != 0)
    throw std::invalid_argument("space_to_depth: block size " + std::to_string(b) + " must be >= 2 and divide H and W of " +
                                shape_str(s));
  const Tensor r = reshape(x, {s[0], s[1] / b, b, s[2] / b, b, s[3]});
  return reshape(transpose(r, {0, 1, 3, 2, 4, 5}), {s[0], s[1] / b, s[2] / b, b * b * s[3]});
}

Tensor depth_to_space(const Tensor& x, std::int64_t b) {
  if (x.dim() != 4) throw std::invalid_argument("depth_to_space: expected [N, H, W, C], got " + shape_str(x.shape()));
  const auto& s = x.shape();
  if (b < 2 || s[3] % (b * b) != 0)
    throw std::invalid_argument("depth_to_space: block size " + std::to_string(b) + " must be >= 2 and b*b must divide C of " +
                                shape_str(s));
  const Tensor r = reshape(x, {s[0], s[1], s[2], b, b, s[3] / (b * b)});
  return reshape(transpose(r, {0, 1, 3, 2, 4, 5}), {s[0], s[1] * b, s[2] * b, s[3] / (b * b)});
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

}  // namespace datamunge::nn
