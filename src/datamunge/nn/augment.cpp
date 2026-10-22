// Data augmentation on float NHWC batches (no autograd).

#include "datamunge/nn/augment.hpp"

#include <cmath>

#include "detail.hpp"

namespace datamunge::nn::augment {

namespace {

/// A packed, private float copy of the images as [N, H, W, C].
struct Batch {
  Tensor       t;
  bool         unbatched{false};
  std::int64_t n{0}, h{0}, w{0}, c{0};

  [[nodiscard]] std::int64_t image_size() const { return h * w * c; }
  /// Back to the caller's rank.
  [[nodiscard]] Tensor result() const { return unbatched ? t.reshape({h, w, c}) : t; }
};

Batch load(const Tensor& images, const char* what) {
  if (images.dim() != 3 && images.dim() != 4)
    throw std::invalid_argument(std::string(what) + ": expected [N, H, W, C] or [H, W, C], got " +
                                shape_str(images.shape()));
  NoGradGuard guard;
  Batch       b;
  Tensor      x = detail::nograd(images);
  if (!is_floating(x.dtype())) x = cast(x, DType::Float32);
  b.unbatched = x.dim() == 3;
  if (b.unbatched) x = x.reshape({1, x.shape()[0], x.shape()[1], x.shape()[2]});
  b.t = x.clone();
  b.n = x.shape()[0];
  b.h = x.shape()[1];
  b.w = x.shape()[2];
  b.c = x.shape()[3];
  return b;
}

/// Calls f(T* image, index) for each image of the batch.
template <typename F>
void each_image(Batch& b, const char* what, F&& f) {
  detail::dispatch_float(b.t.dtype(), what, [&](auto tag) {
    using T = typename decltype(tag)::type;
    T* p    = b.t.template data_ptr<T>();
    for (std::int64_t i = 0; i < b.n; ++i) f(p + i * b.image_size(), i);
  });
}

void require_rgb(const Batch& b, const char* what) {
  if (b.c != 3) throw std::invalid_argument(std::string(what) + ": expected 3 (RGB) channels, got " + std::to_string(b.c));
}

// ---- per-image kernels ---------------------------------------------------------------

/// out[r][c] = src[r + top][(flip ? w-1-c : c) + left], `fill` outside (tflow ImageBatch).
template <typename T>
void shift_flip(T* img, const Batch& b, std::int64_t top, std::int64_t left, bool flip, double fill) {
  const std::vector<T> src(img, img + b.image_size());
  for (std::int64_t r = 0; r < b.h; ++r)
    for (std::int64_t c = 0; c < b.w; ++c) {
      const std::int64_t iy = r + top, ix = (flip ? b.w - 1 - c : c) + left;
      const bool         in = iy >= 0 && iy < b.h && ix >= 0 && ix < b.w;
      for (std::int64_t k = 0; k < b.c; ++k)
        img[(r * b.w + c) * b.c + k] = in ? src[static_cast<std::size_t>((iy * b.w + ix) * b.c + k)] : static_cast<T>(fill);
    }
}

template <typename T>
void flip_rows(T* img, const Batch& b) {
  for (std::int64_t r = 0; r < b.h / 2; ++r)
    std::swap_ranges(img + r * b.w * b.c, img + (r + 1) * b.w * b.c, img + (b.h - 1 - r) * b.w * b.c);
}

template <typename T>
void fill_rect(T* img, const Batch& b, const Rect& rect, double fill) {
  const std::int64_t y0 = std::max<std::int64_t>(rect.top, 0), y1 = std::min(rect.top + rect.height, b.h);
  const std::int64_t x0 = std::max<std::int64_t>(rect.left, 0), x1 = std::min(rect.left + rect.width, b.w);
  for (std::int64_t y = y0; y < y1; ++y)
    for (std::int64_t x = x0; x < x1; ++x)
      for (std::int64_t k = 0; k < b.c; ++k) img[(y * b.w + x) * b.c + k] = static_cast<T>(fill);
}

template <typename T>
void contrast(T* img, const Batch& b, double factor) {
  const std::int64_t pixels = b.h * b.w;
  for (std::int64_t k = 0; k < b.c; ++k) {
    double sum = 0;
    for (std::int64_t i = 0; i < pixels; ++i) sum += static_cast<double>(img[i * b.c + k]);
    const T mean = static_cast<T>(sum / static_cast<double>(pixels));
    const T f    = static_cast<T>(factor);
    for (std::int64_t i = 0; i < pixels; ++i) img[i * b.c + k] = (img[i * b.c + k] - mean) * f + mean;
  }
}

// TF's rgb_to_hsv / hsv_to_rgb (adjust_saturation_op.cc), h in [0, 1).
template <typename T>
void rgb_to_hsv(T r, T g, T bl, T& h, T& s, T& v) {
  const T vv    = std::max(r, std::max(g, bl));
  const T range = vv - std::min(r, std::min(g, bl));
  s             = vv > 0 ? range / vv : T(0);
  const T norm  = T(1) / (T(6) * range);
  T       hh;
  if (r == vv)
    hh = norm * (g - bl);
  else if (g == vv)
    hh = norm * (bl - r) + T(2.0 / 6.0);
  else
    hh = norm * (r - g) + T(4.0 / 6.0);
  if (range <= 0) hh = 0;
  if (hh < 0) hh += 1;
  v = vv;
  h = hh;
}

template <typename T>
void hsv_to_rgb(T h, T s, T v, T& r, T& g, T& bl) {
  const T   c   = s * v;
  const T   m   = v - c;
  const T   dh  = h * 6;
  const int cat = static_cast<int>(dh);
  T         fm  = dh;
  while (fm <= 0) fm += 2;
  while (fm >= 2) fm -= 2;
  const T x = c * (1 - std::abs(fm - 1));
  T       rr = 0, gg = 0, bb = 0;
  switch (cat) {
    case 0: rr = c, gg = x, bb = 0; break;
    case 1: rr = x, gg = c, bb = 0; break;
    case 2: rr = 0, gg = c, bb = x; break;
    case 3: rr = 0, gg = x, bb = c; break;
    case 4: rr = x, gg = 0, bb = c; break;
    case 5: rr = c, gg = 0, bb = x; break;
    default: break;
  }
  r  = rr + m;
  g  = gg + m;
  bl = bb + m;
}

/// Applies f(h, s, v) in place to every RGB pixel.
template <typename T, typename F>
void hsv_map(T* img, const Batch& b, F&& f) {
  for (std::int64_t i = 0; i < b.h * b.w; ++i) {
    T* p = img + i * 3;
    T  h, s, v;
    rgb_to_hsv(p[0], p[1], p[2], h, s, v);
    f(h, s, v);
    hsv_to_rgb(h, s, v, p[0], p[1], p[2]);
  }
}

template <typename T>
void saturation(T* img, const Batch& b, double factor) {
  const T f = static_cast<T>(factor);
  hsv_map(img, b, [f](T&, T& s, T&) { s = std::min(T(1), std::max(T(0), s * f)); });
}

template <typename T>
void hue(T* img, const Batch& b, double delta) {
  const T d = static_cast<T>(delta);
  hsv_map(img, b, [d](T& h, T&, T&) {
    h += d;
    h -= std::floor(h);
  });
}

template <typename T>
void add_scalar(T* img, const Batch& b, double delta) {
  const T d = static_cast<T>(delta);
  for (std::int64_t i = 0; i < b.image_size(); ++i) img[i] += d;
}

/// Gamma(alpha, 1) by Marsaglia and Tsang, boosted for alpha < 1.
double gamma_draw(Generator& gen, double alpha) {
  if (alpha < 1) return gamma_draw(gen, alpha + 1) * std::pow(gen.uniform01(), 1.0 / alpha);
  const double d = alpha - 1.0 / 3.0, c = 1.0 / std::sqrt(9.0 * d);
  while (true) {
    const double x = gen.normal();
    double       v = 1 + c * x;
    if (v <= 0) continue;
    v              = v * v * v;
    const double u = gen.uniform01();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (u > 0 && std::log(u) < 0.5 * x * x + d * (1 - v + std::log(v))) return d * v;
  }
}

// ---- mixing ------------------------------------------------------------------------

void check_mixing(const Batch& b, const Tensor& labels, const std::vector<std::int64_t>& perm, const char* what) {
  if (labels.dim() != 2 || !is_floating(labels.dtype()))
    throw std::invalid_argument(std::string(what) + ": labels must be float [N, K] (one-hot or soft), got " +
                                std::string(dtype_name(labels.dtype())) + shape_str(labels.shape()));
  if (labels.shape()[0] != b.n || static_cast<std::int64_t>(perm.size()) != b.n)
    throw std::invalid_argument(std::string(what) + ": images, labels and permutation need the same batch size");
  std::vector<bool> seen(perm.size(), false);
  for (auto i : perm) {
    if (i < 0 || i >= b.n || seen[static_cast<std::size_t>(i)])
      throw std::invalid_argument(std::string(what) + ": invalid batch permutation");
    seen[static_cast<std::size_t>(i)] = true;
  }
}

/// x * w + x[perm] * (1 - w), recorded-free.
Tensor blend(const Tensor& x, const std::vector<std::int64_t>& perm, const Tensor& w) {
  NoGradGuard  guard;
  const Tensor xd      = detail::nograd(x);
  const Tensor partner = gather(xd, from_vector(perm), 0);
  return add(mul(xd, w), mul(partner, sub(1.0, w)));
}

std::vector<std::int64_t> random_permutation(std::int64_t n, Generator& gen) {
  return permutation(n, gen).to_vector<std::int64_t>();
}

}  // namespace

Tensor flip_left_right(const Tensor& images) {
  Batch b = load(images, "flip_left_right");
  each_image(b, "flip_left_right", [&](auto* img, std::int64_t) { shift_flip(img, b, 0, 0, true, 0.0); });
  return b.result();
}

Tensor flip_up_down(const Tensor& images) {
  Batch b = load(images, "flip_up_down");
  each_image(b, "flip_up_down", [&](auto* img, std::int64_t) { flip_rows(img, b); });
  return b.result();
}

Tensor crop(const Tensor& images, std::int64_t top, std::int64_t left, std::int64_t height, std::int64_t width) {
  Batch b = load(images, "crop");
  if (height <= 0 || width <= 0 || top < 0 || left < 0 || top + height > b.h || left + width > b.w)
    throw std::invalid_argument("crop: window (" + std::to_string(top) + ", " + std::to_string(left) + ", " +
                                std::to_string(height) + ", " + std::to_string(width) + ") does not fit in " +
                                std::to_string(b.h) + " x " + std::to_string(b.w));
  NoGradGuard guard;
  Tensor      out = slice(b.t, {0, top, left, 0}, {b.n, height, width, b.c}).contiguous();
  return b.unbatched ? out.reshape({height, width, b.c}) : out;
}

Tensor cutout(const Tensor& images, const Rect& rect, double fill) {
  if (rect.height < 0 || rect.width < 0) throw std::invalid_argument("cutout: negative rectangle size");
  Batch b = load(images, "cutout");
  each_image(b, "cutout", [&](auto* img, std::int64_t) { fill_rect(img, b, rect, fill); });
  return b.result();
}

MixedBatch mixup(const Tensor& images, const Tensor& labels, const std::vector<std::int64_t>& permutation,
                 double weight) {
  if (!(weight >= 0 && weight <= 1)) throw std::invalid_argument("mixup: weight must be in [0, 1]");
  Batch b = load(images, "mixup");
  check_mixing(b, labels, permutation, "mixup");
  MixedBatch out;
  out.images = blend(b.t, permutation, scalar(weight, b.t.dtype()));
  out.labels = blend(labels, permutation, scalar(weight, labels.dtype()));
  out.weight = weight;
  if (b.unbatched) out.images = out.images.reshape({b.h, b.w, b.c});
  return out;
}

MixedBatch cutmix(const Tensor& images, const Tensor& labels, const std::vector<std::int64_t>& permutation,
                  const Rect& rect) {
  if (rect.height < 0 || rect.width < 0) throw std::invalid_argument("cutmix: negative rectangle size");
  Batch b = load(images, "cutmix");
  check_mixing(b, labels, permutation, "cutmix");
  // Keep mask: 0 inside the clipped rectangle.
  Batch mask = load(ones({1, b.h, b.w, 1}, b.t.dtype()), "cutmix");
  each_image(mask, "cutmix", [&](auto* img, std::int64_t) { fill_rect(img, mask, rect, 0.0); });
  const double kept = mean(mask.t).item();
  MixedBatch   out;
  out.images = blend(b.t, permutation, mask.t);
  out.labels = blend(labels, permutation, scalar(kept, labels.dtype()));
  out.weight = kept;
  if (b.unbatched) out.images = out.images.reshape({b.h, b.w, b.c});
  return out;
}

Tensor adjust_brightness(const Tensor& images, double delta) {
  Batch b = load(images, "adjust_brightness");
  each_image(b, "adjust_brightness", [&](auto* img, std::int64_t) { add_scalar(img, b, delta); });
  return b.result();
}

Tensor adjust_contrast(const Tensor& images, double factor) {
  Batch b = load(images, "adjust_contrast");
  each_image(b, "adjust_contrast", [&](auto* img, std::int64_t) { contrast(img, b, factor); });
  return b.result();
}

Tensor adjust_saturation(const Tensor& images, double factor) {
  Batch b = load(images, "adjust_saturation");
  require_rgb(b, "adjust_saturation");
  each_image(b, "adjust_saturation", [&](auto* img, std::int64_t) { saturation(img, b, factor); });
  return b.result();
}

Tensor adjust_hue(const Tensor& images, double delta) {
  if (delta < -1 || delta > 1) throw std::invalid_argument("adjust_hue: delta must be in [-1, 1]");
  Batch b = load(images, "adjust_hue");
  require_rgb(b, "adjust_hue");
  each_image(b, "adjust_hue", [&](auto* img, std::int64_t) { hue(img, b, delta); });
  return b.result();
}

Tensor color_jitter(const Tensor& images, double brightness, double contrast_factor, double saturation_factor,
                    double hue_delta) {
  Batch b = load(images, "color_jitter");
  require_rgb(b, "color_jitter");
  each_image(b, "color_jitter", [&](auto* img, std::int64_t) {
    using T = std::remove_pointer_t<decltype(img)>;
    add_scalar(img, b, brightness);
    contrast(img, b, contrast_factor);
    saturation(img, b, saturation_factor);
    hue(img, b, hue_delta);
    for (std::int64_t i = 0; i < b.image_size(); ++i) img[i] = std::min(T(1), std::max(T(0), img[i]));
  });
  return b.result();
}

Tensor per_image_standardization(const Tensor& images) {
  Batch b = load(images, "per_image_standardization");
  each_image(b, "per_image_standardization", [&](auto* img, std::int64_t) {
    using T              = std::remove_pointer_t<decltype(img)>;
    const std::int64_t n = b.image_size();
    double             s = 0, s2 = 0;
    for (std::int64_t i = 0; i < n; ++i) s += static_cast<double>(img[i]);
    const double mu = s / static_cast<double>(n);
    for (std::int64_t i = 0; i < n; ++i) s2 += (static_cast<double>(img[i]) - mu) * (static_cast<double>(img[i]) - mu);
    const double sd = std::max(std::sqrt(s2 / static_cast<double>(n)), 1.0 / std::sqrt(static_cast<double>(n)));
    for (std::int64_t i = 0; i < n; ++i) img[i] = static_cast<T>((static_cast<double>(img[i]) - mu) / sd);
  });
  return b.result();
}

// ---- random ---------------------------------------------------------------------------

Tensor random_flip_left_right(const Tensor& images, Generator& gen, double p) {
  Batch b = load(images, "random_flip_left_right");
  each_image(b, "random_flip_left_right", [&](auto* img, std::int64_t) {
    if (gen.uniform01() < p) shift_flip(img, b, 0, 0, true, 0.0);
  });
  return b.result();
}

Tensor random_flip_up_down(const Tensor& images, Generator& gen, double p) {
  Batch b = load(images, "random_flip_up_down");
  each_image(b, "random_flip_up_down", [&](auto* img, std::int64_t) {
    if (gen.uniform01() < p) flip_rows(img, b);
  });
  return b.result();
}

Tensor random_pad_crop(const Tensor& images, std::int64_t pad, Generator& gen, bool flip, double fill) {
  if (pad < 0) throw std::invalid_argument("random_pad_crop: pad must be >= 0");
  Batch b = load(images, "random_pad_crop");
  const auto span = static_cast<std::uint64_t>(2 * pad + 1);
  each_image(b, "random_pad_crop", [&](auto* img, std::int64_t) {
    const std::int64_t top  = static_cast<std::int64_t>(gen.below(span)) - pad;
    const std::int64_t left = static_cast<std::int64_t>(gen.below(span)) - pad;
    const bool         f    = flip && gen.below(2) == 1;
    shift_flip(img, b, top, left, f, fill);
  });
  return b.result();
}

Tensor random_crop(const Tensor& images, std::int64_t height, std::int64_t width, Generator& gen) {
  Batch b = load(images, "random_crop");
  if (height <= 0 || width <= 0 || height > b.h || width > b.w)
    throw std::invalid_argument("random_crop: " + std::to_string(height) + " x " + std::to_string(width) +
                                " window does not fit in " + std::to_string(b.h) + " x " + std::to_string(b.w));
  Tensor out = empty({b.n, height, width, b.c}, b.t.dtype());
  detail::dispatch_float(b.t.dtype(), "random_crop", [&](auto tag) {
    using T      = typename decltype(tag)::type;
    const T* src = b.t.template data_ptr<T>();
    T*       dst = out.data_ptr<T>();
    for (std::int64_t i = 0; i < b.n; ++i) {
      const auto top  = static_cast<std::int64_t>(gen.below(static_cast<std::uint64_t>(b.h - height + 1)));
      const auto left = static_cast<std::int64_t>(gen.below(static_cast<std::uint64_t>(b.w - width + 1)));
      for (std::int64_t r = 0; r < height; ++r) {
        const T* from = src + ((i * b.h + top + r) * b.w + left) * b.c;
        std::copy(from, from + width * b.c, dst + ((i * height + r) * width) * b.c);
      }
    }
  });
  return b.unbatched ? out.reshape({height, width, b.c}) : out;
}

Tensor random_cutout(const Tensor& images, std::int64_t size, Generator& gen, double fill) {
  if (size < 0) throw std::invalid_argument("random_cutout: size must be >= 0");
  Batch b = load(images, "random_cutout");
  each_image(b, "random_cutout", [&](auto* img, std::int64_t) {
    const auto cy = static_cast<std::int64_t>(gen.below(static_cast<std::uint64_t>(b.h)));
    const auto cx = static_cast<std::int64_t>(gen.below(static_cast<std::uint64_t>(b.w)));
    fill_rect(img, b, Rect{cy - size / 2, cx - size / 2, size, size}, fill);
  });
  return b.result();
}

Tensor random_brightness(const Tensor& images, double max_delta, Generator& gen) {
  Batch b = load(images, "random_brightness");
  each_image(b, "random_brightness",
             [&](auto* img, std::int64_t) { add_scalar(img, b, (2 * gen.uniform01() - 1) * max_delta); });
  return b.result();
}

Tensor random_contrast(const Tensor& images, double lower, double upper, Generator& gen) {
  if (lower > upper) throw std::invalid_argument("random_contrast: lower > upper");
  Batch b = load(images, "random_contrast");
  each_image(b, "random_contrast",
             [&](auto* img, std::int64_t) { contrast(img, b, lower + (upper - lower) * gen.uniform01()); });
  return b.result();
}

Tensor random_saturation(const Tensor& images, double lower, double upper, Generator& gen) {
  if (lower > upper || lower < 0) throw std::invalid_argument("random_saturation: need 0 <= lower <= upper");
  Batch b = load(images, "random_saturation");
  require_rgb(b, "random_saturation");
  each_image(b, "random_saturation",
             [&](auto* img, std::int64_t) { saturation(img, b, lower + (upper - lower) * gen.uniform01()); });
  return b.result();
}

Tensor random_hue(const Tensor& images, double max_delta, Generator& gen) {
  if (max_delta < 0 || max_delta > 0.5) throw std::invalid_argument("random_hue: max_delta must be in [0, 0.5]");
  Batch b = load(images, "random_hue");
  require_rgb(b, "random_hue");
  each_image(b, "random_hue", [&](auto* img, std::int64_t) { hue(img, b, (2 * gen.uniform01() - 1) * max_delta); });
  return b.result();
}

double beta_mix_weight(Generator& gen, double alpha) {
  if (!std::isfinite(alpha) || alpha <= 0) throw std::invalid_argument("beta_mix_weight: alpha must be positive");
  for (int attempt = 0; attempt < 100; ++attempt) {
    const double a = gamma_draw(gen, alpha), b = gamma_draw(gen, alpha);
    if (std::isfinite(a + b) && a + b > 0) return a / (a + b);
  }
  throw std::runtime_error("beta_mix_weight: sampling underflow; use a larger alpha");
}

MixedBatch random_mixup(const Tensor& images, const Tensor& labels, double alpha, Generator& gen) {
  const double weight = beta_mix_weight(gen, alpha);
  const auto   n      = images.dim() == 4 ? images.shape()[0] : 1;
  return mixup(images, labels, random_permutation(n, gen), weight);
}

MixedBatch random_cutmix(const Tensor& images, const Tensor& labels, double alpha, Generator& gen) {
  if (images.dim() != 4) throw std::invalid_argument("random_cutmix: expected [N, H, W, C]");
  const std::int64_t h = images.shape()[1], w = images.shape()[2];
  const double       lambda = beta_mix_weight(gen, alpha);
  const double       ratio  = std::sqrt(1 - lambda);
  const auto         ch     = static_cast<std::int64_t>(static_cast<double>(h) * ratio);
  const auto         cw     = static_cast<std::int64_t>(static_cast<double>(w) * ratio);
  const auto         cy     = static_cast<std::int64_t>(gen.below(static_cast<std::uint64_t>(h)));
  const auto         cx     = static_cast<std::int64_t>(gen.below(static_cast<std::uint64_t>(w)));
  const auto         perm   = random_permutation(images.shape()[0], gen);
  return cutmix(images, labels, perm, Rect{cy - ch / 2, cx - cw / 2, ch, cw});
}

}  // namespace datamunge::nn::augment
