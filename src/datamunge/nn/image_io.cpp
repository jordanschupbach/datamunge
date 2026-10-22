// Tensor <-> image::Image conversion, sample sheets and PNG output.

#include "datamunge/nn/image_io.hpp"

#include <cmath>

#include "datamunge/image/png.hpp"
#include "detail.hpp"

namespace datamunge::nn {

namespace {

image::ImageMode mode_for(std::int64_t channels) {
  switch (channels) {
    case 1: return image::ImageMode::Grayscale;
    case 2: return image::ImageMode::GrayscaleAlpha;
    case 3: return image::ImageMode::RGB;
    case 4: return image::ImageMode::RGBA;
    default: throw std::invalid_argument("to_image: expected 1, 2, 3 or 4 channels, got " + std::to_string(channels));
  }
}

}  // namespace

image::Image to_image(const Tensor& x, double low, double high) {
  if (!(high > low)) throw std::invalid_argument("to_image: need high > low");
  Tensor t = x.detach();
  if (t.dim() == 4 && t.shape()[0] == 1) t = t.reshape({t.shape()[1], t.shape()[2], t.shape()[3]});
  if (t.dim() == 2) t = t.reshape({t.shape()[0], t.shape()[1], 1});
  if (t.dim() != 3)
    throw std::invalid_argument("to_image: expected [H, W], [H, W, C] or [1, H, W, C], got " + shape_str(x.shape()));
  const std::int64_t h = t.shape()[0], w = t.shape()[1], c = t.shape()[2];
  image::Image       img(static_cast<int>(w), static_cast<int>(h), mode_for(c));
  const auto         values = t.to_vector<double>();
  auto&              bytes  = img.data();
  const double       k      = 255.0 / (high - low);
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double v = (values[i] - low) * k;
    bytes[i]       = std::isnan(v) ? 0 : static_cast<std::uint8_t>(std::lround(std::min(255.0, std::max(0.0, v))));
  }
  return img;
}

Tensor from_image(const image::Image& img, DType dtype) {
  const auto&         bytes = img.data();
  std::vector<double> values(bytes.begin(), bytes.end());
  return cast(from_vector(values, {img.height(), img.width(), img.channels()}), dtype);
}

Tensor image_grid(const Tensor& batch, std::int64_t cols, std::int64_t padding, double pad_value, std::int64_t scale) {
  NoGradGuard guard;
  Tensor      b = detail::nograd(batch);
  if (b.dim() == 3) b = b.reshape({b.shape()[0], b.shape()[1], b.shape()[2], 1});
  if (b.dim() != 4 || b.shape()[0] == 0)
    throw std::invalid_argument("image_grid: expected a nonempty [N, H, W, C] or [N, H, W] batch, got " +
                                shape_str(batch.shape()));
  if (padding < 0 || scale < 1 || cols < 0) throw std::invalid_argument("image_grid: invalid cols, padding or scale");
  if (scale > 1) b = repeat(repeat(b, scale, 1), scale, 2);
  const std::int64_t n = b.shape()[0], h = b.shape()[1], w = b.shape()[2], c = b.shape()[3];
  if (cols == 0) cols = static_cast<std::int64_t>(std::ceil(std::sqrt(static_cast<double>(n))));
  cols                  = std::min(cols, n);
  const std::int64_t rows = (n + cols - 1) / cols;
  const std::int64_t H = rows * h + (rows - 1) * padding, W = cols * w + (cols - 1) * padding;
  Tensor             out = full({H, W, c}, pad_value, b.dtype());
  const Tensor       src = b.contiguous();
  detail::dispatch(b.dtype(), [&](auto tag) {
    using T     = typename decltype(tag)::type;
    const T* s  = src.data_ptr<T>();
    T*       d  = out.data_ptr<T>();
    for (std::int64_t i = 0; i < n; ++i) {
      const std::int64_t y0 = (i / cols) * (h + padding), x0 = (i % cols) * (w + padding);
      for (std::int64_t y = 0; y < h; ++y) {
        const T* from = s + ((i * h + y) * w) * c;
        std::copy(from, from + w * c, d + ((y0 + y) * W + x0) * c);
      }
    }
  });
  return out;
}

void save_png(const Tensor& x, const std::string& path, double low, double high) {
  image::write_png(to_image(x, low, high), path);
}

void save_image_grid(const Tensor& batch, const std::string& path, std::int64_t cols, double low, double high,
                     std::int64_t padding, double pad_value, std::int64_t scale) {
  save_png(image_grid(batch, cols, padding, pad_value, scale), path, low, high);
}

}  // namespace datamunge::nn
