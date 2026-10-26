// D65/2-degree CIELAB, sRGB transfer functions and genuine grayscale/chroma
// pairs: a port of tflow's common/ml/colorization_data.h. Tensors are
// float32 NHWC nn tensors; CIFAR pixels come from data::CifarDataset (int32
// holding 0..255).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <numeric>
#include <vector>

#include "common/checks.hpp"
#include "common/cifar.hpp"

namespace color {

namespace nn = datamunge::nn;
using ColorTriple = std::array<double, 3>;

inline double ColorLinear(double value) { return value <= .04045 ? value / 12.92 : std::pow((value + .055) / 1.055, 2.4); }
inline double ColorGamma(double value) { return value <= .0031308 ? 12.92 * value : 1.055 * std::pow(value, 1 / 2.4) - .055; }
inline double ColorLabF(double value) {
  constexpr double delta = 6. / 29;
  return value > delta * delta * delta ? std::cbrt(value) : value / (3 * delta * delta) + 4. / 29;
}
inline double ColorLabInverse(double value) {
  constexpr double delta = 6. / 29;
  return value > delta ? value * value * value : 3 * delta * delta * (value - 4. / 29);
}
inline ColorTriple ColorRGBToLab(ColorTriple rgb) {
  for (auto& value : rgb) {
    check::Require(std::isfinite(value) && value >= 0 && value <= 1, "sRGB range");
    value = ColorLinear(value);
  }
  const double x = ColorLabF((.412453 * rgb[0] + .357580 * rgb[1] + .180423 * rgb[2]) / .95047);
  const double y = ColorLabF(.212671 * rgb[0] + .715160 * rgb[1] + .072169 * rgb[2]);
  const double z = ColorLabF((.019334 * rgb[0] + .119193 * rgb[1] + .950227 * rgb[2]) / 1.08883);
  return {116 * y - 16, 500 * (x - y), 200 * (y - z)};
}
/// Unclipped sRGB. Out-of-gamut fractions are reported before display clipping.
inline ColorTriple ColorLabToRGB(const ColorTriple& lab) {
  for (double value : lab) check::Require(std::isfinite(value), "finite Lab reconstruction");
  const double y  = (lab[0] + 16) / 116;
  const double x  = .95047 * ColorLabInverse(y + lab[1] / 500);
  const double yy = ColorLabInverse(y), z = 1.08883 * ColorLabInverse(y - lab[2] / 200);
  return {ColorGamma(3.2404813432005266 * x - 1.5371515162713185 * yy - .4985363261688878 * z),
          ColorGamma(-.9692549499965684 * x + 1.8759900014898907 * yy + .0415559265582928 * z),
          ColorGamma(.0556466391351772 * x - .2040413383665112 * yy + 1.0573110696453443 * z)};
}

/// lightness [N,32,32,1] = L/50-1, chroma [N,32,32,2] = ab/128, rgb [N,32,32,3] in [0,1].
struct ColorData {
  nn::Tensor lightness, chroma, rgb;
};

/// `images` int32 [N,32,32,3] holding 0..255 (data::CifarDataset::images).
inline ColorData ColorPairs(const nn::Tensor& images, const std::vector<std::size_t>& rows) {
  check::Require(!rows.empty() && images.dim() == 4 && images.size(1) == 32 && images.size(2) == 32 && images.size(3) == 3,
                 "CIFAR RGB colorization source");
  const auto         source = images.to_vector<std::int32_t>();
  const std::size_t  n_rows = rows.size();
  std::vector<float> l(n_rows * 1024), ab(n_rows * 2048), rgb_out(n_rows * 3072);
  for (std::size_t n = 0; n < n_rows; ++n) {
    check::Require(rows[n] < std::size_t(images.size(0)), "colorization pair row bounds");
    for (std::size_t i = 0; i < 1024; ++i) {
      ColorTriple rgb;
      for (std::size_t c = 0; c < 3; ++c) {
        // tflow stores the float32 rgb and converts the double value / 255.
        const double v           = source[(rows[n] * 1024 + i) * 3 + c] / 255.;
        rgb[c]                   = v;
        rgb_out[(n * 1024 + i) * 3 + c] = float(v);
      }
      const auto lab = ColorRGBToLab(rgb);
      l[n * 1024 + i] = float(lab[0] / 50 - 1);
      for (std::size_t c = 0; c < 2; ++c) ab[(n * 1024 + i) * 2 + c] = float(lab[c + 1] / 128);
    }
  }
  const auto N = std::int64_t(n_rows);
  return {nn::from_vector(l, {N, 32, 32, 1}), nn::from_vector(ab, {N, 32, 32, 2}), nn::from_vector(rgb_out, {N, 32, 32, 3})};
}

inline nn::Tensor ColorBatch(const nn::Tensor& x, const std::vector<std::size_t>& rows) {
  check::Require(x.dim() == 4 && x.size(0) > 0, "float NHWC colorization batch");
  std::vector<std::int64_t> ids;
  for (auto r : rows) {
    check::Require(r < std::size_t(x.size(0)), "colorization batch bounds");
    ids.push_back(std::int64_t(r));
  }
  return nn::take_rows(x, ids).contiguous();
}

inline std::array<double, 2> ColorMeanChroma(const ColorData& fit) {
  std::array<double, 2> mean{};
  const auto            v = fit.chroma.to_vector<float>();
  for (std::size_t i = 0; i < v.size(); ++i) mean[i % 2] += v[i];
  for (auto& value : mean) value /= double(v.size() / 2);
  return mean;
}

inline nn::Tensor ColorConstantChroma(const ColorData& data, const std::array<double, 2>& mean) {
  std::vector<float> out(std::size_t(data.chroma.numel()));
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = float(mean[i % 2]);
  return nn::from_vector(out, data.chroma.shape());
}

struct ColorMetrics {
  double     ab_mse = 0, delta_e76 = 0, rgb_psnr = 0, clipped_pixels = 0, chroma = 0;
  nn::Tensor rgb;
};

inline ColorMetrics ColorScore(const ColorData& data, const nn::Tensor& prediction) {
  check::Require(prediction.shape() == data.chroma.shape(), "aligned chroma prediction");
  ColorMetrics       result;
  const auto         p = prediction.to_vector<float>(), t = data.chroma.to_vector<float>(), L = data.lightness.to_vector<float>(),
             truth = data.rgb.to_vector<float>();
  const std::int64_t pixels = data.lightness.numel(), per_image = pixels / data.lightness.size(0);
  std::vector<float> rgb_out(std::size_t(pixels) * 3);
  double             image_error = 0;
  for (std::int64_t i = 0; i < pixels; ++i) {
    const double a = 128 * double(p[2 * i]), b = 128 * double(p[2 * i + 1]);
    check::Require(std::isfinite(a) && std::isfinite(b), "finite predicted chroma");
    const double da = a - 128 * double(t[2 * i]), db = b - 128 * double(t[2 * i + 1]);
    result.ab_mse += (da * da + db * db) / (2 * 128 * 128);
    result.delta_e76 += std::hypot(da, db);
    result.chroma += std::hypot(a, b);
    const auto rgb     = ColorLabToRGB({50 * (double(L[i]) + 1), a, b});
    bool       clipped = false;
    for (int c = 0; c < 3; ++c) {
      check::Require(std::isfinite(rgb[c]), "finite decoded RGB");
      clipped |= rgb[c] < 0 || rgb[c] > 1;
      const double value      = std::clamp(rgb[c], 0., 1.);
      rgb_out[3 * i + c]      = float(value);
      const double difference = value - truth[3 * i + c];
      image_error += difference * difference;
    }
    result.clipped_pixels += clipped;
    if ((i + 1) % per_image == 0) {
      result.rgb_psnr += image_error == 0 ? INFINITY : -10 * std::log10(image_error / double(3 * per_image));
      image_error = 0;
    }
  }
  result.ab_mse /= double(pixels);
  result.delta_e76 /= double(pixels);
  result.rgb_psnr /= double(data.lightness.size(0));
  result.clipped_pixels /= double(pixels);
  result.chroma /= double(pixels);
  result.rgb = nn::from_vector(rgb_out, data.rgb.shape());
  return result;
}

}  // namespace color
