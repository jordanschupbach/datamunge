#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_super_resolution_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.8_other_vision_tasks/super_resolution.cpp"
#undef main
namespace oracle {
namespace nn = dm::nn;
using nn::Tensor;
inline double SRCubic(double x) {
  x = std::abs(x);
  if (x <= 1)
    return 1.5 * x * x * x - 2.5 * x * x + 1;
  if (x < 2)
    return -.5 * x * x * x + 2.5 * x * x - 4 * x + 2;
  return 0;
}
inline int SRSymmetric(int index, int length) {
  if (length < 1)
    throw std::invalid_argument("resampling length");
  if (length == 1)
    return 0;
  while (index < 0 || index >= length)
    index = index < 0 ? -index - 1 : 2 * length - 1 - index;
  return index;
}
using SRWeights = std::vector<std::vector<std::pair<int, double>>>;

/// Normalized antialiased Keys contributions of `input` samples to each of `output`.
inline SRWeights SRContributions(int input, int output) {
  if (input < 1 || output < 1)
    throw std::invalid_argument("positive bicubic dimensions");
  SRWeights weights(output);
  const double scale = double(output) / input, filter = std::min(1., scale),
               radius = 2 / filter;
  for (int i = 0; i < output; ++i) {
    const double center = (i + .5) / scale - .5;
    double sum = 0;
    for (int j = int(std::ceil(center - radius)); j <= int(std::floor(center + radius)); ++j) {
      const double value = filter * SRCubic(filter * (center - j));
      if (value != 0) {
        weights[i].push_back({SRSymmetric(j, input), value});
        sum += value;
      }
    }
    check::Require(std::isfinite(sum) && std::abs(sum) > 1e-12,
                   "bicubic normalized kernel mass");
    for (auto &item : weights[i])
      item.second /= sum;
  }
  return weights;
}

/// Separable bicubic resize of float32 NHWC images (rows after columns, float32 intermediate as tflow).
inline Tensor SRBicubic(const Tensor &image, int oh, int ow) {
  check::Require(image.dim() == 4, "bicubic float NHWC");
  const int b = int(image.size(0)), h = int(image.size(1)), w = int(image.size(2)),
            c = int(image.size(3));
  const auto rows = SRContributions(h, oh), columns = SRContributions(w, ow);
  const auto in = nn::cast(image, nn::DType::Float32).contiguous().to_vector<float>();
  std::vector<float> horizontal(std::size_t(b) * h * ow * c),
      out(std::size_t(b) * oh * ow * c);
  for (int n = 0; n < b; ++n)
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < ow; ++x)
        for (int k = 0; k < c; ++k) {
          double value = 0;
          for (auto [j, a] : columns[x])
            value += a * in[((std::size_t(n) * h + y) * w + j) * c + k];
          horizontal[((std::size_t(n) * h + y) * ow + x) * c + k] = float(value);
        }
  for (int n = 0; n < b; ++n)
    for (int y = 0; y < oh; ++y)
      for (int x = 0; x < ow; ++x)
        for (int k = 0; k < c; ++k) {
          double value = 0;
          for (auto [j, a] : rows[y])
            value += a * horizontal[((std::size_t(n) * h + j) * ow + x) * c + k];
          out[((std::size_t(n) * oh + y) * ow + x) * c + k] = float(value);
        }
  return nn::from_vector(out, {b, oh, ow, c});
}

inline double SRSSIM(const std::vector<double> &a, const std::vector<double> &b, int h,
                     int w) {
  check::Require(h >= 11 && w >= 11 && a.size() == std::size_t(h * w) && a.size() == b.size(),
                 "SSIM valid 11 Gaussian window");
  std::array<double, 11> kernel;
  double sum = 0;
  for (int i = 0; i < 11; ++i) {
    kernel[i] = std::exp(-(i - 5) * (i - 5) / (2 * 1.5 * 1.5));
    sum += kernel[i];
  }
  for (auto &v : kernel)
    v /= sum;
  double score = 0;
  int count = 0;
  for (int y = 5; y < h - 5; ++y)
    for (int x = 5; x < w - 5; ++x) {
      double ma = 0, mb = 0, aa = 0, bb = 0, ab = 0;
      for (int j = 0; j < 11; ++j)
        for (int k = 0; k < 11; ++k) {
          const double weight = kernel[j] * kernel[k], p = a[(y + j - 5) * w + x + k - 5],
                       q = b[(y + j - 5) * w + x + k - 5];
          ma += weight * p;
          mb += weight * q;
          aa += weight * p * p;
          bb += weight * q * q;
          ab += weight * p * q;
        }
      const double va = std::max(0., aa - ma * ma), vb = std::max(0., bb - mb * mb),
                   cov = ab - ma * mb;
      score += ((2 * ma * mb + .0001) * (2 * cov + .0009)) /
               ((ma * ma + mb * mb + .0001) * (va + vb + .0009));
      ++count;
    }
  return score / count;
}

} // namespace oracle

TEST(SuperResolution, IndependentShuffle) {
  ShuffleChecks();
}
TEST(SuperResolution, IndependentDataMetrics) {
  DataMetricChecks();
}
TEST(SuperResolution, IndependentAttention) {
  AttentionChecks();
}
TEST(SuperResolution, IndependentResidualClipping) {
  ResidualClipChecks();
}
TEST(SuperResolution, AllFiveFullAndCompactGraphs) {
  ModelChecks();
}
TEST(SuperResolution, BicubicEveryValueAndInputAdjoint) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::set_num_threads(2);
  for (const auto shape : std::vector<n::Shape>{{2, 5, 7, 3}, {1, 1, 5, 2}, {1, 7, 1, 1}}) {
    const auto x =
        (n::sin(n::arange(0, n::shape_numel(shape), 1, n::DType::Float64) * .37) * .8)
            .to(n::DType::Float32)
            .reshape(shape)
            .requires_grad_();
    for (const auto target : std::vector<std::pair<int, int>>{{3, 4}, {9, 11}, {1, 1}}) {
      const auto y = m::SRBicubic(x, target.first, target.second),
                 expected = oracle::SRBicubic(x, target.first, target.second);
      EXPECT_LT(n::max(n::abs(y - expected)).item(), 2e-7);
      const auto up = n::sin(n::arange(0, y.numel(), 1, n::DType::Float64) * .19)
                          .to(n::DType::Float32)
                          .reshape(y.shape());
      n::GradOptions options;
      options.retain_graph = true;
      const auto g = n::grad(n::sum(y * up), {x}, options)[0].to_vector<float>(),
                 uv = up.to_vector<float>();
      const auto rows = oracle::SRContributions(int(shape[1]), target.first),
                 columns = oracle::SRContributions(int(shape[2]), target.second);
      std::vector<double> reference(x.numel());
      for (int b = 0; b < shape[0]; ++b)
        for (int i = 0; i < target.first; ++i)
          for (int j = 0; j < target.second; ++j)
            for (int c = 0; c < shape[3]; ++c)
              for (auto [iy, wy] : rows[i])
                for (auto [ix, wx] : columns[j])
                  reference[((b * shape[1] + iy) * shape[2] + ix) * shape[3] + c] +=
                      wy * wx *
                      uv[((b * target.first + i) * target.second + j) * shape[3] + c];
      for (std::size_t i = 0; i < g.size(); ++i)
        EXPECT_NEAR(g[i], reference[i], 4e-7);
    }
  }
}
TEST(SuperResolution, NonconstantGaussianSsimAndEveryPreviewByte) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::set_num_threads(2);
  const auto x = (n::sin(n::arange(0, 2 * 20 * 23, 1, n::DType::Float64) * .137) + 1.)
                     .reshape({2, 20, 23, 1}) *
                 .45,
             y = (n::cos(n::arange(0, 2 * 20 * 23, 1, n::DType::Float64) * .091) + 1.)
                     .reshape({2, 20, 23, 1}) *
                 .4;
  const auto actual = m::SRSSIMTensor(x, y).to_vector<double>();
  for (int b = 0; b < 2; ++b)
    EXPECT_NEAR(actual[b],
                oracle::SRSSIM(n::select(x, 0, b).to_vector<double>(),
                               n::select(y, 0, b).to_vector<double>(), 20, 23),
                2e-12);
  std::vector<std::int32_t> byte_values(3 * 3072);
  for (std::size_t i = 0; i < byte_values.size(); ++i)
    byte_values[i] = i % 256;
  const auto bytes = n::from_vector(byte_values, {3 * 3072});
  const auto d = m::SRPairs(bytes.reshape({3, 32, 32, 3}), {2, 0});
  const auto p = m::SRCrop(d.hr, 6) * 1.3 - .1;
  const auto tiles = m::SRPreviewTiles(d, p).to_vector<float>(), pv = p.to_vector<float>(),
             hv = d.hr.to_vector<float>(), bv = d.bicubic.to_vector<float>();
  for (int b = 0; b < 2; ++b)
    for (int i = 0; i < 20; ++i)
      for (int j = 0; j < 20; ++j) {
        const auto ref = [&](const auto &v, int side, int row, int col) {
          const int k = ((b * side + row) * side + col) * 3;
          return (16. + 65.481 * v[k] + 128.553 * v[k + 1] + 24.966 * v[k + 2]) / 255.;
        };
        const double prediction = std::clamp(ref(pv, 20, i, j), 0., 1.),
                     truth = ref(hv, 32, i + 6, j + 6),
                     base = std::clamp(ref(bv, 32, i + 6, j + 6), 0., 1.);
        const double values[] = {base, prediction, truth,
                                 std::min(1., 8 * std::abs(prediction - truth))};
        for (int panel = 0; panel < 4; ++panel)
          for (int c = 0; c < 3; ++c) {
            const double v = panel == 3 && c != 0 ? 0 : values[panel];
            EXPECT_NEAR(tiles[((b * 20 + i) * 80 + panel * 20 + j) * 3 + c],
                        std::lround(v * 255) / 255., 4e-8);
          }
      }
}
TEST(SuperResolution, RawInputGradientAndValidation) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::set_num_threads(2);
  for (auto kind : All) {
    const auto cfg = m::SRConfig::Compact(kind);
    const int side = cfg.PreUpsampled() ? 17 : 3;
    n::ParameterStore p(17);
    const auto x = (n::ones({1, side, side + 2, cfg.Channels()}) * .4).requires_grad_();
    const auto out = m::SuperResolution(p, x, cfg);
    const auto g = n::grad(n::sum(out.image), {x})[0];
    EXPECT_TRUE(n::all(n::is_finite(g)).item<bool>());
    EXPECT_GT(n::sum(n::abs(g)).item(), 0.);
  }
  EXPECT_THROW(m::SRPixelShuffle(n::ones({1, 2, 3, 7}), 2, 2), std::invalid_argument);
  EXPECT_THROW(m::VdsrClippedGradient(n::ones({2}), 0.), std::invalid_argument);
  EXPECT_THROW(m::SRSSIMTensor(n::ones({1, 10, 20, 1}), n::ones({1, 10, 20, 1})),
               std::invalid_argument);
}
