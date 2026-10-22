// Tests for datamunge::nn image ops, augmentation and image I/O.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <vector>

#include <datamunge/image/png.hpp>
#include <datamunge/nn/augment.hpp>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/image_io.hpp>
#include <datamunge/nn/image_ops.hpp>
#include <datamunge/nn/ops.hpp>
#include <datamunge/nn/random.hpp>

namespace nn  = datamunge::nn;
namespace aug = datamunge::nn::augment;
using nn::DType;
using nn::Tensor;

namespace {

using Inputs = std::vector<Tensor>;

void expect_close(const Tensor& a, const std::vector<double>& b, double tol = 1e-6) {
  const auto v = a.to_vector<double>();
  ASSERT_EQ(v.size(), b.size());
  for (std::size_t i = 0; i < v.size(); ++i) EXPECT_NEAR(v[i], b[i], tol) << "at " << i;
}

/// [1, 2, 2, 1] image [[1, 2], [3, 4]].
Tensor two_by_two() { return nn::tensor({1, 2, 3, 4}, {1, 2, 2, 1}); }

/// [1, 3, 3, 1] image with value 10 y + x.
Tensor three_by_three() { return nn::tensor({0, 1, 2, 10, 11, 12, 20, 21, 22}, {1, 3, 3, 1}); }

/// Distinct float64 values, requiring grad.
Tensor ramp(const nn::Shape& shape) {
  nn::Generator gen(7);
  Tensor        t = nn::uniform(shape, -1.0, 1.0, gen, DType::Float64);
  t.requires_grad_(true);
  return t;
}

void expect_gradcheck(const nn::TensorFunction& f, const Inputs& inputs) {
  const auto r = nn::gradcheck(f, inputs);
  EXPECT_TRUE(r.ok) << r.message;
}

/// [N, H, W, C] fixture with value n*1000 + y*100 + x*10 + c.
Tensor coded(std::int64_t n, std::int64_t h, std::int64_t w, std::int64_t c) {
  std::vector<float> v;
  for (std::int64_t i = 0; i < n; ++i)
    for (std::int64_t y = 0; y < h; ++y)
      for (std::int64_t x = 0; x < w; ++x)
        for (std::int64_t k = 0; k < c; ++k) v.push_back(static_cast<float>(i * 1000 + y * 100 + x * 10 + k));
  return nn::from_vector(v, {n, h, w, c});
}

}  // namespace

// ---- resize -------------------------------------------------------------------------

TEST(NnImageResize, BilinearLegacy2x2To4x4) {
  const Tensor out = nn::resize_bilinear(two_by_two(), {4, 4});
  EXPECT_EQ(out.shape(), (nn::Shape{1, 4, 4, 1}));
  expect_close(out, {1, 1.5, 2, 2, 2, 2.5, 3, 3, 3, 3.5, 4, 4, 3, 3.5, 4, 4});
}

TEST(NnImageResize, BilinearAlignCorners2x2To4x4) {
  const Tensor        out = nn::resize_bilinear(two_by_two(), {4, 4}, true);
  std::vector<double> want;
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x) want.push_back(1 + x / 3.0 + 2 * y / 3.0);
  expect_close(out, want);
}

TEST(NnImageResize, BilinearHalfPixel2x2To4x4) {
  const Tensor        out = nn::resize_bilinear(two_by_two(), {4, 4}, false, true);
  const double        row[4] = {1, 1.25, 1.75, 2}, wy[4] = {0, 0.25, 0.75, 1};
  std::vector<double> want;
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x) want.push_back(row[x] + 2 * wy[y]);
  expect_close(out, want);
}

TEST(NnImageResize, BilinearDownsample3To2) {
  const Tensor x = nn::tensor({0, 10, 20}, {1, 1, 3, 1});
  expect_close(nn::resize_bilinear(x, {1, 2}), {0, 15});
  expect_close(nn::resize_bilinear(x, {1, 2}, true), {0, 20});
  expect_close(nn::resize_bilinear(x, {1, 2}, false, true), {2.5, 17.5});
}

TEST(NnImageResize, NearestFlags4To6) {
  const Tensor x = nn::tensor({10, 20, 30, 40}, {1, 1, 4, 1});
  expect_close(nn::resize_nearest(x, {1, 6}), {10, 10, 20, 30, 30, 40});
  expect_close(nn::resize_nearest(x, {1, 6}, true), {10, 20, 20, 30, 30, 40});
  expect_close(nn::resize_nearest(x, {1, 6}, false, true), {10, 20, 20, 30, 40, 40});
}

TEST(NnImageResize, Nearest2x2To4x4RepeatsPixels) {
  const Tensor x = two_by_two();
  for (int flags = 0; flags < 3; ++flags) {
    const Tensor out = nn::resize_nearest(x, {4, 4}, flags == 1, flags == 2);
    expect_close(out, {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4});
  }
  expect_close(nn::upsample_nearest(x, 2), {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4});
}

TEST(NnImageResize, SingleImageRankAndValidation) {
  const Tensor img = nn::tensor({1, 2, 3, 4}, {2, 2, 1});
  EXPECT_EQ(nn::resize_bilinear(img, {3, 5}).shape(), (nn::Shape{3, 5, 1}));
  EXPECT_EQ(nn::upsample_nearest(img, 3).shape(), (nn::Shape{6, 6, 1}));
  EXPECT_THROW(nn::resize_bilinear(img, {3, 3}, true, true), std::invalid_argument);
  EXPECT_THROW(nn::resize_nearest(img, {0, 3}), std::invalid_argument);
  // Integer inputs become float32.
  EXPECT_EQ(nn::resize_nearest(nn::cast(img, DType::Int32), {2, 2}).dtype(), DType::Float32);
}

TEST(NnImageResize, Gradcheck) {
  for (int flags = 0; flags < 3; ++flags) {
    const bool ac = flags == 1, hp = flags == 2;
    expect_gradcheck([=](const Inputs& in) { return nn::resize_bilinear(in[0], {5, 3}, ac, hp); },
                     {ramp({2, 3, 4, 2})});
    expect_gradcheck([=](const Inputs& in) { return nn::resize_bilinear(in[0], {2, 7}, ac, hp); },
                     {ramp({1, 4, 3, 1})});
    expect_gradcheck([=](const Inputs& in) { return nn::resize_nearest(in[0], {5, 3}, ac, hp); },
                     {ramp({2, 3, 4, 2})});
  }
}

TEST(NnImageResize, BilinearGradientIsTFResizeBilinearGrad) {
  // d/dx sum(resize(x)) counts how much each source pixel is used.
  Tensor x = nn::tensor({1, 2, 3, 4}, {1, 2, 2, 1});
  x.requires_grad_(true);
  nn::sum(nn::resize_bilinear(x, {4, 4})).backward();
  // Legacy 2 -> 4 taps per axis: weights to source 0: 1 + .5 = 1.5; source 1: .5 + 1 + 1 = 2.5.
  expect_close(x.grad(), {1.5 * 1.5, 1.5 * 2.5, 2.5 * 1.5, 2.5 * 2.5});
}

TEST(NnImageResize, Gradgradcheck) {
  for (int flags = 0; flags < 3; ++flags) {
    const bool ac = flags == 1, hp = flags == 2;
    const auto r  = nn::gradgradcheck(
        [=](const Inputs& in) { return nn::square(nn::resize_bilinear(in[0], {3, 5}, ac, hp)); }, {ramp({1, 2, 3, 2})});
    EXPECT_TRUE(r.ok) << r.message;
  }
  const auto r = nn::gradgradcheck(
      [](const Inputs& in) { return nn::resize_nearest(nn::square(in[0]), {5, 4}); }, {ramp({1, 3, 2, 1})});
  EXPECT_TRUE(r.ok) << r.message;
}

// ---- crop_and_resize -------------------------------------------------------------------

TEST(NnImageCropAndResize, BilinearValues) {
  const Tensor img   = three_by_three();
  const Tensor boxes = nn::tensor({0, 0, 1, 1,  0, 0, 0.5, 0.5,  0.25, 0.25, 0.75, 0.75}, {3, 4});
  const Tensor idx   = nn::from_vector(std::vector<std::int32_t>{0, 0, 0});
  const Tensor out   = nn::crop_and_resize(img, boxes, idx, {3, 3});
  EXPECT_EQ(out.shape(), (nn::Shape{3, 3, 3, 1}));
  const auto v = out.to_vector<double>();
  const std::vector<double> full{0, 1, 2, 10, 11, 12, 20, 21, 22};
  const std::vector<double> quarter{0, 0.5, 1, 5, 5.5, 6, 10, 10.5, 11};
  const std::vector<double> centre{5.5, 6, 6.5, 10.5, 11, 11.5, 15.5, 16, 16.5};
  for (int i = 0; i < 9; ++i) {
    EXPECT_NEAR(v[static_cast<std::size_t>(i)], full[static_cast<std::size_t>(i)], 1e-5);
    EXPECT_NEAR(v[static_cast<std::size_t>(9 + i)], quarter[static_cast<std::size_t>(i)], 1e-5);
    EXPECT_NEAR(v[static_cast<std::size_t>(18 + i)], centre[static_cast<std::size_t>(i)], 1e-5);
  }
  // A 1x1 crop samples the box centre; a 2x2 crop of the full box takes the corners.
  expect_close(nn::crop_and_resize(img, nn::tensor({0.25, 0.25, 0.75, 0.75}, {1, 4}),
                                   nn::from_vector(std::vector<std::int32_t>{0}), {1, 1}),
               {11});
  expect_close(nn::crop_and_resize(img, nn::tensor({0, 0, 1, 1}, {1, 4}), nn::from_vector(std::vector<std::int32_t>{0}),
                                   {2, 2}),
               {0, 2, 20, 22});
  // y2 < y1 flips the crop vertically.
  expect_close(nn::crop_and_resize(img, nn::tensor({1, 0, 0, 1}, {1, 4}), nn::from_vector(std::vector<std::int32_t>{0}),
                                   {2, 2}),
               {20, 22, 0, 2});
}

TEST(NnImageCropAndResize, ExtrapolationNearestAndBoxIndex) {
  const Tensor imgs = nn::concat({three_by_three(), nn::add(three_by_three(), 100.0)}, 0);
  // Rows at in_y = -1, 0, 1: the first falls outside.
  const Tensor out = nn::crop_and_resize(imgs, nn::tensor({-0.5, 0, 0.5, 1}, {1, 4}),
                                         nn::from_vector(std::vector<std::int64_t>{1}), {3, 3},
                                         nn::ResizeMethod::Bilinear, -7);
  expect_close(out, {-7, -7, -7, 100, 101, 102, 110, 111, 112});
  const Tensor near = nn::crop_and_resize(imgs, nn::tensor({0, 0, 0.5, 0.5}, {1, 4}),
                                          nn::from_vector(std::vector<std::int64_t>{0}), {3, 3},
                                          nn::ResizeMethod::Nearest);
  expect_close(near, {0, 1, 1, 10, 11, 11, 10, 11, 11});  // positions 0, .5, 1 round to 0, 1, 1
  EXPECT_THROW(nn::crop_and_resize(imgs, nn::tensor({0, 0, 1, 1}, {1, 4}),
                                   nn::from_vector(std::vector<std::int64_t>{2}), {2, 2}),
               std::out_of_range);
}

TEST(NnImageCropAndResize, Gradcheck) {
  const Tensor boxes = nn::tensor({0.1, 0.2, 0.9, 0.7,  -0.3, 0.1, 0.6, 1.2,  0.8, 0.9, 0.2, 0.3}, {3, 4});
  const Tensor idx   = nn::from_vector(std::vector<std::int32_t>{1, 0, 1});
  for (auto method : {nn::ResizeMethod::Bilinear, nn::ResizeMethod::Nearest})
    expect_gradcheck(
        [=](const Inputs& in) { return nn::crop_and_resize(in[0], boxes, idx, {3, 4}, method, 0.5); },
        {ramp({2, 4, 5, 2})});
  const auto r = nn::gradgradcheck(
      [=](const Inputs& in) { return nn::square(nn::crop_and_resize(in[0], boxes, idx, {2, 3})); }, {ramp({2, 4, 5, 1})});
  EXPECT_TRUE(r.ok) << r.message;
}

// ---- space_to_depth / depth_to_space / mirror pad --------------------------------------------

TEST(NnImageSpaceToDepth, TFOrderAndInverse) {
  // TF docs example: [1, 2, 2, 1] [[1, 2], [3, 4]] -> [1, 1, 1, 4] [1, 2, 3, 4].
  const Tensor s = nn::space_to_depth(two_by_two(), 2);
  EXPECT_EQ(s.shape(), (nn::Shape{1, 1, 1, 4}));
  expect_close(s, {1, 2, 3, 4});
  // TF docs: [1, 2, 2, 3] -> [1, 1, 1, 12] keeps channels innermost.
  const Tensor rgb = nn::reshape(nn::arange(1, 13), {1, 2, 2, 3});
  expect_close(nn::space_to_depth(rgb, 2), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  // 4x4 -> 2x2x4: out[0, 0, 1] = channel (0*2+1) = pixel (0, 1).
  const Tensor big = coded(2, 4, 4, 3);
  const Tensor d   = nn::space_to_depth(big, 2);
  EXPECT_EQ(d.shape(), (nn::Shape{2, 2, 2, 12}));
  EXPECT_EQ(d.at({1, 1, 0, 3 * 3 + 2}), 1000 + 3 * 100 + 1 * 10 + 2);  // block (1,1) of (y,x)=(1,0) -> pixel (3,1)
  expect_close(nn::depth_to_space(d, 2), big.to_vector<double>());
  EXPECT_THROW(nn::space_to_depth(coded(1, 3, 4, 1), 2), std::invalid_argument);
}

TEST(NnImageSpaceToDepth, Gradcheck) {
  expect_gradcheck([](const Inputs& in) { return nn::space_to_depth(in[0], 2); }, {ramp({2, 4, 2, 3})});
  expect_gradcheck([](const Inputs& in) { return nn::depth_to_space(in[0], 2); }, {ramp({1, 2, 3, 8})});
  expect_gradcheck([](const Inputs& in) { return nn::upsample_nearest(in[0], 2); }, {ramp({1, 2, 3, 2})});
}

TEST(NnImageMirrorPad, MatchesPad) {
  const Tensor x = three_by_three();
  const Tensor r = nn::mirror_pad(x, 1, 1);
  EXPECT_EQ(r.shape(), (nn::Shape{1, 5, 5, 1}));
  EXPECT_EQ(r.at({0, 0, 0, 0}), 11);  // reflect excludes the edge
  EXPECT_EQ(nn::mirror_pad(x, 1, 1, nn::PadMode::Symmetric).at({0, 0, 0, 0}), 0);
}

// ---- augmentation -------------------------------------------------------------------------

TEST(NnImageAugment, FlipsAndCrop) {
  const Tensor x = coded(2, 4, 4, 3);
  const Tensor f = aug::flip_left_right(x);
  const Tensor v = aug::flip_up_down(x);
  for (int n = 0; n < 2; ++n)
    for (int y = 0; y < 4; ++y)
      for (int c = 0; c < 4; ++c)
        for (int k = 0; k < 3; ++k) {
          EXPECT_EQ(f.at({n, y, c, k}), x.at({n, y, 3 - c, k}));
          EXPECT_EQ(v.at({n, y, c, k}), x.at({n, 3 - y, c, k}));
        }
  expect_close(aug::flip_left_right(f), x.to_vector<double>(), 0);
  const Tensor cr = aug::crop(x, 1, 1, 2, 2);
  EXPECT_EQ(cr.shape(), (nn::Shape{2, 2, 2, 3}));
  EXPECT_EQ(cr.at({1, 0, 0, 2}), x.at({1, 1, 1, 2}));
  EXPECT_THROW(aug::crop(x, 3, 0, 2, 2), std::invalid_argument);
  // Single image rank is preserved.
  EXPECT_EQ(aug::flip_left_right(nn::select(x, 0, 0)).shape(), (nn::Shape{4, 4, 3}));
}

TEST(NnImageAugment, CutoutClipsRectangle) {
  const Tensor x   = nn::add(coded(2, 4, 4, 3), 1.0);
  const Tensor out = aug::cutout(x, {1, 1, 2, 2});
  const Tensor off = aug::cutout(x, {-1, 2, 3, 4});  // clipped to rows 0-1, cols 2-3
  for (int n = 0; n < 2; ++n)
    for (int y = 0; y < 4; ++y)
      for (int c = 0; c < 4; ++c) {
        const bool in1 = y >= 1 && y < 3 && c >= 1 && c < 3;
        const bool in2 = y < 2 && c >= 2;
        EXPECT_EQ(out.at({n, y, c, 0}), in1 ? 0 : x.at({n, y, c, 0}));
        EXPECT_EQ(off.at({n, y, c, 1}), in2 ? 0 : x.at({n, y, c, 1}));
      }
}

TEST(NnImageAugment, RandomPadCropMatchesImageBatchRule) {
  // Reproduce the draws: top, left, coin per image.
  const Tensor  x = nn::add(coded(3, 5, 6, 2), 1.0);
  nn::Generator g1(42), g2(42);
  const Tensor  out = aug::random_pad_crop(x, 2, g1, true);
  EXPECT_EQ(out.shape(), x.shape());
  int shifted = 0;
  for (int n = 0; n < 3; ++n) {
    const auto top  = static_cast<int>(g2.below(5)) - 2;
    const auto left = static_cast<int>(g2.below(5)) - 2;
    const bool flip = g2.below(2) == 1;
    shifted += top != 0 || left != 0;
    for (int r = 0; r < 5; ++r)
      for (int c = 0; c < 6; ++c)
        for (int k = 0; k < 2; ++k) {
          const int    iy = r + top, ix = (flip ? 5 - c : c) + left;
          const double want = (iy < 0 || iy >= 5 || ix < 0 || ix >= 6) ? 0.0 : x.at({n, iy, ix, k});
          EXPECT_EQ(out.at({n, r, c, k}), want);
        }
  }
  EXPECT_GT(shifted, 0);
}

TEST(NnImageAugment, Reproducible) {
  const Tensor x  = nn::uniform({4, 6, 6, 3}, 0.0, 1.0);
  const Tensor y  = nn::one_hot(nn::from_vector(std::vector<std::int64_t>{0, 1, 2, 1}), 3);
  auto         run = [&](std::uint64_t seed) {
    nn::Generator g(seed);
    std::vector<Tensor> outs{aug::random_flip_left_right(x, g),  aug::random_flip_up_down(x, g),
                             aug::random_pad_crop(x, 4, g, true), aug::random_crop(x, 4, 3, g),
                             aug::random_cutout(x, 3, g),         aug::random_brightness(x, 0.2, g),
                             aug::random_contrast(x, 0.5, 1.5, g), aug::random_saturation(x, 0.5, 1.5, g),
                             aug::random_hue(x, 0.2, g)};
    const auto m = aug::random_mixup(x, y, 0.4, g);
    const auto c = aug::random_cutmix(x, y, 1.0, g);
    outs.insert(outs.end(), {m.images, m.labels, c.images, c.labels});
    return outs;
  };
  const auto a = run(5), b = run(5), c = run(6);
  ASSERT_EQ(a.size(), b.size());
  bool any_diff = false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].to_vector<double>(), b[i].to_vector<double>()) << "output " << i;
    any_diff = any_diff || a[i].to_vector<double>() != c[i].to_vector<double>();
  }
  EXPECT_TRUE(any_diff);
  EXPECT_EQ(a[3].shape(), (nn::Shape{4, 4, 3, 3}));
}

TEST(NnImageAugment, MixupLabels) {
  // tflow data_augmentation: images .2 / .8, labels (1,0) / (0,1), weight .25.
  const Tensor x = nn::concat({nn::full({1, 4, 4, 3}, 0.2), nn::full({1, 4, 4, 3}, 0.8)}, 0);
  const Tensor y = nn::tensor({{1, 0}, {0, 1}});
  const auto   m = aug::mixup(x, y, {1, 0}, 0.25);
  EXPECT_NEAR(m.images.at({0, 0, 0, 0}), 0.65, 1e-6);
  expect_close(m.labels, {0.25, 0.75, 0.75, 0.25});
  EXPECT_THROW(aug::mixup(x, y, {1, 0}, 1.5), std::invalid_argument);
  EXPECT_THROW(aug::mixup(x, y, {1, 1}, 0.5), std::invalid_argument);
  EXPECT_THROW(aug::mixup(x, nn::from_vector(std::vector<std::int64_t>{0, 1}), {1, 0}, 0.5), std::invalid_argument);

  // CutMix rect (-1, 2, 3, 4) clips to 2x2 = 4/16 of the image, so labels keep .75.
  const auto c = aug::cutmix(x, y, {1, 0}, {-1, 2, 3, 4});
  EXPECT_DOUBLE_EQ(c.weight, 0.75);
  expect_close(c.labels, {0.75, 0.25, 0.25, 0.75});
  for (int n = 0; n < 2; ++n)
    for (int r = 0; r < 4; ++r)
      for (int col = 0; col < 4; ++col) {
        const bool pasted = r < 2 && col >= 2;
        EXPECT_NEAR(c.images.at({n, r, col, 1}), ((pasted ? 1 - n : n) == 0) ? 0.2 : 0.8, 1e-7);
      }

  nn::Generator g(7);
  for (int i = 0; i < 20; ++i) {
    const double w = aug::beta_mix_weight(g, 0.4);
    EXPECT_GE(w, 0.0);
    EXPECT_LE(w, 1.0);
  }
  const auto rm = aug::random_mixup(x, y, 0.4, g);
  const auto s  = nn::sum(rm.labels, {1}).to_vector<double>();
  for (double v : s) EXPECT_NEAR(v, 1.0, 1e-6);
}

TEST(NnImageAugment, ColorOps) {
  const Tensor x = nn::reshape(nn::div(nn::arange(0, 96), 120.0), {2, 4, 4, 3});
  // Identity jitter (tflow check).
  expect_close(aug::color_jitter(x, 0, 1, 1, 0), x.to_vector<double>(), 2e-7);
  // Brightness saturates at 1 after the clamp.
  expect_close(aug::color_jitter(nn::full({1, 2, 2, 3}, 0.8), 0.3, 1, 1, 0), std::vector<double>(12, 1.0), 0);
  // Red rotated by a third of a turn is green.
  expect_close(aug::adjust_hue(nn::tensor({1, 0, 0}, {1, 1, 1, 3}), 1.0 / 3), {0, 1, 0}, 2e-7);
  // Saturation 0 gives grey at the max channel value (HSV keeps V).
  expect_close(aug::adjust_saturation(nn::tensor({0.2, 0.6, 0.4}, {1, 1, 1, 3}), 0), {0.6, 0.6, 0.6}, 1e-7);
  expect_close(aug::adjust_saturation(nn::tensor({0.2, 0.6, 0.4}, {1, 1, 1, 3}), 0.5), {0.4, 0.6, 0.5}, 1e-6);
  // Contrast: per image and channel mean over H, W.
  const Tensor img = nn::tensor({0, 10, 2, 20}, {1, 2, 1, 2});  // channel means 1 and 15
  expect_close(aug::adjust_contrast(img, 2), {-1, 5, 3, 25});
  expect_close(aug::adjust_brightness(img, 0.5), {0.5, 10.5, 2.5, 20.5});
  // per_image_standardization: zero mean, unit variance; the stddev floor
  // 1/sqrt(48) applies to x itself (stddev sqrt((48^2 - 1) / 12) / 120) but not to 10 x.
  const Tensor st = aug::per_image_standardization(nn::mul(x, 10.0));
  EXPECT_NEAR(nn::std_dev(nn::select(aug::per_image_standardization(x), 0, 1)).item(),
              std::sqrt((48.0 * 48 - 1) / 12) / 120 * std::sqrt(48.0), 1e-5);
  EXPECT_NEAR(nn::mean(nn::select(st, 0, 1)).item(), 0.0, 1e-5);
  EXPECT_NEAR(nn::std_dev(nn::select(st, 0, 1)).item(), 1.0, 1e-5);
  expect_close(aug::per_image_standardization(nn::full({4, 4, 3}, 0.5)), std::vector<double>(48, 0.0));
}

TEST(NnImageAugment, NoHistory) {
  Tensor x = nn::uniform({2, 4, 4, 3});
  x.requires_grad_(true);
  nn::Generator g(1);
  EXPECT_FALSE(aug::random_pad_crop(x, 2, g, true).requires_grad());
  EXPECT_FALSE(aug::adjust_contrast(x, 0.5).requires_grad());
}

// ---- image I/O --------------------------------------------------------------------------------

TEST(NnImageIo, RoundTripAndRanges) {
  const Tensor x   = nn::reshape(nn::div(nn::arange(0, 24), 23.0), {2, 4, 3});
  const auto   img = nn::to_image(x);
  EXPECT_EQ(img.width(), 4);
  EXPECT_EQ(img.height(), 2);
  EXPECT_EQ(img.mode(), datamunge::image::ImageMode::RGB);
  const Tensor back = nn::from_image(img);
  EXPECT_EQ(back.shape(), (nn::Shape{2, 4, 3}));
  const auto a = x.to_vector<double>(), b = back.to_vector<double>();
  for (std::size_t i = 0; i < a.size(); ++i) EXPECT_EQ(b[i], std::lround(a[i] * 255));
  // [-1, 1] range, clamping, NaN and grayscale.
  const auto g = nn::to_image(nn::tensor({-2, -1, 0, 1, 2, NAN}, {2, 3}), -1, 1);
  EXPECT_EQ(g.mode(), datamunge::image::ImageMode::Grayscale);
  EXPECT_EQ(g.data(), (std::vector<std::uint8_t>{0, 0, 128, 255, 255, 0}));
  EXPECT_THROW(nn::to_image(nn::zeros({2, 2, 5})), std::invalid_argument);
}

TEST(NnImageIo, GridLayout) {
  const Tensor batch = nn::add(coded(5, 2, 3, 1), 1.0);
  const Tensor grid  = nn::image_grid(batch, 3, 1, -1.0);
  EXPECT_EQ(grid.shape(), (nn::Shape{2 * 2 + 1, 3 * 3 + 2, 1}));
  EXPECT_EQ(grid.at({0, 0, 0}), batch.at({0, 0, 0, 0}));
  EXPECT_EQ(grid.at({0, 3, 0}), -1);                          // padding column
  EXPECT_EQ(grid.at({3, 4 + 2, 0}), batch.at({4, 0, 2, 0}));  // tile 4 = row 1, col 1
  EXPECT_EQ(grid.at({4, 8, 0}), -1);                          // empty slot
  const Tensor scaled = nn::image_grid(nn::reshape(batch, {5, 2, 3}), 0, 0, 0.0, 2);  // cols = ceil(sqrt(5)) = 3
  EXPECT_EQ(scaled.shape(), (nn::Shape{8, 18, 1}));
  EXPECT_EQ(scaled.at({1, 1, 0}), batch.at({0, 0, 0, 0}));
}

TEST(NnImageIo, WritesPng) {
  const auto dir = std::filesystem::temp_directory_path() / "datamunge_nn_image_tests";
  std::filesystem::create_directories(dir);
  const Tensor x    = nn::uniform({4, 5, 6, 3}, -1.0, 1.0);
  const auto   path = (dir / "grid.png").string();
  nn::save_image_grid(x, path, 2, -1, 1, 1, 1.0);
  const auto img = datamunge::image::read_png(path);
  EXPECT_EQ(img.width(), 13);
  EXPECT_EQ(img.height(), 11);
  EXPECT_EQ(img.data(), nn::to_image(nn::image_grid(x, 2, 1, 1.0), -1, 1).data());
  const auto single = (dir / "one.png").string();
  nn::save_png(nn::select(x, 0, 0), single, -1, 1);
  const Tensor back = nn::from_image(datamunge::image::read_png(single));
  const auto   a = nn::select(x, 0, 0).to_vector<double>(), b = back.to_vector<double>();
  for (std::size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(b[i] / 255 * 2 - 1, a[i], 1.0 / 255 + 1e-6);
  std::filesystem::remove_all(dir);
}
