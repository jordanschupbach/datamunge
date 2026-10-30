
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_hourglass_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.8_other_vision_tasks/stacked_hourglass.cpp"
#undef main
#define main original_pose_preparation_main
#include "../../scripts/nn_pose_prepare.cpp"
#undef main
TEST(StackedHourglass, IndependentTargetsLossEveryDerivativeDecoderMetrics) {
  TargetLossDecoderChecks();
}
TEST(StackedHourglass, IndependentPreactivationResidualAndBnPolicy) {
  ResidualChecks();
}
TEST(StackedHourglass, FullCompactEveryStageGradientAndFeedback) {
  ModelChecks();
}
TEST(StackedHourglass, EveryOddGaussianAndDecoderFloatBoundary) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::set_num_threads(2);
  const auto joints =
      n::tensor({3.7, 8.2, 1., NAN, NAN, 0., 10., 2.1, 2., 15., 3., 2.}, {2, 2, 3});
  const float sigma = .7f;
  const auto targets = m::PoseHeatmaps(joints, 7, 16, sigma);
  const auto jv = joints.to_vector<float>(), values = targets.heatmap.to_vector<float>();
  for (int b = 0; b < 2; ++b)
    for (int y = 0; y < 7; ++y)
      for (int x = 0; x < 7; ++x)
        for (int j = 0; j < 2; ++j) {
          const int k = (b * 2 + j) * 3;
          double expected = 0;
          if (jv[k + 2] > 0) {
            const double cx = (jv[k] + .5) * 7 / 16 - .5, cy = (jv[k + 1] + .5) * 7 / 16 - .5;
            if (std::abs(x - cx) <= 3.f * sigma && std::abs(y - cy) <= 3.f * sigma)
              expected = std::exp(-((x - cx) * (x - cx) + (y - cy) * (y - cy)) /
                                  double(2.f * sigma * sigma));
          }
          EXPECT_NEAR(values[((b * 7 + y) * 7 + x) * 2 + j], expected, 3e-8);
        }
  const auto raw = (n::sin(n::arange(0, 2 * 7 * 7 * 3) * .37) - 2.).reshape({2, 7, 7, 3});
  const auto decoded = m::DecodePoseHeatmaps(raw, 29).to_vector<float>(),
             hv = raw.to_vector<float>();
  for (int b = 0; b < 2; ++b)
    for (int j = 0; j < 3; ++j) {
      int at = 0;
      float maximum = -INFINITY;
      for (int i = 0; i < 49; ++i)
        if (hv[(b * 49 + i) * 3 + j] > maximum) {
          at = i;
          maximum = hv[(b * 49 + i) * 3 + j];
        }
      EXPECT_EQ(decoded[(b * 3 + j) * 3], (float(at % 7) + .5f) * 29 / 7.f - .5f);
      EXPECT_EQ(decoded[(b * 3 + j) * 3 + 1], (float(at / 7) + .5f) * 29 / 7.f - .5f);
      EXPECT_EQ(decoded[(b * 3 + j) * 3 + 2], maximum);
    }
}
TEST(StackedHourglass, SafeEmptyLossBooleanMaskAndJointWeightedPck) {
  namespace n = dm::nn;
  namespace m = n::models;
  const auto x = n::ones({2, 3, 3, 2}).requires_grad_();
  const auto zero = m::PoseHeatmapMSE(x, n::full(x.shape(), NAN), n::zeros({2, 1, 1, 2}), 3);
  EXPECT_EQ(zero.item(), 0.);
  EXPECT_EQ(n::sum(n::abs(n::grad(zero, {x})[0])).item(), 0.);
  EXPECT_EQ(m::PoseHeatmapMSE(n::ones({1, 3, 3, 2}), n::zeros({1, 3, 3, 2}),
                              n::ones({1, 1, 1, 2}, n::DType::Bool), 3)
                .item(),
            1.);
  m::PoseData d;
  d.joints = n::tensor({0., 0., 1., NAN, NAN, 0., 0., 0., 2., 0., 0., 1.}, {2, 2, 3});
  d.box_size = n::tensor({25., 20., 10., 8.}, {2, 2});
  const auto pred =
      n::tensor({3., 4., -99., NAN, NAN, -99., 0., 0., -99., 3., 4., -99.}, {2, 2, 3});
  const auto score = m::ScoreTensorPose(pred, d);
  EXPECT_DOUBLE_EQ(score.pck.item(), 2. / 3.);
  EXPECT_NEAR(score.normalized_error.item(), (.2 + 0. + .5) / 3., 1e-15);
  EXPECT_DOUBLE_EQ(score.pixels.item(), 10. / 3.);
  EXPECT_EQ(score.joints.item<int>(), 3);
}
TEST(StackedHourglass, EveryPreparedSourceCropPixelAndAnatomicalMirror) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::set_num_threads(2);
  const std::filesystem::path root = "build/cnn-datasets/human_pose";
  if (!std::filesystem::exists(root / "poses.npz"))
    GTEST_SKIP() << "prepare original COCO8-Pose";
  const auto splits = m::LoadPoseCrops(root);
  const auto arrays = m::PoseArrays(root / "poses.npz");
  std::map<int, SourceFrame> sources;
  std::ifstream source_files(root / "source_files.tsv");
  std::string split, image, label;
  int id;
  while (source_files >> split >> id >> image >> label)
    sources.emplace(id, ReadFrame(root / image, root / label, id));
  std::ifstream crops(root / "crops.tsv");
  std::map<std::string, int> offset;
  int pid, left, top, right, bottom, flip;
  while (crops >> split >> id >> pid >> left >> top >> right >> bottom >> flip) {
    const int row = offset[split]++;
    const auto actual = n::select(arrays.at(split + "_rgb"), 0, row).to_vector<float>(),
               raw = sources.at(id).rgb.to_vector<float>();
    const int width = sources.at(id).rgb.size(2);
    for (int y = 0; y < 128; ++y)
      for (int x = 0; x < 128; ++x)
        for (int c = 0; c < 3; ++c) {
          const float sy = float(y) * float(bottom - top) / 128.f,
                      sx = float(flip ? 127 - x : x) * float(right - left) / 128.f;
          const int y0 = int(std::floor(sy)), x0 = int(std::floor(sx)),
                    y1 = std::min(y0 + 1, bottom - top - 1),
                    x1 = std::min(x0 + 1, right - left - 1);
          const float dy = sy - y0, dx = sx - x0;
          auto pixel = [&](int iy, int ix) {
            return raw[((top + iy) * width + left + ix) * 3 + c];
          };
          const float a = pixel(y0, x0) + (pixel(y0, x1) - pixel(y0, x0)) * dx,
                      b = pixel(y1, x0) + (pixel(y1, x1) - pixel(y1, x0)) * dx;
          EXPECT_NEAR(actual[(y * 128 + x) * 3 + c], (a + (b - a) * dy) / 255.f, 2e-7);
        }
    if (flip) {
      const auto previous = n::select(arrays.at(split + "_joints"), 0, row - 1),
                 actual_joints = n::select(arrays.at(split + "_joints"), 0, row);
      const auto original = previous.to_vector<float>(), mirrored = actual_joints.to_vector<float>();
      const int anatomical[] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15};
      for (int joint = 0; joint < 17; ++joint) {
        const int source = anatomical[joint];
        EXPECT_EQ(mirrored[joint * 3 + 2], original[source * 3 + 2]);
        EXPECT_EQ(mirrored[joint * 3 + 1], original[source * 3 + 1]);
        EXPECT_EQ(mirrored[joint * 3], original[source * 3 + 2] > 0 ? 127.f - original[source * 3] : 0.f);
      }
      EXPECT_TRUE(n::all(n::equal(n::select(arrays.at(split + "_box_size"), 0, row),
                                  n::select(arrays.at(split + "_box_size"), 0, row - 1)))
                      .item<bool>());
    } else {
      const auto annotations = sources.at(id).annotations.to_vector<double>();
      const int person = pid - id * 10;
      const int height = sources.at(id).rgb.size(1);
      const auto joints = n::select(arrays.at(split + "_joints"), 0, row).to_vector<float>();
      for (int joint = 0; joint < 17; ++joint) {
        const int k = person * 56 + 5 + joint * 3;
        const bool labeled = annotations[k + 2] > 0;
        const double x = (annotations[k] * width - left) * 128. / (right - left);
        const double y = (annotations[k + 1] * height - top) * 128. / (bottom - top);
        EXPECT_EQ(joints[joint * 3], labeled ? float(std::clamp(x, 0., 127.)) : 0.f);
        EXPECT_EQ(joints[joint * 3 + 1], labeled ? float(std::clamp(y, 0., 127.)) : 0.f);
        EXPECT_EQ(joints[joint * 3 + 2], float(annotations[k + 2]));
      }
      const auto size = n::select(arrays.at(split + "_box_size"), 0, row).to_vector<float>();
      EXPECT_EQ(size[0], float(annotations[person * 56 + 3] * width * 128. / (right - left)));
      EXPECT_EQ(size[1], float(annotations[person * 56 + 4] * height * 128. / (bottom - top)));
    }
  }
  EXPECT_EQ(offset["fit"], 12);
  EXPECT_EQ(offset["validation"], 1);
  EXPECT_EQ(offset["test"], 14);
}
TEST(StackedHourglass, EveryTensorSkeletonAndHeatPreviewPixel) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::set_num_threads(2);
  m::PoseData d;
  d.rgb = n::full({1, 128, 128, 3}, .25);
  std::vector<float> gt(51), pred(51);
  for (int j = 0; j < 17; ++j) {
    gt[j * 3] = float(7 + j * 5);
    gt[j * 3 + 1] = float(9 + j * 4);
    gt[j * 3 + 2] = j % 3 ? 2.f : 0.f;
    pred[j * 3] = float(4 + j * 6);
    pred[j * 3 + 1] = float(11 + j * 3);
    pred[j * 3 + 2] = -1.f;
  }
  gt[0] = NAN;
  gt[1] = NAN;
  d.joints = n::from_vector(gt, {1, 17, 3});
  const auto decoded = n::from_vector(pred, {1, 17, 3}), heat = n::full({1, 32, 32, 17}, .37);
  const auto pixels = m::PosePreviewTiles(d, decoded, heat).to_vector<float>();
  const int edges[18][2] = {{5, 7},  {7, 9},   {6, 8},   {8, 10},  {5, 6},   {5, 11},
                            {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16},
                            {0, 1},  {1, 3},   {0, 2},   {2, 4},   {0, 5},   {0, 6}};
  for (int y = 0; y < 128; ++y)
    for (int x = 0; x < 128; ++x) {
      for (int kind = 0; kind < 2; ++kind) {
        const auto &values = kind ? pred : gt;
        bool covered = false;
        for (int j = 0; j < 17; ++j)
          if (kind || values[j * 3 + 2] > 0)
            covered |= std::abs(x - std::lround(values[j * 3])) <= 1 &&
                       std::abs(y - std::lround(values[j * 3 + 1])) <= 1;
        for (const auto &edge : edges) {
          const int a = edge[0], b = edge[1];
          if (!kind && (values[a * 3 + 2] <= 0 || values[b * 3 + 2] <= 0))
            continue;
          const double ax = values[a * 3], ay = values[a * 3 + 1],
                       dx = double(values[b * 3]) - ax, dy = double(values[b * 3 + 1]) - ay;
          const double t = std::clamp(
              ((x - ax) * dx + (y - ay) * dy) / std::max(dx * dx + dy * dy, 1e-30), 0., 1.);
          covered |=
              (x - ax - t * dx) * (x - ax - t * dx) + (y - ay - t * dy) * (y - ay - t * dy) <=
              1.;
        }
        for (int c = 0; c < 3; ++c) {
          const float expected =
              covered ? (c == (kind ? 0 : 1) ? 1.f : 0.f) : float(64. / 255.);
          EXPECT_NEAR(pixels[(y * 512 + (kind + 1) * 128 + x) * 3 + c], expected, 4e-8);
        }
      }
      for (int c = 0; c < 3; ++c)
        EXPECT_NEAR(pixels[(y * 512 + x) * 3 + c], 64. / 255., 4e-8);
      const double h[] = {.37f, 0., 1.f - .37f};
      for (int c = 0; c < 3; ++c)
        EXPECT_NEAR(pixels[(y * 512 + 384 + x) * 3 + c], std::lround(255. * h[c]) / 255.,
                    4e-8);
    }
}
