
#include <datamunge/image/read_image.hpp>
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_c3d_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.10_3d_and_video_convolutions/c3d.cpp"
#undef main
#include "../fixtures/nn_video_reference.hpp"
TEST(C3d, Every164SignedKernelBiasInputHostDerivative) {
  ConvChecks();
}
TEST(C3d, NegativeOddEdgesCeilPoolAndEveryAdjoint) {
  PoolChecks();
}
TEST(C3d, ActualFullSports1MAndCompactEveryEndpointAnd22Gradients) {
  nn::set_num_threads(2);
  GraphChecks(true);
}
TEST(C3d, EveryDecodedSourceFramePixelAndAnnotationIdentity) {
  namespace m = dm::nn::models;
  const auto root = std::filesystem::path("build/cnn-datasets/action_video");
  const auto pools = m::LoadVideo(root);
  std::ifstream rows(root / "clip_rows.tsv");
  std::string split, path;
  int label, person, video, segment, total = 0;
  std::array<int, 3> offset{};
  while (rows >> split >> label >> person >> video >> segment) {
    const int pool = split == "fit" ? 0 : split == "validation" ? 1 : 2;
    const int row = offset[pool]++;
    const auto &d = pools[pool];
    EXPECT_EQ(d.labels.to_vector<int>()[row], label);
    EXPECT_EQ(d.person.to_vector<int>()[row], person);
    EXPECT_EQ(d.video.to_vector<int>()[row], video);
    EXPECT_EQ(d.segment.to_vector<int>()[row], segment);
    const auto clip = nn::select(d.rgb, 0, row).to_vector<float>();
    for (int frame = 0; frame < 16; ++frame) {
      rows >> path;
      const auto image = nn::from_image(dm::image::read_image((root / path).string()));
      const auto raw = image.to_vector<float>();
      const int height = image.size(0), width = image.size(1);
      for (int y = 0; y < 32; ++y)
        for (int x = 0; x < 32; ++x)
          for (int c = 0; c < 3; ++c) {
            const double sy = y * double(height) / 32, sx = x * double(width) / 32;
            const int y0 = int(sy), x0 = int(sx), y1 = std::min(y0 + 1, height - 1),
                      x1 = std::min(x0 + 1, width - 1);
            const double dy = sy - y0, dx = sx - x0;
            auto at = [&](int yy, int xx) {
              return double(raw[(yy * width + xx) * 3 + c]);
            };
            const float expected =
                float((at(y0, x0) * (1 - dx) * (1 - dy) + at(y0, x1) * dx * (1 - dy) +
                       at(y1, x0) * (1 - dx) * dy + at(y1, x1) * dx * dy)) /
                255.f;
            EXPECT_NEAR(clip[((frame * 32 + y) * 32 + x) * 3 + c], expected, 2e-7);
            ++total;
          }
    }
  }
  EXPECT_EQ(total, 1769472);
  EXPECT_EQ(offset, (std::array<int, 3>{24, 6, 6}));
}
TEST(C3d, EveryStaticMotionReferenceFeatureNeighborScoreAndMetric) {
  namespace m = dm::nn::models;
  const auto pools = m::LoadVideo("build/cnn-datasets/action_video");
  for (bool motion : {false, true}) {
    const auto tensor = m::VideoFeatures(pools[0], motion).to_vector<double>();
    const auto host = video_reference::VideoFeatures(pools[0], motion);
    for (int row = 0; row < 24; ++row)
      for (int c = 0; c < 48; ++c)
        EXPECT_EQ(tensor[row * 48 + c], host[row][c]);
    const auto z = m::VideoNearest(pools[0], pools[2], motion);
    EXPECT_EQ(z.to_vector<float>(),
              video_reference::VideoNearest(pools[0], pools[2], motion).to_vector<float>());
    const auto score = m::ScoreSensor(z, pools[2].labels, 3);
    const auto expected = video_reference::ScoreLogits(z, pools[2].labels, 3);
    EXPECT_NEAR(score.accuracy, expected.accuracy, 1e-12);
    EXPECT_NEAR(score.nll, expected.nll, 1e-12);
    EXPECT_EQ(score.confusion, expected.confusion);
    EXPECT_EQ(score.recall, expected.recall);
    EXPECT_NEAR(score.macro_f1, expected.macro_f1, 1e-12);
  }
}
TEST(C3d, DropoutGeneratorAndUnitRgbGeometryContracts) {
  namespace m = dm::nn::models;
  nn::ParameterStore p;
  const auto x = nn::zeros({1, 16, 32, 32, 3});
  EXPECT_THROW(m::C3d(p, x, {4, 64, 3, 32}, true, nullptr, .5), std::invalid_argument);
  EXPECT_THROW(m::C3d(p, x + 2., {4, 64, 3, 32}), std::invalid_argument);
  EXPECT_THROW(m::C3d(p, nn::zeros({1, 15, 32, 32, 3}), {4, 64, 3, 32}),
               std::invalid_argument);
  nn::NoGradGuard guard;
  EXPECT_NO_THROW(m::C3d(p, x, {4, 64, 3, 32}, true, nullptr, 1.));
}
