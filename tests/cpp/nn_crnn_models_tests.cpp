#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main cnn_crnn_original_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.8_other_vision_tasks/crnn_ctc.cpp"
#undef main
TEST(CrnnTensor, ExhaustiveRepeatedCtcEveryLogitJacobian) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(CtcChecks());
}
TEST(CrnnTensor, ExhaustiveVariableTimeMeanAndEveryFiniteDifference) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(VariableCtcChecks());
}
TEST(CrnnTensor, FullGraphEveryConvolutionShapeAndLearnedGradient) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(FullChecks());
}

TEST(CrnnTensor, CompactGraphAll36GradientsAndDoubleDirectionalProjection) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::ParameterStore p(307), state(311);
  const auto x = (.5 + .4 * n::sin(n::arange(0, 2 * 32 * 100, 1, n::DType::Float64) * .17))
                     .to(n::DType::Float32)
                     .reshape({2, 32, 100, 1})
                     .requires_grad_();
  const auto result = m::Crnn(p, state, x, {8, 32}, true, false);
  const auto labels = n::tensor({10, 11, 12, 12}, {2, 2}, n::DType::Int32);
  auto leaves = p.parameters();
  leaves.push_back(x);
  const auto g =
      n::grad(n::mean(n::ctc_loss(result.logits, labels, n::full({2}, 2, n::DType::Int32),
                                  n::full({2}, 26, n::DType::Int32))),
              leaves);
  EXPECT_EQ(p.size(), 36);
  EXPECT_EQ(state.size(), 6);
  EXPECT_EQ(result.logits.shape(), (n::Shape{26, 2, 37}));
  for (const auto &v : g) {
    EXPECT_TRUE(n::all(n::less(n::abs(v), INFINITY)).item<bool>());
    EXPECT_GT(n::sum(n::square(v.to(n::DType::Float64))).item(), 0);
  }
  n::ParameterStore recurrent(5, n::DType::Float64);
  const auto input = n::ones({2, 3, 4}, n::DType::Float64).requires_grad_();
  const auto projected = m::CrnnBiLstm(recurrent, "b", input, 5, 6);
  const auto dx = n::grad(n::sum(n::square(projected)), {input})[0];
  EXPECT_EQ(projected.dtype(), n::DType::Float64);
  EXPECT_EQ(projected.shape(), (n::Shape{2, 3, 6}));
  EXPECT_GT(n::sum(n::abs(dx)).item(), 0);
}
TEST(CrnnTensor, IndependentCropEveryGrayResizePixelAndMetadata) {
  namespace n = dm::nn;
  namespace m = n::models;
  const auto root = std::filesystem::temp_directory_path() / "dm-ocr-native-crops";
  std::filesystem::create_directories(root);
  const auto raw = n::arange(0, 36).reshape({3, 4, 3});
  n::save_png(raw, (root / "000000000009.jpg").string(), 0, 255);
  const auto d = m::ReadOcrCrops(root, {{9, 17, "ab1", {1.2, .2, 2.1, 1.7}, 4, 3}});
  const auto pixels = d.pixels.to_vector<float>();
  EXPECT_EQ(m::OcrText(d, 0), "ab1");
  EXPECT_EQ(d.image_ids.item<int>(), 9);
  EXPECT_EQ(d.annotation_ids.item<int>(), 17);
  float gray[2][3];
  for (int y = 0; y < 2; ++y)
    for (int x = 0; x < 3; ++x) {
      const int i = (y * 4 + x + 1) * 3;
      gray[y][x] =
          float(.299 * (i / 255.) + .587 * ((i + 1) / 255.) + .114 * ((i + 2) / 255.));
    }
  for (int y = 0; y < 32; ++y)
    for (int x = 0; x < 100; ++x) {
      const double sy = std::clamp((y + .5) * 2 / 32 - .5, 0., 1.),
                   sx = std::clamp((x + .5) * 3 / 100 - .5, 0., 2.);
      const int y0 = int(std::floor(sy)), x0 = int(std::floor(sx));
      double expected = 0;
      for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx)
          expected += (dy ? sy - y0 : 1 - sy + y0) * (dx ? sx - x0 : 1 - sx + x0) *
                      gray[std::min(y0 + dy, 1)][std::min(x0 + dx, 2)];
      EXPECT_NEAR(pixels[y * 100 + x], expected, 1e-7);
    }
  const auto other = m::ReadOcrCrops(root, {{9, 18, "coffee", {1.2, .2, 2.1, 1.7}, 4, 3}});
  EXPECT_EQ(pixels, other.pixels.to_vector<float>());
  EXPECT_THROW(m::ReadOcrCrops(root, {{9, 17, "x", {-20., 0., 1., 2.}, 4, 3}}),
               std::invalid_argument);
  EXPECT_THROW(m::ReadOcrCrops(root, {{9, 17, "x", {0., 0., 2., 2.}, 5, 3}}),
               std::invalid_argument);
  std::filesystem::remove(root / "000000000009.jpg");
  std::filesystem::remove(root);
}
TEST(CrnnTensor, TensorEditDpEmptyRepeatedWordsAndGreedyTieRules) {
  namespace n = dm::nn;
  namespace m = n::models;
  const std::vector<std::string> words = {"", "a", "aa", "ab", "ba", "coffee", "cofee"};
  for (const auto &a : words)
    for (const auto &b : words) {
      std::vector<int> row(b.size() + 1);
      std::iota(row.begin(), row.end(), 0);
      for (size_t i = 0; i < a.size(); ++i) {
        auto old = row;
        row[0] = i + 1;
        for (size_t j = 0; j < b.size(); ++j)
          row[j + 1] = std::min({old[j + 1] + 1, row[j] + 1, old[j] + int(a[i] != b[j])});
      }
      EXPECT_EQ(m::OcrDistance(a, b), row.back());
    }
  const auto decoded = m::OcrDecode(n::zeros({26, 2, 37}));
  EXPECT_EQ(decoded, (std::vector<std::string>{"0", "0"}));
  EXPECT_THROW(m::CtcCollapse({36}, 37), std::invalid_argument);
}
TEST(CrnnTensor, PreparedHumanSourceImagesDisjointAndTargetsAligned) {
  namespace n = dm::nn;
  namespace m = n::models;
  const auto root = std::filesystem::current_path() / "build/cnn-datasets/ocr";
  if (!std::filesystem::exists(root / "ocr.npz"))
    GTEST_SKIP() << "Optional prepared human COCO-Text cache absent";
  const auto data = m::LoadOcr(root);
  std::set<int> previous, annotations;
  const int expected[] = {43, 7, 6};
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(data[i].pixels.size(0), expected[i]);
    std::set<int> current;
    const auto ids = data[i].image_ids.to_vector<int>(),
               notes = data[i].annotation_ids.to_vector<int>();
    std::vector<std::string> truth;
    for (int j = 0; j < expected[i]; ++j) {
      EXPECT_FALSE(previous.contains(ids[j]));
      EXPECT_TRUE(annotations.insert(notes[j]).second);
      current.insert(ids[j]);
      truth.push_back(m::OcrText(data[i], j));
    }
    previous.insert(current.begin(), current.end());
    const auto score = m::ScoreTensorOcr(data[i], truth);
    EXPECT_EQ(score.exact.item<int>(), expected[i]);
    EXPECT_EQ(score.errors.item<int>(), 0);
    EXPECT_EQ(score.character_error_rate.item(), 0);
  }
  EXPECT_EQ(annotations.size(), 56);
  EXPECT_EQ(previous.size(), 16);
}
