
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_vanilla_saliency_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.11_interpretability/vanilla_saliency.cpp"
#undef main
TEST(Saliency, EveryRawScoreSignedInputHostFiniteDifference) {
  DerivativeChecks();
}
TEST(Saliency, EveryClassifierAndInputDerivativeEndpoint) {
  GraphChecks();
}
TEST(Saliency, AllOriginalByteNormalizationMetadataAndHost49CellDescriptors) {
  namespace n = dm::nn;
  namespace m = n::models;
  const auto root =
      std::filesystem::path(std::getenv("HOME")) / ".cache/datamunge/datasets/saliency";
  std::map<std::string, n::Tensor> a;
  for (const auto &[key, v] : n::load_npz((root / "saliency_data.npz").string()))
    a.emplace(key, v);
  const auto pools = m::LoadSaliency(root);
  int total = 0;
  for (int split = 0; split < 3; ++split) {
    const std::string name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    const auto bytes=a.at(name+"_images").to_vector<int>();const auto x=pools[split].images.to_vector<float>();
    EXPECT_EQ(pools[split].labels.to_vector<int>(), a.at(name + "_labels").to_vector<int>());
    EXPECT_EQ(pools[split].source.to_vector<int>(),
              a.at(name + "_source_ids").to_vector<int>());
    for (std::size_t i = 0; i < x.size(); ++i) {
      EXPECT_EQ(x[i], float(bytes[i]) / 255.f);
      ++total;
    }
    const auto f = m::SaliencyFeatures(pools[split].images).to_vector<double>();
    for (int row = 0; row < pools[split].images.size(0); ++row)
      for (int gy = 0; gy < 7; ++gy)
        for (int gx = 0; gx < 7; ++gx) {
          double expected = 0;
          for (int y = 0; y < 4; ++y)
            for (int xx = 0; xx < 4; ++xx)
              expected += x[row * 784 + (gy * 4 + y) * 28 + gx * 4 + xx] / 16.f;
          EXPECT_EQ(f[row * 49 + gy * 7 + gx], expected);
        }
  }
  EXPECT_EQ(total, 1097600);
}
TEST(Saliency, EveryInitialRealImageLearnedGradientAndInferenceImmutability) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::set_num_threads(2);
  const auto pools = m::LoadSaliency(std::filesystem::path(std::getenv("HOME")) /
                                     ".cache/datamunge/datasets/saliency");
  const auto d = m::SaliencyRows(pools[0], {0, 1, 2, 3, 4, 5, 6, 7});
  n::ParameterStore p(1103);
  const auto out = m::SaliencyCnn(p, d.images);
  const auto before = p.flatten();
  const auto gs =
      n::grad(n::sparse_softmax_cross_entropy(out.logits, d.labels), p.parameters());
  ASSERT_EQ(gs.size(), 8);
  for (const auto &g : gs) {
    EXPECT_TRUE(n::all(n::is_finite(g)).item<bool>());
    EXPECT_GT(n::sum(n::square(g)).item(), 0);
  }
  const auto targets = n::argmax(out.logits, 1).to(n::DType::Int32);
  const auto g = m::VanillaInputGradient(
      [&](const n::Tensor &x) {
        return m::SaliencyCnn(p, x).logits;
      },
      d.images, targets);
  const auto single = m::VanillaInputGradient(
      [&](const n::Tensor &x) {
        return m::SaliencyCnn(p, x).logits;
      },
      n::slice_axis(d.images, 0, 0, 1), n::slice_axis(targets, 0, 0, 1));
  EXPECT_LE(n::max(n::abs(single - n::slice_axis(g, 0, 0, 1))).item(), 1e-6);
  EXPECT_EQ(before, p.flatten());
}
TEST(Saliency, InvalidTargetDtypeRangeAndImageGeometry) {
  namespace n = dm::nn;
  namespace m = n::models;
  EXPECT_THROW(m::SaliencyScore(n::zeros({2, 10}), n::zeros({2})), std::invalid_argument);
  EXPECT_THROW(m::SaliencyScore(n::zeros({2, 10}), n::full({2}, 10, n::DType::Int32)),
               std::invalid_argument);
  n::ParameterStore p;
  EXPECT_THROW(m::SaliencyCnn(p, n::zeros({1, 24, 24, 1})), std::invalid_argument);
}
