/// @file nn_word_embeddings_tests.cpp
/// @brief Tests for embedding bags, co-occurrence counts, PMI and skip-gram negative sampling.

#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>

#include <cmath>

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {
const auto F64 = nn::DType::Float64;
}

TEST(NnWordEmbeddings, EmbeddingBagModes) {
  const Tensor table = nn::from_vector(std::vector<double>{1, 2, 3, 4, 5, 6, -1, 0}, {4, 2});
  const Tensor ids = nn::from_vector(std::vector<std::int64_t>{0, 2, 1, 3, 3}, {5});
  const Tensor off = nn::from_vector(std::vector<std::int64_t>{0, 2, 2}, {3});  // bags {0, 2}, {}, {1, 3, 3}
  const Tensor s = nn::embedding_bag(table, ids, off, nn::BagMode::Sum);
  EXPECT_DOUBLE_EQ(s.at({0, 0}), 6);
  EXPECT_DOUBLE_EQ(s.at({0, 1}), 8);
  EXPECT_DOUBLE_EQ(s.at({1, 0}), 0);
  EXPECT_DOUBLE_EQ(s.at({2, 0}), 1);
  EXPECT_DOUBLE_EQ(s.at({2, 1}), 4);
  const Tensor m = nn::embedding_bag(table, ids, off, nn::BagMode::Mean);
  EXPECT_DOUBLE_EQ(m.at({0, 0}), 3);
  EXPECT_DOUBLE_EQ(m.at({1, 1}), 0);
  EXPECT_NEAR(m.at({2, 1}), 4.0 / 3, 1e-15);
  const Tensor x = nn::embedding_bag(table, ids, off, nn::BagMode::Max);
  EXPECT_DOUBLE_EQ(x.at({0, 0}), 5);
  EXPECT_DOUBLE_EQ(x.at({0, 1}), 6);
  EXPECT_DOUBLE_EQ(x.at({1, 0}), 0);
  EXPECT_DOUBLE_EQ(x.at({2, 0}), 3);
  EXPECT_DOUBLE_EQ(x.at({2, 1}), 4);
  const Tensor w = nn::from_vector(std::vector<double>{1, 3, 1, 1, 2}, {5});
  const Tensor wm = nn::embedding_bag(table, ids, off, nn::BagMode::Mean, w);
  EXPECT_DOUBLE_EQ(wm.at({0, 0}), (1 + 15) / 4.0);
  // Gradient of the sum mode: each row receives the number of times it was used.
  Tensor t = table.clone();
  t.requires_grad_();
  const Tensor g = nn::grad(nn::sum(nn::embedding_bag(t, ids, off, nn::BagMode::Sum)), {t})[0];
  EXPECT_DOUBLE_EQ(g.at({3, 0}), 2);
  EXPECT_DOUBLE_EQ(g.at({0, 1}), 1);
  EXPECT_THROW((void)nn::embedding_bag(table, ids, nn::from_vector(std::vector<std::int64_t>{1}, {1})), std::invalid_argument);
  EXPECT_THROW((void)nn::embedding_bag(table, ids, off, nn::BagMode::Max, w), std::invalid_argument);
}

TEST(NnWordEmbeddings, CooccurrenceAndPmi) {
  const Tensor c = nn::cooccurrence_matrix({{0, 1, 2}, {2, 2}}, 3, 1);
  EXPECT_DOUBLE_EQ(c.at({0, 1}), 1);
  EXPECT_DOUBLE_EQ(c.at({1, 0}), 1);
  EXPECT_DOUBLE_EQ(c.at({0, 2}), 0);
  EXPECT_DOUBLE_EQ(c.at({2, 2}), 2);
  const Tensor h = nn::cooccurrence_matrix({{0, 1, 2}}, 3, 2, true);
  EXPECT_DOUBLE_EQ(h.at({0, 2}), 0.5);
  EXPECT_THROW((void)nn::cooccurrence_matrix({{0, 5}}, 3, 1), std::invalid_argument);
  // PMI against the definition, with a shift and context smoothing.
  const Tensor counts = nn::from_vector(std::vector<double>{4, 1, 0, 2, 2, 3}, {2, 3});
  const Tensor p = nn::pmi(counts, std::log(2.0), false, 0.75);
  const double col[3] = {6, 3, 3}, row[2] = {5, 7};
  const double z = std::pow(6, 0.75) + 2 * std::pow(3, 0.75);
  EXPECT_NEAR(p.at({0, 0}), std::log(4 / (row[0] * std::pow(col[0], 0.75) / z)) - std::log(2.0), 1e-14);
  EXPECT_NEAR(p.at({1, 2}), std::log(3 / (row[1] * std::pow(col[2], 0.75) / z)) - std::log(2.0), 1e-14);
  EXPECT_TRUE(std::isinf(p.at({0, 2})) && p.at({0, 2}) < 0);
  const Tensor pp = nn::pmi(counts, 0.0, true);
  EXPECT_EQ(pp.at({0, 2}), 0.0);
  EXPECT_GE(nn::min(pp).item(), 0.0);
}

TEST(NnWordEmbeddings, NegativeSamplingLossAndSampler) {
  nn::Generator g(5);
  const Tensor t = nn::normal({3, 4}, 0, 1, g, F64), c = nn::normal({3, 4}, 0, 1, g, F64), n = nn::normal({3, 2, 4}, 0, 1, g, F64);
  double ref = 0;
  for (std::int64_t b = 0; b < 3; ++b) {
    double pos = 0;
    for (std::int64_t k = 0; k < 4; ++k) pos += t.at({b, k}) * c.at({b, k});
    ref += -std::log(1 / (1 + std::exp(-pos)));
    for (std::int64_t j = 0; j < 2; ++j) {
      double neg = 0;
      for (std::int64_t k = 0; k < 4; ++k) neg += t.at({b, k}) * n.at({b, j, k});
      ref += -std::log(1 / (1 + std::exp(neg)));
    }
  }
  EXPECT_NEAR(nn::negative_sampling_loss(t, c, n).item(), ref / 3, 1e-13);
  EXPECT_THROW((void)nn::negative_sampling_loss(t, c, nn::zeros({3, 2, 5}, F64)), std::invalid_argument);
  // Frequencies of sample_negatives follow counts^0.75.
  const Tensor counts = nn::from_vector(std::vector<double>{1, 16, 0, 81}, {4});
  const Tensor s = nn::sample_negatives(counts, {200000}, 0.75, g);
  double f[4] = {0, 0, 0, 0};
  for (std::int64_t i = 0; i < 200000; ++i) f[static_cast<int>(s.at({i}))] += 1.0 / 200000;
  const double z = 1 + 8 + 27;
  EXPECT_NEAR(f[0], 1 / z, 0.003);
  EXPECT_NEAR(f[1], 8 / z, 0.004);
  EXPECT_EQ(f[2], 0.0);
  EXPECT_NEAR(f[3], 27 / z, 0.004);
}
