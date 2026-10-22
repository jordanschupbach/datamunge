// Extra gtests for library additions made while porting tflow examples (P1).
// Each port group appends its own TESTs; suite names are prefixed Nn.

#include <gtest/gtest.h>

#include <cmath>

#include <datamunge/nn/nn.hpp>

namespace nn = datamunge::nn;

// layers group (activations port): at exactly x = 0 the SELU and ELU
// derivatives take the linear branch, as TF SeluGrad / EluGrad do.
TEST(NnActivationGradAtZero, SeluAndEluUseLinearBranch) {
  const double scale = 1.0507009873554804934193349852946;
  nn::Tensor   x     = nn::tensor({-1.0, 0.0, 1.0}).requires_grad_();
  const auto   ds    = nn::grad(nn::sum(nn::selu(x)), {x})[0].to_vector<double>();
  EXPECT_NEAR(ds[1], scale, 1e-6);
  EXPECT_NEAR(ds[0], scale * 1.6732632423543772848170429916717 * std::exp(-1.0), 1e-6);
  nn::Tensor x2 = nn::tensor({-1.0, 0.0, 1.0}).requires_grad_();
  const auto de = nn::grad(nn::sum(nn::elu(x2, 0.5)), {x2})[0].to_vector<double>();
  EXPECT_NEAR(de[1], 1.0, 1e-6);
  EXPECT_NEAR(de[0], 0.5 * std::exp(-1.0), 1e-6);
}

// data group (idx_loader port): IDX dimensions whose product wraps around
// 64 bits (2^22 * 2^22 * 2^20 = 2^64 == 0 elements) are rejected, not read
// as an empty array with a bogus shape.
#include <cstdio>
#include <stdexcept>
#include <string>
TEST(NnReadIdxOverflow, RejectsWrappingDimensions) {
  const std::string path = ::testing::TempDir() + "nn_idx_overflow.idx";
  {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    const unsigned char bytes[] = {0, 0, 8, 3, 0, 0x40, 0, 0, 0, 0x40, 0, 0, 0, 0x10, 0, 0};
    std::fwrite(bytes, 1, sizeof bytes, f);
    std::fclose(f);
  }
  try {
    (void)nn::read_idx(path);
    FAIL() << "wrapping IDX dimensions were accepted";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("overflow"), std::string::npos) << e.what();
  }
  std::remove(path.c_str());
}

// optim group (gradient_clipping_accumulation port): sample-weighted
// accumulation reproduces the full-batch mean gradient; bad weights and
// empty accumulators are rejected without changing state.
TEST(NnGradientAccumulatorWeights, WeightedMeanMatchesFullBatch) {
  nn::GradientAccumulator acc;
  EXPECT_THROW((void)acc.mean(), std::logic_error);
  acc.add({nn::tensor({1.0, 2.0})}, 2.0);
  acc.add({nn::tensor({4.0, -1.0})}, 3.0);
  EXPECT_THROW(acc.add({nn::tensor({9.0, 9.0})}, 0.0), std::invalid_argument);
  EXPECT_THROW(acc.add({nn::tensor({9.0, 9.0})}, -1.0), std::invalid_argument);
  EXPECT_EQ(acc.count(), 2);
  EXPECT_DOUBLE_EQ(acc.total_weight(), 5.0);
  const auto mean = acc.mean()[0].to_vector<double>();
  EXPECT_NEAR(mean[0], (2.0 * 1 + 3.0 * 4) / 5, 1e-6);
  EXPECT_NEAR(mean[1], (2.0 * 2 - 3.0 * 1) / 5, 1e-6);
  const auto sum = acc.sum()[0].to_vector<double>();
  EXPECT_NEAR(sum[0], 14.0, 1e-6);
  acc.reset();
  EXPECT_EQ(acc.count(), 0);
  EXPECT_DOUBLE_EQ(acc.total_weight(), 0.0);
  acc.add({nn::tensor({3.0})});
  acc.add({nn::tensor({5.0})});
  EXPECT_NEAR(acc.mean()[0].item(), 4.0, 1e-6);  // unit weights: plain mean
}
