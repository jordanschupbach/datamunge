
#include <gtest/gtest.h>

#include <dm/dm.hpp>

#include "../fixtures/nn_guided_backprop_checks.hpp"
TEST(GuidedBackprop, AllNineSignZeroRulesNonlinearVjpAndActualForwardFiniteDifference) {
  GateChecks();
}
TEST(GuidedBackprop, All2000RealForwardLogitsUnchangedAndThreeGateCount) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::set_num_threads(2);
  n::ParameterStore p(1103);
  m::SaliencyCnn(p, n::zeros({1, 28, 28, 1}));
  const auto pools = m::LoadSaliency(std::filesystem::path(dm::datasets::data_home()) / "saliency");
  const auto before = p.flatten();
  n::NoGradGuard guard;
  const auto     ordinary = m::SaliencyCnn(p, pools[2].images).logits;
  const auto     count    = m::GuidedReluCount().load();
  const auto     guided   = m::GuidedSaliencyCnn(p, pools[2].images).logits;
  EXPECT_EQ(m::GuidedReluCount().load() - count, 3);
  EXPECT_EQ(n::max(n::abs(ordinary - guided)).item(), 0);
  EXPECT_EQ(before, p.flatten());
}
