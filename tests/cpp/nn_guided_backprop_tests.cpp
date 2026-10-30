
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_guided_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.11_interpretability/guided_backprop.cpp"
#undef main
TEST(GuidedBackprop, AllNineSignZeroRulesNonlinearVjpAndActualForwardFiniteDifference) {
  GateChecks();
}
TEST(GuidedBackprop, All2000RealForwardLogitsUnchangedAndThreeGateCount) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::set_num_threads(2);
  n::ParameterStore p(1103);
  m::SaliencyCnn(p, n::zeros({1, 28, 28, 1}));
  for (const auto &[key, v] :
       n::load_npz("output/nn/quick/models/vanilla_saliency/model_arrays.npz"))
    p.set(key, v);
  const auto pools = m::LoadSaliency(std::filesystem::path(std::getenv("HOME")) /
                                     ".cache/datamunge/datasets/saliency");
  const auto before = p.flatten();
  n::NoGradGuard guard;
  const auto ordinary = m::SaliencyCnn(p, pools[2].images).logits;
  const auto count = m::GuidedReluCount().load();
  const auto guided = m::GuidedSaliencyCnn(p, pools[2].images).logits;
  EXPECT_EQ(m::GuidedReluCount().load() - count, 3);
  EXPECT_EQ(n::max(n::abs(ordinary - guided)).item(), 0);
  EXPECT_EQ(before, p.flatten());
}
