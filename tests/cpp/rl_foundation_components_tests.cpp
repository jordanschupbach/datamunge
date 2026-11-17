#include <gtest/gtest.h>

#include <dm/dm.hpp>
#include <limits>
namespace {
namespace f = dm::rl::foundations;
namespace n = dm::nn;
TEST(RlFoundations, TableOwnsCopiesSnapshotsAndMoves) {
  f::QTable q(3, 2, 1);
  q(1, 1)    = 4;
  auto copy  = q;
  copy(1, 1) = 7;
  EXPECT_EQ(q(1, 1), 4);
  auto snapshot                 = q.tensor();
  snapshot.data_ptr<float>()[3] = 9;
  EXPECT_EQ(q(1, 1), 4);
  f::QTable assigned(1, 1);
  assigned       = q;
  assigned(0, 0) = 3;
  EXPECT_EQ(q(0, 0), 1);
  auto moved = std::move(assigned);
  EXPECT_EQ(moved(0, 0), 3);
  EXPECT_THROW(assigned.Row(0), std::logic_error);
  EXPECT_THROW(f::QTable bad(0, 1), std::invalid_argument);
  q.S = 4;
  EXPECT_THROW(q.Row(0), std::logic_error);
}
TEST(RlFoundations, TieAndEpsilonDrawSchedule) {
  float           row[] = {1, 3, 3};
  std::mt19937_64 a(37), b(37);
  for (int i = 0; i < 100; ++i)
    EXPECT_EQ(f::ArgMaxRandomTie(row, 3, a), 1 + int(b() % 2));
  EXPECT_EQ(a(), b());
  f::QTable q(1, 3);
  q(0, 1) = 3;
  q(0, 2) = 3;
  std::mt19937_64 c(11), d(11);
  for (int i = 0; i < 100; ++i) {
    (void)std::uniform_real_distribution<float>(0, 1)(d);
    EXPECT_EQ(q.EpsilonGreedy(0, 0, c), 1 + int(d() % 2));
  }
  EXPECT_EQ(c(), d());
  q(0, 1) = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(q.Max(0), std::invalid_argument);
}
TEST(RlFoundations, BanditPosteriorStreamsAndExtremeUcb) {
  EXPECT_EQ(f::UcbArm({1, 0}, {3.e35f, 0}, 2, 1), 1);
  EXPECT_EQ(f::UcbArm({1, 1}, {-3.e35f, -2.e35f}, 2, 0), 1);
  std::mt19937_64    a(71), b(71);
  std::vector<int>   counts{0, 1, 7};
  std::vector<float> sums{0, 1, -2};
  for (int trial = 0; trial < 100; ++trial) {
    int   best = 0;
    float top  = -INFINITY;
    for (int arm = 0; arm < 3; ++arm) {
      float draw = std::normal_distribution<float>(
          sums[arm] / (counts[arm] + 1), 1.f / std::sqrt(counts[arm] + 1.f))(b);
      if (draw > top)
        top = draw, best = arm;
    }
    EXPECT_EQ(f::ThompsonGaussianArm(counts, sums, a), best);
  }
  EXPECT_EQ(a(), b());
  std::vector<float> wins{0, 2, 7}, loss{3, 1, 0};
  for (int trial = 0; trial < 100; ++trial) {
    int   best = 0;
    float top  = -1;
    for (int arm = 0; arm < 3; ++arm) {
      auto x     = std::gamma_distribution<float>(wins[arm] + 1, 1)(b),
           y     = std::gamma_distribution<float>(loss[arm] + 1, 1)(b);
      float draw = x / (x + y);
      if (draw > top)
        top = draw, best = arm;
    }
    EXPECT_EQ(f::ThompsonBetaArm(wins, loss, a), best);
  }
  EXPECT_EQ(a(), b());
  EXPECT_THROW(f::UcbArm({-1}, {0}, 1, 1), std::invalid_argument);
}
TEST(RlFoundations, TilesIndependentCoordinateFormulaAndGatherDerivative) {
  f::TileCoder coder(8, 8, 3, -1.2f, .6f, -.07f, .07f);
  for (float x : {-1.3f, -1.2f, -.5f, .6f, .7f})
    for (float v : {-.08f, -.07f, 0.f, .07f, .08f})
      for (int a = 0; a < 3; ++a) {
        auto idx = coder.Active(x, v, a);
        for (int t = 0; t < 8; ++t) {
          float u = (x + 1.2f) / (.6f + 1.2f) * 8, z = (v + .07f) / (.07f + .07f) * 8;
          int   c = std::clamp(int(u + t / 8.f), 0, 7), r = std::clamp(int(z + 3.f * t / 8), 0, 7);
          EXPECT_EQ(idx[t], ((a * 8 + t) * 8 + c) * 8 + r);
        }
      }
  auto idx = coder.Active(-.5, 0, 1);
  auto w   = n::zeros({1536}).requires_grad_();
  auto g   = n::grad(n::sum(n::gather(w, n::from_vector(idx), 0)), {w})[0].to_vector<float>();
  for (int i = 0; i < 1536; ++i)
    EXPECT_EQ(g[i], std::find(idx.begin(), idx.end(), i) != idx.end() ? 1 : 0);
  EXPECT_THROW(coder.Active(NAN, 0, 0), std::invalid_argument);
}
TEST(RlFoundations, PseudoCountEmpiricalAndPriorEvidence) {
  for (int total : {10, 100, 10000})
    for (int count : {0, 1, total / 2, total - 1})
      EXPECT_NEAR(
          f::PseudoCount(double(count) / total, double(count + 1) / (total + 1)),
          count,
          1e-7 * (count + 1));
  f::FactorizedDensity density({4});
  int                  counts[4] = {};
  for (int value : {0, 1, 0, 3, 0, 1, 2})
    EXPECT_NEAR(density.Observe({value}), counts[value]++ + .5, 1e-10);
  auto copy = density;
  copy.Observe({0});
  EXPECT_NE(copy.Probability({0}), density.Probability({0}));
  auto before = density.Probability({0});
  EXPECT_THROW(density.Observe({4}), std::out_of_range);
  EXPECT_EQ(before, density.Probability({0}));
  EXPECT_THROW(f::PseudoCount(.3, .2), std::invalid_argument);
  EXPECT_TRUE(std::isinf(f::PseudoCount(.2, .2)));
  EXPECT_EQ(f::CountBonus(INFINITY, 1), 0);
}
TEST(RlFoundations, SparseRoomsTransitionAndReward) {
  dm::rl::environments::SparseRooms env;
  EXPECT_EQ(env.num_states(), 81);
  EXPECT_TRUE(env.IsWall(4, 4));
  EXPECT_FALSE(env.IsWall(4, 1));
  EXPECT_FALSE(env.IsWall(7, 4));
  auto blocked = env.Apply(3, 3, 2);
  EXPECT_EQ(blocked.row, 3);
  EXPECT_EQ(blocked.col, 3);
  EXPECT_EQ(blocked.reward, 0);
  auto goal = env.Apply(8, 7, 1);
  EXPECT_TRUE(goal.terminal);
  EXPECT_EQ(goal.reward, 1);
}
TEST(RlFoundations, MaterializedModelIncludesTerminalRewardOnly) {
  dm::rl::environments::CliffWalking env;
  auto                               model = f::BuildMdpModel(env);
  auto                               p = model.P.to_vector<float>(), r = model.R.to_vector<float>();
  for (int s = 0; s < model.S; ++s)
    for (int a = 0; a < model.A; ++a) {
      double              reward = 0;
      std::vector<double> row(model.S);
      for (auto t : env.transitions(s, a)) {
        reward += t.prob * t.reward;
        if (!t.terminal)
          row[t.next_state] += t.prob;
      }
      EXPECT_NEAR(r[s * model.A + a], reward, 1e-6);
      for (int k = 0; k < model.S; ++k)
        EXPECT_NEAR(p[(s * model.A + a) * model.S + k], row[k], 1e-7);
    }
}
TEST(RlFoundations, BellmanPolicyResidualAndLiveDerivatives) {
  auto P      = n::from_vector(std::vector<double>{.2, .3, .1, .2}, {2, 1, 2}).requires_grad_();
  auto R      = n::from_vector(std::vector<double>{2, 1}, {2, 1}).requires_grad_();
  auto policy = n::from_vector(std::vector<int>{0, 0});
  auto V      = f::EvaluateMdpPolicy(P, R, policy, .9);
  auto backup = n::squeeze(f::BellmanActionValues(P, R, V, .9), {1});
  EXPECT_LT(n::max(n::abs(V - backup)).item(), 1e-12);
  auto grad = n::grad(n::sum(V), {P, R});
  for (const auto& t : grad)
    for (double v : t.to_vector<double>())
      EXPECT_TRUE(std::isfinite(v));
  auto values = V.to_vector<double>();
  EXPECT_NEAR(values[0], (2 * .82 + .27) / (.82 * .82 - .27 * .09), 1e-12);
  auto x = n::from_vector(std::vector<double>{1, 2}, {2}).requires_grad_();
  auto g = n::grad(n::sum(f::BellmanActionValues(P, R, x, .9)), {x})[0].to_vector<double>();
  EXPECT_NEAR(g[0], .27, 1e-12);
  EXPECT_NEAR(g[1], .45, 1e-12);
}
TEST(RlFoundations, NeuralPolicyFiniteDifferencesEveryInputAndParameter) {
  std::vector<n::Tensor> t{
      n::from_vector(std::vector<double>{.2, -.1, .3, .4}, {2, 2}).requires_grad_(),
      n::from_vector(std::vector<double>{.1, .2, -.2, .3}, {2, 2}).requires_grad_(),
      n::from_vector(std::vector<double>{.01, -.02}, {2}).requires_grad_(),
      n::from_vector(std::vector<double>{.2, -.3, .4, .1}, {2, 2}).requires_grad_(),
      n::from_vector(std::vector<double>{.1, -.1}, {2}).requires_grad_()};
  auto actions = n::from_vector(std::vector<int>{0, 1});
  auto loss    = [&](const std::vector<n::Tensor>& p) {
    return n::sum(
        f::CategoricalActionLogProbability(
            f::TanhCategoricalPolicy(p[0], p[1], p[2], p[3], p[4]).log_probability, actions));
  };
  auto grad = n::grad(loss(t), t);
  for (size_t j = 0; j < t.size(); ++j)
    for (int64_t i = 0; i < t[j].numel(); ++i) {
      auto plus = t, minus = t;
      plus[j]  = t[j].detach().clone();
      minus[j] = t[j].detach().clone();
      plus[j].data_ptr<double>()[i] += 1e-5;
      minus[j].data_ptr<double>()[i] -= 1e-5;
      double numerical = (loss(plus).item() - loss(minus).item()) / 2e-5;
      EXPECT_NEAR(grad[j].to_vector<double>()[i], numerical, 1e-9);
    }
}
TEST(RlFoundations, CategoricalSamplerAndObservationOwnership) {
  float           p[] = {.1f, .2f, .7f};
  std::mt19937_64 a(7), b(7);
  for (int i = 0; i < 100; ++i) {
    float u = std::uniform_real_distribution<float>(0, 1)(b), sum = 0;
    int   expected = 2;
    for (int k = 0; k < 3; ++k) {
      sum += p[k];
      if (u < sum) {
        expected = k;
        break;
      }
    }
    EXPECT_EQ(f::SampleCategoricalAction(p, 3, a), expected);
  }
  EXPECT_EQ(a(), b());
  std::vector<dm::rl::environments::Obs> rows{{1, 2}, {3, 4}};
  auto                                   packed = f::ObservationMatrix(rows);
  rows[0][0]                                    = 9;
  EXPECT_EQ(packed.to_vector<float>()[0], 1);
  EXPECT_THROW(f::ObservationMatrix({{1}, {1, 2}}), std::invalid_argument);
  EXPECT_EQ(f::ObservationMatrix({}).shape(), (n::Shape{0, 0}));
}
TEST(RlFoundations, AttentionConcentrationAndRemovalIndependentNorm) {
  auto weights = n::from_vector(std::vector<float>{.5, .5, 0, 0, 1, 0}, {3, 2});
  auto stats   = n::models::AttentionConcentration(weights);
  auto support = stats.effective_support.to_vector<float>();
  EXPECT_NEAR(support[0], 2, 1e-6);
  EXPECT_EQ(support[1], 0);
  EXPECT_EQ(support[2], 1);
  auto q = n::ones({1, 3, 2}), k = n::ones({1, 2, 2}),
       v     = n::from_vector(std::vector<float>{1, 2, 3, 6}, {1, 2, 2});
  auto  mask = n::zeros({1, 3, 2}, n::DType::Bool);
  bool* bits = mask.data_ptr<bool>();
  bits[0] = bits[1] = bits[4] = true;
  auto result                 = n::models::AttentionKeyRemoval(q, k, v, mask);
  auto effects                = result.effects.to_vector<float>();
  EXPECT_NEAR(effects[0], std::sqrt(5.f), 1e-6);
  EXPECT_NEAR(effects[1], std::sqrt(5.f), 1e-6);
  EXPECT_EQ(effects[2], 0);
  EXPECT_EQ(effects[3], 0);
  EXPECT_NEAR(effects[4], std::sqrt(5.f), 1e-6);
  EXPECT_EQ(effects[5], 0);
  EXPECT_FALSE(result.output.requires_grad());
  auto same  = n::ones({1, 2, 2});
  auto equal = n::models::AttentionKeyRemoval(q, k, same, mask).effects.to_vector<float>();
  EXPECT_EQ(equal[0], 0);
  EXPECT_EQ(equal[1], 0);
  EXPECT_NEAR(equal[4], std::sqrt(2.f), 1e-6);
}
} // namespace
