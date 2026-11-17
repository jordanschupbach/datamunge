#include <gtest/gtest.h>

#include <dm/dm.hpp>
namespace {
namespace n = dm::nn;
namespace m = n::models;
namespace p = dm::rl::planning;
namespace e = dm::rl::environments;
TEST(PlanningComponents, SearchClonesTerminalBackupAndWinningMove) {
  e::TicTacToe game;
  for (int a : {0, 3, 1, 4})
    game.Apply(a);
  const auto      before = game.Render();
  std::mt19937_64 rng(11);
  p::Mcts         search(p::RolloutEvaluator(&rng));
  auto            visits = search.Search(game, 2000, rng);
  EXPECT_EQ(std::accumulate(visits.begin(), visits.end(), 0), 2000);
  EXPECT_EQ(std::max_element(visits.begin(), visits.end()) - visits.begin(), 2);
  EXPECT_EQ(before, game.Render());
  EXPECT_GT(search.last_root_value(), .8);
  game.Apply(2);
  EXPECT_EQ(p::TerminalValue(game), -1);
  auto terminal = search.Search(game, 4, rng);
  EXPECT_EQ(std::accumulate(terminal.begin(), terminal.end(), 0), 0);
  EXPECT_EQ(search.last_root_value(), -1);
}
TEST(PlanningComponents, SearchZeroBudgetAndInvalidPriors) {
  e::TicTacToe    game;
  std::mt19937_64 rng(7), unchanged(7);
  int             calls = 0;
  p::Mcts         search([&](const p::Game& g) {
    ++calls;
    return p::Evaluation{std::vector<float>(g.num_actions(), 0), .5};
  });
  auto            visits = search.Search(game, 0, rng);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(search.last_root_value(), 0);
  EXPECT_EQ(rng(), unchanged());
  p::Mcts bad(
      [](const p::Game& g) { return p::Evaluation{std::vector<float>(g.num_actions(), -1), 0}; });
  EXPECT_THROW(bad.Search(game, 1, rng), std::invalid_argument);
  EXPECT_THROW(p::RolloutEvaluator(nullptr), std::invalid_argument);
  EXPECT_THROW(search.Search(game, -1, rng), std::invalid_argument);
}
TEST(PlanningComponents, SearchNoiseDeterminismAndCallerStream) {
  e::TicTacToe game;
  auto         eval = [](const p::Game& g) {
    return p::Evaluation{std::vector<float>(g.num_actions(), 1), 0};
  };
  p::Mcts         a(eval), b(eval);
  std::mt19937_64 x(91), y(91);
  EXPECT_EQ(a.Search(game, 70, x, true), b.Search(game, 70, y, true));
  EXPECT_EQ(x(), y());
}
TEST(PlanningComponents, TransitionsCopyRowsAndUseExactlyOneModuloDrawPerRow) {
  p::TransitionSet data;
  e::Obs           row{1, 2};
  data.Add(row, {3}, {4, 5}, 6);
  row[0] = 99;
  data.Add({7, 8}, {9}, {10, 11}, 12);
  std::mt19937_64 x(31), y(31);
  auto            batch = data.Sample(20, x);
  auto            s = batch.state.to_vector<float>(), a = batch.action.to_vector<float>(),
       r = batch.reward.to_vector<float>();
  for (int i = 0; i < 20; ++i) {
    int index = y() % 2;
    EXPECT_EQ(s[2 * i], index ? 7 : 1);
    EXPECT_EQ(a[i], index ? 9 : 3);
    EXPECT_EQ(r[i], index ? 12 : 6);
  }
  EXPECT_EQ(x(), y());
  auto copy = data;
  copy.Add({0, 0}, {0}, {0, 0}, 0);
  EXPECT_EQ(data.size(), 2);
  EXPECT_EQ(copy.size(), 3);
  EXPECT_THROW(data.Add({1}, {1}, {2}, 0), std::invalid_argument);
  EXPECT_EQ(data.size(), 2);
  EXPECT_THROW(data.Add({1, 2}, {1}, {2, 3}, NAN), std::invalid_argument);
  EXPECT_THROW(data.Sample(0, x), std::invalid_argument);
}
TEST(PlanningComponents, DynamicsSamplingAndAnalyticMeanDerivative) {
  auto                mean = n::from_vector(std::vector<float>{.5, 2}, {1, 2}).requires_grad_();
  m::DynamicsGaussian g{mean, n::zeros({1, 2})};
  auto                next =
      m::DynamicsNext(n::ones({1, 1}), g, n::from_vector(std::vector<float>{2, -1}, {1, 2}));
  EXPECT_EQ(next.next_s.item(), 3.5);
  EXPECT_EQ(next.r.item(), 1);
  auto grad = n::grad(n::sum(next.next_s) + n::sum(next.r), {mean})[0];
  EXPECT_EQ(grad.to_vector<float>(), (std::vector<float>{1, 1}));
  EXPECT_THROW(m::DynamicsNext(n::zeros({2, 1}), g), std::invalid_argument);
}
TEST(PlanningComponents, DynamicsForwardExactSourceNamesAndSoftBounds) {
  n::ParameterStore actual(17), ref(17);
  auto              s = n::ones({2, 3}), a = n::zeros({2, 1});
  auto              out  = m::DynamicsMember(actual, s, a, 8);
  auto              h    = n::mlp(ref, "mlp", n::concat({s, a}, 1), {8, 8}, n::Activation::Relu);
  auto              mean = n::dense(ref, "mean", h, 4);
  auto              raw  = n::dense(ref, "logvar", h, 4);
  auto              expected = -10. + n::softplus(.5 - n::softplus(.5 - raw) + 10.);
  EXPECT_EQ(actual.names(), ref.names());
  EXPECT_EQ(out.mean.to_vector<float>(), mean.to_vector<float>());
  EXPECT_EQ(out.logvar.to_vector<float>(), expected.to_vector<float>());
}
TEST(PlanningComponents, RssmKlClosedFormAndLivePartials) {
  n::ParameterStore store(0);
  m::Rssm           model(store, {});
  auto              mean = n::ones({1, 16}).requires_grad_();
  m::RssmGauss      q{mean, n::zeros({1, 16})}, prior{n::zeros({1, 16}), n::zeros({1, 16})};
  auto              kl = model.Kl(q, prior);
  EXPECT_NEAR(kl.item(), 8, 1e-6);
  auto gradient = n::grad(n::sum(kl), {mean})[0];
  EXPECT_EQ(gradient.to_vector<float>(), std::vector<float>(16, 1));
}
TEST(PlanningComponents, RssmObserveAlignmentAndUnclippedKl) {
  n::ParameterStore store(8);
  m::RssmConfig     cfg;
  cfg.det    = 4;
  cfg.stoch  = 2;
  cfg.hidden = 5;
  m::Rssm model(store, cfg);
  auto    state = model.Zero(2);
  auto    out   = model.Observe(
      state,
      {n::ones({2, 3}), n::zeros({2, 3})},
      {n::zeros({2, 1}), n::ones({2, 1})},
      {n::zeros({2, 2}), n::zeros({2, 2})});
  ASSERT_EQ(out.recon.size(), 2);
  EXPECT_EQ(out.recon[0].shape(), (n::Shape{2, 3}));
  EXPECT_EQ(out.states[1].h.shape(), (n::Shape{2, 4}));
  auto gradients =
      n::grad(n::sum(out.recon[1]) + n::sum(out.kl[1]) + n::sum(out.reward[1]), store.parameters());
  EXPECT_FALSE(gradients.empty());
  EXPECT_GT(n::global_norm(gradients), 0);
  EXPECT_THROW(
      model.Observe(state, {n::zeros({2, 3})}, {}, {n::zeros({2, 2})}), std::invalid_argument);
}
TEST(PlanningComponents, MuZeroScalingTransformAndHeadGeometry) {
  auto scaled = m::MuZeroScale(n::from_vector(std::vector<float>{2, 4, 6, 7, 7, 7}, {2, 3}))
                    .to_vector<float>();
  EXPECT_NEAR(scaled[1], .5, 2e-7);
  EXPECT_EQ(scaled[3], 0);
  for (float v : {-100.f, -1.f, 0.f, 1.f, 100.f}) {
    EXPECT_EQ(m::MuZeroValueTransform(v), v);
    EXPECT_NEAR(m::MuZeroInverseValueTransform(m::MuZeroValueTransform(v, true), true), v, .005);
  }
  n::ParameterStore store(0);
  auto              s = m::MuZeroRepresentation(store, n::zeros({2, 4}), 32);
  auto              d = m::MuZeroDynamics(store, s, n::zeros({2, 2}), 32);
  auto              o = m::MuZeroPrediction(store, d.state, 2);
  EXPECT_EQ(o.logits.shape(), (n::Shape{2, 2}));
  EXPECT_EQ(d.reward.shape(), (n::Shape{2}));
}
TEST(PlanningComponents, MdnClosedDensityAndMixtureGradient) {
  auto output = n::zeros({1, 6}).requires_grad_();
  auto target = n::ones({1, 1});
  auto loss   = m::WorldMdnNll(output, target, 1, 2);
  EXPECT_NEAR(loss.item(), .5 + .5 * std::log(2 * 3.14159265f), 2e-7);
  auto grad = n::grad(n::sum(loss), {output})[0].to_vector<float>();
  EXPECT_NEAR(grad[0], -.5, 1e-7);
  EXPECT_NEAR(grad[1], -.5, 1e-7);
  EXPECT_NEAR(grad[4], 0, 1e-7);
  EXPECT_THROW(m::WorldMdnNll(n::zeros({1, 5}), target, 1, 2), std::invalid_argument);
}
TEST(PlanningComponents, PixelWrapperExactPhysicsAndBinaryFrame) {
  e::WorldPixelCartPole pixels;
  e::CartPole           physics;
  auto                  raw = physics.reset(13);
  EXPECT_EQ(pixels.reset(13), e::WorldPixelCartPole::Render(raw));
  for (int a : {0, 1, 0, 1}) {
    auto actual = pixels.step(a), expected = physics.step(a);
    EXPECT_EQ(actual.obs, e::WorldPixelCartPole::Render(expected.obs));
    EXPECT_EQ(actual.reward, expected.reward);
    EXPECT_EQ(actual.done(), expected.done());
  }
  auto flat = e::WorldPixelCartPole::Render({0, 0, 0, 0});
  EXPECT_EQ(std::accumulate(flat.begin(), flat.end(), 0.f), 13);
  EXPECT_EQ(e::WorldPixelCartPole::Ascii(flat).size(), 12 * (6 + 24 + 1));
  EXPECT_THROW(e::WorldPixelCartPole::Render({NAN, 0, 0, 0}), std::invalid_argument);
}
TEST(PlanningComponents, BlockingMazeSwitchesGapWithoutReset) {
  e::BlockingMaze maze;
  maze.reset(0);
  auto before = maze.Apply(4, 8, 0);
  maze.SetWall(true);
  auto after = maze.Apply(4, 8, 0);
  EXPECT_NE(before.row, after.row);
  EXPECT_EQ(maze.state(), 5 * 9 + 3);
  EXPECT_EQ(after.reward, 0);
}
} // namespace
