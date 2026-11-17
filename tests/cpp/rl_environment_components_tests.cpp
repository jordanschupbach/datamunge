#include <gtest/gtest.h>

#include <cmath>
#include <dm/dm.hpp>
#include <limits>
namespace {
namespace e = dm::rl::environments;
TEST(RlEnvironmentComponents, ActionSpacesRejectMalformedInputs) {
  EXPECT_THROW(e::ActionSpace::Discrete(0), std::invalid_argument);
  EXPECT_THROW(e::ActionSpace::Continuous({}, {}), std::invalid_argument);
  EXPECT_THROW(e::ActionSpace::Continuous({1}, {0}), std::invalid_argument);
  EXPECT_THROW(
      e::ActionSpace::Continuous({0}, {std::numeric_limits<float>::infinity()}),
      std::invalid_argument);
  EXPECT_THROW(e::ValidateAction(e::ActionSpace::Discrete(2), e::Action(2)), std::invalid_argument);
  EXPECT_THROW(
      e::ValidateAction(e::ActionSpace::Continuous({-1}, {1}), e::Action(std::vector<float>{})),
      std::invalid_argument);
  EXPECT_THROW(e::CartPole(-1), std::invalid_argument);
  EXPECT_THROW(e::Snake(1, 8), std::invalid_argument);
  EXPECT_THROW(e::Corridor(2), std::invalid_argument);
}
TEST(RlEnvironmentComponents, SeedRepeatOwnsObservationsAndInvalidActionDoesNotAdvance) {
  e::CartPole a, b;
  auto        first = a.reset(17);
  EXPECT_EQ(first, b.reset(17));
  auto saved = first;
  EXPECT_THROW(a.step(9), std::invalid_argument);
  for (int t = 0; t < 30; ++t) {
    auto x = a.step(t % 2), y = b.step(t % 2);
    EXPECT_EQ(x.obs, y.obs);
    EXPECT_EQ(x.reward, y.reward);
    EXPECT_EQ(x.terminated, y.terminated);
    EXPECT_EQ(x.truncated, y.truncated);
    if (x.done())
      break;
  }
  EXPECT_EQ(first, saved);
}
TEST(RlEnvironmentComponents, CorridorTimeLimitVersusTerminal) {
  e::Corridor short_limit(7, 1);
  short_limit.reset(1);
  auto cut = short_limit.step(1);
  EXPECT_FALSE(cut.terminated);
  EXPECT_TRUE(cut.truncated);
  EXPECT_EQ(cut.reward, 0);
  e::Corridor terminal(3, 20);
  terminal.reset(1);
  auto end = terminal.step(1);
  EXPECT_TRUE(end.terminated);
  EXPECT_FALSE(end.truncated);
  EXPECT_EQ(std::abs(end.reward), 1);
  EXPECT_THROW(terminal.step(1), std::logic_error);
}
TEST(RlEnvironmentComponents, CartPoleIndependentEulerEquations) {
  e::CartPole env;
  env.reset(123);
  auto   s = env.state();
  double x = s[0], xd = s[1], th = s[2], td = s[3];
  double cs = std::cos(th), sn = std::sin(th);
  double temp   = (10 + .05 * td * td * sn) / 1.1;
  double ta     = (9.8 * sn - cs * temp) / (.5 * (4. / 3. - .1 * cs * cs / 1.1));
  double xa     = temp - .05 * ta * cs / 1.1;
  auto   result = env.step(1);
  EXPECT_NEAR(result.obs[0], x + .02 * xd, 1e-7);
  EXPECT_NEAR(result.obs[1], xd + .02 * xa, 2e-7);
  EXPECT_NEAR(result.obs[2], th + .02 * td, 1e-7);
  EXPECT_NEAR(result.obs[3], td + .02 * ta, 3e-7);
}
TEST(RlEnvironmentComponents, MountainCarForceClippingAndLeftBoundary) {
  e::MountainCar env;
  env.reset(11);
  double x = env.position(), v = env.velocity();
  auto   result = env.step(2);
  double next   = std::clamp(v + .001 - .0025 * std::cos(3 * x), -.07, .07);
  EXPECT_NEAR(result.obs[1], next, 3e-8);
  EXPECT_NEAR(result.obs[0], x + next, 7e-8);
  e::MountainCarContinuous a, b;
  EXPECT_EQ(a.reset(1), b.reset(1));
  EXPECT_EQ(
      a.step(e::Action(std::vector<float>{10})).obs, b.step(e::Action(std::vector<float>{1})).obs);
  for (int i = 0; i < 400; ++i) {
    auto r = env.step(0);
    EXPECT_GE(r.obs[0], -1.2f);
    EXPECT_LE(r.obs[0], .6f);
    EXPECT_LE(std::abs(r.obs[1]), .070001f);
    if (r.done())
      break;
  }
}
TEST(RlEnvironmentComponents, PendulumCostAndTorqueSaturation) {
  e::Pendulum a, b;
  EXPECT_EQ(a.reset(23), b.reset(23));
  double th = e::WrapAngle(a.theta()), v = a.theta_dot();
  auto x = a.step(e::Action(std::vector<float>{10})), y = b.step(e::Action(std::vector<float>{2}));
  EXPECT_EQ(x.obs, y.obs);
  EXPECT_EQ(x.reward, y.reward);
  EXPECT_NEAR(x.reward, -(th * th + .1 * v * v + .001 * 4), 2e-6);
}
using V = std::array<double, 4>;
V AcrobotForce(const V& y, double torque) {
  double t1 = y[0], t2 = y[1], v1 = y[2], v2 = y[3];
  double d1 = .25 + 1 + .25 + std::cos(t2) + 2, d2 = .25 + .5 * std::cos(t2) + 1;
  double phi2 = .5 * 9.8 * std::cos(t1 + t2 - double(e::kPi) / 2);
  double phi1 = -.5 * v2 * v2 * std::sin(t2) - v2 * v1 * std::sin(t2) +
                1.5 * 9.8 * std::cos(t1 - double(e::kPi) / 2) + phi2;
  double a2 =
      (torque + d2 / d1 * phi1 - .5 * v1 * v1 * std::sin(t2) - phi2) / (1.25 - d2 * d2 / d1);
  return {v1, v2, -(d2 * a2 + phi1) / d1, a2};
}
TEST(RlEnvironmentComponents, AcrobotIndependentRk4AndTipGeometry) {
  e::Acrobot env;
  auto       obs = env.reset(19);
  V          y{env.theta1(), env.theta2(), obs[4], obs[5]};
  auto       axpy = [](V a, double h, V b) {
    for (int i = 0; i < 4; ++i)
      a[i] += h * b[i];
    return a;
  };
  double dt = double(.2f);
  auto   a = AcrobotForce(y, -1), b = AcrobotForce(axpy(y, dt / 2, a), -1),
       c = AcrobotForce(axpy(y, dt / 2, b), -1), d = AcrobotForce(axpy(y, dt, c), -1);
  for (int i = 0; i < 4; ++i)
    y[i] += dt / 6 * (a[i] + 2 * b[i] + 2 * c[i] + d[i]);
  auto result = env.step(0);
  EXPECT_NEAR(env.theta1(), y[0], 5e-7);
  EXPECT_NEAR(env.theta2(), y[1], 5e-7);
  EXPECT_NEAR(result.obs[4], y[2], 4e-7);
  EXPECT_NEAR(result.obs[5], y[3], 4e-7);
  EXPECT_NEAR(env.tip_height(), -std::cos(y[0]) - std::cos(y[0] + y[1]), 3e-7);
}
TEST(RlEnvironmentComponents, GridTransitionMassAndCliffNonterminalPenalty) {
  e::GridWorld grid(.3f);
  for (int state = 0; state < grid.num_states(); ++state)
    for (int action = 0; action < 4; ++action) {
      auto   v    = grid.transitions(state, action);
      double mass = 0;
      for (auto& r : v) {
        EXPECT_GE(r.prob, 0);
        EXPECT_GE(r.next_state, 0);
        EXPECT_LT(r.next_state, grid.num_states());
        mass += r.prob;
      }
      if (!v.empty())
        EXPECT_NEAR(mass, 1, 5e-8);
    }
  e::CliffWalking cliff;
  cliff.reset(1);
  auto r = cliff.step(1);
  EXPECT_EQ(r.obs[0], 36);
  EXPECT_EQ(r.reward, -100);
  EXPECT_FALSE(r.terminated);
  EXPECT_FALSE(r.truncated);
  EXPECT_THROW(grid.transitions(-1, 0), std::out_of_range);
  EXPECT_THROW(grid.Apply(0, 0, 7), std::out_of_range);
}
TEST(RlEnvironmentComponents, BanditEndpointsAndIndependentInitialization) {
  e::MultiArmedBandit zero({.3f, .7f}, 0.f);
  zero.reset(1);
  for (int i = 0; i < 20; ++i)
    EXPECT_EQ(zero.step(i % 2).reward, i % 2 ? .7f : .3f);
  auto b = e::MultiArmedBandit::Bernoulli({0, 1});
  b.reset(1);
  for (int i = 0; i < 20; ++i) {
    EXPECT_EQ(b.step(0).reward, 0);
    EXPECT_EQ(b.step(1).reward, 1);
  }
  EXPECT_EQ(b.best_arm(), 1);
  EXPECT_THROW(b.Pull(2), std::out_of_range);
  EXPECT_THROW(e::MultiArmedBandit::Bernoulli({1.1f}), std::invalid_argument);
  EXPECT_THROW(e::MultiArmedBandit({}, 1), std::invalid_argument);
  auto a = e::MultiArmedBandit::Testbed(10, 42), c = e::MultiArmedBandit::Testbed(10, 42);
  for (int i = 0; i < 10; ++i)
    EXPECT_EQ(a.mean(i), c.mean(i));
  a.reset(11);
  c.reset(11);
  for (int i = 0; i < 30; ++i)
    EXPECT_EQ(a.step(i % 10).reward, c.step(i % 10).reward);
}
TEST(RlEnvironmentComponents, ContextualBanditRequiresResetAndStableLogisticRewards) {
  e::ContextualBandit a(3, 4, 19), b(3, 4, 19);
  EXPECT_THROW(a.ExpectedReward(0), std::logic_error);
  EXPECT_EQ(a.reset(31), b.reset(31));
  for (int i = 0; i < 4; ++i) {
    double p = a.ExpectedReward(i);
    EXPECT_GE(p, 0);
    EXPECT_LE(p, 1);
    EXPECT_EQ(p, b.ExpectedReward(i));
  }
  EXPECT_EQ(a.BestArm(), b.BestArm());
  EXPECT_THROW(e::ContextualBandit(-1, 2, 1), std::invalid_argument);
}
TEST(RlEnvironmentComponents, GamesLegalMovesPerspectiveAndCloneIndependence) {
  e::TicTacToe game;
  game.Reset();
  auto clone = game.Clone();
  game.Apply(0);
  EXPECT_EQ(clone->LegalActions().size(), 9);
  EXPECT_EQ(game.Observation()[0], -1);
  EXPECT_THROW(game.Apply(0), std::invalid_argument);
  for (int move : {3, 1, 4, 2})
    game.Apply(move);
  EXPECT_TRUE(game.Terminal());
  EXPECT_EQ(game.Winner(), 1);
  EXPECT_TRUE(game.LegalActions().empty());
  EXPECT_THROW(game.Apply(8), std::invalid_argument);
  e::ConnectFour connect;
  connect.Reset();
  for (int move : {0, 1, 0, 1, 0, 1, 0})
    connect.Apply(move);
  EXPECT_TRUE(connect.Terminal());
  EXPECT_EQ(connect.Winner(), 1);
  e::Othello o;
  EXPECT_EQ(o.LegalActions().size(), 4);
  auto counts = o.Counts();
  EXPECT_EQ(counts.first, 2);
  EXPECT_EQ(counts.second, 2);
  o.Apply(o.LegalActions().front());
  counts = o.Counts();
  EXPECT_EQ(counts.first + counts.second, 5);
  EXPECT_THROW(o.Apply(e::Othello::kPass), std::invalid_argument);
}
TEST(RlEnvironmentComponents, SnakeReversalAndFrameOwnership) {
  e::Snake env;
  env.reset(1);
  auto head = env.head();
  auto obs  = env.step(3).obs;
  EXPECT_EQ(env.head(), std::make_pair(head.first, head.second + 1));
  EXPECT_EQ(obs.size(), 192);
  EXPECT_EQ(std::count(obs.begin(), obs.begin() + 64, 1.f), 1);
  e::SnakeFilmstrip film;
  EXPECT_THROW(film.tensor(), std::logic_error);
  film.Add(env);
  auto first = film.tensor().to_vector<float>();
  env.step(0);
  EXPECT_EQ(film.tensor().to_vector<float>(), first);
  film.Add(env);
  EXPECT_EQ(film.frames(), 2);
  e::Snake other(4, 4);
  other.reset(1);
  EXPECT_THROW(film.Add(other), std::invalid_argument);
}
TEST(RlEnvironmentComponents, VectorSerialThreadedIdentityAndFinalObservations) {
  auto      factory = [] { return std::make_unique<e::CartPole>(5); };
  e::VecEnv a(factory, 8, 31, 1), b(factory, 8, 31, 4);
  EXPECT_EQ(a.reset(), b.reset());
  int ended = 0;
  for (int t = 0; t < 40; ++t) {
    std::vector<e::Action> actions(8);
    for (int i = 0; i < 8; ++i)
      actions[i] = e::Action((i + t) % 2);
    auto x = a.step(actions), y = b.step(actions);
    EXPECT_EQ(x.obs, y.obs);
    EXPECT_EQ(x.reward, y.reward);
    EXPECT_EQ(x.terminated, y.terminated);
    EXPECT_EQ(x.truncated, y.truncated);
    EXPECT_EQ(x.final_obs, y.final_obs);
    for (int i = 0; i < 8; ++i)
      if (x.done(i)) {
        ++ended;
        EXPECT_EQ(x.final_obs[i].size(), 4);
        EXPECT_EQ(x.obs[i].size(), 4);
      }
  }
  EXPECT_GT(ended, 0);
  EXPECT_THROW(a.step({0}), std::invalid_argument);
  EXPECT_THROW(a.env(8), std::out_of_range);
}
class ThrowingEnv : public e::EnvBase {
 public:
  std::string    name() const override { return "throwing"; }
  int            obs_size() const override { return 1; }
  e::ActionSpace action_space() const override { return e::ActionSpace::Discrete(2); }
  e::Obs         Reset() override {
    value = 0;
    return {0};
  }
  e::StepResult step(const e::Action& a) override {
    if (a.discrete == 1)
      throw std::runtime_error("worker probe");
    ++value;
    return {{float(value)}, 0, false, false};
  }
  int value = 0;
};
TEST(RlEnvironmentComponents, VectorWorkerExceptionPropagatesAndPoolRemainsUsable) {
  e::VecEnv env([] { return std::make_unique<ThrowingEnv>(); }, 4, 1, 2);
  env.reset();
  EXPECT_THROW(env.step({0, 1, 0, 0}), std::runtime_error);
  auto reset = env.reset();
  EXPECT_EQ(reset, std::vector<e::Obs>(4, e::Obs{0}));
  auto out = env.step({0, 0, 0, 0});
  EXPECT_EQ(out.obs, std::vector<e::Obs>(4, e::Obs{1}));
  EXPECT_THROW(e::VecEnv([] { return std::unique_ptr<e::Env>{}; }, 2, 1), std::invalid_argument);
  EXPECT_THROW(
      e::VecEnv([] { return std::make_unique<e::CartPole>(); }, 0, 1), std::invalid_argument);
}
} // namespace
