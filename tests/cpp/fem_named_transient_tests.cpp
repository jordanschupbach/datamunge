#include <gtest/gtest.h>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <datamunge/fem/fem1d.hpp>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
using namespace datamunge::fem;
namespace {
class Field final : public ScalarField1D {
  std::function<double(double)> f_;

 public:
  explicit Field(std::function<double(double)> f) : f_(std::move(f)) {}
  double evaluate(double x) override { return f_(x); }
};
void same(const FEM1DTimeSeries& a, const FEM1DTimeSeries& b) {
  EXPECT_EQ(a.nodes, b.nodes);
  EXPECT_EQ(a.times, b.times);
  EXPECT_EQ(a.u, b.u);
}
std::function<double(double)> function(std::size_t kind, const std::vector<double>& p) {
  return [=](double x) {
    switch (kind) {
      case 0:
        return 0.;
      case 1:
        return p[0];
      case 2:
        return p[0] * x + p[1];
      case 3: {
        double value = 0, power = 1;
        for (double c : p) {
          value += c * power;
          power *= x;
        }
        return value;
      }
      default:
        return p[0] * std::sin(p[1] * x + p[2]);
    }
  };
}
} // namespace
TEST(FemNamedTransient, EverySourceMatchesLiveForSteadyAndActualTimeHistory) {
  std::mt19937                   rng(617);
  const std::vector<std::string> names{"zero", "constant", "linear", "polynomial", "sine"};
  for (int trial = 0; trial < 400; ++trial) {
    auto                                   m = make_uniform_mesh1d(-.5, 1.5, 3 + rng() % 7);
    const std::vector<std::vector<double>> params{
        {},
        {double(rng() % 51) / 10 - 2},
        {double(rng() % 21) / 10, double(rng() % 11) / 10 - .5},
        {.3, -.2, .75},
        {1.2, 2.3, .1}};
    std::vector<double> p(m.num_elements()), q(m.num_elements());
    for (auto& v : p)
      v = .5 + double(rng() % 21) / 10;
    for (auto& v : q)
      v = double(rng() % 11) / 10;
    std::vector<double> u0(m.num_nodes());
    for (auto& v : u0)
      v = double(rng() % 11) / 10;
    BoundaryCondition1D left{}, right{};
    for (std::size_t j = 0; j < names.size(); ++j) {
      Field field(function(j, params[j]));
      auto  named =
          FEM1D::solve_transient_builtin(m, p, q, names[j], params[j], left, right, u0, .25, .1);
      auto live = FEM1D::solve_transient(m, p, q, field, left, right, u0, .25, .1);
      same(named, live);
      auto a = FEM1D::solve_builtin(m, p, q, names[j], params[j], left, right);
      auto b = FEM1D::solve(m, p, q, field, left, right);
      EXPECT_EQ(a.nodes, b.nodes);
      EXPECT_EQ(a.u, b.u);
    }
  }
}
TEST(FemNamedTransient, DefaultsTrailingParametersAndEmptyPolynomial) {
  auto                                   m = make_uniform_mesh1d(0, 1, 3);
  const std::vector<std::string>         names{"zero", "constant", "linear", "polynomial", "sine"};
  const std::vector<std::vector<double>> defaults{{}, {1}, {1, 0}, {}, {1, 1, 0}};
  for (std::size_t i = 0; i < names.size(); ++i) {
    same(
        FEM1D::solve_transient_builtin(m, {1}, {0}, names[i], {}, {}, {}, {0, 0, 0, 0}, .2, .1),
        FEM1D::solve_transient_builtin(
            m, {1}, {0}, names[i], defaults[i], {}, {}, {0, 0, 0, 0}, .2, .1));
    if (i != 3) {
      auto extra = defaults[i];
      extra.push_back(19);
      if (i == 0 || i == 1 || i == 2 || i == 4)
        same(
            FEM1D::solve_transient_builtin(
                m, {1}, {0}, names[i], extra, {}, {}, {0, 0, 0, 0}, .2, .1),
            FEM1D::solve_transient_builtin(
                m, {1}, {0}, names[i], defaults[i], {}, {}, {0, 0, 0, 0}, .2, .1));
    }
  }
  auto sine =
      FEM1D::solve_transient_builtin(m, {1}, {0}, "sine", {2}, {}, {}, {0, 0, 0, 0}, .2, .1);
  auto explicit_sine =
      FEM1D::solve_transient_builtin(m, {1}, {0}, "sine", {2, 1, 0}, {}, {}, {0, 0, 0, 0}, .2, .1);
  same(sine, explicit_sine);
  auto poly =
      FEM1D::solve_transient_builtin(m, {1}, {0}, "polynomial", {}, {}, {}, {0, 0, 0, 0}, .2, .1);
  for (const auto& row : poly.u)
    for (double v : row)
      EXPECT_EQ(v, 0);
  EXPECT_THROW(
      FEM1D::solve_transient_builtin(m, {1}, {0}, "Constant", {}, {}, {}, {0, 0, 0, 0}, .2, .1),
      std::invalid_argument);
}
TEST(FemNamedTransient, OneInteriorModeMatchesClosedBackwardEulerProduct) {
  for (double mid : {.2, .5, .8})
    for (double reaction : {0., .5, 2.}) {
      Mesh1D       m{{0, mid, 1}};
      const double lambda = 3 * (1 / mid + 1 / (1 - mid)) + reaction, steady = 1.5 / lambda;
      auto         t = FEM1D::solve_transient_builtin(
          m, {1}, {reaction}, "constant", {1}, {}, {}, {0, 0, 0}, .25, .1);
      ASSERT_EQ(t.times, (std::vector<double>{0, .1, .2, .25}));
      long double product = 1;
      for (std::size_t j = 1; j < t.times.size(); ++j) {
        const double dt = t.times[j] - t.times[j - 1];
        product /= 1 + static_cast<long double>(lambda) * dt;
        EXPECT_NEAR(t.u[j][1], steady * (1 - static_cast<double>(product)), 2e-15);
        EXPECT_EQ(t.u[j][0], 0);
        EXPECT_EQ(t.u[j][2], 0);
      }
    }
  auto m = make_uniform_mesh1d(0, 1, 2);
  auto sourcezero =
      FEM1D::solve_transient_builtin(m, {1}, {0}, "zero", {}, {}, {}, {0, 1, 0}, .5, .5);
  EXPECT_NEAR(sourcezero.u[1][1], 1. / 7, 1e-15);
}
TEST(FemNamedTransient, RoundedFinalTimeNeverDuplicatesAndPartialStepsAdvance) {
  auto m = make_uniform_mesh1d(0, 1, 2);
  for (double dt : {.02, .1, .2, .3, .125})
    for (int n = 1; n <= 20; ++n)
      for (int shift : {-1, 0, 1}) {
        const double nominal = n * dt,
                     end     = shift < 0 ? std::nextafter(nominal, 0.)
                               : shift > 0
                                   ? std::nextafter(nominal, std::numeric_limits<double>::infinity())
                                   : nominal;
        auto t               = FEM1D::solve_transient_builtin(
            m, {1}, {0}, "constant", {1}, {}, {}, {0, 0, 0}, end, dt);
        ASSERT_GE(t.times.size(), 2u);
        EXPECT_EQ(t.times.front(), 0);
        EXPECT_EQ(t.times.back(), end);
        ASSERT_EQ(t.times.size(), t.u.size());
        for (std::size_t j = 1; j < t.times.size(); ++j) {
          EXPECT_GT(t.times[j], t.times[j - 1]);
          EXPECT_LE(t.times[j], end);
          for (double v : t.u[j])
            EXPECT_TRUE(std::isfinite(v));
        }
      }
  auto edge = FEM1D::solve_transient_builtin(
      m, {1}, {0}, "constant", {1}, {}, {}, {0, 0, 0}, .30000000000000004, .1);
  EXPECT_EQ(edge.times.size(), 4u);
  EXPECT_TRUE(std::isfinite(edge.u.back()[1]));
}
TEST(FemNamedTransient, FiniteControlsAndCapacityFailBeforeCallbacks) {
  auto         m     = make_uniform_mesh1d(0, 1, 2);
  std::size_t  calls = 0;
  Field        f([&](double) {
    ++calls;
    return 1.;
  });
  const double inf = std::numeric_limits<double>::infinity(),
               nan = std::numeric_limits<double>::quiet_NaN();
  for (double bad : {0., -1., inf, -inf, nan}) {
    EXPECT_THROW(
        FEM1D::solve_transient(m, {1}, {0}, f, {}, {}, {0, 0, 0}, bad, .1), std::invalid_argument);
    EXPECT_THROW(
        FEM1D::solve_transient(m, {1}, {0}, f, {}, {}, {0, 0, 0}, 1, bad), std::invalid_argument);
  }
  EXPECT_THROW(
      FEM1D::solve_transient(m, {1}, {0}, f, {}, {}, {0, 0}, 1, .1), std::invalid_argument);
  EXPECT_THROW(
      FEM1D::solve_transient(Mesh1D{}, {1}, {0}, f, {}, {}, {}, 1, .1), std::invalid_argument);
  EXPECT_THROW(
      FEM1D::solve_transient(Mesh1D{{0}}, {1}, {0}, f, {}, {}, {0}, 1, .1), std::invalid_argument);
  std::feclearexcept(FE_ALL_EXCEPT);
  EXPECT_THROW(
      FEM1D::solve_transient(
          m,
          {1},
          {0},
          f,
          {},
          {},
          {0, 0, 0},
          std::numeric_limits<double>::max(),
          std::numeric_limits<double>::denorm_min()),
      std::length_error);
  EXPECT_EQ(std::fetestexcept(FE_OVERFLOW | FE_INVALID), 0);
  EXPECT_THROW(
      FEM1D::solve_transient(m, {1}, {0}, f, {}, {}, {0, 0, 0}, 1, 1e-20), std::length_error);
  EXPECT_EQ(calls, 0u);
}
TEST(FemNamedTransient, HugeFiniteTimesAvoidIntermediateOverflow) {
  auto            m       = make_uniform_mesh1d(0, 1, 2);
  volatile double maximum = std::numeric_limits<double>::max();
  const double    M       = maximum;
  std::feclearexcept(FE_ALL_EXCEPT);
  auto t =
      FEM1D::solve_transient_builtin(m, {1}, {0}, "constant", {1}, {}, {}, {0, 0, 0}, M, .75 * M);
  const int bad = std::fetestexcept(FE_OVERFLOW | FE_INVALID);
  EXPECT_EQ(bad, 0);
  ASSERT_EQ(t.times.size(), 3u);
  EXPECT_EQ(t.times[1], .75 * M);
  EXPECT_EQ(t.times[2], M);
  EXPECT_TRUE(std::isfinite(t.u.back()[1]));
  EXPECT_NEAR(t.u.back()[1], .125, 1e-15);
}
TEST(FemNamedTransient, ActualQuadratureCallCountAndCallbackExceptions) {
  auto                m = make_uniform_mesh1d(0, 1, 5);
  std::vector<double> coordinates;
  Field               f([&](double x) {
    coordinates.push_back(x);
    return 1.;
  });
  auto                t = FEM1D::solve_transient(
      m, {1}, {0}, f, {}, {}, std::vector<double>(6, 0), .30000000000000004, .1);
  EXPECT_EQ(coordinates.size(), 3 * m.num_elements() * (t.num_steps() - 1));
  for (std::size_t i = 0; i < coordinates.size(); ++i)
    EXPECT_EQ(coordinates[i], coordinates[i % (3 * m.num_elements())]);
  Field throwing([](double) -> double { throw std::runtime_error("source"); });
  EXPECT_THROW(
      FEM1D::solve_transient(m, {1}, {0}, throwing, {}, {}, std::vector<double>(6, 0), .2, .1),
      std::runtime_error);
}
TEST(FemNamedTransient, ResultsOwnValuesAndAccessorsCheckStoredBounds) {
  auto                mesh = make_uniform_mesh1d(0, 1, 2);
  std::vector<double> p{1}, q{0}, u0{0, 0, 0};
  auto t = FEM1D::solve_transient_builtin(mesh, p, q, "constant", {1}, {}, {}, u0, .2, .1);
  EXPECT_EQ(p, (std::vector<double>{1}));
  EXPECT_EQ(q, (std::vector<double>{0}));
  EXPECT_EQ(u0, (std::vector<double>{0, 0, 0}));
  auto copy     = t;
  mesh.nodes[1] = 99;
  t.u[1][1]     = 99;
  t.times[1]    = 99;
  t.nodes[1]    = 99;
  EXPECT_EQ(copy.nodes[1], .5);
  EXPECT_EQ(copy.times[1], .1);
  EXPECT_LT(copy.u[1][1], 1);
  EXPECT_THROW(copy.node_at(99), std::out_of_range);
  EXPECT_THROW(copy.time_at(99), std::out_of_range);
  EXPECT_THROW(copy.value_at(99, 0), std::out_of_range);
  EXPECT_THROW(copy.value_at(0, 99), std::out_of_range);
}
