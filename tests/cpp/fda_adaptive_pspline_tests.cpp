/// @file fda_adaptive_pspline_tests.cpp
/// @brief Analytic, independent QR, basis-invariance and leakage oracles for greedy HP splines.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/fda/adaptive_pspline.hpp>
#include <datamunge/linalg/qr.hpp>
#include <limits>
#include <map>
namespace {
namespace f = datamunge::fda;
std::vector<double> grid(int n) {
  std::vector<double> x;
  for (int i = 0; i < n; ++i)
    x.push_back(double(i) / (n - 1));
  return x;
}
std::vector<double> signal(const std::vector<double>& x) {
  std::vector<double> y;
  for (double t : x)
    y.push_back(std::sin(6 * t) + .8 * std::exp(-1000 * (t - .7) * (t - .7)));
  return y;
}
TEST(AdaptivePSpline, ThreePointAnalyticPenaltyScaling) {
  f::AdaptiveSplineOptions o;
  o.degree     = 1;
  o.root_basis = 3;
  o.max_depth  = 0;
  o.lambda     = 1. / 24;
  auto a       = f::fit_hierarchical_pspline({0, .5, 1}, {0, 1, 0}, o);
  auto p       = a.predict({0, .5, 1});
  EXPECT_NEAR(p[0], 2. / 7, 1e-13);
  EXPECT_NEAR(p[1], 3. / 7, 1e-13);
  EXPECT_NEAR(p[2], 2. / 7, 1e-13);
  EXPECT_NEAR(a.training_mse, 8. / 49, 1e-13);
  EXPECT_NEAR(a.roughness, 32. / 49, 1e-12);
  o.lambda = 0;
  auto b   = f::fit_hierarchical_pspline({0, .5, 1}, {0, 1, 0}, o);
  EXPECT_LT(b.training_mse, 1e-28);
  EXPECT_TRUE(std::isnan(b.validation_mse));
}
TEST(AdaptivePSpline, AugmentedQrIndependentRootOracle) {
  auto                     x = grid(41), y = signal(x);
  f::AdaptiveSplineOptions o;
  o.max_depth                                 = 2;
  o.lambda                                    = 3e-5;
  auto                                   fit  = f::fit_hierarchical_pspline(x, y, o);
  auto                                   base = f::BSplineBasis::open_uniform(3, 7);
  auto                                   a = f::dyadic_refine(base), b = f::dyadic_refine(a.basis);
  datamunge::linalg::DenseMatrix<double> R(b.basis.basis_size(), 7, 0.);
  for (int j = 0; j < 7; ++j) {
    std::vector<double> u(7);
    u[j]   = 1;
    auto c = b.apply(a.apply(u));
    for (std::size_t i = 0; i < c.size(); ++i)
      R(i, j) = c[i];
  }
  auto                                   X    = base.evaluate_dense(x);
  const std::size_t                      rows = x.size() + R.rows() - 2;
  datamunge::linalg::DenseMatrix<double> A(rows, 7, 0.);
  std::vector<double>                    target(rows, 0.);
  for (std::size_t i = 0; i < x.size(); ++i) {
    target[i] = y[i] / std::sqrt(double(x.size()));
    for (int j = 0; j < 7; ++j)
      A(i, j) = X(i, j) / std::sqrt(double(x.size()));
  }
  double              h = 1. / (R.rows() - 3), weight = std::sqrt(o.lambda / (h * h * h));
  std::vector<double> xi(R.rows(), 0.);
  for (std::size_t i = 0; i < xi.size(); ++i)
    for (int q = 1; q <= 3; ++q)
      xi[i] += b.basis.knots()[i + q] / 3.;
  for (std::size_t i = 0; i + 2 < R.rows(); ++i)
    for (int j = 0; j < 7; ++j) {
      const double dl = xi[i + 1] - xi[i], dr = xi[i + 2] - xi[i + 1];
      A(x.size() + i, j) = weight * 2 * h * h / (dl + dr) *
                           ((R(i + 2, j) - R(i + 1, j)) / dr - (R(i + 1, j) - R(i, j)) / dl);
    }
  auto beta = datamunge::linalg::qr(A).solve(target);
  auto pred = fit.predict(x);
  for (std::size_t i = 0; i < x.size(); ++i) {
    double oracle = 0;
    for (int j = 0; j < 7; ++j)
      oracle += X(i, j) * beta[j];
    EXPECT_NEAR(pred[i], oracle, 1e-11);
  }
}
TEST(AdaptivePSpline, HbAndThbCoordinatesGiveSameFunctionAndPenalty) {
  auto                     x = grid(161), y = signal(x), g = grid(301);
  f::AdaptiveSplineOptions o;
  o.max_depth = 4;
  for (double lambda : {0., 1e-7, 1e-3, 1.}) {
    o.lambda    = lambda;
    o.truncated = false;
    auto hb     = f::fit_hierarchical_pspline(x, y, o, {0, 2, 3});
    o.truncated = true;
    auto thb    = f::fit_hierarchical_pspline(x, y, o, {0, 2, 3});
    EXPECT_EQ(hb.parameters, thb.parameters);
    auto a = hb.predict(g), b = thb.predict(g);
    for (std::size_t i = 0; i < g.size(); ++i)
      EXPECT_NEAR(a[i], b[i], 2e-8);
    EXPECT_NEAR(hb.roughness, thb.roughness, 1e-7 * std::max(1., hb.roughness));
  }
}
TEST(AdaptivePSpline, GreedyChoiceIsLargestPositiveLocalMseDrop) {
  auto                     x = grid(161), y = signal(x);
  f::AdaptiveSplineOptions o;
  o.max_splits                                             = 6;
  o.lambda                                                 = 1e-6;
  auto                                                   a = f::fit_adaptive_pspline(x, y, x, y, o);
  std::map<int, std::vector<f::AdaptiveSplineCandidate>> groups;
  for (const auto& c : a.candidates)
    groups[c.step].push_back(c);
  for (const auto& [iteration, candidates] : groups) {
    auto chosen = std::find_if(
        candidates.begin(), candidates.end(), [](const auto& c) { return c.selected; });
    if (chosen == candidates.end())
      continue;
    EXPECT_GT(chosen->gain, 0);
    EXPECT_TRUE(chosen->fitted);
    EXPECT_NEAR(chosen->gain, chosen->mse_before - chosen->mse_after, 1e-15);
    for (const auto& c : candidates)
      if (c.fitted)
        EXPECT_LE(c.gain, chosen->gain + 1e-12);
  }
  EXPECT_GT(a.path.size(), 2u);
}
TEST(AdaptivePSpline, AuditLocalMseMatchesIndependentCounterfactualReplay) {
  auto                     x = grid(161), y = signal(x);
  f::AdaptiveSplineOptions o;
  o.max_splits          = 4;
  o.lambda              = 1e-6;
  auto             path = f::fit_adaptive_pspline(x, y, x, y, o);
  std::vector<int> accepted;
  for (int iteration = 1; iteration <= 4; ++iteration) {
    auto current = f::fit_hierarchical_pspline(x, y, o, accepted);
    int  chosen  = -1;
    for (const auto& audit : path.candidates)
      if (audit.step == iteration && audit.fitted) {
        auto proposed = accepted;
        proposed.push_back(audit.cell);
        auto   after    = f::fit_hierarchical_pspline(x, y, o, proposed);
        auto   old_pred = current.predict(x), new_pred = after.predict(x);
        auto   cell   = current.cells[audit.cell];
        double old_ss = 0, new_ss = 0;
        int    count = 0;
        for (std::size_t i = 0; i < x.size(); ++i)
          if (x[i] >= cell.lower && (x[i] < cell.upper || cell.upper == 1)) {
            old_ss += std::pow(y[i] - old_pred[i], 2);
            new_ss += std::pow(y[i] - new_pred[i], 2);
            ++count;
          }
        EXPECT_EQ(audit.observations, count);
        EXPECT_NEAR(audit.mse_before, old_ss / count, 1e-14);
        EXPECT_NEAR(audit.mse_after, new_ss / count, 1e-14);
        if (audit.selected)
          chosen = audit.cell;
      }
    if (chosen < 0)
      break;
    accepted.push_back(chosen);
  }
}
TEST(AdaptivePSpline, ValidationNeverChangesTrainingGrowthOrder) {
  auto x = grid(129), y = signal(x);
  auto vy = y;
  for (double& v : vy)
    v = 3 - v;
  f::AdaptiveSplineOptions o;
  o.max_splits = 5;
  o.lambda     = 1e-5;
  auto a = f::fit_adaptive_pspline(x, y, x, y, o), b = f::fit_adaptive_pspline(x, y, x, vy, o);
  ASSERT_EQ(a.candidates.size(), b.candidates.size());
  ASSERT_EQ(a.path.size(), b.path.size());
  for (std::size_t i = 0; i < a.candidates.size(); ++i) {
    EXPECT_EQ(a.candidates[i].cell, b.candidates[i].cell);
    EXPECT_EQ(a.candidates[i].selected, b.candidates[i].selected);
    EXPECT_DOUBLE_EQ(a.candidates[i].gain, b.candidates[i].gain);
  }
  double min = std::numeric_limits<double>::infinity();
  for (auto s : a.path)
    min = std::min(min, s.validation_mse);
  EXPECT_NEAR(a.validation_mse, min, 1e-12);
  EXPECT_NE(a.splits.size(), b.splits.size());
}
TEST(AdaptivePSpline, MeanLossIsInvariantToReplicatingTrainingRows) {
  auto                     x = grid(65), y = signal(x);
  f::AdaptiveSplineOptions o;
  o.lambda = .001;
  auto a   = f::fit_hierarchical_pspline(x, y, o, {0, 2});
  auto xx = x, yy = y;
  xx.insert(xx.end(), x.begin(), x.end());
  yy.insert(yy.end(), y.begin(), y.end());
  auto b  = f::fit_hierarchical_pspline(xx, yy, o, {0, 2});
  auto ap = a.predict(x), bp = b.predict(x);
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(ap[i], bp[i], 1e-11);
}
TEST(AdaptivePSpline, AffineFunctionsSurviveLargePenaltyAndRefinement) {
  auto                x = grid(129);
  std::vector<double> y;
  for (double t : x)
    y.push_back(2.5 + 2 * t);
  f::AdaptiveSplineOptions o;
  o.lambda  = 10.;
  auto a    = f::fit_hierarchical_pspline(x, y, o, {0, 1, 4});
  auto pred = a.predict(x);
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(pred[i], y[i], 1e-6);
  EXPECT_LT(a.roughness, 1e-12);
}
TEST(AdaptivePSpline, InvalidControlsAndInputsThrowWithoutMutatingCaller) {
  auto                     x = grid(21), y = signal(x);
  f::AdaptiveSplineOptions o;
  o.lambda = -1;
  EXPECT_THROW(f::fit_hierarchical_pspline(x, y, o), std::invalid_argument);
  o.lambda     = 0;
  o.root_basis = std::numeric_limits<int>::max();
  EXPECT_THROW(f::fit_hierarchical_pspline(x, y, o), std::invalid_argument);
  o.root_basis = 7;
  EXPECT_THROW(f::fit_hierarchical_pspline(x, y, o, {2}), std::invalid_argument);
  EXPECT_THROW(f::fit_hierarchical_pspline(x, y, o, {0, 0}), std::invalid_argument);
  EXPECT_THROW(f::fit_adaptive_pspline(x, y, {0}, {}, o), std::invalid_argument);
  auto a = f::fit_hierarchical_pspline(x, y, o);
  EXPECT_EQ(a.predict({-1}).front(), a.predict({0}).front());
  EXPECT_THROW(a.predict({std::numeric_limits<double>::quiet_NaN()}), std::invalid_argument);
  EXPECT_EQ(x, grid(21));
}
TEST(AdaptivePSpline, AnalyticDerivativesAndExactCurvatureIndicator) {
  auto                x = grid(81);
  std::vector<double> y;
  for (double t : x)
    y.push_back(1 + 3 * t + 2 * t * t);
  f::AdaptiveSplineOptions o;
  o.max_depth = 2;
  auto fit    = f::fit_hierarchical_pspline(x, y, o);
  auto first = fit.derivative(x), second = fit.derivative(x, 2), zero = fit.derivative(x, 0);
  for (std::size_t i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(first[i], 3 + 4 * x[i], 1e-10);
    EXPECT_NEAR(second[i], 4, 1e-8);
    EXPECT_DOUBLE_EQ(zero[i], fit.predict({x[i]})[0]);
  }
  f::JointAdaptiveSplineOptions search;
  search.spline             = o;
  search.curvature_priority = true;
  search.lambdas            = {0};
  auto joint                = f::fit_joint_adaptive_pspline(x, y, x, y, search);
  ASSERT_FALSE(joint.candidates.empty());
  EXPECT_NEAR(joint.candidates.front().mean_curvature_squared, 16, 1e-8);
  EXPECT_NEAR(joint.candidates.front().priority, 16, 1e-8);
  EXPECT_THROW(fit.derivative(x, -1), std::invalid_argument);
  EXPECT_THROW(fit.derivative(x, 4), std::invalid_argument);
  EXPECT_THROW(fit.derivative({NAN}), std::invalid_argument);
}
TEST(AdaptivePSpline, JointSingletonLambdaReproducesFixedLambdaPath) {
  auto                          x = grid(129), y = signal(x);
  f::JointAdaptiveSplineOptions search;
  search.lambdas           = {1e-6};
  search.spline.lambda     = 1e-6;
  search.spline.max_splits = 6;
  auto joint               = f::fit_joint_adaptive_pspline(x, y, x, y, search);
  auto fixed               = f::fit_adaptive_pspline(x, y, x, y, search.spline);
  EXPECT_EQ(joint.splits, fixed.splits);
  ASSERT_EQ(joint.path.size(), fixed.path.size());
  for (std::size_t i = 0; i < joint.path.size(); ++i) {
    EXPECT_DOUBLE_EQ(joint.path[i].lambda, 1e-6);
    EXPECT_NEAR(joint.path[i].validation_mse, fixed.path[i].validation_mse, 1e-14);
  }
}
TEST(AdaptivePSpline, JointRetunesEachTreeAndMatchesIndependentGridOracle) {
  auto x = grid(97), y = signal(x), vx = grid(151), vy = signal(vx);
  for (std::size_t i = 0; i < y.size(); ++i)
    y[i] += .08 * std::sin(83. * x[i]);
  f::JointAdaptiveSplineOptions search;
  search.spline.max_splits             = 5;
  search.spline.min_child_observations = 4;
  search.lambdas                       = {0, 1e-8, 1e-5, .01};
  search.lambda_refinements            = 2;
  auto joint                           = f::fit_joint_adaptive_pspline(x, y, vx, vy, search);
  ASSERT_GT(joint.path.size(), 2u);
  std::vector<int> splits;
  double           best = INFINITY;
  for (const auto& prefix : joint.path) {
    if (prefix.refined_cell >= 0)
      splits.push_back(prefix.refined_cell);
    auto o        = search.spline;
    o.lambda      = prefix.lambda;
    auto   replay = f::fit_hierarchical_pspline(x, y, o, splits);
    auto   pred   = replay.predict(vx);
    double score  = 0;
    for (std::size_t i = 0; i < vy.size(); ++i)
      score += std::pow(vy[i] - pred[i], 2) / vy.size();
    EXPECT_NEAR(prefix.validation_mse, score, 1e-13);
    best         = std::min(best, score);
    int selected = 0;
    for (const auto& trial : joint.tuning)
      if (trial.splits == prefix.splits) {
        if (trial.selected) {
          ++selected;
          EXPECT_DOUBLE_EQ(trial.lambda, prefix.lambda);
        }
        if (trial.fitted)
          EXPECT_LE(score, trial.validation_mse + 1e-12);
      }
    EXPECT_EQ(selected, 1);
    for (double lambda : search.lambdas) {
      o.lambda         = lambda;
      auto   candidate = f::fit_hierarchical_pspline(x, y, o, splits);
      auto   p         = candidate.predict(vx);
      double oracle    = 0;
      for (std::size_t i = 0; i < vy.size(); ++i)
        oracle += std::pow(vy[i] - p[i], 2) / vy.size();
      EXPECT_LE(score, oracle + 1e-12);
    }
  }
  EXPECT_NEAR(joint.validation_mse, best, 1e-12);
  EXPECT_EQ(joint.fit_evaluations, int(joint.tuning.size() + joint.candidates.size()));
}
TEST(AdaptivePSpline, JointRecursiveCurvaturePriorityAndMatchedLambdaAudit) {
  auto                          x = grid(257), y = signal(x);
  f::JointAdaptiveSplineOptions search;
  search.curvature_priority            = true;
  search.lambdas                       = {0, 1e-8, 1e-5};
  search.spline.max_depth              = 4;
  search.spline.max_splits             = 10;
  search.spline.min_child_observations = 4;
  auto joint                           = f::fit_joint_adaptive_pspline(x, y, x, y, search);
  ASSERT_GT(joint.path.size(), 4u);
  std::vector<int> splits;
  int              deepest = 0;
  for (std::size_t i = 1; i < joint.path.size(); ++i) {
    auto o                                     = search.spline;
    o.lambda                                   = joint.path[i - 1].lambda;
    auto                              parent   = f::fit_hierarchical_pspline(x, y, o, splits);
    const f::AdaptiveSplineCandidate* selected = nullptr;
    for (const auto& entry : joint.candidates)
      if (entry.step == int(i) && entry.selected)
        selected = &entry;
    ASSERT_NE(selected, nullptr);
    const auto cell = parent.cells[selected->cell];
    deepest         = std::max(deepest, cell.depth + 1);
    EXPECT_DOUBLE_EQ(selected->lambda, o.lambda);
    EXPECT_GT(selected->gain, 0);
    for (const auto& entry : joint.candidates)
      if (entry.step == int(i) && entry.fitted && entry.gain > 1e-12)
        EXPECT_LE(entry.priority, selected->priority + 1e-8);
    splits.push_back(selected->cell);
    auto   child = f::fit_hierarchical_pspline(x, y, o, splits);
    auto   a = parent.predict(x), b = child.predict(x);
    double before = 0, after = 0;
    int    n = 0;
    for (std::size_t j = 0; j < x.size(); ++j)
      if (x[j] >= cell.lower && (x[j] < cell.upper || cell.upper == 1)) {
        before += std::pow(y[j] - a[j], 2);
        after += std::pow(y[j] - b[j], 2);
        ++n;
      }
    EXPECT_NEAR(selected->mse_before, before / n, 1e-13);
    EXPECT_NEAR(selected->mse_after, after / n, 1e-13);
  }
  EXPECT_GE(deepest, 3);
}
TEST(AdaptivePSpline, JointSearchRejectsInvalidControls) {
  auto                          x = grid(33), y = signal(x);
  f::JointAdaptiveSplineOptions o;
  o.lambdas = {};
  EXPECT_THROW(f::fit_joint_adaptive_pspline(x, y, x, y, o), std::invalid_argument);
  o.lambdas = {-1};
  EXPECT_THROW(f::fit_joint_adaptive_pspline(x, y, x, y, o), std::invalid_argument);
  o.lambdas = {NAN};
  EXPECT_THROW(f::fit_joint_adaptive_pspline(x, y, x, y, o), std::invalid_argument);
  o.lambdas            = {1e-5};
  o.lambda_refinements = 9;
  EXPECT_THROW(f::fit_joint_adaptive_pspline(x, y, x, y, o), std::invalid_argument);
  o.lambda_refinements = 0;
  o.curvature_priority = true;
  o.spline.degree      = 1;
  EXPECT_THROW(f::fit_joint_adaptive_pspline(x, y, x, y, o), std::invalid_argument);
}
TEST(AdaptivePSpline, DeepHierarchiesRespectTheFinestGridBudget) {
  const auto          x = grid(129);
  std::vector<double> y;
  for (double t : x)
    y.push_back(1 + .3 * t);
  for (const auto [root, depth] : {std::pair{7, 7}, std::pair{5, 8}, std::pair{4, 9}}) {
    f::AdaptiveSplineOptions o;
    o.root_basis = root;
    o.max_depth  = depth;
    o.lambda     = 1e-5;
    std::vector<int> splits{0};
    for (int d = 1; d < depth; ++d)
      splits.push_back(2 * d - 1);
    auto fit = f::fit_hierarchical_pspline(x, y, o, splits);
    EXPECT_EQ(fit.basis.basis_size(), 515u);
    EXPECT_EQ(fit.cells.back().depth, depth);
    const auto pred = fit.predict(x);
    for (std::size_t i = 0; i < x.size(); ++i)
      EXPECT_NEAR(pred[i], y[i], 1e-6);
  }
  f::AdaptiveSplineOptions o;
  o.max_depth = 9; // 4 * 512 + 3 exceeds the 1024 cap.
  EXPECT_THROW(f::fit_hierarchical_pspline(x, y, o), std::invalid_argument);
  o.root_basis = 4;
  o.max_depth  = 10;
  EXPECT_THROW(f::fit_hierarchical_pspline(x, y, o), std::invalid_argument);
}
} // namespace
