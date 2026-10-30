#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/bayes/distributions.hpp>
#include <datamunge/bayes/inla_diagnostics.hpp>
#include <datamunge/stats/inla_mixed_model.hpp>
#include <limits>
namespace b = datamunge::bayes;
using M     = datamunge::linalg::DenseMatrix<double>;
using DF    = datamunge::dstruct::DataFrame;
namespace {
b::INLALikelihoodFn normal() {
  return b::inla_gaussian_likelihood([](const auto&) { return 1.; });
}
b::INLAOptions quick() {
  b::INLAOptions o;
  o.mode_population_size = 12;
  o.mode_max_generations = 35;
  return o;
}
DF groups() {
  DF f;
  f.add_column("y", std::vector<double>{0, 0, 0, 1, 0, 1, 1, 1});
  f.add_column("x", std::vector<double>{-1, 0, 1, 2, -1, 0, 1, 2});
  f.add_column("g", std::vector<std::string>{"a", "a", "a", "a", "b", "b", "b", "b"});
  return f;
}
} // namespace
TEST(INLAConditional, ExactScalarNormalMeanVarianceAndNormalizedEvidence) {
  const auto fit = b::INLA().conditional(M(3, 1, 1), {1, 2, 3}, normal(), M(1, 1, 1));
  EXPECT_NEAR(fit.mode[0], 1.5, 1e-14);
  EXPECT_NEAR(fit.covariance(0, 0), .25, 1e-14);
  EXPECT_NEAR(
      fit.log_marginal_likelihood,
      -1.5 * std::log(2 * std::acos(-1.)) - .5 * std::log(4.) - 2.5,
      1e-14);
  EXPECT_LT(fit.gradient_inf, 1e-12);
  EXPECT_EQ(fit.iterations, 2);
  EXPECT_NEAR(b::inla_conditional_table(fit).double_at("sd", 0), .5, 1e-14);
  const auto ref = b::inla_scalar_reference({1, 2, 3}, normal(), 1, {}, -6, 8, 4001);
  EXPECT_NEAR(ref.mean[0], fit.mode[0], 1e-12);
  EXPECT_NEAR(ref.sd[0], .5, 1e-12);
  EXPECT_NEAR(ref.log_evidence, fit.log_marginal_likelihood, 1e-12);
  EXPECT_LT(ref.edge_mass, 1e-30);
}
TEST(INLAConditional, ExactMultivariateGaussianCovariance) {
  M          d(3, 2, {1, -1, 1, 0, 1, 1}), q(2, 2, {1, 0, 0, 2});
  const auto f = b::INLA().conditional(d, {1, 2, 3}, normal(), q);
  EXPECT_NEAR(f.mode[0], 1.5, 1e-14);
  EXPECT_NEAR(f.mode[1], .5, 1e-14);
  EXPECT_NEAR(f.covariance(0, 0), .25, 1e-14);
  EXPECT_NEAR(f.covariance(1, 1), .25, 1e-14);
  EXPECT_NEAR(f.covariance(0, 1), 0, 1e-14);
  EXPECT_NEAR(
      f.log_marginal_likelihood,
      -1.5 * std::log(2 * std::acos(-1.)) + .5 * std::log(2.) - .5 * std::log(16.) - 2,
      1e-14);
}
TEST(INLAConditional, SmallBernoulliSkewDiffersFromModeGaussian) {
  const auto f = b::INLA().conditional(
      M(4, 1, 1), {1, 1, 1, 0}, b::inla_binomial_logit_likelihood(), M(1, 1, .25));
  const auto r = b::inla_scalar_reference(
      {1, 1, 1, 0}, b::inla_binomial_logit_likelihood(), .25, {}, -12, 12, 12001);
  EXPECT_GT(r.mean[0], f.mode[0] + .1);
  EXPECT_GT(r.sd[0], std::sqrt(f.covariance(0, 0)));
  EXPECT_LT(f.gradient_inf, 1e-8);
  EXPECT_LT(r.edge_mass, 1e-10);
}
TEST(INLAConditional, RejectsBudgetExhaustionAndInvalidPrecision) {
  auto o                  = quick();
  o.newton_max_iterations = 1;
  EXPECT_THROW(
      b::INLA(o).conditional(M(3, 1, 1), {1, 2, 3}, normal(), M(1, 1, 1)), std::runtime_error);
  EXPECT_THROW(b::INLA().conditional(M(2, 1, 1), {0, 1}, normal(), M(1, 1, 0)), std::runtime_error);
  EXPECT_THROW(
      b::INLA().conditional(M(2, 2, 1), {0, 1}, normal(), M(2, 2, {1, 0, 1, 2})),
      std::runtime_error);
  EXPECT_THROW(b::INLA().conditional(M(0, 1), {}, normal(), M(1, 1, 1)), std::invalid_argument);
  EXPECT_THROW(
      b::INLA().conditional(M(1, 1, 1), {NAN}, normal(), M(1, 1, 1)), std::invalid_argument);
}
TEST(INLAGrid, MixtureCovarianceAndLawOfTotalVariance) {
  auto o                = quick();
  o.grid_points_per_dim = 31;
  o.grid_span           = 5;
  const auto f          = b::INLA(o).fit(
      M(3, 1, 1),
      {1, 2, 3},
      normal(),
      [](const auto& t) { return M(1, 1, std::exp(t[0])); },
      [](const auto& t) { return b::normal_lpdf(t[0], 0, 1); },
      {-6},
      {6},
      {0});
  double mean = 0, within = 0, between = 0, sum = 0;
  for (std::size_t k = 0; k < f.theta_grid_weights.size(); ++k) {
    const double w = f.theta_grid_weights[k], q = std::exp(f.theta_grid[k]), v = 1 / (3 + q),
                 m = 6 * v;
    sum += w;
    mean += w * m;
    within += w * v;
    EXPECT_NEAR(f.conditional_means[k][0], m, 1e-12);
    EXPECT_NEAR(f.conditional_covariances[k](0, 0), v, 1e-12);
  }
  for (std::size_t k = 0; k < f.theta_grid_weights.size(); ++k)
    between += f.theta_grid_weights[k] * std::pow(f.conditional_means[k][0] - mean, 2);
  EXPECT_NEAR(sum, 1, 1e-14);
  EXPECT_NEAR(f.latent_mean[0], mean, 1e-12);
  EXPECT_NEAR(f.latent_within_variance[0], within, 1e-12);
  EXPECT_NEAR(f.latent_between_variance[0], between, 1e-12);
  EXPECT_NEAR(f.latent_covariance(0, 0), within + between, 1e-12);
  EXPECT_NEAR(f.latent_sd[0] * f.latent_sd[0], within + between, 1e-12);
  EXPECT_EQ(b::inla_grid_table(f).nrows(), f.theta_grid_weights.size());
  EXPECT_EQ(b::inla_latent_summary(f).nrows(), 1);
  EXPECT_GT(f.latent_between_variance[0], 0);
  const auto lo = b::inla_latent_quantile(f, 0, .025), median = b::inla_latent_quantile(f, 0, .5),
             hi = b::inla_latent_quantile(f, 0, .975);
  EXPECT_LT(lo, median);
  EXPECT_LT(median, hi);
  EXPECT_THROW(b::inla_latent_quantile(f, 0, 1), std::invalid_argument);
  EXPECT_EQ(b::inla_latent_density(f, 0, {lo, median, hi}).nrows(), 3);
  EXPECT_THROW(b::inla_latent_density(f, 1, {0}), std::invalid_argument);
  EXPECT_THROW(b::inla_latent_density(f, 0, {NAN}), std::invalid_argument);
}
TEST(INLAGrid, BoundsContainAllPrecisionProbesAndFlatCurvatureFallback) {
  auto o                = quick();
  o.grid_points_per_dim = 11;
  std::size_t calls     = 0;
  const auto  f         = b::INLA(o).fit(
      M(2, 1, 1),
      {0, 1},
      normal(),
      [&](const auto& t) {
        ++calls;
        EXPECT_GE(t[0], .1);
        EXPECT_LE(t[0], .8);
        return M(1, 1, 1);
      },
      [](const auto& t) { return t[0]; },
      {.1},
      {.8},
      {.5});
  EXPECT_GT(calls, 0);
  EXPECT_GT(f.grid_skipped_points, 0);
  EXPECT_TRUE(f.grid_covariance_fallback);
  for (double t : f.theta_grid) {
    EXPECT_GE(t, .1);
    EXPECT_LE(t, .8);
  }
  EXPECT_TRUE(std::isfinite(f.log_marginal_likelihood));
}
TEST(INLAGrid, EBHasNoIntegratedEvidenceOrBetweenVariance) {
  auto o       = quick();
  o.strategy   = b::INLAIntegrationStrategy::EmpiricalBayes;
  const auto f = b::INLA(o).fit(
      M(2, 1, 1),
      {0, 1},
      normal(),
      [](const auto&) { return M(1, 1, 1); },
      [](const auto& t) { return b::normal_lpdf(t[0], 0, 1); },
      {-4},
      {4},
      {0});
  EXPECT_TRUE(std::isnan(f.log_marginal_likelihood));
  EXPECT_TRUE(std::isfinite(f.log_posterior_at_mode));
  EXPECT_EQ(f.theta_grid_weights.size(), 1);
  EXPECT_EQ(f.latent_between_variance[0], 0);
  EXPECT_EQ(f.theta_sd[0], 0);
  EXPECT_NEAR(b::inla_latent_quantile(f, 0, .5), f.latent_mean[0], 1e-14);
  const auto d = b::inla_latent_density(f, 0, {f.latent_mean[0]});
  EXPECT_NEAR(
      d.double_at("density", 0),
      1 / std::sqrt(2 * std::acos(-1.) * f.latent_covariance(0, 0)),
      1e-14);
}
TEST(INLAGrid, ValidatesControlsBoxesAndInputs) {
  auto o                = quick();
  o.grid_points_per_dim = 1;
  EXPECT_THROW(b::INLA{o}, std::invalid_argument);
  o                  = quick();
  o.newton_tolerance = NAN;
  EXPECT_THROW(b::INLA{o}, std::invalid_argument);
  o                      = quick();
  o.mode_population_size = 3;
  EXPECT_THROW(b::INLA{o}, std::invalid_argument);
  auto q     = [](const auto&) { return M(1, 1, 1); };
  auto prior = [](const auto&) { return 0.; };
  EXPECT_THROW(
      b::INLA().fit(M(1, 1, 1), {0}, normal(), q, prior, {1}, {0}, {.5}), std::invalid_argument);
  EXPECT_THROW(
      b::INLA().fit(M(1, 1, NAN), {0}, normal(), q, prior, {0}, {1}, {.5}), std::invalid_argument);
  EXPECT_THROW(
      b::INLA().fit(M(1, 1, 1), {0}, normal(), q, prior, {0}, {1}, {2}), std::invalid_argument);
  o                     = quick();
  o.grid_points_per_dim = 101;
  EXPECT_THROW(
      b::INLA(o).fit(
          M(1, 1, 1), {0}, normal(), q, prior, {0, 0, 0, 0}, {1, 1, 1, 1}, {.5, .5, .5, .5}),
      std::invalid_argument);
}
TEST(INLALikelihood, StableTailsAndStrictObservationSupport) {
  auto f = b::inla_binomial_logit_likelihood();
  EXPECT_TRUE(std::isfinite(f(0, 1000, {}).log_density));
  EXPECT_EQ(f(1, -1000, {}).score, 1);
  EXPECT_GT(f(1, 40, {}).curvature, 0);
  EXPECT_THROW(f(.5, 0, {}), std::invalid_argument);
  EXPECT_THROW(b::inla_poisson_log_likelihood()(.5, 0, {}), std::invalid_argument);
  EXPECT_THROW(b::inla_poisson_log_likelihood()(-1, 0, {}), std::invalid_argument);
  EXPECT_THROW(
      b::inla_gaussian_likelihood([](const auto&) { return 0.; })(0, 0, {}), std::invalid_argument);
}
TEST(INLAReference, RejectsInvalidInputs) {
  EXPECT_THROW(b::inla_scalar_reference({}, normal(), 1, {}, -1, 1), std::invalid_argument);
  EXPECT_THROW(b::inla_scalar_reference({0}, normal(), 0, {}, -1, 1), std::invalid_argument);
  EXPECT_THROW(b::inla_scalar_reference({0}, normal(), 1, {}, 1, -1), std::invalid_argument);
  EXPECT_THROW(b::inla_conditional_table({}), std::invalid_argument);
  EXPECT_THROW(b::inla_grid_table({}), std::invalid_argument);
}
TEST(INLAMixedDiagnostics, RetainedRowsProperPriorsAndNamedMixtureIntervals) {
  auto f = groups();
  f.set_value("g", 1, std::optional<std::string>{});
  datamunge::stats::INLAMixedModelOptions o;
  o.family                = datamunge::stats::INLAMixedModelFamily::Binomial;
  o.fixed_effect_prior_sd = 2.5;
  o.grid_points_per_dim   = 21;
  o.mode_population_size  = 12;
  o.mode_max_generations  = 35;
  const datamunge::stats::INLAMixedModel m(f, "y ~ x + (1 | g)", o);
  EXPECT_EQ(m.observations(), 7);
  EXPECT_EQ(m.used_rows(), (std::vector<std::size_t>{0, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(m.latent_design().rows(), 7);
  EXPECT_EQ(m.response().size(), 7);
  EXPECT_EQ(m.fixed_effects_frame().nrows(), 2);
  const auto r = m.random_effects_frame();
  EXPECT_EQ(r.nrows(), 2);
  EXPECT_LT(r.double_at("q025", 0), r.double_at("q975", 0));
  const auto v = m.variance_components_frame();
  EXPECT_EQ(v.nrows(), 1);
  EXPECT_GT(v.double_at("mean_sd", 0), 0);
  EXPECT_GT(v.double_at("sd_of_sd", 0), 0);
  EXPECT_NEAR(v.double_at("mean_sd", 0), m.posterior().theta_mean[0], 1e-12);
  auto       q = f.head(1).mutate("g", std::vector<std::string>{"unseen"});
  const auto p = m.predict(q)[0];
  EXPECT_NEAR(
      p, 1 / (1 + std::exp(-(m.fixed_effects_mean()[0] - m.fixed_effects_mean()[1]))), 1e-14);
  EXPECT_THROW(m.predict(q.mutate("g", std::vector<double>{1})), std::invalid_argument);
  auto bad = f;
  bad.set_value("y", 0, .5);
  EXPECT_THROW(
      (datamunge::stats::INLAMixedModel(bad, "y ~ x + (1 | g)", o)), std::invalid_argument);
}
TEST(INLAMixedDiagnostics, DistinctNumericKeysAndPriorValidation) {
  auto f = groups().mutate(
      "g",
      std::vector<double>{
          1.0000001, 1.0000001, 1.0000001, 1.0000001, 1.0000002, 1.0000002, 1.0000002, 1.0000002});
  datamunge::stats::INLAMixedModelOptions o;
  o.family               = datamunge::stats::INLAMixedModelFamily::Binomial;
  o.strategy             = b::INLAIntegrationStrategy::EmpiricalBayes;
  o.mode_population_size = 8;
  o.mode_max_generations = 8;
  const datamunge::stats::INLAMixedModel m(f, "y ~ x + (1 | g)", o);
  EXPECT_EQ(m.num_groups(), 2);
  o.lambda_bound = .5;
  EXPECT_NO_THROW((datamunge::stats::INLAMixedModel(f, "y ~ x + (1 | g)", o)));
  o.fixed_effect_prior_sd = 0;
  EXPECT_THROW((datamunge::stats::INLAMixedModel(f, "y ~ x + (1 | g)", o)), std::invalid_argument);
}
TEST(INLAGrid, ClippedScalarCellsIntegrateAConstantHyperparameterExactly) {
  auto o                = quick();
  o.grid_points_per_dim = 11;
  const auto f          = b::INLA(o).fit(
      M(3, 1, 1),
      {1, 2, 3},
      normal(),
      [](const auto&) { return M(1, 1, 1); },
      [](const auto&) { return -std::log(.7); },
      {.1},
      {.8},
      {.5});
  const auto exact  = b::INLA().conditional(M(3, 1, 1), {1, 2, 3}, normal(), M(1, 1, 1));
  double     volume = 0;
  for (double v : f.theta_grid_cell_volumes)
    volume += v;
  EXPECT_NEAR(volume, .7, 1e-14);
  EXPECT_NEAR(f.log_marginal_likelihood, exact.log_marginal_likelihood, 1e-13);
  for (std::size_t i = 0; i < f.theta_grid_weights.size(); ++i)
    EXPECT_NEAR(f.theta_grid_weights[i], f.theta_grid_cell_volumes[i] / .7, 1e-14);
}
TEST(INLAMixedDiagnostics, MissingRandomOnlySlopeNeverSilentlyBecomesZero) {
  auto                                    f = groups();
  datamunge::stats::INLAMixedModelOptions o;
  o.family               = datamunge::stats::INLAMixedModelFamily::Binomial;
  o.strategy             = b::INLAIntegrationStrategy::EmpiricalBayes;
  o.mode_population_size = 8;
  o.mode_max_generations = 8;
  const datamunge::stats::INLAMixedModel m(f, "y ~ 1 + (0 + x | g)", o);
  auto                                   query = f.head(1);
  query.set_value("x", 0, std::optional<double>{});
  EXPECT_TRUE(std::isnan(m.predict(query)[0]));
  EXPECT_THROW(m.predict(query.select({"g"})), std::invalid_argument);
  EXPECT_TRUE(m.predict(f.head(0)).empty());
}
