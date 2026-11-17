#include <gtest/gtest.h>

#include <datamunge/algorithms/nested_sampling.hpp>
#include <numeric>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(NestedSamplingTrace, ConstantLikelihoodConservesAllPriorMassIncludingZeroIterations) {
  for (std::size_t n : {2, 3, 10, 100})
    for (std::size_t budget : {0, 1, 7, 50})
      for (double ll : {-1e300, -100., 0., 3., 1e300}) {
        auto t = nested_sampling_trace(
            [&](const auto&) { return ll; }, {-1., 2.}, {1., 4.}, n, budget, 7, 3, {0, 0});
        EXPECT_EQ(t.result.iterations, budget);
        EXPECT_NEAR(t.result.log_evidence, ll, 3e-13);
        long double mass = 0, weight = 0;
        for (auto& r : t.iterations) {
          mass += std::exp((long double)r.log_shell);
          EXPECT_EQ(r.accepts, 0);
          EXPECT_FALSE(r.strict_replacement);
        }
        for (auto& f : t.final_live)
          mass += std::exp((long double)f.log_prior_weight);
        for (auto w : t.result.posterior_log_weights)
          weight += std::exp((long double)w);
        EXPECT_NEAR(double(mass), 1., 1e-14);
        EXPECT_NEAR(double(weight), 1., 3e-13);
        EXPECT_NEAR(t.result.information, 0., 3e-13);
        std::size_t d = 2;
        EXPECT_EQ(
            t.captured_entries,
            2 * d + 6 + n * (1 + d) + budget * (1 + 4 * d) + budget * 3 * (1 + d) + n * (1 + d) +
                (n + budget) * (2 + d));
      }
}
TEST(NestedSamplingTrace, IndependentActualRngWalkMassAndWeightedInformation) {
  auto                  likelihood = [](const std::vector<double>& p) { return -2 * p[0] * p[0]; };
  constexpr std::size_t n = 5, budget = 20, steps = 4;
  auto            t = nested_sampling_trace(likelihood, {-1.}, {1.}, n, budget, 42, steps, {0, 0});
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> unit(0., 1.);
  std::normal_distribution<double>       normal(0., 1.);
  std::vector<double>                    live(n), ll(n);
  for (std::size_t i = 0; i < n; ++i) {
    live[i] = -1. + unit(rng) * 2.;
    ll[i]   = -2 * live[i] * live[i];
    EXPECT_EQ(t.initial_live[i].point[0], live[i]);
    EXPECT_EQ(t.initial_live[i].log_likelihood, ll[i]);
  }
  double      scale    = .2;
  long double evidence = 0, weighted_ll = 0;
  std::size_t callbacks = n;
  for (std::size_t k = 0; k < budget; ++k) {
    auto&       r     = t.iterations[k];
    std::size_t worst = std::min_element(ll.begin(), ll.end()) - ll.begin();
    EXPECT_EQ(r.worst, worst);
    EXPECT_EQ(r.dead_point[0], live[worst]);
    double mass = std::exp(-double(k) / n) - std::exp(-double(k + 1) / n);
    evidence += mass * std::exp(ll[worst]);
    weighted_ll += mass * std::exp(ll[worst]) * ll[worst];
    EXPECT_NEAR(std::exp(r.log_shell), mass, 2e-16);
    std::size_t survivor = worst;
    while (survivor == worst)
      survivor = std::size_t(unit(rng) * n);
    EXPECT_EQ(r.survivor, survivor);
    double      cur = live[survivor], value = ll[survivor];
    std::size_t accepted = 0;
    for (std::size_t s = 0; s < steps; ++s) {
      auto&  p      = t.proposals[k * steps + s];
      double x      = cur + scale * normal(rng);
      bool   inside = x >= -1 && x <= 1;
      EXPECT_EQ(p.point[0], x);
      EXPECT_EQ(p.evaluated, inside);
      if (inside) {
        ++callbacks;
        double v = -2 * x * x;
        EXPECT_EQ(p.log_likelihood, v);
        EXPECT_EQ(p.accepted, v > ll[worst]);
        if (v > ll[worst]) {
          cur   = x;
          value = v;
          ++accepted;
        }
      }
    }
    scale *= double(accepted) / steps > .5 ? 1.1 : .9;
    EXPECT_EQ(r.step_after[0], scale);
    EXPECT_EQ(r.replacement[0], cur);
    EXPECT_EQ(r.accepts, accepted);
    live[worst] = cur;
    ll[worst]   = value;
  }
  for (std::size_t i = 0; i < n; ++i) {
    double mass = std::exp(-double(budget) / n) / n;
    EXPECT_EQ(t.final_live[i].point[0], live[i]);
    evidence += mass * std::exp(ll[i]);
    weighted_ll += mass * std::exp(ll[i]) * ll[i];
  }
  EXPECT_EQ(t.likelihood_calls, callbacks);
  EXPECT_NEAR(t.result.log_evidence, std::log(double(evidence)), 2e-15);
  EXPECT_NEAR(
      t.raw_information, double(weighted_ll / evidence) - std::log(double(evidence)), 3e-15);
  auto same = nested_sampling(likelihood, {-1.}, {1.}, n, budget, 42, steps);
  EXPECT_EQ(same.posterior_samples, t.result.posterior_samples);
  EXPECT_EQ(same.log_evidence, t.result.log_evidence);
}
TEST(NestedSamplingTrace, CompleteValidationZeroLikelihoodQuotasAndCallbackEffects) {
  std::size_t calls = 0;
  auto        f     = [&](const auto&) {
    ++calls;
    return 0.;
  };
  EXPECT_THROW(nested_sampling(f, {}, {}), std::invalid_argument);
  EXPECT_THROW(nested_sampling(f, {0.}, {1., 2.}), std::invalid_argument);
  EXPECT_THROW(nested_sampling(f, {0.}, {0.}), std::invalid_argument);
  EXPECT_THROW(nested_sampling(f, {0.}, {1.}, 1), std::invalid_argument);
  EXPECT_THROW(nested_sampling(f, {0.}, {1.}, 2, 0, 0, 0), std::invalid_argument);
  EXPECT_THROW(nested_sampling_trace(f, {0.}, {1.}, 2, 0, 0, 1, {1, 0}), std::length_error);
  EXPECT_EQ(calls, 0);
  EXPECT_THROW(
      nested_sampling([](const auto&) { return std::nan(""); }, {0.}, {1.}, 2, 0),
      std::domain_error);
  EXPECT_THROW(
      nested_sampling([](const auto&) { return INFINITY; }, {0.}, {1.}, 2, 0), std::domain_error);
  auto zero = nested_sampling_trace([](const auto&) { return -INFINITY; }, {0.}, {1.}, 2, 3, 0, 1);
  EXPECT_TRUE(zero.zero_evidence);
  EXPECT_EQ(zero.result.log_evidence, -INFINITY);
  for (double w : zero.result.posterior_log_weights)
    EXPECT_EQ(w, -INFINITY);
  EXPECT_EQ(zero.result.information, 0.);
  auto t = nested_sampling_trace(f, {0.}, {1.}, 2, 3, 0, 1, {0, 0});
  EXPECT_NO_THROW(nested_sampling_trace(f, {0.}, {1.}, 2, 3, 0, 1, {t.captured_entries, 0}));
  EXPECT_THROW(
      nested_sampling_trace(f, {0.}, {1.}, 2, 3, 0, 1, {t.captured_entries - 1, 0}),
      std::length_error);
  auto stopped = nested_sampling_trace(f, {0.}, {1.}, 2, 100, 0, 1, {0, .5});
  EXPECT_TRUE(stopped.heuristic_stopped);
  EXPECT_TRUE(stopped.iterations.back().stop);
  EXPECT_NEAR(stopped.result.log_evidence, 0., 1e-14);
  EXPECT_THROW(
      nested_sampling(
          [](const auto&) -> double { throw std::runtime_error("caller"); }, {0.}, {1.}, 2, 0),
      std::runtime_error);
  EXPECT_NEAR(nested_sampling_demo_log_evidence(), std::log(2 * std::acos(-1.) / 400.), 1e-14);
}
