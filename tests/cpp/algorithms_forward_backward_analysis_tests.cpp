#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <datamunge/algorithms/hmm.hpp>
#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
using namespace datamunge::algorithms;
namespace {
struct Reference {
  long double                           mass = 0, best = 0;
  std::vector<std::vector<long double>> gamma;
};
// Independent recursive state tree; no scaled recursions or integer path-code decoder.
Reference reference(const HiddenMarkovModel& h, const std::vector<std::size_t>& obs) {
  Reference r;
  r.gamma.assign(obs.size(), std::vector<long double>(h.num_states()));
  std::vector<std::size_t>                      path;
  std::function<void(std::size_t, long double)> visit = [&](std::size_t t, long double p) {
    if (t == obs.size()) {
      r.mass += p;
      r.best = std::max(r.best, p);
      for (std::size_t k = 0; k < t; ++k)
        r.gamma[k][path[k]] += p;
      return;
    }
    for (std::size_t s = 0; s < h.num_states(); ++s) {
      const long double prior = t ? h.transition[path.back()][s] : h.initial[s];
      path.push_back(s);
      visit(t + 1, p * prior * static_cast<long double>(h.emission[s][obs[t]]));
      path.pop_back();
    }
  };
  visit(0, 1);
  if (r.mass > 0)
    for (auto& row : r.gamma)
      for (auto& x : row)
        x /= r.mass;
  return r;
}
void compare(const HiddenMarkovModel& h, const std::vector<std::size_t>& o) {
  const auto expected   = reference(h, o);
  const auto actual     = forward_backward(h, o);
  const auto enumerated = hmm_enumerate_paths(h, o);
  EXPECT_NEAR(double(enumerated.probability), double(expected.mass), 1e-15);
  EXPECT_NEAR(double(enumerated.best_probability), double(expected.best), 1e-15);
  EXPECT_EQ(enumerated.paths, hmm_enumeration_paths(h.num_states(), o.size()));
  if (expected.mass > 0) {
    EXPECT_NEAR(actual.log_likelihood, double(std::log(expected.mass)), 2e-12);
    EXPECT_NEAR(double(enumerated.log_likelihood), double(std::log(expected.mass)), 1e-14);
    EXPECT_NEAR(
        double(hmm_path_probability(h, o, enumerated.best_states)), double(expected.best), 1e-15);
  } else {
    EXPECT_EQ(actual.log_likelihood, -std::numeric_limits<double>::infinity());
    EXPECT_EQ(enumerated.log_likelihood, -std::numeric_limits<long double>::infinity());
    EXPECT_TRUE(enumerated.best_states.empty());
  }
  ASSERT_EQ(actual.gamma.size(), o.size());
  for (std::size_t t = 0; t < o.size(); ++t)
    for (std::size_t s = 0; s < h.num_states(); ++s) {
      EXPECT_NEAR(actual.gamma[t][s], double(expected.gamma[t][s]), 2e-13);
      EXPECT_NEAR(double(enumerated.gamma[t][s]), double(expected.gamma[t][s]), 1e-15);
    }
}
long double suffix(
    const HiddenMarkovModel&        h,
    const std::vector<std::size_t>& o,
    std::size_t                     t,
    std::size_t                     s) {
  if (t + 1 == o.size())
    return 1;
  long double sum = 0;
  for (std::size_t v = 0; v < h.num_states(); ++v)
    sum += (long double)h.transition[s][v] * h.emission[v][o[t + 1]] * suffix(h, o, t + 1, v);
  return sum;
}
} // namespace
TEST(ForwardBackwardAnalysis, ExhaustiveDyadicModelsAgainstRecursivePathTree) {
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      for (int c = 0; c < 3; ++c)
        for (int d = 0; d < 3; ++d)
          for (int e = 0; e < 3; ++e) {
            const HiddenMarkovModel h{
                {a * .5, 1 - a * .5},
                {{b * .5, 1 - b * .5}, {c * .5, 1 - c * .5}},
                {{d * .5, 1 - d * .5}, {e * .5, 1 - e * .5}}};
            for (std::size_t t = 0; t <= 5; ++t)
              for (unsigned code = 0; code < (1u << t); ++code) {
                std::vector<std::size_t> o(t);
                for (std::size_t k = 0; k < t; ++k)
                  o[k] = (code >> k) & 1;
                compare(h, o);
              }
          }
}
TEST(ForwardBackwardAnalysis, RandomModelsAndFilteringBackwardIdentities) {
  std::mt19937_64 rng(120);
  for (int fixture = 0; fixture < 200; ++fixture) {
    const auto        n = 1 + rng() % 3, m = 1 + rng() % 4;
    HiddenMarkovModel h;
    h.initial.resize(n);
    h.transition.assign(n, std::vector<double>(n));
    h.emission.assign(n, std::vector<double>(m));
    auto normalize = [&](auto& row) {
      double sum = 0;
      for (auto& v : row) {
        v = 1 + double(rng() % 30);
        sum += v;
      }
      for (auto& v : row)
        v /= sum;
    };
    normalize(h.initial);
    for (auto& r : h.transition)
      normalize(r);
    for (auto& r : h.emission)
      normalize(r);
    std::vector<std::size_t> o(1 + rng() % 6);
    for (auto& v : o)
      v = rng() % m;
    compare(h, o);
    const auto r = forward_backward(h, o);
    for (std::size_t t = 0; t < o.size(); ++t) {
      const std::vector<std::size_t> prefix(o.begin(), o.begin() + t + 1);
      const auto                     expected = reference(h, prefix);
      long double                    norm = 0, scale = 1;
      for (std::size_t k = t; k < o.size(); ++k)
        scale *= r.scale[k];
      for (std::size_t s = 0; s < n; ++s) {
        EXPECT_NEAR(r.alpha[t][s], double(expected.gamma.back()[s]), 1e-13);
        EXPECT_NEAR(r.beta[t][s], double(scale * suffix(h, o, t, s)), 1e-11);
        norm += (long double)r.alpha[t][s] * r.beta[t][s];
      }
      EXPECT_NEAR(double(norm), r.scale[t], r.scale[t] * 1e-12);
    }
  }
}
TEST(ForwardBackwardAnalysis, AnalyticTwoTimesAndFutureInformation) {
  const HiddenMarkovModel h{{.5, .5}, {{.8, .2}, {.2, .8}}, {{.9, .1}, {.1, .9}}};
  const auto              r = forward_backward(h, {0, 1});
  const auto              e = hmm_enumerate_paths(h, {0, 1});
  EXPECT_NEAR(double(e.probability), .154, 1e-16);
  EXPECT_NEAR(r.alpha[0][1], .1, 1e-16);
  EXPECT_NEAR(r.gamma[0][1], .037 / .154, 1e-15);
  EXPECT_NEAR(r.gamma[1][1], .117 / .154, 1e-15);
  EXPECT_DOUBLE_EQ(r.scale[0], 2);
  EXPECT_NEAR(r.scale[1], 1 / .308, 1e-15);
  EXPECT_GT(r.beta[0][1], 1);
  EXPECT_NEAR(double(hmm_path_probability(h, {0, 1}, {0, 0})), .036, 1e-16);
  EXPECT_NEAR(double(hmm_path_probability(h, {0, 1}, {0, 1})), .081, 1e-16);
  EXPECT_NEAR(double(hmm_path_probability(h, {0, 1}, {1, 0})), .001, 1e-16);
  EXPECT_NEAR(double(hmm_path_probability(h, {0, 1}, {1, 1})), .036, 1e-16);
}
TEST(ForwardBackwardAnalysis, ImpossibleSequenceHasUndefinedPosteriorPlaceholders) {
  const HiddenMarkovModel h{{1}, {{1}}, {{1, 0}}};
  for (const std::vector<std::size_t> o :
       std::vector<std::vector<std::size_t>>{{1}, {0, 1}, {0, 0, 1, 0}}) {
    const auto r = forward_backward_work(h, o);
    EXPECT_EQ(r.result.log_likelihood, -std::numeric_limits<double>::infinity());
    EXPECT_EQ(r.work.posterior_divisions, 0u);
    EXPECT_EQ(r.work.support_initial_tests, 1u);
    for (auto& row : r.result.gamma)
      for (auto p : row)
        EXPECT_EQ(p, 0);
    for (auto& row : r.result.beta)
      for (auto p : row)
        EXPECT_EQ(p, 0);
    compare(h, o);
  }
  const HiddenMarkovModel conflict{{.5, .5}, {{1, 0}, {0, 1}}, {{1, 0}, {0, 1}}};
  const auto              r = forward_backward_work(conflict, {0, 1, 0});
  EXPECT_EQ(r.result.log_likelihood, -std::numeric_limits<double>::infinity());
  EXPECT_EQ(r.work.support_pair_visits, 4u);
  EXPECT_DOUBLE_EQ(r.result.alpha[0][0], 1);
  EXPECT_DOUBLE_EQ(r.result.scale[0], 2);
  EXPECT_DOUBLE_EQ(r.result.scale[1], 0);
  EXPECT_DOUBLE_EQ(r.result.scale[2], 0);
}
TEST(ForwardBackwardAnalysis, PositiveUnderflowAndReciprocalOverflowAreDistinct) {
  const auto              tiny = std::numeric_limits<double>::denorm_min();
  const HiddenMarkovModel lost{{.5, .5}, {{1, 0}, {0, 1}}, {{1, tiny}, {1, 0}}};
  EXPECT_THROW(forward_backward(lost, {1}), std::underflow_error);
  EXPECT_THROW(forward_backward(lost, {0, 1}), std::underflow_error);
  EXPECT_GT(hmm_enumerate_paths(lost, {1}).probability, 0);
  const HiddenMarkovModel reciprocal{{1}, {{1}}, {{1, 1e-320}}};
  EXPECT_THROW(forward_backward(reciprocal, {1}), std::overflow_error);
  EXPECT_GT(hmm_enumerate_paths(reciprocal, {1}).probability, 0);
  const HiddenMarkovModel finite{{1}, {{1}}, {{1, 1e-308}}};
  const auto              r = forward_backward(finite, {1});
  EXPECT_TRUE(std::isfinite(r.log_likelihood));
  EXPECT_DOUBLE_EQ(r.alpha[0][0], 1);
  EXPECT_DOUBLE_EQ(r.gamma[0][0], 1);
  const HiddenMarkovModel widened{{1}, {{1}}, {{1, 1e-300}}};
  EXPECT_THROW(
      hmm_path_probability(
          widened, std::vector<std::size_t>(100, 1), std::vector<std::size_t>(100, 0)),
      std::underflow_error);
}
TEST(ForwardBackwardAnalysis, FullModelAndObservationDomainBeforeQuotaAndEmpty) {
  const auto                     valid = forward_backward_example_model();
  std::vector<HiddenMarkovModel> bad;
  auto                           h = valid;
  h.transition.pop_back();
  bad.push_back(h);
  h = valid;
  h.transition[1].pop_back();
  bad.push_back(h);
  h = valid;
  h.emission[1].clear();
  bad.push_back(h);
  h = valid;
  h.emission.pop_back();
  bad.push_back(h);
  h            = valid;
  h.initial[0] = -.1;
  bad.push_back(h);
  h                = valid;
  h.emission[0][0] = 1.1;
  bad.push_back(h);
  h                  = valid;
  h.transition[0][0] = std::numeric_limits<double>::infinity();
  bad.push_back(h);
  h                = valid;
  h.emission[1][2] = std::numeric_limits<double>::quiet_NaN();
  bad.push_back(h);
  h         = valid;
  h.initial = {.7, .7};
  bad.push_back(h);
  h               = valid;
  h.transition[0] = {0, 0};
  bad.push_back(h);
  h             = valid;
  h.emission[0] = {0, 0, 0};
  bad.push_back(h);
  h          = {};
  h.emission = {{1}};
  bad.push_back(h);
  for (const auto& b : bad) {
    EXPECT_THROW(forward_backward(b, {}), std::invalid_argument);
    EXPECT_THROW(forward_backward_work(b, {0, 1}, 1), std::invalid_argument);
    EXPECT_THROW(hmm_enumerate_paths(b, {0, 1}, 1), std::invalid_argument);
    EXPECT_THROW(hmm_path_probability(b, {}, {}), std::invalid_argument);
  }
  EXPECT_THROW(forward_backward_work(valid, {0, 3}, 1), std::out_of_range);
  EXPECT_THROW(hmm_enumerate_paths(valid, {3}, 1), std::out_of_range);
  EXPECT_THROW(hmm_path_probability(valid, {0}, {2}), std::out_of_range);
  EXPECT_THROW(hmm_path_probability(valid, {0}, {}), std::invalid_argument);
  EXPECT_THROW(forward_backward({}, {0}), std::out_of_range);
  auto close    = valid;
  close.initial = {.5, .5 + 5e-13};
  EXPECT_NO_THROW(forward_backward(close, {0}));
  EXPECT_DOUBLE_EQ(close.initial[1], .5 + 5e-13);
  close.initial = {.5, .5 + 2e-12};
  EXPECT_THROW(forward_backward(close, {}), std::invalid_argument);
}
TEST(ForwardBackwardAnalysis, WholeEntryAndEnumerationAdmissionOwnership) {
  EXPECT_EQ(forward_backward_entries(2, 2), 20u);
  EXPECT_EQ(forward_backward_entries(2, 14), 140u);
  EXPECT_EQ(forward_backward_entries(0, SIZE_MAX), 0u);
  EXPECT_EQ(forward_backward_entries(SIZE_MAX, 0), 0u);
  EXPECT_THROW(forward_backward_entries(SIZE_MAX, 1), std::length_error);
  EXPECT_THROW(forward_backward_entries(1, SIZE_MAX), std::length_error);
  auto                     h = forward_backward_example_model();
  std::vector<std::size_t> o{0, 1};
  EXPECT_THROW(forward_backward_work(h, o, 19), std::length_error);
  auto result = forward_backward_work(h, o, 20);
  EXPECT_NO_THROW(forward_backward_work(h, o, 0));
  EXPECT_THROW(hmm_enumerate_paths(h, {0, 1, 2, 2}, 15), std::length_error);
  EXPECT_EQ(hmm_enumerate_paths(h, {0, 1, 2, 2}, 16).paths, 16u);
  EXPECT_EQ(hmm_enumerate_paths(h, {0, 1, 2, 2}, 0).paths, 16u);
  const auto e      = hmm_enumerate_paths(h, o);
  const auto before = result.result.gamma[0][1];
  h.initial         = {1, 0};
  o[0]              = 2;
  EXPECT_DOUBLE_EQ(result.result.gamma[0][1], before);
  EXPECT_NEAR(double(e.probability), .1246, 1e-15);
  result.result.alpha[0][0] = 999;
  EXPECT_DOUBLE_EQ(result.result.gamma[0][1], before);
  EXPECT_EQ(hmm_enumeration_paths(0, 0), 1u);
  EXPECT_EQ(hmm_enumeration_paths(0, SIZE_MAX), 0u);
  EXPECT_EQ(hmm_enumeration_paths(1, SIZE_MAX), 1u);
  EXPECT_EQ(hmm_enumeration_paths(2, 10), 1024u);
  EXPECT_THROW(hmm_enumeration_paths(2, SIZE_MAX), std::length_error);
  EXPECT_THROW(hmm_enumeration_paths(SIZE_MAX, 2), std::length_error);
  EXPECT_THROW(
      hmm_enumerate_paths(forward_backward_example_model(), std::vector<std::size_t>(100, 0), 0),
      std::length_error);
}
TEST(ForwardBackwardAnalysis, ActualCountsParityEmptyAndSingleTime) {
  const auto h = forward_backward_example_model();
  for (std::size_t T : {0u, 1u, 2u, 14u, 100u}) {
    const std::vector<std::size_t> o(T, 0);
    const auto                     r = forward_backward_work(h, o);
    const auto                     s = forward_backward(h, o);
    EXPECT_EQ(r.result.alpha, s.alpha);
    EXPECT_EQ(r.result.beta, s.beta);
    EXPECT_EQ(r.result.gamma, s.gamma);
    EXPECT_EQ(r.result.scale, s.scale);
    EXPECT_EQ(
        std::bit_cast<std::uint64_t>(r.result.log_likelihood),
        std::bit_cast<std::uint64_t>(s.log_likelihood));
    const auto& w = r.work;
    EXPECT_EQ(w.initial_products, T ? 2u : 0u);
    EXPECT_EQ(w.forward_transition_terms, T ? (T - 1) * 4 : 0u);
    EXPECT_EQ(w.backward_transition_terms, T ? (T - 1) * 4 : 0u);
    EXPECT_EQ(w.forward_emission_products, T ? (T - 1) * 2 : 0u);
    EXPECT_EQ(w.forward_scalings, T * 2);
    EXPECT_EQ(w.backward_initializations, T ? 2u : 0u);
    EXPECT_EQ(w.backward_scalings, T ? (T - 1) * 2 : 0u);
    EXPECT_EQ(w.posterior_products, T * 2);
    EXPECT_EQ(w.posterior_divisions, T * 2);
    EXPECT_EQ(w.scale_logs, T);
    EXPECT_EQ(w.completed_forward_steps, T);
    EXPECT_EQ(w.support_pair_visits, 0u);
  }
  const auto empty = forward_backward({}, {});
  EXPECT_TRUE(empty.alpha.empty());
  EXPECT_DOUBLE_EQ(empty.log_likelihood, 0);
  const auto e = hmm_enumerate_paths({}, {});
  EXPECT_EQ(e.paths, 1u);
  EXPECT_EQ(e.probability, 1);
  EXPECT_EQ(e.best_probability, 1);
  EXPECT_EQ(e.log_likelihood, 0);
  EXPECT_TRUE(e.gamma.empty());
  EXPECT_EQ(hmm_path_probability({}, {}, {}), 1);
  const auto single = forward_backward(h, {0});
  for (std::size_t i = 0; i < h.num_states(); ++i)
    EXPECT_NEAR(single.alpha[0][i], single.gamma[0][i], 2e-16);
}
TEST(ForwardBackwardAnalysis, LongProductsUnderflowWhileScaledLogRemainsFinite) {
  const HiddenMarkovModel        h{{1}, {{1}}, {{.5, .5}}};
  const std::vector<std::size_t> o(10000, 0);
  const auto                     r = forward_backward_work(h, o);
  EXPECT_NEAR(r.result.log_likelihood, -10000 * std::log(2.), 1e-8);
  EXPECT_DOUBLE_EQ(std::exp(r.result.log_likelihood), 0);
  EXPECT_EQ(r.work.forward_transition_terms, 9999u);
  EXPECT_EQ(r.work.backward_transition_terms, 9999u);
  EXPECT_EQ(r.work.posterior_divisions, 10000u);
  for (std::size_t t = 0; t < o.size(); ++t) {
    EXPECT_DOUBLE_EQ(r.result.alpha[t][0], 1);
    EXPECT_DOUBLE_EQ(r.result.beta[t][0], 2);
    EXPECT_DOUBLE_EQ(r.result.gamma[t][0], 1);
    EXPECT_DOUBLE_EQ(r.result.scale[t], 2);
  }
}
TEST(ForwardBackwardAnalysis, MarginalModesCanUseForbiddenTransitionAndEnumerationTies) {
  const HiddenMarkovModel  h{{.4, .3, .3}, {{0, 0, 1}, {0, 1, 0}, {0, 1, 0}}, {{1}, {1}, {1}}};
  const auto               r = forward_backward(h, {0, 0});
  std::vector<std::size_t> modes;
  for (const auto& row : r.gamma)
    modes.push_back(std::max_element(row.begin(), row.end()) - row.begin());
  EXPECT_EQ(modes, (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(hmm_path_probability(h, {0, 0}, modes), 0);
  const auto v = viterbi(h, {0, 0});
  EXPECT_EQ(v.states, (std::vector<std::size_t>{0, 2}));
  EXPECT_NEAR(double(hmm_path_probability(h, {0, 0}, v.states)), .4, 1e-16);
  EXPECT_NEAR(std::exp(v.log_probability), .4, 1e-16);
  const HiddenMarkovModel tied{{.5, .5}, {{.5, .5}, {.5, .5}}, {{1}, {1}}};
  const auto              e = hmm_enumerate_paths(tied, {0, 0, 0});
  EXPECT_EQ(e.best_states, (std::vector<std::size_t>{0, 0, 0}));
  EXPECT_EQ(e.best_probability, .125L);
}
