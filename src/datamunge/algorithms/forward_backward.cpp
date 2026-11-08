/// @file forward_backward.cpp
/// @brief Checked original scaled recursion and independent widened path definitions.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/forward_backward.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Internal validation, numerical guards and once-only inference kernel.
namespace forward_backward_detail {
/// @brief Add entry counts without wrap.
/// @param a First count.
/// @param b Second count.
/// @return Exact sum.
/// @throws std::length_error Overflow.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::size_t(-1) - a)
    throw std::length_error("HMM count overflow");
  return a + b;
}
/// @brief Multiply entry/work counts without wrap.
/// @param a First count.
/// @param b Second count.
/// @return Exact product.
/// @throws std::length_error Overflow.
std::size_t mul(std::size_t a, std::size_t b) {
  if (a && b > std::size_t(-1) / a)
    throw std::length_error("HMM count overflow");
  return a * b;
}
/// @brief Reject nonfinite actual numeric arithmetic.
/// @tparam T Double or long-double type.
/// @param x Actual intermediate value.
/// @throws std::overflow_error Nonfinite value.
template<class T>
void finite(T x) {
  if (!std::isfinite(x))
    throw std::overflow_error("HMM nonfinite arithmetic");
}
/// @brief Validate one complete stochastic row without changing its weights.
/// @param row Finite probabilities expected to sum to1 within absolute1e-12.
/// @throws std::invalid_argument Nonfinite/out-of-range weight or invalid sum.
void distribution(const std::vector<double>& row) {
  long double sum = 0;
  for (double x : row) {
    if (!std::isfinite(x) || x < 0 || x > 1)
      throw std::invalid_argument("HMM probability outside finite [0,1]");
    sum += static_cast<long double>(x);
  }
  if (!std::isfinite(sum) || std::abs(sum - 1.L) > 1e-12L)
    throw std::invalid_argument("HMM distribution must sum to1");
}
/// @brief Validate all model shapes, then all weights, then all observations.
/// @param h Caller-owned model.
/// @param obs Zero-based symbols.
/// @throws std::invalid_argument Invalid model shape/probability/sum.
/// @throws std::out_of_range Invalid symbol.
void validate(const HiddenMarkovModel& h, const std::vector<std::size_t>& obs) {
  const auto n = h.num_states(), m = h.num_symbols();
  if (h.transition.size() != n || h.emission.size() != n || (n && m == 0))
    throw std::invalid_argument("HMM invalid model shape");
  for (const auto& r : h.transition)
    if (r.size() != n)
      throw std::invalid_argument("HMM transition must be square");
  for (const auto& r : h.emission)
    if (r.size() != m)
      throw std::invalid_argument("HMM emissions must be rectangular");
  if (n) {
    distribution(h.initial);
    for (const auto& r : h.transition)
      distribution(r);
    for (const auto& r : h.emission)
      distribution(r);
  }
  for (auto o : obs)
    if (o >= m)
      throw std::out_of_range("hmm: observation symbol out of range");
}
/// @brief Probe structural support only after an actual zero predictive mass.
/// @param h Validated model.
/// @param obs Validated symbols.
/// @param length Prefix length, at least1.
/// @param w Actual Boolean state/pair visit counts updated.
/// @return Whether any strictly positive-probability prefix path exists.
/// @throws std::bad_alloc Boolean scratch allocation fails.
bool supported(
    const HiddenMarkovModel&        h,
    const std::vector<std::size_t>& obs,
    std::size_t                     length,
    ForwardBackwardWork&            w) {
  const auto        n = h.num_states();
  std::vector<bool> a(n), b(n);
  for (std::size_t i = 0; i < n; ++i) {
    a[i] = h.initial[i] > 0 && h.emission[i][obs[0]] > 0;
    ++w.support_initial_tests;
  }
  for (std::size_t t = 1; t < length; ++t) {
    std::fill(b.begin(), b.end(), false);
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t i = 0; i < n; ++i) {
        b[j] = b[j] || (a[i] && h.transition[i][j] > 0 && h.emission[j][obs[t]] > 0);
        ++w.support_pair_visits;
      }
    a.swap(b);
  }
  return std::any_of(a.begin(), a.end(), [](bool x) { return x; });
}
/// @brief Execute the original scaled recursion once, with checked admission and actual counts.
/// @param hmm Model validated before allocation/admission.
/// @param obs Symbols validated after the complete model.
/// @param cap Whole owned entry cap; zero disables optional cap.
/// @return Owned result and actual work.
/// @throws std::invalid_argument Model invalid.
/// @throws std::out_of_range Symbol invalid.
/// @throws std::length_error Counts/capacity/cap invalid.
/// @throws std::overflow_error Nonfinite actual arithmetic.
/// @throws std::underflow_error Positive support lost entirely to underflow.
/// @throws std::bad_alloc Allocation fails.
ForwardBackwardOutcome
run(const HiddenMarkovModel& hmm, const std::vector<std::size_t>& obs, std::size_t cap) {
  validate(hmm, obs);
  const std::size_t n = hmm.num_states();
  const std::size_t t = obs.size();

  ForwardBackwardOutcome outcome;
  auto&                  r = outcome.result;
  auto&                  w = outcome.work;
  w.states                 = n;
  w.symbols                = hmm.num_symbols();
  w.observations           = t;
  const auto entries       = forward_backward_entries(n, t);
  if (cap && entries > cap)
    throw std::length_error("forward backward whole result cap");
  if (n && t) {
    mul(mul(t - 1, n), n);
    mul(t, n);
  }
  bool impossible  = false;
  r.log_likelihood = 0.0;
  if (t == 0 || n == 0)
    return outcome;

  r.alpha.assign(t, std::vector<double>(n, 0.0));
  r.beta.assign(t, std::vector<double>(n, 0.0));
  r.gamma.assign(t, std::vector<double>(n, 0.0));
  r.scale.assign(t, 0.0);

  // Forward pass with scaling.
  double sum0 = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    r.alpha[0][i] = hmm.initial[i] * hmm.emission[i][obs[0]];
    finite(r.alpha[0][i]);
    sum0 += r.alpha[0][i];
    finite(sum0);
    ++w.initial_products;
  }
  if (sum0 == 0 && !supported(hmm, obs, 1, w))
    impossible = true;
  else if (sum0 == 0)
    throw std::underflow_error("forward backward positive initial mass underflow");
  r.scale[0] = sum0 > 0.0 ? 1.0 / sum0 : 0.0;
  finite(r.scale[0]);
  for (std::size_t i = 0; i < n; ++i) {
    r.alpha[0][i] *= r.scale[0];
    finite(r.alpha[0][i]);
    ++w.forward_scalings;
  }
  ++w.completed_forward_steps;

  for (std::size_t step = 1; step < t; ++step) {
    double sum = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        acc += r.alpha[step - 1][i] * hmm.transition[i][j];
        finite(acc);
        ++w.forward_transition_terms;
      }
      r.alpha[step][j] = acc * hmm.emission[j][obs[step]];
      finite(r.alpha[step][j]);
      sum += r.alpha[step][j];
      finite(sum);
      ++w.forward_emission_products;
    }
    if (sum == 0 && !impossible) {
      if (supported(hmm, obs, step + 1, w))
        throw std::underflow_error("forward backward positive predictive mass underflow");
      impossible = true;
    }
    r.scale[step] = sum > 0.0 ? 1.0 / sum : 0.0;
    finite(r.scale[step]);
    for (std::size_t j = 0; j < n; ++j) {
      r.alpha[step][j] *= r.scale[step];
      finite(r.alpha[step][j]);
      ++w.forward_scalings;
    }
    ++w.completed_forward_steps;
  }

  // Backward pass, reusing the forward scaling factors.
  for (std::size_t i = 0; i < n; ++i) {
    r.beta[t - 1][i] = r.scale[t - 1];
    ++w.backward_initializations;
  }
  for (std::size_t step = t - 1; step > 0; --step)
    for (std::size_t i = 0; i < n; ++i) {
      double acc = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        acc += hmm.transition[i][j] * hmm.emission[j][obs[step]] * r.beta[step][j];
        finite(acc);
        ++w.backward_transition_terms;
      }

      r.beta[step - 1][i] = acc * r.scale[step - 1];
      finite(r.beta[step - 1][i]);
      ++w.backward_scalings;
    }

  // Posteriors gamma_t(i), normalized per time step.
  for (std::size_t step = 0; step < t; ++step) {
    double norm = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      r.gamma[step][i] = r.alpha[step][i] * r.beta[step][i];
      finite(r.gamma[step][i]);
      norm += r.gamma[step][i];
      finite(norm);
      ++w.posterior_products;
    }
    if (norm > 0.0)
      for (std::size_t i = 0; i < n; ++i) {
        r.gamma[step][i] /= norm;
        finite(r.gamma[step][i]);
        ++w.posterior_divisions;
      }
    else if (!impossible)
      throw std::underflow_error("forward backward positive posterior norm underflow");
  }

  for (std::size_t step = 0; step < t; ++step) {
    r.log_likelihood -= (r.scale[step] > 0.0 ? std::log(r.scale[step]) : 0.0);
    finite(r.log_likelihood);
    w.scale_logs += r.scale[step] > 0;
  }
  if (impossible)
    r.log_likelihood = -std::numeric_limits<double>::infinity();
  return outcome;
}

/// @brief Multiply a joint prefix by one validated probability with underflow detection.
/// @param a Widened nonnegative joint prefix.
/// @param b Finite probability in[0,1].
/// @return Widened product.
/// @throws std::overflow_error Product nonfinite.
/// @throws std::underflow_error Positive product rounds to zero.
long double product(long double a, double b) {
  const auto p = a * static_cast<long double>(b);
  finite(p);
  if (a > 0 && b > 0 && p == 0)
    throw std::underflow_error("HMM widened positive joint product underflow");
  return p;
}
/// @brief Evaluate a fully validated path with widened products in time order.
/// @param h Validated model.
/// @param obs Validated symbols.
/// @param q Validated same-length state path.
/// @return Widened joint mass,1 for empty observations.
/// @throws std::overflow_error Nonfinite arithmetic.
/// @throws std::underflow_error Positive product underflows.
long double joint(
    const HiddenMarkovModel&        h,
    const std::vector<std::size_t>& obs,
    const std::vector<std::size_t>& q) {
  if (obs.empty())
    return 1;
  long double p = product(h.initial[q[0]], h.emission[q[0]][obs[0]]);
  for (std::size_t t = 1; t < obs.size(); ++t) {
    p = product(p, h.transition[q[t - 1]][q[t]]);
    p = product(p, h.emission[q[t]][obs[t]]);
  }
  return p;
}
} // namespace forward_backward_detail
std::size_t forward_backward_entries(std::size_t n, std::size_t t) {
  using namespace forward_backward_detail;
  if (!n || !t)
    return 0;
  if (n > std::vector<double>().max_size() || t > std::vector<std::vector<double>>().max_size() ||
      t > std::vector<double>().max_size())
    throw std::length_error("HMM result capacity");
  return add(mul(4, t), mul(3, mul(t, n)));
}
ForwardBackwardResult forward_backward(
    const HiddenMarkovModel&        h,
    const std::vector<std::size_t>& obs) {
  return forward_backward_detail::run(h, obs, 0).result;
}
ForwardBackwardOutcome forward_backward_work(
    const HiddenMarkovModel&        h,
    const std::vector<std::size_t>& obs,
    std::size_t                     cap) {
  return forward_backward_detail::run(h, obs, cap);
}
std::size_t hmm_enumeration_paths(std::size_t n, std::size_t t) {
  if (!t)
    return 1;
  if (n < 2)
    return n;
  std::size_t count = 1;
  for (std::size_t i = 0; i < t; ++i)
    count = forward_backward_detail::mul(count, n);
  return count;
}
long double hmm_path_probability(
    const HiddenMarkovModel&        h,
    const std::vector<std::size_t>& obs,
    const std::vector<std::size_t>& q) {
  forward_backward_detail::validate(h, obs);
  if (q.size() != obs.size())
    throw std::invalid_argument("HMM path length differs from observation length");
  for (auto s : q)
    if (s >= h.num_states())
      throw std::out_of_range("HMM state ID out of range");
  return forward_backward_detail::joint(h, obs, q);
}
HmmPathEnumeration hmm_enumerate_paths(
    const HiddenMarkovModel&        h,
    const std::vector<std::size_t>& obs,
    std::size_t                     cap) {
  using namespace forward_backward_detail;
  validate(h, obs);
  const auto n = h.num_states(), t = obs.size(), paths = hmm_enumeration_paths(n, t);
  mul(t, paths);
  mul(t, n);
  if (n > std::vector<long double>().max_size() ||
      t > std::vector<std::vector<long double>>().max_size() ||
      t > std::vector<std::size_t>().max_size())
    throw std::length_error("HMM enumeration capacity");
  if (cap && paths > cap)
    throw std::length_error("HMM enumeration path cap");
  HmmPathEnumeration r;
  r.gamma.assign(t, std::vector<long double>(n));
  std::vector<std::size_t> q(t);
  for (std::size_t code = 0; code < paths; ++code) {
    auto digits = code;
    for (std::size_t step = 0; step < t; ++step) {
      q[step] = digits % n;
      digits /= n;
    }
    const auto p = joint(h, obs, q);
    r.probability += p;
    finite(r.probability);
    if (p > r.best_probability) {
      r.best_probability = p;
      r.best_states      = q;
    }
    for (std::size_t step = 0; step < t; ++step) {
      r.gamma[step][q[step]] += p;
      finite(r.gamma[step][q[step]]);
    }
    ++r.paths;
  }
  if (r.probability > 0) {
    r.log_likelihood = std::log(r.probability);
    finite(r.log_likelihood);
    for (auto& row : r.gamma)
      for (auto& x : row)
        x /= r.probability;
  } else
    r.log_likelihood = -std::numeric_limits<long double>::infinity();
  return r;
}
HiddenMarkovModel forward_backward_example_model() {
  return {{.6, .4}, {{.7, .3}, {.4, .6}}, {{.5, .4, .1}, {.1, .3, .6}}};
}
std::vector<std::size_t> forward_backward_example_observations() {
  return {0, 0, 1, 2, 2, 2, 1, 0, 1, 2, 2, 1, 0, 0};
}
} // namespace datamunge::algorithms
