/// @file nested_sampling.cpp
/// @brief Canonical finite-walk nested sampling with complete guards and corrected mass accounting.
#include <datamunge/algorithms/nested_sampling.hpp>
#include <numbers>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private checked history accounting and shared producer.
namespace {
/// @brief Add nonnegative record counts without wrapping.
/// @param a First count.
/// @param b Second count.
/// @return Checked sum.
/// @throws std::length_error If count overflows.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("nested sampling: count overflow");
  return a + b;
}
/// @brief Multiply nonnegative geometry counts without wrapping.
/// @param a First count.
/// @param b Second count.
/// @return Checked product.
/// @throws std::length_error If count overflows.
std::size_t mul(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::length_error("nested sampling: geometry overflow");
  return a * b;
}
/// @brief Execute admitted live sampling, actual finite walks and corrected final quadrature.
/// @param log_likelihood Complete checked caller callback.
/// @param lo Original lower bounds.
/// @param hi Original upper bounds.
/// @param num_live Actual live count at least2.
/// @param max_iterations Actual replacement budget.
/// @param seed Local RNG seed.
/// @param walk_steps Positive proposal budget per replacement.
/// @param options Heuristic stop and history policies.
/// @param out Optional actual owned-history sink.
/// @return Actual evidence and normalized weighted points.
/// @throws std::invalid_argument If complete inputs are invalid before callbacks.
/// @throws std::length_error If count geometry/history policy is exceeded.
/// @throws std::domain_error If a callback returns NaN/+infinity.
/// @throws std::overflow_error If numeric scale/evidence accounting is unrepresentable.
NestedSamplingResult run(
    const std::function<double(const std::vector<double>&)>& log_likelihood,
    const std::vector<double>&                               lo,
    const std::vector<double>&                               hi,
    std::size_t                                              num_live,
    std::size_t                                              max_iterations,
    std::uint64_t                                            seed,
    std::size_t                                              walk_steps,
    NestedSamplingOptions                                    options,
    NestedSamplingTrace*                                     out) {
  const std::size_t d = lo.size();
  if (!log_likelihood || !d || hi.size() != d || num_live < 2 || !walk_steps ||
      !std::isfinite(options.remaining_fraction) || options.remaining_fraction < 0)
    throw std::invalid_argument("nested sampling: invalid complete controls");
  std::vector<double> width(d);
  for (std::size_t k = 0; k < d; ++k) {
    width[k] = hi[k] - lo[k];
    if (!std::isfinite(lo[k]) || !std::isfinite(hi[k]) || !std::isfinite(width[k]) || width[k] <= 0)
      throw std::invalid_argument("nested sampling: finite positive widths required");
  }
  const auto count = add(num_live, max_iterations);
  mul(count, d);
  mul(max_iterations, walk_steps);
  if (count > std::vector<std::vector<double>>{}.max_size() ||
      num_live > std::vector<double>{}.max_size())
    throw std::length_error("nested sampling: vector geometry");
  /// @brief Admit actual logical history units before their allocation or callback.
  auto admit = [&](std::size_t n) {
    if (!out)
      return;
    auto next = add(out->captured_entries, n);
    if (options.max_trace_entries && next > options.max_trace_entries)
      throw std::length_error("nested sampling: capture quota");
    out->captured_entries = next;
  };
  if (out) {
    admit(add(mul(2, d), 6));
    out->num_live           = num_live;
    out->max_iterations     = max_iterations;
    out->seed               = seed;
    out->walk_steps         = walk_steps;
    out->max_trace_entries  = options.max_trace_entries;
    out->remaining_fraction = options.remaining_fraction;
    out->lo                 = lo;
    out->hi                 = hi;
  }
  /// @brief Evaluate one admitted callback and reject invalid log-likelihood outputs.
  auto evaluate = [&](const std::vector<double>& p) {
    double v = log_likelihood(p);
    if (out)
      out->likelihood_calls = add(out->likelihood_calls, 1);
    if (std::isnan(v) || v == std::numeric_limits<double>::infinity())
      throw std::domain_error("nested sampling: callback must return finite or -infinity");
    return v;
  };
  std::mt19937_64                        rng(seed);
  std::uniform_real_distribution<double> unit(0., 1.);
  std::normal_distribution<double>       gauss(0., 1.);
  std::vector<std::vector<double>>       live(num_live, std::vector<double>(d));
  std::vector<double>                    live_logl(num_live), step(d);
  for (std::size_t k = 0; k < d; ++k)
    step[k] = .1 * width[k];
  for (std::size_t i = 0; i < num_live; ++i) {
    admit(add(1, d));
    for (std::size_t k = 0; k < d; ++k) {
      live[i][k] = lo[k] + unit(rng) * width[k];
      if (!std::isfinite(live[i][k]))
        throw std::overflow_error("nested sampling: prior draw overflow");
    }
    live_logl[i] = evaluate(live[i]);
    if (out)
      out->initial_live.push_back({live[i], live_logl[i]});
  }
  NestedSamplingResult result;
  std::vector<double>  values, prior_log_weights;
  const double         invN = 1. / double(num_live), logN = std::log(double(num_live)),
               shell_offset = std::log(-std::expm1(-invN));
  double logX = 0, logZ = -std::numeric_limits<double>::infinity();
  double likelihood_offset     = -std::numeric_limits<double>::infinity();
  double relative_log_evidence = -std::numeric_limits<double>::infinity();
  /// @brief Retain an actual quadrature point and accumulate its unnormalized contribution.
  auto retain = [&](const std::vector<double>& p, double ll, double log_prior_weight) {
    admit(add(2, d));
    result.posterior_samples.push_back(p);
    values.push_back(ll);
    prior_log_weights.push_back(log_prior_weight);
    if (ll != -std::numeric_limits<double>::infinity()) {
      if (ll > likelihood_offset) {
        relative_log_evidence += likelihood_offset - ll;
        likelihood_offset = ll;
      }
      relative_log_evidence =
          detail::logaddexp(relative_log_evidence, log_prior_weight + (ll - likelihood_offset));
      logZ = likelihood_offset + relative_log_evidence;
      if (!std::isfinite(logZ))
        throw std::overflow_error("nested sampling: log evidence overflow");
    }
  };
  for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    admit(add(1, mul(4, d)));
    NestedSamplingIteration rec;
    std::size_t             worst = 0;
    for (std::size_t i = 1; i < num_live; ++i)
      if (live_logl[i] < live_logl[worst])
        worst = i;
    rec.worst           = worst;
    rec.threshold       = live_logl[worst];
    rec.log_mass_before = logX;
    rec.log_shell       = logX + shell_offset;
    rec.log_weight      = rec.log_shell + rec.threshold;
    if (out) {
      rec.dead_point  = live[worst];
      rec.step_before = step;
    }
    retain(live[worst], rec.threshold, rec.log_shell);
    rec.log_evidence_dead = logZ;
    std::size_t survivor  = worst;
    while (survivor == worst) {
      survivor = static_cast<std::size_t>(unit(rng) * double(num_live));
      if (survivor >= num_live)
        survivor = num_live - 1;
    }
    rec.survivor    = survivor;
    auto   cur      = live[survivor];
    double cur_logl = live_logl[survivor];
    for (std::size_t s = 0; s < walk_steps; ++s) {
      admit(add(1, d));
      NestedSamplingProposal proposal;
      proposal.iteration = iter;
      proposal.step      = s;
      proposal.point.resize(d);
      proposal.in_box = true;
      for (std::size_t k = 0; k < d; ++k) {
        proposal.point[k] = cur[k] + step[k] * gauss(rng);
        if (!std::isfinite(proposal.point[k]) || proposal.point[k] < lo[k] ||
            proposal.point[k] > hi[k])
          proposal.in_box = false;
      }
      if (proposal.in_box) {
        proposal.evaluated      = true;
        proposal.log_likelihood = evaluate(proposal.point);
        proposal.accepted       = proposal.log_likelihood > rec.threshold;
        if (proposal.accepted) {
          cur      = proposal.point;
          cur_logl = proposal.log_likelihood;
          ++rec.accepts;
        }
      }
      if (out)
        out->proposals.push_back(std::move(proposal));
    }
    const double factor = double(rec.accepts) / double(walk_steps) > .5 ? 1.1 : .9;
    for (auto& st : step) {
      st *= factor;
      if (!std::isfinite(st))
        throw std::overflow_error("nested sampling: adaptive proposal scale overflow");
    }
    live[worst]                    = cur;
    live_logl[worst]               = cur_logl;
    result.iterations              = iter + 1;
    logX                           = -double(result.iterations) / double(num_live);
    rec.log_mass_after             = logX;
    rec.replacement_log_likelihood = cur_logl;
    rec.strict_replacement         = cur_logl > rec.threshold;
    rec.log_remaining_estimate     = *std::max_element(live_logl.begin(), live_logl.end()) + logX;
    const double remaining_ratio =
        (*std::max_element(live_logl.begin(), live_logl.end()) - likelihood_offset) + logX -
        relative_log_evidence;
    rec.stop = options.remaining_fraction > 0 && std::isfinite(logZ) &&
               remaining_ratio < std::log(options.remaining_fraction);
    const bool stop = rec.stop;
    if (out) {
      rec.replacement = cur;
      rec.step_after  = step;
      out->iterations.push_back(std::move(rec));
      if (out->iterations.back().stop)
        out->heuristic_stopped = true;
    }
    if (stop)
      break;
  }
  for (std::size_t i = 0; i < num_live; ++i) {
    admit(add(1, d));
    const double lp = logX - logN, lw = lp + live_logl[i];
    if (out)
      out->final_live.push_back({live[i], live_logl[i], lp, lw});
    retain(live[i], live_logl[i], lp);
  }
  result.log_evidence = logZ;
  const bool zero     = logZ == -std::numeric_limits<double>::infinity();
  if (out)
    out->zero_evidence = zero;
  long double information = 0;
  for (std::size_t i = 0; i < prior_log_weights.size(); ++i) {
    const double normalized =
        zero ? -std::numeric_limits<double>::infinity()
             : prior_log_weights[i] + (values[i] - likelihood_offset) - relative_log_evidence;
    result.posterior_log_weights.push_back(normalized);
    if (std::isfinite(normalized))
      information +=
          std::exp(static_cast<long double>(normalized)) *
          ((static_cast<long double>(values[i]) - likelihood_offset) - relative_log_evidence);
  }
  if (!std::isfinite(information) || std::abs(information) > std::numeric_limits<double>::max())
    throw std::overflow_error("nested sampling: information overflow");
  if (out)
    out->raw_information = double(information);
  result.information        = std::max(0., double(information));
  result.log_evidence_error = std::sqrt(result.information / double(num_live));
  return result;
}
} // namespace
NestedSamplingResult nested_sampling(
    const std::function<double(const std::vector<double>&)>& f,
    const std::vector<double>&                               lo,
    const std::vector<double>&                               hi,
    std::size_t                                              n,
    std::size_t                                              iterations,
    std::uint64_t                                            seed,
    std::size_t                                              steps) {
  return run(f, lo, hi, n, iterations, seed, steps, {0, 1e-4}, nullptr);
}
NestedSamplingTrace nested_sampling_trace(
    const std::function<double(const std::vector<double>&)>& f,
    const std::vector<double>&                               lo,
    const std::vector<double>&                               hi,
    std::size_t                                              n,
    std::size_t                                              iterations,
    std::uint64_t                                            seed,
    std::size_t                                              steps,
    NestedSamplingOptions                                    options) {
  NestedSamplingTrace t;
  t.result = run(f, lo, hi, n, iterations, seed, steps, options, &t);
  return t;
}
double nested_sampling_demo_log_likelihood(const std::vector<double>& p) {
  if (p.size() != 2 || !std::isfinite(p[0]) || !std::isfinite(p[1]))
    throw std::invalid_argument("nested Gaussian: two finite coordinates required");
  const double x = p[0] - 2., y = p[1] + 1.;
  return -.5 * (x * x + y * y);
}
double nested_sampling_demo_log_evidence() {
  double z = 1;
  for (double mu : {2., -1.})
    z *= std::sqrt(std::numbers::pi / 2.) *
         (std::erf((10. - mu) / std::sqrt(2.)) - std::erf((-10. - mu) / std::sqrt(2.))) / 20.;
  return std::log(z);
}
} // namespace datamunge::algorithms
