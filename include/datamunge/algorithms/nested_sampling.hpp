#pragma once
/// @file nested_sampling.hpp
/// @brief Checked box-prior nested sampling with actual finite-walk and mass-accounting observations.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <vector>
namespace datamunge::algorithms {
/// @brief Evidence and normalized weighted dead-plus-final-live points; field order retained.
/// @par Example
/// @code
/// dm::algorithms::NestedSamplingResult r;std::print("{}\n",r.iterations); // 0
/// @endcode
/// @see nested_sampling_trace()
struct NestedSamplingResult {
  double log_evidence{
      -std::numeric_limits<double>::
          infinity()}; ///< Estimated log evidence; negative infinity for all-zero observed likelihood.
  double log_evidence_error{
      0}; ///< Heuristic sqrt(max(H,0)/N), not a calibrated confidence interval.
  double information{
      0}; ///< Clipped weighted information estimate in nats, including final live points.
  std::size_t iterations{0}; ///< Actual discarded/replaced live points.
  std::vector<std::vector<double>>
      posterior_samples; ///< Owned dead points followed by all final live points; correlated weighted records.
  std::vector<double>
      posterior_log_weights; ///< Normalized log weights, or all negative infinity when evidence is zero.
};
/// @brief Actual capture and heuristic stopping policies.
/// @par Example
/// @code
/// dm::algorithms::NestedSamplingOptions o;std::print("{}\n",o.max_trace_entries); // 1000000
/// @endcode
/// @see nested_sampling_trace()
struct NestedSamplingOptions {
  std::size_t max_trace_entries{1000000}; ///< Owned logical history units; zero disables.
  double      remaining_fraction{
      1e-4}; ///< Positive relative observed-live remaining-evidence heuristic, or zero to disable.
};
/// @brief Actual callback-evaluated point with its log likelihood.
/// @par Example
/// @code
/// dm::algorithms::NestedSamplingPoint p;std::print("{}\n",p.point.size()); // 0
/// @endcode
/// @see nested_sampling_trace()
struct NestedSamplingPoint {
  std::vector<double> point;             ///< Owned actual coordinates.
  double              log_likelihood{0}; ///< Actual finite or negative-infinite callback value.
};
/// @brief Actual finite-walk proposal and its likelihood decision.
/// @par Example
/// @code
/// dm::algorithms::NestedSamplingProposal p;std::print("{}\n",p.evaluated); // false
/// @endcode
/// @see nested_sampling_trace()
struct NestedSamplingProposal {
  std::size_t iteration{0}; ///< Zero-based actual replacement index.
  std::size_t step{0};      ///< Zero-based actual walk step.
  std::vector<double>
       point;         ///< Actual proposed coordinates, including out-of-box/nonfinite coordinates.
  bool in_box{false}; ///< All coordinates finite and within the closed prior box.
  bool evaluated{false}; ///< Whether callback ran; false leaves log_likelihood as placeholder zero.
  double log_likelihood{0}; ///< Actual callback value when evaluated.
  bool   accepted{false};   ///< Actual strict log-likelihood improvement over discarded threshold.
};
/// @brief Actual discarded point, replacement, mass and adaptation decision.
/// @par Example
/// @code
/// dm::algorithms::NestedSamplingIteration i;std::print("{}\n",i.accepts); // 0
/// @endcode
/// @see nested_sampling_trace()
struct NestedSamplingIteration {
  std::size_t         worst{0};     ///< Actual first lowest-likelihood live index.
  std::size_t         survivor{0};  ///< Actual uniformly selected different surviving index.
  std::vector<double> dead_point;   ///< Actual discarded coordinates.
  std::vector<double> replacement;  ///< Actual final walk coordinates.
  std::vector<double> step_before;  ///< Actual per-dimension proposal scales before walk.
  std::vector<double> step_after;   ///< Actual scales after acceptance adaptation.
  double              threshold{0}; ///< Actual discarded log likelihood.
  double              replacement_log_likelihood{
      0};    ///< Actual replacement value, possibly tied if the walk did not move.
  std::size_t accepts{0}; ///< Actual accepted proposals.
  double      log_mass_before{0}; ///< Deterministic estimated prior mass log X before discard.
  double      log_mass_after{0};  ///< Estimated log X after discard, -(iteration+1)/N.
  double      log_shell{0};       ///< Actual log(X_before-X_after), computed stably with expm1.
  double      log_weight{0};      ///< Actual unnormalized dead-point log evidence contribution.
  double log_evidence_dead{0};    ///< Actual cumulative dead-point log evidence after this discard.
  double log_remaining_estimate{
      0}; ///< Actual maximum observed live log likelihood plus log X after replacement.
  bool strict_replacement{
      false};       ///< Whether replacement log likelihood exceeds the discarded threshold.
  bool stop{false}; ///< Whether the observed-live heuristic actually stopped this run.
};
/// @brief Actual final-live contribution using remaining prior mass.
/// @par Example
/// @code
/// dm::algorithms::NestedSamplingFinal f;std::print("{}\n",f.point.size()); // 0
/// @endcode
/// @see nested_sampling_trace()
struct NestedSamplingFinal {
  std::vector<double> point;             ///< Owned final live coordinates in live-index order.
  double              log_likelihood{0}; ///< Actual final live likelihood value.
  double log_prior_weight{0};            ///< Actual log X_final-log N, not a discarded shell width.
  double log_weight{0};                  ///< Actual unnormalized final log evidence contribution.
};
/// @brief Owned actual initialization, every proposal, replacement and final contribution.
/// Capture counts2d+6 initial scalar/bound units; each initial/proposal/final
/// record1+d; each iteration1+4d; each returned posterior record2+d.
/// Ordinary result storage and transient live/walk arrays are not peak-RAM claims.
/// @par Example
/// @code
/// auto t=dm::algorithms::nested_sampling_trace([](const auto&){return 0.;},{0.},{1.},2,0);std::print("{}\n",t.final_live.size()); // 2
/// @endcode
/// @see nested_sampling_trace()
struct NestedSamplingTrace {
  std::size_t   num_live{0};           ///< Original validated live count.
  std::size_t   max_iterations{0};     ///< Original replacement budget.
  std::uint64_t seed{0};               ///< Original local MT19937_64 seed.
  std::size_t   walk_steps{0};         ///< Original positive proposal budget.
  std::size_t   max_trace_entries{0};  ///< Original history policy, zero disables.
  double        remaining_fraction{0}; ///< Original observed-live heuristic policy, zero disables.

  std::vector<double>              lo;     ///< Owned validated lower bounds.
  std::vector<double>              hi;     ///< Owned validated upper bounds.
  NestedSamplingResult             result; ///< Actual weighted result from this same producer.
  std::vector<NestedSamplingPoint> initial_live; ///< Actual prior draws and callback values.
  std::vector<NestedSamplingProposal>
      proposals; ///< Actual proposals in iteration/step order, no replay.
  std::vector<NestedSamplingIteration> iterations; ///< Actual complete replacement decisions.
  std::vector<NestedSamplingFinal>
              final_live; ///< All actual final live points and corrected mass weights.
  std::size_t likelihood_calls{
      0}; ///< Actual callbacks, including initialization, excluding outside proposals.
  std::size_t captured_entries{0}; ///< Actual admitted logical history units.
  bool        zero_evidence{
      false}; ///< All retained likelihood contributions are zero; no normalized posterior exists.
  bool heuristic_stopped{
      false}; ///< Actual observed-live heuristic stop, never a proven remaining bound.
  double raw_information{
      0}; ///< Unclipped normalized weighted information sum, including final live points.
};
/// @brief Shared logarithmic addition helper retained for existing native/generated consumers.
namespace detail {
/// @brief Compute log(exp(a)+exp(b)) for finite values or negative infinity.
/// @param a First log summand.
/// @param b Second log summand.
/// @return Stable log sum, including both-negative-infinity case.
/// @par Example
/// @code
/// std::print("{:.6f}\n",dm::algorithms::detail::logaddexp(0.,0.)); // 0.693147
/// @endcode
/// @see nested_sampling()
inline double logaddexp(double a, double b) {
  if (a == -std::numeric_limits<double>::infinity())
    return b;
  if (b == -std::numeric_limits<double>::infinity())
    return a;
  return std::max(a, b) + std::log1p(std::exp(-std::fabs(a - b)));
}
} // namespace detail
/// @brief Estimate box-prior evidence with deterministic mean-log-mass shrinkage and finite walks.
/// Uses X_i=exp(-i/N), dead shell X_before-X_after, final live weights X_final/N.
/// Fixes the legacy final-shell/remaining-mass error and includes final points in H.
/// Starting survivor can tie the threshold; no-move/tied replacements are retained,
/// not advertised as independent constrained-prior draws. Gaussian finite walks and
/// adaptive scales provide no multimodal mixing guarantee. The maximum observed live
/// likelihood times remaining mass is a heuristic, not an upper bound. sqrt(H/N)
/// is an approximate diagnostic, not a confidence certificate for this sampler.
/// Zero iterations gives a prior Monte Carlo mean. Negative infinity denotes zero
/// likelihood; if all contributions are zero, log Z and all output log weights are
/// negative infinity, information/error are zero, and no normalized posterior exists.
/// Complete finite positive-width bounds, N>=2, positive walk_steps, callable and
/// representable N+iterations/dimensional geometry are checked before callbacks.
/// O(Nd+I*(N+Wd)) producer work plus likelihood costs, O((N+I)d) result/live storage.
/// Sequential callbacks may have effects before a later callback/numeric/quota error;
/// no partial result is returned. Thread safety follows caller callback state.
/// @param log_likelihood Sequential callback accepting d coordinates; finite or -infinity result.
/// @param lo Nonempty finite lower bounds, owned by trace only.
/// @param hi Matching finite upper bounds, finite positive widths.
/// @param num_live Number N of live points, default100, at least2.
/// @param max_iterations Replacement budget I, default5000; zero valid.
/// @param seed Local MT19937_64 seed, default0; caller RNG unaffected.
/// @param walk_steps Positive W Gaussian proposals per replacement, default30.
/// @return Owned evidence estimate and correlated weighted dead-plus-final records.
/// @throws std::invalid_argument If complete controls/bounds are invalid before callbacks.
/// @throws std::domain_error If a callback returns NaN or positive infinity.
/// @throws std::overflow_error If finite numeric proposal-scale/evidence geometry fails.
/// @throws std::length_error If count geometry exceeds limits.
/// @throws std::bad_alloc If allocation fails; callback exceptions propagate unchanged.
/// @par Example
/// @code
/// auto r=dm::algorithms::nested_sampling([](const auto&){return 0.;},{0.},{1.},2,0);
/// std::print("{:.1f}\n",r.log_evidence); // 0.0
/// @endcode
/// @par References
/// - J. Skilling (2006), Nested sampling for general Bayesian computation, Bayesian Analysis1,833–859, doi:10.1214/06-BA127.
/// @see nested_sampling_trace(), NestedSamplingResult
NestedSamplingResult nested_sampling(
    const std::function<double(const std::vector<double>&)>& log_likelihood,
    const std::vector<double>&                               lo,
    const std::vector<double>&                               hi,
    std::size_t                                              num_live       = 100,
    std::size_t                                              max_iterations = 5000,
    std::uint64_t                                            seed           = 0,
    std::size_t                                              walk_steps     = 30);
/// @brief Capture the actual checked finite-walk nested-sampling producer.
/// Uses X_i=exp(-i/N), dead shell X_before-X_after, final live weights X_final/N.
/// Fixes the legacy final-shell/remaining-mass error and includes final points in H.
/// Starting survivor can tie the threshold; no-move/tied replacements are retained,
/// not advertised as independent constrained-prior draws. Gaussian finite walks and
/// adaptive scales provide no multimodal mixing guarantee. The maximum observed live
/// likelihood times remaining mass is a heuristic, not an upper bound. sqrt(H/N)
/// is an approximate diagnostic, not a confidence certificate for this sampler.
/// Zero iterations gives a prior Monte Carlo mean. Negative infinity denotes zero
/// likelihood; if all contributions are zero, log Z and all output log weights are
/// negative infinity, information/error are zero, and no normalized posterior exists.
/// Complete finite positive-width bounds, N>=2, positive walk_steps, callable and
/// representable N+iterations/dimensional geometry are checked before callbacks.
/// O(Nd+I*(N+Wd)) producer work plus likelihood costs, O((N+I)d) result/live storage.
/// Sequential callbacks may have effects before a later callback/numeric/quota error;
/// no partial result is returned. Thread safety follows caller callback state.
/// @param log_likelihood Sequential callback accepting d coordinates; finite or -infinity result.
/// @param lo Nonempty finite lower bounds, owned by trace only.
/// @param hi Matching finite upper bounds, finite positive widths.
/// @param num_live Number N of live points, default100, at least2.
/// @param max_iterations Replacement budget I, default5000; zero valid.
/// @param seed Local MT19937_64 seed, default0; caller RNG unaffected.
/// @param walk_steps Positive W Gaussian proposals per replacement, default30.
/// @param options Capture quota and remaining-fraction heuristic; zero disables either.
/// @return Owned actual history and same-producer weighted result.
/// @throws std::invalid_argument If complete controls/bounds are invalid before callbacks.
/// @throws std::domain_error If a callback returns NaN or positive infinity.
/// @throws std::overflow_error If finite numeric proposal-scale/evidence geometry fails.
/// @throws std::length_error If count geometry or capture quota exceeds limits.
/// @throws std::bad_alloc If allocation fails; callback exceptions propagate unchanged.
/// @par Example
/// @code
/// auto r=dm::algorithms::nested_sampling_trace([](const auto&){return 0.;},{0.},{1.},2,0);
/// std::print("{:.1f}\n",r.result.log_evidence); // 0.0
/// @endcode
/// @par References
/// - J. Skilling (2006), Nested sampling for general Bayesian computation, Bayesian Analysis1,833–859, doi:10.1214/06-BA127.
/// @see nested_sampling(), NestedSamplingTrace
NestedSamplingTrace nested_sampling_trace(
    const std::function<double(const std::vector<double>&)>& log_likelihood,
    const std::vector<double>&                               lo,
    const std::vector<double>&                               hi,
    std::size_t                                              num_live       = 100,
    std::size_t                                              max_iterations = 5000,
    std::uint64_t                                            seed           = 0,
    std::size_t                                              walk_steps     = 30,
    NestedSamplingOptions                                    options        = {});
/// @brief Evaluate the original unnormalized two-dimensional unit Gaussian centered at (2,-1).
/// @param point Exactly two finite coordinates.
/// @return -0.5*((x-2)^2+(y+1)^2).
/// @throws std::invalid_argument If shape or coordinates are invalid.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::nested_sampling_demo_log_likelihood({2.,-1.})); // -0
/// @endcode
/// @see nested_sampling_demo_log_evidence()
double nested_sampling_demo_log_likelihood(const std::vector<double>& point);
/// @brief Analytic finite-box log evidence of the original Gaussian under uniform [-10,10]^2.
/// Includes exact erf truncation factors rather than replacing finite bounds with infinity.
/// @return log of the product of two truncated Gaussian integrals divided by400.
/// @par Example
/// @code
/// std::print("{:.5f}\n",dm::algorithms::nested_sampling_demo_log_evidence()); // -4.15359
/// @endcode
/// @see nested_sampling_demo_log_likelihood()
double nested_sampling_demo_log_evidence();
} // namespace datamunge::algorithms
