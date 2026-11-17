#pragma once
/// @file jackknife.hpp
/// @brief Checked delete-one/delete-d jackknife with actual owned deletion histories.
#include <cstddef>
#include <cstdint>
#include <datamunge/stats/resample_types.hpp>
#include <vector>
namespace datamunge::stats {
/// @brief Owned delete-one statistics, estimated bias/SE and actual leave-one-out values.
/// Order matches original observation indices; pseudo-values are algebraic
/// transforms of actual callback results, not independent observations. Bias
/// correction and variance estimation need statistical assumptions beyond finite
/// callbacks. No sample/callback references retained. Original field layout kept.
/// @par Example
/// @code
/// const auto r=dm::stats::jackknife({1,2,3},[](const auto& v){return dm::mean(dm::c(v));});
/// std::print("{:.0f} {}\n",r.estimate,r.leave_one_out.size()); // 2 3
/// @endcode
/// @see jackknife(), jackknife_checked()
struct JackknifeResult {
  double estimate{0}; ///< Actual full-sample statistic.
  double bias{0}; ///< (n-1)*(mean(leave_one_out)-estimate), evaluated with centred wide arithmetic.
  double bias_corrected{0};          ///< estimate-bias; mean of pseudo-values in exact arithmetic.
  double standard_error{0};          ///< sqrt((n-1)/n * sum squared centred leave-one-out values).
  std::vector<double> leave_one_out; ///< Actual deletion callbacks in original index order.
  std::vector<double> pseudo_values; ///< estimate+(n-1)*(estimate-leave_one_out[i]).
};
/// @brief Delete-d subset budget and deterministic engine seed.
/// Enumerates lexicographic sorted deletion indices if exact C(n,d)<=max_subsets;
/// otherwise draws max_subsets uniform size-d sets independently with replacement
/// across draws. Each draw has no repeated index, but duplicate sets across draws
/// are allowed. Uses retained partial Fisher-Yates permutations, MT19937-64 and
/// std::uniform_int_distribution, so sequences can depend on standard library.
/// Finite B Monte Carlo dispersion uses denominator B to retain legacy semantics;
/// B=1 therefore gives zero dispersion, not evidence of zero sampling variance.
/// @par Example
/// @code
/// const dm::stats::DeleteDJackknifeOptions o;
/// std::print("{} {} {}\n",o.d,o.max_subsets,o.seed); // 1 2000 42
/// @endcode
/// @see jackknife_delete_d(), jackknife_delete_d_trace()
struct DeleteDJackknifeOptions {
  std::size_t d{1}; ///< Deleted count, require1<=d<n; retains n-d in original order.
  std::size_t max_subsets{
      2000};              ///< Positive exact-enumeration ceiling or number of Monte Carlo draws.
  std::uint64_t seed{42}; ///< Local MT19937-64 seed; ignored for exact enumeration.
};
/// @brief Delete-d standard-error estimate and actual subset count/mode.
/// Complete variance uses (n-d)/d times population variance over all subsets.
/// In Monte Carlo mode this same formula uses the B sampled replicate values;
/// it is a conditional approximation with a (B-1)/B finite-B variance factor.
/// Larger d alone does not guarantee valid inference for nonsmooth statistics.
/// Original field layout kept; no retained sample/callback or shared mutable state.
/// @par Example
/// @code
/// const auto r=dm::stats::jackknife_delete_d({1,2,3},[](const auto& v){return dm::mean(dm::c(v));},{1,3,7});
/// std::print("{} {}\n",r.complete,r.num_subsets); // true 3
/// @endcode
/// @see jackknife_delete_d_trace()
struct DeleteDJackknifeResult {
  double      estimate{0};       ///< Actual full-sample callback.
  double      standard_error{0}; ///< sqrt((n-d)/d * mean squared centred subset statistic).
  std::size_t d{1};              ///< Actual deleted count.
  std::size_t num_subsets{
      0}; ///< Actually evaluated deletion sets, counting repeated Monte Carlo sets.
  bool complete{
      false}; ///< True only for complete exact enumeration, never inferred from random coverage.
};
/// @brief Optional complete input/retained-scalar ceilings checked before callbacks/copies.
/// Zero disables either cap. Delete-one counts 2*n output doubles; delete-d
/// traces count B*(d+1) deletion-index/replicate entries. These are element counts
/// of mixed scalar types, not bytes. Excludes borrowed sample, summaries,
/// temporary kept/permutation vectors, vector capacities and byte/peak RAM.
/// No partial result returns on error. Legacy convenience functions disable caps.
/// @par Example
/// @code
/// const dm::stats::JackknifeLimits l;
/// std::print("{} {}\n",l.max_elements,l.max_retained_values); // 1000000 1000000
/// @endcode
/// @see jackknife_checked(), jackknife_delete_d_trace()
struct JackknifeLimits {
  std::size_t max_elements{1000000}; ///< Complete borrowed sample count; zero disables.
  std::size_t max_retained_values{
      1000000}; ///< Complete owned history scalar-entry count; zero disables.
};
/// @brief Actual same-execution delete-d sets, replicate statistics and summary.
/// Each sorted deletion set pairs with its actual callback value; no RNG replay
/// or repeated statistic fits. All vectors are owned. Mean/variance describe
/// subset statistics conditional on this input, not the statistic's sampling law.
/// @par Example
/// @code
/// const auto t=dm::stats::jackknife_delete_d_trace({1,2,3},[](const auto& v){return dm::mean(dm::c(v));},{1,3,7});
/// std::print("{} {}\n",t.deletion_sets[0][0],t.evaluations); // 0 4
/// @endcode
/// @see jackknife_delete_d_trace()
struct DeleteDJackknifeTrace {
  DeleteDJackknifeResult                result; ///< Actual same-run final estimate/SE/mode/count.
  std::vector<std::vector<std::size_t>> deletion_sets; ///< Actual sorted deleted original indices.
  std::vector<double>                   replicates;    ///< Actual matching callback values.
  double      mean_replicate{0}; ///< Wide-centred mean of actual subset statistic values.
  long double replicate_population_variance{
      0};                     ///< Actual squared spread divided by evaluated subset count.
  std::size_t evaluations{0}; ///< Actual statistic calls including the full-sample call.
};
/// @brief Compute the delete-one jackknife from actual callbacks in observation order.
/// Requires finite sample of n>=2 and nonempty statistic; every returned statistic
/// must be finite. Preserves input, callback ownership and result layout; calls
/// statistic on the original sample then n original-order deletion samples.
/// Wide centred summaries and robust norms avoid avoidable uncentred overflow;
/// finite-double outputs are checked. No symmetry/smoothness/independence check.
/// Pseudo-values need not be independent or yield a valid interval. Exact mean
/// identities and plug-in-variance bias removal hold in exact arithmetic, not
/// as guarantees of arbitrary estimator bias. Callback exceptions propagate and
/// side effects persist. O(n^2) copying plus callback costs, O(n) owned output
/// and temporary storage, no shared mutable state or retained input references.
/// @param sample Borrowed finite observations, at least2, never modified.
/// @param statistic Borrowed nonempty scalar callback, valid for n and n-1 values.
/// @return Owned actual leave-one-out values, pseudo-values and bias/SE summary.
/// @throws std::invalid_argument On n<2, empty statistic or nonfinite input before callbacks.
/// @throws std::domain_error On nonfinite callback return; callback exceptions propagate.
/// @throws std::overflow_error On unrepresentable summary/pseudo-values.
/// @throws std::length_error On impossible output count/container sizes.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::stats::jackknife({1,2,3},[](const auto& v){return dm::mean(dm::c(v));});
/// std::print("{:.6f}\n",r.standard_error); // 0.577350
/// @endcode
/// @par References
/// - Shao and Wu (1989), A General Theory for Jackknife Variance Estimation, Annals of Statistics17(3):1176-1197: https://doi.org/10.1214/aos/1176347263.
/// @see jackknife_checked(), JackknifeResult
[[nodiscard]] JackknifeResult jackknife(
    const std::vector<double>& sample,
    const SampleStatistic&     statistic);
/// @brief Compute the same delete-one result with complete input/output admission.
/// Same numerical/callback/ownership semantics as jackknife(). Complete 2*n
/// retained-double admission precedes callbacks and copying; zero optional caps
/// disable them. O(n^2) copying plus n+1 callbacks, O(n) output/working storage.
/// @param sample Borrowed finite n>=2 observations.
/// @param statistic Borrowed callback valid for n and n-1 finite values.
/// @param limits Complete input and retained-entry caps, defaults above.
/// @return Owned actual jackknife output, never a truncated prefix.
/// @throws std::invalid_argument On sample/callback violations before callbacks.
/// @throws std::domain_error On nonfinite callback return; callback exceptions propagate.
/// @throws std::overflow_error On summary/pseudo-value range failure.
/// @throws std::length_error On enabled cap or output count/container overflow.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::stats::jackknife_checked({1,2,3},[](const auto& v){return dm::mean(dm::c(v));},{3,6});
/// std::print("{}\n",r.pseudo_values.size()); // 3
/// @endcode
/// @see jackknife()
[[nodiscard]] JackknifeResult jackknife_checked(
    const std::vector<double>& sample,
    const SampleStatistic&     statistic,
    JackknifeLimits            limits = {});
/// @brief Compute complete or Monte Carlo delete-d jackknife SE from actual subsets.
/// Finite n>=2 input, nonempty callback,1<=d<n and positive max_subsets required.
/// Exact capped integer binomial admission chooses mode without a floating count.
/// Enumeration is lexicographic; sampled sets preserve the original MT19937-64/
/// partial-shuffle draw policy and original retained-value order. Repeated sets
/// are valid Monte Carlo draws. Uses a centred streaming population variance,
/// retaining no subset history. Same formula/field layout as before; final digits
/// can change with stable arithmetic. Sample/callback unchanged and unretained;
/// callback side effects/exceptions propagate. No general consistency guarantee
/// for nonsmooth statistics: d, estimator regularity and sampling design matter.
/// O(B*(n+d log d)) copying/selection plus B+1 callback costs, O(n+d) workspace.
/// @param sample Borrowed finite sample of n>=2 values.
/// @param statistic Borrowed callback valid for n and n-d values.
/// @param options Deleted count, positive subset ceiling/draw count and seed.
/// @return Owned summary/mode/count; no partial prefix on failure.
/// @throws std::invalid_argument On sample/callback/options violations before callbacks.
/// @throws std::domain_error On nonfinite callback return; callback exceptions propagate.
/// @throws std::overflow_error On unrepresentable summary arithmetic.
/// @throws std::length_error On count/container-size overflow.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::stats::jackknife_delete_d({1,2,3},[](const auto& v){return dm::mean(dm::c(v));},{2,3,7});
/// std::print("{} {:.6f}\n",r.complete,r.standard_error); // true 0.577350
/// @endcode
/// @par References
/// - Shao and Wu (1989), Annals of Statistics17(3):1176-1197: https://doi.org/10.1214/aos/1176347263.
/// @see DeleteDJackknifeOptions, jackknife_delete_d_trace()
[[nodiscard]] DeleteDJackknifeResult jackknife_delete_d(
    const std::vector<double>& sample,
    const SampleStatistic&     statistic,
    DeleteDJackknifeOptions    options = {});
/// @brief Capture every actual delete-d set/statistic from the producing execution.
/// Same mode/RNG/variance/callback policies as jackknife_delete_d(). Complete
/// B*(d+1) retained-entry admission occurs before callbacks/copies, including
/// every deletion index and replicate value. Optional caps are not byte/RAM
/// bounds; exclusions are listed in JackknifeLimits. Captures no kept samples
/// or RNG internals. O(B*d) additional owned scalar entries, no replay or retained
/// callback/input references. Copying the result is deep. No partial trace on error.
/// @param sample Borrowed finite n>=2 observations.
/// @param statistic Borrowed callback valid for n and n-d observations.
/// @param options Deleted count, subset budget and seed, defaults above.
/// @param limits Complete input and actual known history-entry policy.
/// @return Owned chronological deletion sets/replicates and same-run final summary.
/// @throws std::invalid_argument On invalid sample/callback/options before callbacks.
/// @throws std::domain_error On nonfinite callback returns; callback exceptions propagate.
/// @throws std::overflow_error On unrepresentable summary arithmetic.
/// @throws std::length_error On enabled caps or count/container overflow.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::stats::jackknife_delete_d_trace({1,2,3},[](const auto& v){return dm::mean(dm::c(v));},{1,3,7},{3,6});
/// std::print("{}\n",t.replicates.size()); // 3
/// @endcode
/// @see jackknife_delete_d(), DeleteDJackknifeTrace
[[nodiscard]] DeleteDJackknifeTrace jackknife_delete_d_trace(
    const std::vector<double>& sample,
    const SampleStatistic&     statistic,
    DeleteDJackknifeOptions    options = {},
    JackknifeLimits            limits  = {});
} // namespace datamunge::stats
