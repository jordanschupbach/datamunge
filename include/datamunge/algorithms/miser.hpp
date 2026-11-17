#pragma once
/// @file miser.hpp
/// @brief Checked recursive stratified integration with actual pilot and leaf samples.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
namespace datamunge::algorithms {
/// @brief Integral estimate and the native plug-in estimator variance.
/// Leaf variance uses (mean(f squared)-mean(f) squared)/n, clipped to zero;
/// it is not an unbiased sample-variance estimate or a confidence guarantee.
/// @par Example
/// @code
/// auto r=dm::algorithms::miser_integrate([](const auto&){return 2.;},{0},{1},20);
/// std::print("{} {}\n",r.value,r.variance); // 2 0
/// @endcode
/// @see miser_integrate()
struct MiserResult {
  double value{0};    ///< Estimated integral in function-times-volume units.
  double variance{0}; ///< Native estimated variance in squared integral units.
};
/// @brief One actual callback evaluation, labeled pilot versus retained leaf.
/// @par Example
/// @code
/// auto t=dm::algorithms::miser_trace([](const auto&){return 2.;},{0},{1},2);
/// std::print("{}\n",t.samples.size()); // 2
/// @endcode
/// @see MiserTrace
struct MiserSample {
  std::size_t node{0}; ///< Actual preorder recursion-node ID.
  bool pilot{false};   ///< True for an axis/allocation pilot discarded from the final estimator.
  std::vector<double> point;    ///< Owned actual coordinates passed to the callback.
  double              value{0}; ///< Actual returned finite callback value.
};
/// @brief Actual region, split/allocation and returned estimate for one recursion call.
/// Leaf nodes have pilot_calls=0 and split_dimension=0 placeholder. Child budgets
/// sum to calls minus pilot_calls; pilot observations are not reused in leaves.
/// @par Example
/// @code
/// auto t=dm::algorithms::miser_trace([](const auto&){return 1.;},{0},{1},128);
/// std::print("{}\n",t.nodes.front().pilot_calls); // 16
/// @endcode
/// @see MiserTrace
struct MiserNode {
  std::vector<double> lower;          ///< Owned actual lower region bounds.
  std::vector<double> upper;          ///< Owned actual upper region bounds.
  long                calls{0};       ///< Actual requested evaluations for this region, at least2.
  long                pilot_calls{0}; ///< Actual pilot evaluations, zero for leaf.
  std::size_t         split_dimension{0}; ///< Actual minimum-sum-stddev axis for a split.
  double              left_sd{0};         ///< Actual selected left-half pilot population SD.
  double              right_sd{0};        ///< Actual selected right-half pilot population SD.
  long                left_calls{0};  ///< Actual allocated left recursion budget, zero for leaf.
  long                right_calls{0}; ///< Actual allocated right recursion budget, zero for leaf.
  MiserResult         result;         ///< Actual returned region estimate/variance.
};
/// @brief Owned actual sample and recursive allocation history.
/// Capture units are original2d bounds, each node record plus its2d bounds,
/// and each sample record plus its d coordinates. Quota excludes scalar fields
/// within records and temporary pilot accumulators. All callbacks are observable;
/// a quota failure can occur after earlier evaluations and returns no partial trace.
/// @par Example
/// @code
/// auto t=dm::algorithms::miser_trace([](const auto&){return 1.;},{0},{1},2);
/// std::print("{}\n",t.evaluations); // 2
/// @endcode
/// @see miser_trace()
struct MiserTrace {
  std::vector<double>      lower;    ///< Owned original lower bounds.
  std::vector<double>      upper;    ///< Owned original upper bounds.
  long                     calls{0}; ///< Original complete sample budget.
  std::uint64_t            seed{0};  ///< Original xorshift seed, zero uses documented fallback.
  MiserResult              result;   ///< Actual final estimate.
  std::vector<MiserNode>   nodes;    ///< Actual preorder region records, completed on return.
  std::vector<MiserSample> samples;  ///< Actual callback order, pilots and leaves.
  std::size_t evaluations{0};        ///< Actual callback evaluations, exactly calls on success.
  std::size_t captured_entries{0};   ///< Actual admitted capture units.
};
/// @brief Private deterministic stream and common checked recursive producer.
namespace detail {
/// @brief Validate box geometry and complete sample budget before any callback.
/// @param a Nonempty finite lower bounds.
/// @param b Matching finite strictly higher bounds.
/// @param calls Requested evaluations, at least2.
/// @throws std::invalid_argument If shape, bounds or sample count is invalid.
/// @throws std::overflow_error If widths or volume are nonfinite/underflow to zero.
inline void miser_validate(const std::vector<double>& a, const std::vector<double>& b, long calls) {
  if (a.empty() || a.size() != b.size() || calls < 2)
    throw std::invalid_argument("MISER: nonempty matching box and at least two calls required");
  double volume = 1;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || !(a[i] < b[i]))
      throw std::invalid_argument("MISER: finite increasing bounds required");
    volume *= b[i] - a[i];
    if (!std::isfinite(volume) || volume == 0)
      throw std::overflow_error("MISER: unrepresentable box volume");
  }
}
/// @brief Original xorshift13/7/17 stream with top53-bit unit conversion.
struct MiserRng {
  std::uint64_t s; ///< Current owned stream state.

  /// @brief Seed the stream, replacing zero with its nonzero fallback.
  /// @param seed Initial state; zero maps to0x2545F4914F6CDD1D.
  explicit MiserRng(std::uint64_t seed) : s(seed ? seed : 0x2545F4914F6CDD1DULL) {}
  /// @brief Advance once and map top53 bits into [0,1).
  /// @return Actual deterministic unit variate.
  double next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return (s >> 11) * (1.0 / 9007199254740992.0);
  }
};
/// @brief Admit actual history units before allocating or calling the integrand.
/// @param out Optional history sink.
/// @param cap Optional quota, zero disables.
/// @param amount New owned capture units.
/// @throws std::length_error If admission exceeds quota or size_t.
inline void miser_admit(MiserTrace* out, std::size_t cap, std::size_t amount) {
  if (!out)
    return;
  if (amount > std::numeric_limits<std::size_t>::max() - out->captured_entries ||
      (cap && (out->captured_entries > cap || amount > cap - out->captured_entries)))
    throw std::length_error("MISER: capture quota exceeded");
  out->captured_entries += amount;
}
/// @brief Execute one actual recursive pilot/split or plain leaf.
/// @tparam F Callable receiving a const coordinate vector.
/// @param f Callback, whose exceptions propagate.
/// @param a Mutable lower bounds, restored after recursion.
/// @param b Mutable upper bounds, restored after recursion.
/// @param calls Complete region budget.
/// @param rng Shared actual stream.
/// @param pt Reusable coordinate scratch.
/// @param out Optional owned capture sink.
/// @param cap Capture limit, zero disables.
/// @return Actual region integral/variance.
/// @throws std::domain_error If callback returns nonfinite data.
/// @throws std::overflow_error If squared values/statistics are unrepresentable.
/// @throws std::length_error If capture quota is exceeded.
template<class F>
MiserResult miser_rec(
    F&                   f,
    std::vector<double>& a,
    std::vector<double>& b,
    long                 calls,
    MiserRng&            rng,
    std::vector<double>& pt,
    MiserTrace*          out = nullptr,
    std::size_t          cap = 0) {
  const auto dim    = a.size();
  double     volume = 1;
  for (std::size_t i = 0; i < dim; ++i)
    volume *= b[i] - a[i];
  if (!std::isfinite(volume) || volume == 0)
    throw std::overflow_error("MISER: region volume unrepresentable");
  std::size_t id = 0;
  if (out) {
    miser_admit(out, cap, 1);
    miser_admit(out, cap, dim);
    miser_admit(out, cap, dim);
    id = out->nodes.size();
    MiserNode node;
    node.lower = a;
    node.upper = b;
    node.calls = calls;
    out->nodes.push_back(std::move(node));
  }
  /// @brief Draw and evaluate one actual point, optionally recording its role.
  auto evaluate = [&](bool pilot) {
    if (out) {
      miser_admit(out, cap, 1);
      miser_admit(out, cap, dim);
    }
    for (std::size_t i = 0; i < dim; ++i)
      pt[i] = a[i] + (b[i] - a[i]) * rng.next();
    const double value = f(pt);
    if (!std::isfinite(value))
      throw std::domain_error("MISER: nonfinite callback result");
    if (!std::isfinite(value * value))
      throw std::overflow_error("MISER: squared callback overflow");
    if (out) {
      ++out->evaluations;
      out->samples.push_back({id, pilot, pt, value});
    }
    return value;
  };
  MiserResult result;
  if (calls < 64) {
    double sum = 0, sum2 = 0;
    for (long k = 0; k < calls; ++k) {
      auto value = evaluate(false);
      sum += value;
      sum2 += value * value;
    }
    const double mean = sum / calls, var = (sum2 / calls - mean * mean) / calls;
    result = {volume * mean, volume * volume * (var > 0 ? var : 0)};
  } else {
    long                pre = std::max(16L, static_cast<long>(calls * .1));
    std::vector<double> mid(dim), sl(dim), sl2(dim), sr(dim), sr2(dim);
    std::vector<long>   nl(dim), nr(dim);
    for (std::size_t i = 0; i < dim; ++i)
      mid[i] = .5 * a[i] + .5 * b[i];
    for (long k = 0; k < pre; ++k) {
      auto value = evaluate(true);
      for (std::size_t i = 0; i < dim; ++i)
        if (pt[i] < mid[i]) {
          sl[i] += value;
          sl2[i] += value * value;
          ++nl[i];
        } else {
          sr[i] += value;
          sr2[i] += value * value;
          ++nr[i];
        }
    }
    /// @brief Original population-SD pilot estimate with cancellation clipped at zero.
    auto sd = [](double sum, double sum2, long count) {
      if (count < 2)
        return 0.;
      double mean = sum / count, var = sum2 / count - mean * mean;
      if (!std::isfinite(var))
        throw std::overflow_error("MISER: pilot statistics overflow");
      return var > 0 ? std::sqrt(var) : 0.;
    };
    std::size_t axis = 0;
    double      best = std::numeric_limits<double>::infinity(), left_sd = 0, right_sd = 0;
    for (std::size_t i = 0; i < dim; ++i) {
      auto left = sd(sl[i], sl2[i], nl[i]), right = sd(sr[i], sr2[i], nr[i]);
      if (left + right < best) {
        best     = left + right;
        axis     = i;
        left_sd  = left;
        right_sd = right;
      }
    }
    const long   remain   = calls - pre;
    const double fraction = best > 0 ? left_sd / best : .5;
    long         left     = static_cast<long>(remain * fraction);
    if (left < 32)
      left = 32;
    if (left > remain - 32)
      left = remain - 32;
    const long right = remain - left;
    if (out) {
      auto& node           = out->nodes[id];
      node.pilot_calls     = pre;
      node.split_dimension = axis;
      node.left_sd         = left_sd;
      node.right_sd        = right_sd;
      node.left_calls      = left;
      node.right_calls     = right;
    }
    const double save = b[axis];
    b[axis]           = mid[axis];
    auto l            = miser_rec(f, a, b, left, rng, pt, out, cap);
    b[axis]           = save;
    const double old  = a[axis];
    a[axis]           = mid[axis];
    auto r            = miser_rec(f, a, b, right, rng, pt, out, cap);
    a[axis]           = old;
    result            = {l.value + r.value, l.variance + r.variance};
  }
  if (!std::isfinite(result.value) || !std::isfinite(result.variance))
    throw std::overflow_error("MISER: result statistics overflow");
  if (out)
    out->nodes[id].result = result;
  return result;
}
} // namespace detail
/// @brief Integrate a callable over a finite box with an exact evaluation budget.
/// Uses the original xorshift stream,10percent pilots (minimum16), midpoint
/// axis minimizing the sum of pilot SDs, and proportional remaining allocation.
/// Calls below64 are plain leaves. The legacy32-sample allocation clamps are
/// preserved, including leaves smaller than32 when a split budget is64..79.
/// Pilots guide allocation but are discarded from the final estimate. At least
/// two calls are required, correcting legacy silent overspending below two.
/// Successful calls evaluate f exactly calls times. No error-rate or confidence
/// guarantee; quasi-random stream correlations and adaptation remain relevant.
/// Work O(d*calls) excluding callbacks; recursion and scratch O(d*log calls).
/// @tparam F Callable accepting const std::vector<double>& and returning double.
/// @param f Owned callable; callback exceptions propagate, side effects remain.
/// @param a Nonempty finite lower bounds, copied.
/// @param b Matching finite strictly higher bounds, copied.
/// @param calls Complete evaluation budget, at least2.
/// @param seed Xorshift seed, default1; zero uses a fixed nonzero fallback.
/// @return Integral and native clipped plug-in variance estimate.
/// @throws std::invalid_argument If box or budget is invalid, before any callback.
/// @throws std::domain_error If callback returns nonfinite data.
/// @throws std::overflow_error If geometry or statistics cannot be represented.
/// @par Example
/// @code
/// auto r=dm::algorithms::miser_integrate([](const auto&){return 2.;},{0},{1},20);
/// std::print("{}\n",r.value); // 2
/// @endcode
/// @par References
/// - W. H. Press and G. R. Farrar (1990), Recursive stratified sampling for multidimensional Monte Carlo integration, Computers in Physics4,190–195, doi:10.1063/1.4822899.
/// @see miser_trace()
template<class F>
MiserResult miser_integrate(
    F                   f,
    std::vector<double> a,
    std::vector<double> b,
    long                calls,
    std::uint64_t       seed = 1) {
  detail::miser_validate(a, b, calls);
  detail::MiserRng    rng(seed);
  std::vector<double> pt(a.size());
  return detail::miser_rec(f, a, b, calls, rng, pt);
}
/// @brief Capture actual callback points and region allocations of the same integrator.
/// Same stream, scheduling and result as miser_integrate(), without replay.
/// Actual quota admission precedes each stored evaluation/region. Earlier callback
/// side effects remain if a later quota or callback fails. History O(d*calls).
/// @tparam F Callable accepting a coordinate vector and returning double.
/// @param f Owned integrand, exceptions propagate.
/// @param a Owned original lower bounds.
/// @param b Owned original upper bounds.
/// @param calls Exact callback budget, at least2.
/// @param seed Xorshift seed, default1, zero uses fallback.
/// @param max_entries Capture cap, default1000000; zero disables.
/// @return Owned actual original bounds, samples, nodes and estimate.
/// @throws std::invalid_argument If box/budget is invalid, before callbacks.
/// @throws std::domain_error If callback returns nonfinite data.
/// @throws std::overflow_error If geometry/statistics are unrepresentable.
/// @throws std::length_error If capture admission exceeds quota.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// auto t=dm::algorithms::miser_trace([](const auto&){return 1.;},{0},{1},2);
/// std::print("{}\n",t.samples.size()); // 2
/// @endcode
/// @see MiserTrace, miser_integrate()
template<class F>
MiserTrace miser_trace(
    F                   f,
    std::vector<double> a,
    std::vector<double> b,
    long                calls,
    std::uint64_t       seed        = 1,
    std::size_t         max_entries = 1000000) {
  detail::miser_validate(a, b, calls);
  MiserTrace t;
  t.calls = calls;
  t.seed  = seed;
  detail::miser_admit(&t, max_entries, a.size());
  detail::miser_admit(&t, max_entries, b.size());
  t.lower = a;
  t.upper = b;
  detail::MiserRng    rng(seed);
  std::vector<double> pt(a.size());
  t.result = detail::miser_rec(f, a, b, calls, rng, pt, &t, max_entries);
  return t;
}
/// @brief One native replicated original Gaussian MISER-versus-plain comparison.
/// @par Example
/// @code
/// auto r=dm::algorithms::miser_gaussian_comparison({2000},2);
/// std::print("{}\n",r.front().calls); // 2000
/// @endcode
/// @see miser_gaussian_comparison()
struct MiserComparison {
  long   calls{0};      ///< Exact calls per independent method/run.
  int    replicates{0}; ///< Seeds1..replicates for MISER,1001..1000+replicates for plain.
  double miser_rms{0};  ///< Root mean squared error against exact finite-box integral.
  double plain_rms{0};  ///< Same metric for original MT19937-64 plain comparator.
};
/// @brief Evaluate the original centered narrow Gaussian over two coordinates.
/// @param point Exactly two finite coordinates.
/// @return exp(-200*((x-.5)^2+(y-.5)^2)).
/// @throws std::invalid_argument If shape or coordinates are invalid.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::miser_gaussian({.5,.5})); // 1
/// @endcode
/// @see miser_gaussian_integral()
double miser_gaussian(const std::vector<double>& point);
/// @brief Return the analytic finite-unit-square integral of the original Gaussian.
/// @return pi/200 times erf(sqrt(50)) squared; not the infinite-plane approximation alone.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::miser_gaussian_integral()>0); // true
/// @endcode
/// @see miser_gaussian()
double miser_gaussian_integral();
/// @brief Run the original budget/seed comparison in the native library.
/// Plain comparator preserves MT19937-64 seed1000+seed and uniform_real_distribution.
/// Finite-replicate RMS values are observations, not universal superiority/rates.
/// @param budgets Exact per-run calls, each at least2; empty allowed.
/// @param replicates Positive replicate count, default16, at most10000.
/// @return Owned original-order budget summaries.
/// @throws std::invalid_argument If budget or replicate count is invalid.
/// @throws std::length_error If replicate count exceeds10000.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::miser_gaussian_comparison({2000},2).size()); // 1
/// @endcode
/// @see MiserComparison
std::vector<MiserComparison> miser_gaussian_comparison(
    const std::vector<long>& budgets,
    int                      replicates = 16);
} // namespace datamunge::algorithms
