#pragma once
/// @file samplesort.hpp
/// @brief Checked deterministic oversampled bucket sorting with actual comparison histories.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
/// @brief Canonical checked native algorithm producers and owned outputs.
namespace datamunge::algorithms {
/// @brief Input/work/capture policies and bounded recursion.
/// @par Example
/// @code
/// const dm::algorithms::SampleSortOptions value{};
/// std::print("{}\n",value.depth_limit); // 64
/// @endcode
struct SampleSortOptions {
  std::size_t max_keys{1000000};         ///< Complete key count; zero disables.
  std::size_t max_comparisons{20000000}; ///< Actual operator-less calls; zero disables.
  std::size_t max_trace_entries{
      1000000}; ///< Actual copied keys, source indices, bucket sizes and event headers; zero disables.
  std::size_t depth_limit{64}; ///< Depth fallback threshold, required in [1,256].
};
/// @brief Actual operation performed on one visited subrange.
/// @par Example
/// @code
/// const auto value=dm::algorithms::SampleSortStage::Insertion;
/// std::print("{}\n",static_cast<int>(value)); // 0
/// @endcode
enum class SampleSortStage {
  Insertion,           ///< At most sixteen keys, including empty root.
  Partition,           ///< Actual sampled split and concatenation before child sorting.
  WholeBucketFallback, ///< No progress: one bucket contains the complete parent range.
  DepthFallback        ///< Recursion policy reached: counted std::sort.
};
/// @brief Owned actual subrange event; partition after is before recursive child work.
/// @tparam T Owned copyable/movable key type.
/// @par Example
/// @code
/// const dm::algorithms::SampleSortStep<int> value{};
/// std::print("{}\n",value.offset); // 0
/// @endcode
template<class T>
struct SampleSortStep {
  std::size_t offset{}; ///< Original array offset and actual recursion depth.
  std::size_t depth{}; ///< Original array offset and actual recursion depth.
  SampleSortStage stage{};           ///< Actual performed stage.
  std::vector<T> before; ///< Owned actual keys; sample sorted by the real sampling sort.
  std::vector<T> after; ///< Owned actual keys; sample sorted by the real sampling sort.
  std::vector<T> sample; ///< Owned actual keys; sample sorted by the real sampling sort.
  std::vector<T> splitters; ///< Owned actual keys; sample sorted by the real sampling sort.
  std::vector<std::size_t> sample_indices; ///< Actual original-array sample addresses and eight sizes.
  std::vector<std::size_t> bucket_sizes; ///< Actual original-array sample addresses and eight sizes.
};
/// @brief Owned sorted keys, actual comparator work and optional preorder events.
/// @tparam T Owned copyable/movable key type.
/// @par Example
/// @code
/// const dm::algorithms::SampleSortRun<int> value{};
/// std::print("{}\n",value.comparisons); // 0
/// @endcode
template<class T>
struct SampleSortRun {
  std::vector<T> values; ///< Actual sorted output; unstable among equal keys.
  std::size_t comparisons{}; ///< Actual operator-less calls, total and disjoint categories.
  std::size_t sample_comparisons{}; ///< Actual operator-less calls, total and disjoint categories.
  std::size_t splitter_comparisons{}; ///< Actual operator-less calls, total and disjoint categories.
  std::size_t leaf_comparisons{}; ///< Actual operator-less calls, total and disjoint categories.
  std::size_t sample_reads{}; ///< Actual selected algorithm operations.
  std::size_t classification_writes{}; ///< Actual selected algorithm operations.
  std::size_t partitions{}; ///< Actual selected algorithm operations.
  std::size_t fallbacks{}; ///< Actual selected algorithm operations.
  std::size_t maximum_depth{}; ///< Actual selected algorithm operations.
  std::vector<SampleSortStep<T>> steps; ///< Optional actual preorder events.
};
/// @brief Private single generic producer shared by all public sorting adapters.
namespace samplesort_detail {
/// @brief Borrowed mutable range and owned counters/history for one independent invocation.
/// @tparam T Copyable/movable key obeying strict weak operator-less ordering.
template<class T>
struct Worker {
  std::vector<T>&          a;          ///< Borrowed invocation-owned or caller-owned working array.
  const SampleSortOptions& o;          ///< Borrowed immutable policy.
  bool                     capture;    ///< Retain optional events.
  SampleSortRun<T>         out;        ///< Owned actual counters and histories.
  std::size_t              retained{}; ///< Actual admitted history units.

  /// @brief Admit history scalars/keys before copying.
  /// @param n Next actual units.
  void admit(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() - retained ||
        (o.max_trace_entries &&
         (retained > o.max_trace_entries || n > o.max_trace_entries - retained)))
      throw std::length_error("samplesort: trace policy");
    retained += n;
  }
  /// @brief Count and execute one actual strict key comparison.
  /// @param x Borrowed first operand.
  /// @param y Borrowed second operand.
  /// @param category Borrowed category counter.
  /// @return Actual x less than y.
  bool less(const T& x, const T& y, std::size_t& category) {
    if (out.comparisons == std::numeric_limits<std::size_t>::max() ||
        (o.max_comparisons && out.comparisons >= o.max_comparisons))
      throw std::length_error("samplesort: comparison policy");
    ++out.comparisons;
    ++category;
    return x < y;
  }
  /// @brief Execute one real visited range and recurse only after bucket scratch release.
  /// @param lo Inclusive array address.
  /// @param hi Exclusive array address.
  /// @param depth Actual recursion depth.
  /// @param forced Whether parent classification made no progress.
  void sort(std::size_t lo, std::size_t hi, std::size_t depth, bool forced = false) {
    const auto n      = hi - lo;
    out.maximum_depth = std::max(out.maximum_depth, depth);
    SampleSortStep<T> event;
    event.offset    = lo;
    event.depth     = depth;
    const bool leaf = n <= 16 || forced || depth >= o.depth_limit;
    event.stage     = n <= 16                  ? SampleSortStage::Insertion
                      : forced                 ? SampleSortStage::WholeBucketFallback
                      : depth >= o.depth_limit ? SampleSortStage::DepthFallback
                                               : SampleSortStage::Partition;
    if (capture) {
      if (n > (std::numeric_limits<std::size_t>::max() - 1) / 2)
        throw std::length_error("samplesort: capture geometry");
      admit(2 * n + 1);
      event.before.assign(a.begin() + lo, a.begin() + hi);
    }
    if (leaf) {
      if (n <= 16) {
        for (std::size_t i = lo + (n ? 1 : 0); i < hi; ++i) {
          T    key = a[i];
          auto j   = i;
          while (j > lo && less(key, a[j - 1], out.leaf_comparisons)) {
            a[j] = a[j - 1];
            --j;
          }
          a[j] = std::move(key);
        }
      } else {
        ++out.fallbacks;
        /// @brief Count actual standard-library leaf comparisons.
        /// @param x First key.
        /// @param y Second key.
        /// @return Actual strict ordering.
        auto cmp = [&](const T& x, const T& y) { return less(x, y, out.leaf_comparisons); };
        std::sort(a.begin() + lo, a.begin() + hi, cmp);
      }
      if (capture) {
        event.after.assign(a.begin() + lo, a.begin() + hi);
        out.steps.push_back(std::move(event));
      }
      return;
    }
    ++out.partitions;
    std::vector<T> sample;
    sample.reserve(24);
    std::vector<std::size_t> indices;
    for (std::size_t i = 1; i <= 24; ++i) {
      const auto index = lo + i * (n / 25) + (i * (n % 25)) / 25;
      sample.push_back(a[index]);
      indices.push_back(index);
      ++out.sample_reads;
    }
    /// @brief Count the actual sample sorting comparisons.
    /// @param x First key.
    /// @param y Second key.
    /// @return Actual strict ordering.
    auto cmp = [&](const T& x, const T& y) { return less(x, y, out.sample_comparisons); };
    std::sort(sample.begin(), sample.end(), cmp);
    std::vector<T> splitters;
    for (std::size_t i = 1; i < 8; ++i)
      splitters.push_back(sample[3 * i]);
    std::vector<std::vector<T>> buckets(8);
    for (std::size_t i = lo; i < hi; ++i) {
      std::size_t left = 0, count = 7;
      while (count) {
        const auto half = count / 2, mid = left + half;
        if (less(a[i], splitters[mid], out.splitter_comparisons))
          count = half;
        else {
          left = mid + 1;
          count -= half + 1;
        }
      }
      buckets[left].push_back(a[i]);
      ++out.classification_writes;
    }
    std::array<std::size_t, 8> sizes{};
    auto                       pos = lo;
    for (std::size_t b = 0; b < 8; ++b) {
      sizes[b] = buckets[b].size();
      for (const auto& x : buckets[b]) {
        a[pos++] = x;
        ++out.classification_writes;
      }
    }
    if (capture) {
      admit(24 + 24 + 7 + 8);
      event.sample         = sample;
      event.sample_indices = indices;
      event.splitters      = splitters;
      event.bucket_sizes.assign(sizes.begin(), sizes.end());
      event.after.assign(a.begin() + lo, a.begin() + hi);
      out.steps.push_back(std::move(event));
    }
    buckets.clear();
    sample.clear();
    splitters.clear();
    indices.clear();
    pos = lo;
    for (auto size : sizes) {
      if (size)
        sort(pos, pos + size, depth + 1, size == n);
      pos += size;
    }
  }
};
/// @brief Validate complete input/policies before constructing a worker.
/// @tparam T Key type with strict weak operator-less ordering.
/// @param a Borrowed complete keys.
/// @param o Borrowed policies.
template<class T>
void validate(const std::vector<T>& a, const SampleSortOptions& o) {
  if (!o.depth_limit || o.depth_limit > 256)
    throw std::invalid_argument("samplesort: depth policy");
  if ((o.max_keys && a.size() > o.max_keys) ||
      a.size() > std::numeric_limits<std::size_t>::max() / 512)
    throw std::length_error("samplesort: input/work capacity");
  if constexpr (std::is_floating_point_v<T>)
    for (auto x : a)
      if (!std::isfinite(x))
        throw std::invalid_argument("samplesort: finite keys required");
}
} // namespace samplesort_detail
/// @tparam T Copyable/movable key type with strict weak operator-less ordering.
/// @brief Sort owned keys using actual deterministic sampled bucket partitions.
/// Samples 24 evenly spaced addresses, sorts that sample and selects seven splitters at indices 3,6,...,21. Equal-to-splitter keys enter the higher bucket. Ranges of at most sixteen use insertion sort; one-whole-bucket or configured depth uses counted std::sort. All actual key comparisons are counted in disjoint categories; library-dependent std::sort counts need not be portable. Unstable. Trace owns actual before/after states; partition after is concatenation before children. Units are two range copies plus event header, and for each partition 24 sampled keys+24 source indices+7 splitters+8 bucket sizes. Typical O(n log n), bounded-depth fallback prevents unbounded recursion; no worst-case linear distribution claim. Input is passed by value, so caller storage is unchanged. Keys/comparator exceptions propagate; independent calls share no state.
/// @param values Owned working keys; floating types must all be finite.
/// @param options Complete input, actual comparison, history and depth bounds.
/// @return Owned sorted values and actual work/history.
/// @throws std::invalid_argument Nonfinite floating key or depth outside [1,256].
/// @throws std::length_error Input, work-counter or actual comparison/history bounds exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::samplesort_run(std::vector<int>{3,1,2});
/// std::print("{}\n",r.values); // [1, 2, 3]
/// @endcode
/// @par References
/// - Frazer and McKellar (1970), Samplesort; deterministic bounded-depth variant here. https://research.ibm.com/publications/samplesort-a-sampling-approach-to-minimal-storage-tree-sorting
/// @see samplesort()
template<class T>
SampleSortRun<T> samplesort_run(std::vector<T> values, const SampleSortOptions& options = {}) {
  samplesort_detail::validate(values, options);
  samplesort_detail::Worker<T> w{values, options, false};
  w.sort(0, values.size(), 0);
  w.out.values = std::move(values);
  return std::move(w.out);
}
/// @tparam T Copyable/movable key type with strict weak operator-less ordering.
/// @brief Sort owned keys using actual deterministic sampled bucket partitions and retain their native events.
/// Samples 24 evenly spaced addresses, sorts that sample and selects seven splitters at indices 3,6,...,21. Equal-to-splitter keys enter the higher bucket. Ranges of at most sixteen use insertion sort; one-whole-bucket or configured depth uses counted std::sort. All actual key comparisons are counted in disjoint categories; library-dependent std::sort counts need not be portable. Unstable. Trace owns actual before/after states; partition after is concatenation before children. Units are two range copies plus event header, and for each partition 24 sampled keys+24 source indices+7 splitters+8 bucket sizes. Typical O(n log n), bounded-depth fallback prevents unbounded recursion; no worst-case linear distribution claim. Input is passed by value, so caller storage is unchanged. Keys/comparator exceptions propagate; independent calls share no state.
/// @param values Owned working keys; floating types must all be finite.
/// @param options Complete input, actual comparison, history and depth bounds.
/// @return Owned sorted values and actual work/history.
/// @throws std::invalid_argument Nonfinite floating key or depth outside [1,256].
/// @throws std::length_error Input, work-counter or actual comparison/history bounds exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::samplesort_trace(std::vector<int>{3,1,2});
/// std::print("{}\n",r.values); // [1, 2, 3]
/// @endcode
/// @par References
/// - Frazer and McKellar (1970), Samplesort; deterministic bounded-depth variant here. https://research.ibm.com/publications/samplesort-a-sampling-approach-to-minimal-storage-tree-sorting
/// @see samplesort()
template<class T>
SampleSortRun<T> samplesort_trace(std::vector<T> values, const SampleSortOptions& options = {}) {
  samplesort_detail::validate(values, options);
  samplesort_detail::Worker<T> w{values, options, true};
  w.sort(0, values.size(), 0);
  w.out.values = std::move(values);
  return std::move(w.out);
}
/// @tparam T Copyable/movable key with strict weak operator-less ordering.
/// @brief Sort caller keys in place through the same checked native worker.
/// Preserves the legacy signature and deterministic splitter scheme. No optional history or final output copy. Exceptions during comparison/copy/move give the underlying operations basic guarantee; validation occurs before mutation.
/// @param a Borrowed mutable complete key vector.
/// @throws std::invalid_argument Nonfinite floating key.
/// @throws std::length_error Default input/comparison limits exceeded.
/// @par Example
/// @code
/// std::vector<int> a{3,1,2};dm::algorithms::samplesort(a);
/// std::print("{}\n",a); // [1, 2, 3]
/// @endcode
/// @par References
/// - Frazer and McKellar (1970), Samplesort; deterministic bounded-depth variant here. https://research.ibm.com/publications/samplesort-a-sampling-approach-to-minimal-storage-tree-sorting
/// @see samplesort_run()
template<class T>
void samplesort(std::vector<T>& a) {
  const SampleSortOptions o;
  samplesort_detail::validate(a, o);
  samplesort_detail::Worker<T> w{a, o, false};
  w.sort(0, a.size(), 0);
}
/// @brief Generate the original deterministic unsigned-key study workload.
/// Uses local mt19937_64 and remainder 1000000; values are nonnegative int keys. No shared random state.
/// @param n Complete requested keys, default 8000.
/// @param seed Local engine seed, default 21.
/// @param max_keys Complete allocation cap; zero disables.
/// @return Owned original controlled workload.
/// @throws std::length_error Requested key capacity/policy exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::samplesort_example_values(3).size()); // 3
/// @endcode
/// @par References
/// - Frazer and McKellar (1970), Samplesort; deterministic bounded-depth variant here. https://research.ibm.com/publications/samplesort-a-sampling-approach-to-minimal-storage-tree-sorting
/// @see samplesort_trace()
std::vector<int> samplesort_example_values(
    std::size_t   n        = 8000,
    std::uint64_t seed     = 21,
    std::size_t   max_keys = 1000000);
} // namespace datamunge::algorithms
