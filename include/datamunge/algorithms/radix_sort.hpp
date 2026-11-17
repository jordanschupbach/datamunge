#pragma once
/// @file radix_sort.hpp
/// @brief Stable signed-word byte-LSD radix sort with actual pass provenance.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

/// @brief Canonical integer sorting and actual pass records.
namespace datamunge::algorithms {

/// @brief Allocation/history limits for owned radix runs; zero disables a cap.
/// @par Example
/// @code
/// const dm::algorithms::RadixSortOptions x{};
/// std::print("{}\n",x.max_keys); // 1000000
/// @endcode
/// @see radix_sort_run()
struct RadixSortOptions {
  std::size_t max_keys{1000000};          ///< Maximum admitted keys.
  std::size_t max_trace_entries{1000000}; ///< Stored values/origins and histogram/header entries.
};
/// @brief One actual completed stable byte pass on sign-biased keys.
/// @par Example
/// @code
/// const dm::algorithms::RadixSortPass x{};
/// std::print("{}\n",x.shift); // 0
/// @endcode
/// @see radix_sort_run()
struct RadixSortPass {
  unsigned                     shift{};  ///< Actual selected bit shift: 0,8,...,56.
  std::array<std::size_t, 256> counts{}; ///< Actual digit histogram before prefix accumulation.
  std::array<std::size_t, 257>
      offsets{};                    ///< Actual prefix starts plus final n, before scatter mutation.
  std::vector<std::int64_t> values; ///< Actual pass output decoded back to signed keys.
  std::vector<std::size_t>  origins; ///< Corresponding original input positions after this pass.
};
/// @brief Owned ascending keys, stable identities and actual selected pass work.
/// @par Example
/// @code
/// const dm::algorithms::RadixSortResult x{};
/// std::print("{}\n",x.digit_reads); // 0
/// @endcode
/// @see radix_sort_run()
struct RadixSortResult {
  std::vector<std::int64_t>  values;  ///< Final ascending keys.
  std::vector<std::size_t>   origins; ///< Stable original input positions.
  std::vector<RadixSortPass> passes;  ///< Actual records, populated only by radix_sort_trace().
  std::size_t digit_reads{};          ///< Actual counting/scattering digit reads, 16n for n>=2.
  std::size_t scatters{};             ///< Actual stable key/origin scatter pairs, 8n for n>=2.
  std::size_t prefix_steps{};         ///< Actual histogram-prefix updates, 2048 for n>=2.
};
/// @brief Stably sort signed 64-bit keys with the shared byte-LSD producer.
/// Flips the sign bit, then executes eight stable counting/scatter passes. Origin identities are moved alongside keys, exposing stability. Empty/singleton inputs perform zero passes. O(8*(n+256)) time and O(n+256) working storage, plus owned output. Input is unchanged. Counters cover selected digit reads, scatter pairs and prefix updates, excluding allocation/initialization and sign conversion.
/// @param data Owned signed keys.
/// @param options Input allocation bound; zero disables.
/// @return Sorted values, stable original indices and actual selected work.
/// @throws std::length_error Input or representable selected-work capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::radix_sort_run({2,-1,2});
/// std::print("{} {}\n",r.origins[1],r.origins[2]); // 0 2
/// @endcode
/// @see radix_sort_trace(), radix_sort()
RadixSortResult radix_sort_run(
    std::vector<std::int64_t> data,
    const RadixSortOptions&   options = {});
/// @brief Retain every actual histogram, prefix array and stable pass output.
/// Uses the same producer as radix_sort_run(). Each pass costs 2n+514 scalar/header entries, admitted before capture. Pass values are signed decodings of intermediate biased words, not sorted signed arrays until the final pass. History O(8n+8*256), bounded by policy; zero cap disables.
/// @param data Owned signed input keys.
/// @param options Input and retained scalar policies.
/// @return Final stable ordering and eight actual pass records for n>=2.
/// @throws std::length_error Input, work capacity or retained history exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::radix_sort_trace({0,-1});
/// std::print("{}\n",r.passes.size()); // 8
/// @endcode
/// @see radix_sort_run()
RadixSortResult radix_sort_trace(
    std::vector<std::int64_t> data,
    const RadixSortOptions&   options = {});

/// @brief LSD (least-significant-digit) radix sort for signed 64-bit integers, sorting @p data
///        ascending *in place*. Radix sort is a *non-comparison* sort: instead of asking "is a <
///        b?" it distributes keys by their digits, sidestepping the \(\Omega(n \log n)\) lower
///        bound that binds every comparison sort. This implementation treats each key as eight
///        base-256 "digits" (its bytes) and makes one *stable counting-sort* pass per byte, from
///        least to most significant. Because each pass is stable, keys that agree on the current
///        byte keep the order established by the lower bytes already processed -- and after all
///        eight passes the array is fully ordered. Signed keys are handled by flipping the sign
///        bit (equivalently, biasing by \(2^{63}\)) so that two's-complement order maps onto plain
///        unsigned byte order; the bias is undone at the end. Runs in \(O(d\,(n+b))\) time with
///        \(d = 8\) passes and radix \(b = 256\), i.e. linear in @p data 's size, using scratch
///        buffers of size \(n\). Empty and singleton inputs are returned unchanged.
///
/// @param data the integers to sort; overwritten with the ascending-sorted sequence.
/// @par Example
/// @code
/// std::vector<std::int64_t> v{1,-1,0};
/// dm::algorithms::radix_sort(v);
/// std::print("{}\n",v[0]); // -1
/// @endcode
/// @see radix_sort_run()
void radix_sort(std::vector<std::int64_t>& data);

} // namespace datamunge::algorithms
