#pragma once
/// @file quine_mccluskey.hpp
/// @brief Checked exact Boolean minimization, actual merge rounds and native Petrick covering.
#include <cstddef>
#include <cstdint>
#include <datamunge/algorithms/petrick.hpp>
#include <string>
#include <vector>
/// @brief Canonical Boolean minimization and owned execution records.
namespace datamunge::algorithms {
/// @brief Product term: bits fix non-dashed positions; one mask bit denotes a dash.
/// @par Example
/// @code
/// const dm::algorithms::Implicant x{};
/// std::print("{}\n",x.bits); // 0
/// @endcode
/// @see quine_mccluskey_run()
struct Implicant {
  std::uint32_t bits; ///< Fixed values; masked positions are ignored by covers().
  std::uint32_t mask; ///< Dashed positions, with one indicating a don't-care variable.

                      /// @brief Compare the stored bit and mask representation.
                      /// @param other Other implicant.
                      /// @return Whether both stored words are equal.
                      /// @par Example
                      /// @code
                      /// const dm::algorithms::Implicant a{1,0};
                      /// std::print("{}\n",a==dm::algorithms::Implicant{1,0}); // true
                      /// @endcode
                      /// @see covers()
  bool operator==(const Implicant& other) const { return bits == other.bits && mask == other.mask; }
  /// @brief Test whether a raw minterm agrees in every fixed position.
  /// Pure 32-bit comparison; no variable-count policy, allocation or mutation.
  /// @param m Raw 32-bit minterm.
  /// @return True when every nondashed bit agrees.
  /// @par Example
  /// @code
  /// const dm::algorithms::Implicant a{0,1};
  /// std::print("{}\n",a.covers(1)); // true
  /// @endcode
  /// @see implicant_to_string()
  bool covers(std::uint32_t m) const { return ((m ^ bits) & ~mask) == 0; }
};
/// @brief Exact minimum sum-of-products result, preserving the historical field layout.
/// @par Example
/// @code
/// const dm::algorithms::QuineMcCluskeyResult x{};
/// std::print("{}\n",x.minimal_cover.size()); // 0
/// @endcode
/// @see quine_mccluskey_run()
struct QuineMcCluskeyResult {
  std::vector<Implicant>
      prime_implicants; ///< All deduplicated prime implicants from minterms and don't-cares.
  std::vector<Implicant>
      essential; ///< Prime implicants that alone cover at least one required minterm.
  std::vector<Implicant> minimal_cover; ///< Cover minimizing term count, then total literal count.
};
/// @brief Finite selected-search and allocation policies; zero size/work caps disable them.
/// @par Example
/// @code
/// const dm::algorithms::QuineMcCluskeyOptions x{};
/// std::print("{}\n",x.max_variables); // 16
/// @endcode
/// @see quine_mccluskey_run()
struct QuineMcCluskeyOptions {
  std::size_t max_variables{16}; ///< Policy bound on variables; hard representable range is0..32.
  std::size_t max_terms{
      4096}; ///< Bound on input/working/generated/prime term vectors before deduplication.
  std::size_t    max_pair_tests{10000000};   ///< Cumulative actual merge pair tests.
  std::size_t    max_chart_entries{1000000}; ///< Actual prime ids stored in the coverage chart.
  std::size_t    max_trace_entries{1000000}; ///< Retained round scalar/header entries.
  PetrickOptions covering{}; ///< Independent native Petrick candidate and stage-history bounds.
};
/// @brief One actual completed prime-generation merge round.
/// @par Example
/// @code
/// const dm::algorithms::QuineMcCluskeyRound x{};
/// std::print("{}\n",x.terms.size()); // 0
/// @endcode
/// @see quine_mccluskey_run()
struct QuineMcCluskeyRound {
  std::vector<Implicant> terms;  ///< Actual deduplicated terms considered this round.
  std::vector<bool>      used;   ///< Whether each term participated in an actual single-bit merge.
  std::vector<Implicant> next;   ///< Actual deduplicated next round.
  std::vector<Implicant> primes; ///< Actual unmerged terms appended to the prime list this round.
};
/// @brief Owned exact minimization, chart and actual selected search work.
/// @par Example
/// @code
/// const dm::algorithms::QuineMcCluskeyRun x{};
/// std::print("{}\n",x.pair_tests); // 0
/// @endcode
/// @see quine_mccluskey_run()
struct QuineMcCluskeyRun {
  QuineMcCluskeyResult       result;    ///< Exact term-count/literal-count minimized expression.
  std::vector<std::uint32_t> minterms;  ///< Sorted unique required minterms.
  std::vector<std::uint32_t> dontcares; ///< Sorted unique optional minterms; overlap is permitted.
  std::vector<std::vector<int>>
      chart; ///< Actual covering prime ids for each normalized required minterm.
  std::vector<QuineMcCluskeyRound> rounds; ///< Actual merge histories, populated only by the trace.
  PetrickTrace
      covering; ///< Actual residual covering expansion; empty if essentials already cover all.
  std::size_t pair_tests{}; ///< Actual term pairs tested for possible merging.
  std::size_t merges{};     ///< Actual successful pair merges, including duplicate generated terms.
};
/// @brief Render an implicant as an n-character high-bit-first Boolean pattern.
/// Positions masked by one render as a dash. Permits n=0 (empty string) and n=32, with no undefined shifts. Rejects any stored bit/mask outside the declared n-variable domain; masked bits within the domain are allowed. O(n) time and owned string storage.
/// @param imp Bit/mask product term.
/// @param n Variable count in0..32.
/// @return Owned string of zeros, ones and dashes.
/// @throws std::invalid_argument Invalid variable count or out-of-domain stored bit.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::implicant_to_string({8,5},4)); // 1-0-
/// @endcode
/// @see Implicant::covers()
std::string implicant_to_string(const Implicant& imp, int n);
/// @brief Minimize a Boolean function with native prime generation and exact Petrick covering.
/// Validates0..32 variables and every minterm before any shifts/search. Duplicate required/optional terms normalize; overlap is allowed, and required values remain required. Empty required set returns an empty cover; n=0 permits the single empty-variable minterm0. Uses exponential worst-case time/storage, independently bounded by input/working terms, pair tests, chart ids, and native Petrick candidate/history policies. Chooses minimum product count then minimum literal count; residual ties retain deterministic native Petrick order. No global mutation, borrowed inputs unchanged, outputs owned. The residual native Petrick stages are retained even in this non-round-capturing entry point.
/// @param n Variable count in0..32, also bounded by policy.
/// @param minterms Required one-valued domain assignments.
/// @param dontcares Optional assignments, not required to be covered.
/// @param options Independent work/allocation and covering policies.
/// @return Exact minimized expression, complete chart, actual covering expansion and selected counts.
/// @throws std::invalid_argument Variable count or minterm outside the declared domain.
/// @throws std::length_error Any search/allocation policy or representable id/history bound exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::quine_mccluskey_run(2,{0,1});
/// std::print("{}\n",dm::algorithms::implicant_to_string(r.result.minimal_cover[0],2)); // 0-
/// @endcode
/// @par References
/// - W. V. Quine (1952), https://doi.org/10.1080/00029890.1952.11988183; E. J. McCluskey (1956), https://doi.org/10.1002/j.1538-7305.1956.tb03835.x
/// @see quine_mccluskey_trace(), petrick_trace()
QuineMcCluskeyRun quine_mccluskey_run(
    int                          n,
    std::vector<std::uint32_t>   minterms,
    std::vector<std::uint32_t>   dontcares = {},
    const QuineMcCluskeyOptions& options   = {});
/// @brief Capture actual prime merge rounds from the same minimization producer.
/// Retains before/used/next/new-prime arrays for each real round. A record costs3*current+2*next+2*newprimes+1 scalar/header entries, admitted before copying. Other semantics and independent covering-history policy match quine_mccluskey_run().
/// @param n Variable count0..32.
/// @param minterms Required domain assignments.
/// @param dontcares Optional domain assignments, default empty.
/// @param options Input, search, chart and retained-history policies.
/// @return Owned exact minimized result and actual rounds/chart/native covering.
/// @throws std::invalid_argument Invalid variable count or minterm.
/// @throws std::length_error Any policy or representable capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::quine_mccluskey_trace(1,{0,1});
/// std::print("{}\n",r.rounds.size()); // 2
/// @endcode
/// @see quine_mccluskey_run()
QuineMcCluskeyRun quine_mccluskey_trace(
    int                          n,
    std::vector<std::uint32_t>   minterms,
    std::vector<std::uint32_t>   dontcares = {},
    const QuineMcCluskeyOptions& options   = {});
/// @brief Return the exact minimum sum-of-products result with default checked policies.
/// Signature and three-vector result layout preserved. Delegates to the shared producer; invalid domain/shift inputs now throw. Use quine_mccluskey_run() to configure exponential search policies.
/// @param n Variable count0..32, default policy allows at most16.
/// @param minterms Required assignments.
/// @param dontcares Optional assignments, default empty.
/// @return All primes, essentials and a term/literal-minimal cover.
/// @throws std::invalid_argument Invalid domain.
/// @throws std::length_error Default policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::quine_mccluskey(1,{0,1});
/// std::print("{}\n",dm::algorithms::implicant_to_string(r.minimal_cover[0],1)); // -
/// @endcode
/// @see quine_mccluskey_run()
QuineMcCluskeyResult quine_mccluskey(
    int                        n,
    std::vector<std::uint32_t> minterms,
    std::vector<std::uint32_t> dontcares = {});
} // namespace datamunge::algorithms
