#pragma once
/// @file dice_coefficient.hpp
/// @brief Byte-bigram multiset Dice/Jaccard overlap and actual occurrence consumption.
/// Extends Dice's set-overlap normalization to literal adjacent byte occurrences;
/// the original ecological coefficient was not a word-bigram algorithm.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Checked multiset sizes, intersection/union counts and normalized overlap.
/// Counts are occurrence counts, not numbers of distinct keys. For a nonempty
/// combined multiset, Dice=2*shared/total and Jaccard=shared/union_count. For two
/// empty multisets both values use the caller's explicit empty-equality convention.
/// Scores use binary64 rounding; integer fields retain exact checked counts.
/// @par Example
/// @code
/// const auto s=dm::algorithms::dice_overlap_counts(4,4,1);
/// std::print("{} {} {:.6f} {:.6f}\n",s.shared,s.union_count,s.coefficient,s.jaccard); // 1 7 0.250000 0.142857
/// @endcode
/// @see dice_overlap_counts(), DiceTrace
struct DiceOverlap {
  std::size_t left_bigrams{0}; ///< Number of left occurrences, including repeats.
  std::size_t right_bigrams{0}; ///< Number of right occurrences, including repeats.
  std::size_t shared{0}; ///< Multiset intersection: sum of per-key minimum counts.
  std::size_t total{0}; ///< Checked sum of both occurrence counts.
  std::size_t union_count{0}; ///< Multiset union: total minus shared, not distinct-key count.
  double coefficient{0}; ///< Dice score, or the explicit both-empty convention.
  double jaccard{0}; ///< Multiset Jaccard score, or the same both-empty convention.
};
/// @brief One unsigned-byte bigram and its actual left/right/shared multiplicities.
/// key=(first_byte<<8)|second_byte. All 65,536 byte pairs are distinct, including
/// NUL and bytes 128–255. Native trace rows are sorted by this unsigned key.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dice_coefficient_trace("aa","aaaa");
/// std::print("{} {} {}\n",t.counts[0].left_count,t.counts[0].right_count,t.counts[0].shared_count); // 1 3 1
/// @endcode
/// @see DiceTrace
struct DiceBigramCount {
  std::uint16_t key{0}; ///< Packed first/second unsigned bytes; high byte comes first.
  std::size_t left_count{0}; ///< Occurrences in the left input.
  std::size_t right_count{0}; ///< Occurrences in the right input.
  std::size_t shared_count{0}; ///< Actually consumed left occurrences; minimum of both counts.
};
/// @brief Actual remaining-count transition for one right-hand bigram occurrence.
/// One record per ascending right byte offset, including failed matches. Before
/// and after refer only to this key's unconsumed left count. A miss has both zero.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dice_coefficient_trace("aa","aaaa");
/// std::print("{} {} {}\n",t.events[0].remaining_before,t.events[0].remaining_after,t.events[1].matched); // 1 0 false
/// @endcode
/// @see dice_coefficient_trace(), DiceBigramCount
struct DiceConsumeEvent {
  std::size_t right_offset{0}; ///< Zero-based first-byte offset in the right string.
  std::uint16_t key{0}; ///< Actual adjacent unsigned-byte pair at that offset.
  std::size_t remaining_before{0}; ///< Unconsumed left multiplicity before this lookup.
  std::size_t remaining_after{0}; ///< Remaining left multiplicity after this lookup.
  bool matched{false}; ///< True exactly when one left occurrence was consumed.
};
/// @brief Actual occurrence-level operations of the shared consuming kernel.
/// One left dictionary update per left bigram and one right lookup per right
/// bigram. Counts exclude hash collisions/probes, reserve/rehash, equality checks,
/// copies, trace-only row insertion/counting, sorting and score arithmetic.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dice_coefficient_trace("aa","aaaa");
/// std::print("{} {} {} {}\n",t.work.left_updates,t.work.right_lookups,t.work.matches,t.work.misses); // 1 3 1 2
/// @endcode
/// @see DiceTrace
struct DiceWork {
  std::size_t left_updates{0}; ///< Actual kernel left occurrence updates.
  std::size_t right_lookups{0}; ///< Actual right occurrence match lookups.
  std::size_t matches{0}; ///< Successful one-occurrence consumptions.
  std::size_t misses{0}; ///< Missing-key or exhausted-count lookups.
};
/// @brief Owned actual bigram counts, consumed occurrences, inputs and scores.
/// No replay or independent reconstruction: sorted rows and events come from the
/// executing dictionary kernel. Rows include keys seen on either side. The rich
/// trace scans even identical/short inputs, unlike the scalar early-return paths.
/// Containers own their values, including byte strings; no shared mutable state.
/// There is no case folding, trimming, padding, UTF-8 decoding or normalization.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dice_coefficient_trace("night","nacht");
/// std::print("{} {} {} {:.6f}\n",t.counts.size(),t.events.size(),t.overlap.shared,t.overlap.coefficient); // 7 4 1 0.250000
/// @endcode
/// @see dice_coefficient_trace(), dice_coefficient()
struct DiceTrace {
  std::string left; ///< Owned unchanged first byte string.
  std::string right; ///< Owned unchanged second byte string.
  DiceOverlap overlap; ///< Checked totals and both overlap scores.
  std::vector<DiceBigramCount> counts; ///< Union of distinct keys, sorted unsigned.
  std::vector<DiceConsumeEvent> events; ///< Actual transitions, in right-offset order.
  DiceWork work; ///< Occurrence-level kernel counts, not timings or hash probes.
};
/// @brief Normalize supplied multiset overlap counts without constructing strings.
/// Validates shared<=min(left,right), then checks the total sum. No allocation,
/// mutation or shared state; O(1) work. Does not verify that counts came from
/// actual strings. The empty convention is supplied separately from bag sizes.
/// @param left Number of left bigram occurrences, including multiplicities.
/// @param right Number of right bigram occurrences, including multiplicities.
/// @param shared Multiset intersection count, at most each side's count.
/// @param empty_equal Value for both scores when left=right=0, default true.
/// @return Exact integer counts plus floating Dice/Jaccard; empty scores are 1 or 0.
/// @throws std::invalid_argument If shared exceeds either side, before overflow checks.
/// @throws std::length_error If left+right is not representable in size_t.
/// @par Example
/// @code
/// const auto s=dm::algorithms::dice_overlap_counts(1,3,1);
/// std::print("{} {:.6f} {:.6f}\n",s.total,s.coefficient,s.jaccard); // 4 0.500000 0.333333
/// @endcode
/// @par References
/// - L. R. Dice (1945), Measures of the Amount of Ecologic Association Between Species, Ecology 26(3), 297–302, https://doi.org/10.2307/1932409. The present API extends the normalization to multiset counts.
/// @see dice_coefficient(), DiceOverlap
DiceOverlap dice_overlap_counts(std::size_t left,std::size_t right,std::size_t shared,bool empty_equal=true);
/// @brief Compute the original literal byte-bigram multiset Dice coefficient.
/// Identical strings give 1; any other pair with a length below two gives 0.
/// Otherwise uses 2*sum(min(count_a,count_b))/((a.size()-1)+(b.size()-1)). Equal
/// bigram bags can belong to different strings: score one does not prove equality.
/// Bytes, case, whitespace and embedded NULs are literal; there is no Unicode
/// decoding, normalization or padding. Inputs are borrowed and unchanged.
/// A size_t-count dictionary packs each key into 16 bits, without substring keys.
/// Expected O(|a|+|b|) work; O(min(|a|,65536)) auxiliary storage. Hash collision
/// behavior can increase work; the identity/short early paths allocate no table.
/// Calls share no state. Scores use binary64 rounding, not a semantic probability.
/// @param a First arbitrary byte string, including empty strings and embedded NULs.
/// @param b Second arbitrary byte string.
/// @return Original [0,1] score and short-input conventions, up to floating rounding.
/// @throws std::length_error If the complete nontrivial occurrence sum overflows.
/// @throws std::bad_alloc If dictionary allocation fails.
/// @par Example
/// @code
/// std::print("{:.6f} {:.6f} {:.6f}\n",dm::algorithms::dice_coefficient("night","nacht"),dm::algorithms::dice_coefficient("aa","aaaa"),dm::algorithms::dice_coefficient("a","b")); // 0.250000 0.500000 0.000000
/// @endcode
/// @par References
/// - L. R. Dice (1945), Measures of the Amount of Ecologic Association Between Species, https://doi.org/10.2307/1932409 (normalization, not word matching).
/// - G. Kondrak, D. Marcu and K. Knight (2003), Cognates Can Improve Statistical Translation Models, https://aclanthology.org/N03-2016/ (bigram-based lexical similarity; this API explicitly counts multiplicities).
/// @see dice_coefficient_trace(), dice_overlap_counts(), levenshtein_distance()
double dice_coefficient(const std::string& a,const std::string& b);
/// @brief Record actual byte-bigram consumption and sorted multiplicity rows.
/// Uses the same consuming kernel as the nontrivial scalar path, without replay.
/// Scans both inputs even if equal/short. Complete A+B cap is checked before any
/// copying/allocation/scanning; zero disables only the optional cap, not checked
/// arithmetic. Expected O(|a|+|b|+K*log(K+1)) work including the final row sort,
/// K<=65536 distinct union keys. Owned inputs/events require O(|a|+|b|+K) storage.
/// @param a First literal byte string, copied into the successful result.
/// @param b Second literal byte string, copied into the successful result.
/// @param max_bigrams Complete A+B occurrence cap, default 1000000; zero disables.
/// @return Owned actual events and rows, work counts and legacy-compatible scores.
/// @throws std::length_error For complete cap, count-sum overflow or container capacity.
/// @throws std::bad_alloc If a container allocation fails; no partial trace is returned.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dice_coefficient_trace("aa","aaaa",4);
/// std::print("{} {} {:.6f}\n",t.events.size(),t.work.misses,t.overlap.coefficient); // 3 2 0.500000
/// @endcode
/// @par References
/// - L. R. Dice (1945), Measures of the Amount of Ecologic Association Between Species, https://doi.org/10.2307/1932409; literal-byte multiset extension as specified in dice_coefficient().
/// @see dice_coefficient(), dice_overlap_counts(), DiceTrace
DiceTrace dice_coefficient_trace(const std::string& a,const std::string& b,std::size_t max_bigrams=1000000);
} // namespace datamunge::algorithms
