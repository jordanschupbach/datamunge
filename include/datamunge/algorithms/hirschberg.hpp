#pragma once
/// @file hirschberg.hpp
/// @brief Checked global Hirschberg alignment, actual split profiles and released rolling rows.
#include <cstddef>
#include <datamunge/algorithms/alignment_types.hpp>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Complete input admission and online producing-work/history policies.
/// Counts are not bytes/peak RAM: input counts bytes of both strings, work counts
/// rolling DP cells and base-case pair scans, capture counts one record plus
/// three score arrays per split. Fields/capacity/output/frame overhead excluded.
/// Zero disables each optional cap. Orientation can change tied optimal rows.
/// @par Example
/// @code
/// const dm::algorithms::HirschbergOptions o;
/// std::print("{}\n",o.orient_shorter); // true
/// @endcode
/// @see hirschberg_run(), hirschberg_trace()
struct HirschbergOptions {
  bool        orient_shorter{true}; ///< Transpose when needed so second/row dimension is shorter.
  std::size_t max_input_bytes{1000000}; ///< Complete sum of input bytes; zero disables.
  std::size_t max_work_cells{
      10000000}; ///< Online complete sweep/base scan work cap; zero disables.
  std::size_t max_trace_entries{1000000}; ///< Online split record/score-value cap; zero disables.
};
/// @brief Selected actual producing work and live rolling score-value peak.
/// Peak excludes output, stack frames, views, trace copies, capacity and bytes.
/// Left result row is held during the reverse sweep, so up to three rows live,
/// then all score rows are destroyed before either recursive child is entered.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hirschberg_run("AB","AB");
/// std::print("{}\n",r.work.peak_score_slots); // 9
/// @endcode
/// @see HirschbergRun
struct HirschbergWork {
  std::size_t calls{0};         ///< Actual recursive invocations including empty/base cases.
  std::size_t rolling_cells{0}; ///< Actual scored cells in forward/reverse rolling sweeps.
  std::size_t base_pairs{0};    ///< Actual character-pair score scans in one-character base cases.
  std::size_t profile_positions{0}; ///< Actual candidate split columns tested.
  std::size_t peak_score_slots{0}; ///< Actual live score-vector element peak, not capacities/bytes.
  std::size_t max_depth{0};        ///< Actual recursive depth, root1.
};
/// @brief Owned alignment, actual work and orientation from the producing execution.
/// Aligned output rows are always in original caller order, even if internal
/// orientation transposes the inputs. No references retained; copies are deep.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hirschberg_run("A","ABC");
/// std::print("{}\n",r.transposed); // true
/// @endcode
/// @see hirschberg_run()
struct HirschbergRun {
  Alignment      alignment;         ///< Actual optimal global alignment in caller order.
  HirschbergWork work;              ///< Actual selected work and live score-row peak.
  bool           transposed{false}; ///< Internal first string is caller's second string.
};
/// @brief Actual forward/reverse profiles and first maximizing split from one recursive call.
/// Ranges and midpoints are zero-based offsets in internally oriented inputs.
/// reverse[j] scores the suffix beginning at column j, already reindexed.
/// All vectors own actual values; total is the producing split sum, not replay.
/// @par Example
/// @code
/// const auto t=dm::algorithms::hirschberg_trace("AB","AB");
/// std::print("{} {}\n",t.splits[0].a_mid,t.splits[0].b_mid); // 1 1
/// @endcode
/// @see HirschbergTrace
struct HirschbergSplit {
  std::size_t      a_begin{0}; ///< Inclusive first-input subrange start in internal orientation.
  std::size_t      a_end{0};   ///< Exclusive first-input subrange end.
  std::size_t      b_begin{0}; ///< Inclusive second-input subrange start.
  std::size_t      b_end{0};   ///< Exclusive second-input subrange end.
  std::size_t      depth{0};   ///< Actual call depth, root1.
  std::size_t      a_mid{0};   ///< Actual global midpoint of first-input range.
  std::size_t      b_mid{0};   ///< Actual first-maximizing global crossing column.
  std::vector<int> forward;    ///< Actual left-half score row, length subrange_b+1.
  std::vector<int> reverse;    ///< Actual right-half suffix scores in original column order.
  std::vector<int> total;      ///< Actual forward+reverse sums used to select split.
};
/// @brief Owned complete split histories and producing run, with online whole-record admission.
/// Only nonbase split calls have records, in actual preorder. Input views and
/// temporary score rows are not retained. Capture failure returns no partial
/// trace. Additional retained histories do not obey the untraced row-space bound.
/// @par Example
/// @code
/// const auto t=dm::algorithms::hirschberg_trace("AB","AB");
/// std::print("{}\n",t.captured_entries); // 10
/// @endcode
/// @see HirschbergOptions, hirschberg_trace()
struct HirschbergTrace {
  HirschbergRun                run;    ///< Actual producing result/work/orientation.
  std::vector<HirschbergSplit> splits; ///< Owned actual nonbase split records.
  std::size_t captured_entries{0};     ///< One record plus3*(subrange_b+1) values per split.
};
/// @brief Align whole byte strings globally with the preserved legacy input orientation and ties.
/// Splits first input at its midpoint and takes first maximizing column. Base
/// traceback preserves diagonal/up/left preference without a dense DP table.
/// Uses views, releases profiles before children and appends output once; no
/// substring/reversal copies or recursive alignment concatenation. Time
/// O(nm+n+m); auxiliary score values O(m), stack O(log(n+1)), owned output O(n+m).
/// This legacy orientation does not promise O(min(n,m)) score rows. No optional
/// input/work/history caps; representability checks still precede execution.
/// Inputs must exclude literal '-' to make gap rows unambiguous. Other bytes,
/// including NUL, compare literally. Conservative score admission requires
/// (n+m)*max(abs(match),abs(mismatch),abs(gap))<=INT_MAX. Any integer scoring signs
/// are otherwise supported. Tied alignment rows may differ from dense NW.
/// Independent calls share no mutable state and retain no references.
/// @param a Borrowed first byte string, no literal '-'.
/// @param b Borrowed second byte string, no literal '-'.
/// @param scoring Linear symmetric integer scores, default2/-1/-2.
/// @return Owned optimal global score and gapped rows, in caller order.
/// @throws std::invalid_argument On reserved input gap bytes.
/// @throws std::length_error On input/output/count/container representability.
/// @throws std::overflow_error On conservative int score-domain rejection.
/// @throws std::bad_alloc On score/output allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::hirschberg("AB","AB").score); // 4
/// @endcode
/// @par References
/// - D. S. Hirschberg (1975), A linear space algorithm for computing maximal common
///   subsequences, https://doi.org/10.1145/360825.360861 . This applies the split
///   principle to additive global alignment scores, not only unweighted LCS.
/// @see hirschberg_run(), hirschberg_trace(), AlignmentScoring
Alignment hirschberg(const std::string& a, const std::string& b, AlignmentScoring scoring = {});
/// @brief Execute checked Hirschberg with optional shorter score-row orientation and actual work.
/// Same domains, base ties and producing kernel as hirschberg(). Default internal
/// second input is shorter, giving O(min(n,m)) score values plus O(log(max(n,m)+1))
/// frames and O(n+m) output. Transposition can choose a different tied optimum,
/// but output rows return to caller order. Complete input domain/caps precede
/// output allocation; each full score sweep/base scan is admitted before it runs.
/// @param a Borrowed first gap-free byte string.
/// @param b Borrowed second gap-free byte string.
/// @param scoring Linear integer scores, default2/-1/-2.
/// @param options Complete input/online-work policies, default shorter orientation/1m/10m.
/// @return Owned actual alignment and selected producing work.
/// @throws std::invalid_argument On reserved input '-'.
/// @throws std::length_error On input/work/count/container caps.
/// @throws std::overflow_error On conservative score-domain rejection.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::hirschberg_run("AB","AB").alignment.score); // 4
/// @endcode
/// @see HirschbergRun, HirschbergOptions
HirschbergRun hirschberg_run(
    const std::string& a,
    const std::string& b,
    AlignmentScoring   scoring = {},
    HirschbergOptions  options = {});
/// @brief Capture the same execution's actual split profiles without replay or dense tables.
/// Complete record units1+3*(subrange_b+1) admitted before profile capture/call
/// work. Known row-work units also admitted before their producing sweeps.
/// No truncation/partial result or caller mutation. Histories own copies and add
/// their retained memory beyond untraced score rows, frames and ordinary output.
/// @param a Borrowed first gap-free byte string.
/// @param b Borrowed second gap-free byte string.
/// @param scoring Linear integer scores, default2/-1/-2.
/// @param options Complete input/online-work/capture policies, default1m/10m/1m.
/// @return Owned complete actual split profiles, run and logical capture count.
/// @throws std::invalid_argument On reserved input '-'.
/// @throws std::length_error On input/work/capture/count/container caps.
/// @throws std::overflow_error On conservative score-domain rejection.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::hirschberg_trace("AB","AB",{}, {false,0,0,10});
/// std::print("{}\n",t.splits.size()); // 1
/// @endcode
/// @see HirschbergTrace, HirschbergSplit
HirschbergTrace hirschberg_trace(
    const std::string& a,
    const std::string& b,
    AlignmentScoring   scoring = {},
    HirschbergOptions  options = {});
} // namespace datamunge::algorithms
