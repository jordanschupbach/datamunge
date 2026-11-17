#pragma once
/// @file jaro_winkler.hpp
/// @brief Byte-based Jaro and unconditional Winkler scores with actual phase events.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Complete byte admission and optional actual event-record ceiling.
/// Complete first.size()+second.size() is checked before flags/input copies.
/// Events are admitted online before each candidate/order/prefix inspection;
/// zero disables either optional cap, and untraced calls ignore max_events.
/// Counts exclude flag workspace, scalar summaries, vector/string capacities,
/// and byte/peak RAM; the byte count admits borrowed inputs and their trace
/// copies once, not as a total allocation quota. No partial trace on failure.
/// @par Example
/// @code
/// const dm::algorithms::JaroWinklerOptions o;
/// std::print("{} {}\n",o.max_total_bytes,o.max_events); // 1000000 1000000
/// @endcode
/// @see jaro_winkler_run(), jaro_winkler_trace()
struct JaroWinklerOptions {
  std::size_t max_total_bytes{
      1000000};                    ///< Complete sum of borrowed input byte lengths; zero disables.
  std::size_t max_events{1000000}; ///< Actual chronological event count; zero disables.
};
/// @brief Phase of one actual Jaro/Winkler inspection.
/// No hidden matching replay; candidate means a visited window slot, order
/// means comparison of matched byte sequences, prefix means an actual prefix test.
/// @par Example
/// @code
/// const auto t=dm::algorithms::jaro_winkler_trace("A","A");
/// std::print("{}\n",t.events[0].phase==dm::algorithms::JaroWinklerPhase::Candidate); // true
/// @endcode
/// @see JaroWinklerEvent
enum class JaroWinklerPhase {
  Candidate, ///< One actually visited candidate slot, including occupied slots skipped without reading bytes.
  Order, ///< One actual comparison between the two ordered matched-byte sequences.
  Prefix ///< One actual equality test while scanning a prefix of at most four bytes.
};
/// @brief Owned scalar snapshot of one actually executed phase inspection.
/// Occupied candidate slots have compared=false and both byte fields zero;
/// no extra input byte reads are introduced for recording. Other events have
/// compared=true and equal records actual byte equality. Candidate equal events
/// are actual accepted matches. Order pairing can differ from match partners.
/// @par Example
/// @code
/// const auto t=dm::algorithms::jaro_winkler_trace("A","A");
/// std::print("{} {}\n",t.events[0].first_index,t.events[0].equal); // 0 true
/// @endcode
/// @see JaroWinklerTrace
struct JaroWinklerEvent {
  JaroWinklerPhase phase{JaroWinklerPhase::Candidate}; ///< Actual executing phase.
  std::size_t      first_index{0};  ///< Visited original first-string byte index.
  std::size_t      second_index{0}; ///< Visited original second-string byte index.
  bool             occupied{
      false}; ///< Candidate second-slot was already matched; unused/false for other phases.
  bool         compared{false}; ///< Whether an actual byte-equality test occurred.
  std::uint8_t first_byte{0};   ///< Actual unsigned byte, meaningful only if compared=true.
  std::uint8_t second_byte{0};  ///< Actual unsigned byte, meaningful only if compared=true.
  bool         equal{false};    ///< Actual equality outcome, meaningful only if compared=true.
};
/// @brief Selected actual inspections, excluding bounds/arithmetic/flag writes/allocation.
/// Candidate slots include occupied skips, which do not execute byte equality.
/// Order tests equal matched count; prefix tests include a terminating mismatch.
/// These are logical operations, not physical memory/cache accesses or time.
/// @par Example
/// @code
/// const auto r=dm::algorithms::jaro_winkler_run("A","A");
/// std::print("{} {} {}\n",r.work.candidate_slots,r.work.order_byte_tests,r.work.prefix_byte_tests); // 1 1 1
/// @endcode
/// @see jaro_winkler_run()
struct JaroWinklerWork {
  std::size_t candidate_slots{0};      ///< Actually visited candidate positions.
  std::size_t candidate_byte_tests{0}; ///< Actual unoccupied-candidate equality tests.
  std::size_t order_byte_tests{0};     ///< Actual comparisons in matched-byte order.
  std::size_t prefix_byte_tests{0};    ///< Actual prefix equality tests, at most four.
};
/// @brief Actual Jaro ingredients, unconditional prefix score and selected work.
/// No input ownership/references. Scores are numerical similarities, not identity
/// probabilities or a metric distance. Odd ordered-mismatch counts retain a
/// half-integer t=mismatches/2, not integer division. Bytes are case sensitive,
/// include NUL/high bytes, and receive no Unicode decoding or normalization.
/// @par Example
/// @code
/// const auto r=dm::algorithms::jaro_winkler_run("MARTHA","MARHTA");
/// std::print("{} {} {}\n",r.matches,r.order_mismatches,r.prefix); // 6 2 3
/// @endcode
/// @see jaro_winkler_run()
struct JaroWinklerRun {
  double      jaro{0};    ///< Actual unboosted score; both empty1, one empty0.
  double      winkler{0}; ///< jaro+prefix*scale*(1-jaro), with no boost threshold.
  std::size_t window{
      0}; ///< max(floor(max length/2)-1,0) byte displacement, zero for empty early exits.
  std::size_t matches{0};          ///< Actual accepted one-to-one window byte matches.
  std::size_t order_mismatches{0}; ///< Actual unequal positions in ordered matched byte sequences.
  std::size_t prefix{0};           ///< Actual common byte-prefix length capped at four.
  JaroWinklerWork work;            ///< Actual selected logical inspections.
};
/// @brief Owned original bytes and chronological events from the producing loop.
/// Inputs are copied once for a self-contained trace, preserving NUL/high bytes.
/// No repeated-prefix scoring or reconstructed matching decisions. Events cover
/// candidate slots, ordered comparisons and actual prefix tests in execution order.
/// @par Example
/// @code
/// const auto t=dm::algorithms::jaro_winkler_trace("A","A");
/// std::print("{} {}\n",t.first,t.events.size()); // A 3
/// @endcode
/// @see jaro_winkler_trace()
struct JaroWinklerTrace {
  JaroWinklerRun                run;    ///< Same-execution actual scores, ingredients and work.
  std::string                   first;  ///< Owned complete first input bytes.
  std::string                   second; ///< Owned complete second input bytes.
  std::vector<JaroWinklerEvent> events; ///< Actual chronological phase inspections.
};
/// @brief Compute raw-byte Jaro similarity using the original left-to-right greedy rule.
/// Each first-string byte takes the leftmost still-unmatched equal second byte
/// within window max(floor(max length/2)-1,0). Compare the two matched sequences
/// in their own original index order; t is half their disagreement count.
/// Returns0 if no window matches, even when equal bytes occur outside windows;
/// both empty gives1, one empty0. No case folding, Unicode decoding, normalization
/// or C-string termination at NUL. No input mutation, retained references or
/// shared state. Shares the traced kernel but omits prefix inspections/history.
/// O(n+m+n*min(m,2w+1)) work and O(n+m) flag workspace for nonempty inputs;
/// empty exits allocate no flags. Original signature/arithmetic order preserved;
/// overflow-safe window bounds replace unchecked i+window+1.
/// @param a Borrowed first byte string, including NUL/high bytes.
/// @param b Borrowed second byte string.
/// @return Similarity in[0,1], not an identity probability or metric distance.
/// @throws std::length_error On complete size/count/container overflow.
/// @throws std::bad_alloc On flag allocation failure.
/// @par Example
/// @code
/// std::print("{:.6f}\n",dm::algorithms::jaro_similarity("MARTHA","MARHTA")); // 0.944444
/// @endcode
/// @par References
/// - Jaro (1989), JASA84(406):414-420: https://doi.org/10.1080/01621459.1989.10478785.
/// @see jaro_winkler_similarity(), jaro_winkler_trace()
double jaro_similarity(const std::string& a, const std::string& b);
/// @brief Compute the retained unconditional raw-byte Winkler prefix refinement.
/// Returns j+l*p*(1-j) for capped byte prefix l<=4. This implementation has no
/// boost threshold. Finite p in[0,.25] is now required before any allocation,
/// correcting previously unbounded/nonfinite outputs. This domain ensures a
/// convex combination with1; at p=.25, a shared four-byte prefix can yield1
/// for unequal strings. Default p=.1 retains all legacy valid-p arithmetic.
/// Same byte/matching/workspace/ownership policies as jaro_similarity(), plus
/// at most four actual prefix byte tests. No identity/probability guarantee.
/// @param a Borrowed first complete byte string.
/// @param b Borrowed second complete byte string.
/// @param prefix_scale Finite scale in[0,.25], default .1.
/// @return Bounded unconditional similarity, both empty1 and one empty0.
/// @throws std::invalid_argument On nonfinite/out-of-domain scale before allocation.
/// @throws std::length_error On complete size/count/container overflow.
/// @throws std::bad_alloc On flag allocation failure.
/// @par Example
/// @code
/// std::print("{:.6f}\n",dm::algorithms::jaro_winkler_similarity("MARTHA","MARHTA")); // 0.961111
/// @endcode
/// @par References
/// - Winkler (1990), String Comparator Metrics and Enhanced Decision Rules in the Fellegi-Sunter Model of Record Linkage: https://files.eric.ed.gov/fulltext/ED325505.pdf.
/// @see jaro_winkler_run()
double
jaro_winkler_similarity(const std::string& a, const std::string& b, double prefix_scale = .1);
/// @brief Return both byte-based scores, actual ingredients and work without history.
/// Same retained matching/score/scale policy as jaro_winkler_similarity(). Complete
/// combined input byte admission precedes flags/reads, optional max_events ignored.
/// No copied inputs, mutation, retained references or shared state. O(n+m) flag
/// workspace; quadratic worst-case candidate work, plus at most four prefix tests.
/// @param a Borrowed first complete byte string.
/// @param b Borrowed second complete byte string.
/// @param prefix_scale Finite scale in[0,.25], default .1.
/// @param options Complete byte cap and ignored event cap; defaults above.
/// @return Actual same-run base/refined scores, matching ingredients and selected counts.
/// @throws std::invalid_argument On invalid scale before allocation/reads.
/// @throws std::length_error On enabled input cap or size/count overflow.
/// @throws std::bad_alloc On flag allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::jaro_winkler_run("AB","BA");
/// std::print("{} {}\n",r.matches,r.winkler); // 0 0
/// @endcode
/// @see jaro_winkler_trace()
JaroWinklerRun jaro_winkler_run(
    const std::string& a,
    const std::string& b,
    double             prefix_scale = .1,
    JaroWinklerOptions options      = {});
/// @brief Capture actual candidate, order and prefix inspections without scoring replay.
/// Uses the same producing kernel as jaro_winkler_run(). Owns complete input
/// bytes and scalar records; no retained caller references. Complete byte cap
/// precedes copying/flags; each actual event is admitted before its inspection.
/// Zero optional caps disable them; count/container checks remain. No partial
/// output returns on failure. Additional O(E+n+m) owned trace data for E events;
/// default event ceiling one million, not a byte/peak-RAM bound.
/// @param a Borrowed complete first bytes, copied into result after admission.
/// @param b Borrowed complete second bytes, copied into result after admission.
/// @param prefix_scale Finite scale in[0,.25], default .1.
/// @param options Complete byte and online actual-event policies.
/// @return Owned same-execution input snapshots, scores/work and actual event chronology.
/// @throws std::invalid_argument On invalid scale before allocation/reads.
/// @throws std::length_error On enabled cap or size/count/container overflow.
/// @throws std::bad_alloc On flag/input/event allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::jaro_winkler_trace("MARTHA","MARHTA");
/// std::print("{}\n",t.run.prefix); // 3
/// @endcode
/// @see jaro_winkler_run(), JaroWinklerEvent
JaroWinklerTrace jaro_winkler_trace(
    const std::string& a,
    const std::string& b,
    double             prefix_scale = .1,
    JaroWinklerOptions options      = {});
} // namespace datamunge::algorithms
