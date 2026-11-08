#pragma once
/// @file stable_matching.hpp
/// @brief Complete strict one-to-one deferred acceptance, validated stability and actual FIFO history.
#include <cstddef>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Sentinel for an unmatched agent in either matching orientation.
/// This is size_t(-1), never a valid participant ID. Proposal-count metadata is independent of map feasibility.
/// @par Example
/// @code
/// std::cout << (dm::algorithms::kUnmatched == std::size_t(-1)); // 1
/// @endcode
/// @see gale_shapley(), gale_shapley_trace()
inline constexpr std::size_t kUnmatched = static_cast<std::size_t>(-1);
/// @brief Owned preference rows with favorite first.
/// For n agents each row must be a permutation of0..n-1; the two sides have equal n. Empty sides are allowed. Ties, truncated lists, unequal sizes, capacities and couples are outside this model.
/// @par Example
/// @code
/// dm::algorithms::StablePreferences p{{0,1},{1,0}};
/// std::cout << p.size(); // 2
/// @endcode
/// @see gale_shapley(), gale_shapley_trace()
using StablePreferences = std::vector<std::vector<std::size_t>>;
/// @brief Owned mutual partner maps and proposal-count metadata.
/// Legacy field order/layout retained. A structurally feasible partial matching uses kUnmatched symmetrically; complete strict equal-size inputs produce a perfect matching. Stability ignores proposals, allowing manually constructed candidates. Reversing orientation exchanges the maps and preserves the original execution count.
/// @par Example
/// @code
/// dm::algorithms::StableMatchingResult m;
/// std::cout << m.proposals; // 0
/// @endcode
/// @see gale_shapley(), gale_shapley_trace()
struct StableMatchingResult {
    std::vector<std::size_t> proposer_match; ///< Reviewer ID for each proposer, or kUnmatched.
    std::vector<std::size_t> reviewer_match; ///< Proposer ID for each reviewer, or kUnmatched.
    std::size_t proposals{0}; ///< Actual proposals for a matcher result;0 for enumeration candidates.
};
/// @brief Owned complete strict preferences for two equal sides.
/// Both matrices retain original agent and rank order. The fixture and generators produce independent owned rows without shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::StableMatchingProfile p;
/// std::cout << p.proposer_preferences.size(); // 0
/// @endcode
/// @see gale_shapley(), gale_shapley_trace()
struct StableMatchingProfile {
    StablePreferences proposer_preferences{}; ///< Favorite-first reviewer IDs for every proposer.
    StablePreferences reviewer_preferences{}; ///< Favorite-first proposer IDs for every reviewer.
};
/// @brief Actual rank construction and FIFO decision counts.
/// Excludes input validation, allocation, history copies, checking stability, rank summaries and enumeration. rank_entries counts only the reviewer rank table used by the matcher. A queue push occurs initially or after rejection/displacement; each real proposal pops one free proposer.
/// @par Example
/// @code
/// dm::algorithms::GaleShapleyWork w;
/// std::cout << w.replacements; // 0
/// @endcode
/// @see gale_shapley(), gale_shapley_trace()
struct GaleShapleyWork {
    std::size_t rank_entries{}; ///< Reviewer inverse-rank entries written.
    std::size_t proposals{}; ///< Actual proposals, equal to result.proposals.
    std::size_t accepted_free{}; ///< Proposals held by previously free reviewers.
    std::size_t replacements{}; ///< Proposals replacing an incumbent.
    std::size_t rejections{}; ///< Proposals rejected immediately.
    std::size_t queue_pushes{}; ///< Initial and subsequent FIFO appends.
    std::size_t queue_pops{}; ///< FIFO removals for actual proposals.
    std::size_t peak_free{}; ///< Maximum number of queued free proposers.
    std::size_t next_choice_advances{}; ///< Actual increments of a proposer next-choice index.
};
/// @brief One matching with actual kernel work.
/// Both ordinary and diagnostic fitting use the same FIFO traversal without replay. The theorem concerns submitted strict complete rankings and stable matchings, not maximum total utility or real-world program constraints.
/// @par Example
/// @code
/// dm::algorithms::GaleShapleyRun r;
/// std::cout << r.work.proposals; // 0
/// @endcode
/// @see gale_shapley(), gale_shapley_trace()
struct GaleShapleyRun {
    StableMatchingResult result{}; ///< Owned proposer-oriented result.
    GaleShapleyWork work{}; ///< Actual rank and proposal operations.
};
/// @brief Owned actual proposal decision and before/after FIFO state.
/// Captured after the actual acceptance/replacement/rejection. next_choice and both maps are post-decision; queue_before is before removing this proposer, queue_after is after any append. decision is accept-free,replace or reject. No regenerated history. kUnmatched denotes no prior incumbent or no newly freed proposer.
/// @par Example
/// @code
/// dm::algorithms::GaleShapleyEvent e;
/// std::cout << (e.held == dm::algorithms::kUnmatched); // 1
/// @endcode
/// @see gale_shapley(), gale_shapley_trace()
struct GaleShapleyEvent {
    std::size_t number{}; ///< One-based actual proposal number.
    std::size_t proposer{}; ///< Proposer removed from queue front.
    std::size_t reviewer{}; ///< Reviewer actually proposed to.
    std::size_t held{kUnmatched}; ///< Reviewer incumbent before this proposal.
    std::size_t newly_free{kUnmatched}; ///< Displaced incumbent, rejected proposer, or kUnmatched.
    std::string decision{}; ///< One of the three actual decision names.
    std::vector<std::size_t> next_choice{}; ///< Actual next rank index for every proposer.
    std::vector<std::size_t> proposer_match{}; ///< Actual post-decision proposer map.
    std::vector<std::size_t> reviewer_match{}; ///< Actual post-decision reviewer map.
    std::vector<std::size_t> queue_before{}; ///< Actual FIFO contents before the pop.
    std::vector<std::size_t> queue_after{}; ///< Actual FIFO contents after the decision.
};
/// @brief Owned inputs, actual reviewer ranks and actual FIFO proposal events.
/// Whole-request conservative admission follows full preference validation, which may allocate O(n) scratch. Admission precedes quadratic rank tables, matching state and history/input copies. No replay, partial trace, truncation, caller mutation or shared state.
/// @par Example
/// @code
/// dm::algorithms::GaleShapleyTrace t;
/// std::cout << t.events.size(); // 0
/// @endcode
/// @see gale_shapley(), gale_shapley_trace()
struct GaleShapleyTrace : public GaleShapleyRun {
    StablePreferences proposer_preferences{}; ///< Owned raw proposer rankings.
    StablePreferences reviewer_preferences{}; ///< Owned raw reviewer rankings.
    StablePreferences reviewer_rank{}; ///< Actual zero-based reviewer inverse-rank table used by the kernel.
    std::vector<GaleShapleyEvent> events{}; ///< Actual performed proposals, no padding to upper bound.
    std::size_t captured_entries{}; ///< Input/rank rows and cells plus actual retained event records/vector elements.
};
/// @brief A genuine mutually preferred pair from a feasible candidate.
/// All ranks are one-based, with smaller better. An unmatched incumbent has rank n+1. Candidate pair ranks are strictly less than incumbent ranks on both sides. This witness is meaningful only after feasibility validation.
/// @par Example
/// @code
/// dm::algorithms::StableBlockingPair b;
/// std::cout << b.proposer; // 0
/// @endcode
/// @see gale_shapley(), gale_shapley_trace()
struct StableBlockingPair {
    std::size_t proposer{}; ///< Blocking proposer ID.
    std::size_t reviewer{}; ///< Blocking reviewer ID.
    std::size_t proposer_rank{}; ///< Reviewer rank in proposer list,1..n.
    std::size_t reviewer_rank{}; ///< Proposer rank in reviewer list,1..n.
    std::size_t proposer_partner_rank{}; ///< Current partner rank, or n+1 if unmatched.
    std::size_t reviewer_partner_rank{}; ///< Current partner rank, or n+1 if unmatched.
};
/// @brief Structural feasibility and all actual blocking pairs for a candidate.
/// A malformed candidate gives feasible=false,complete=false,stable=false and no witnesses. Empty witnesses alone do not certify stability. Valid partial maps may be feasible but are unstable under complete equal-side preferences. Empty valid maps are complete and stable. Proposal metadata is ignored.
/// @par Example
/// @code
/// dm::algorithms::StabilityDiagnostics d;
/// std::cout << d.feasible; // 0
/// @endcode
/// @see gale_shapley(), gale_shapley_trace()
struct StabilityDiagnostics {
    bool feasible{}; ///< Both maps have correct dimensions, ranges and mutual inverse partial assignments.
    bool complete{}; ///< Feasible and every participant matched; true for empty maps.
    bool stable{}; ///< Feasible with no blocking pair.
    std::size_t pairs_examined{}; ///< Actual proposer-reviewer pairs examined;0 for malformed candidates.
    std::vector<StableBlockingPair> blocking_pairs{}; ///< Actual witnesses in proposer-major/reviewer-major order.
};
/// @brief One-based assigned-partner ranks for a perfect candidate.
/// Requires mutual perfect maps; does not itself certify stability. Means are descriptive averages of ordinal ranks, not utility or probabilities. Empty input has has_agents=false and zero mean placeholders, not observed mean satisfaction.
/// @par Example
/// @code
/// dm::algorithms::StableMatchingRanks r;
/// std::cout << r.has_agents; // 0
/// @endcode
/// @see gale_shapley(), gale_shapley_trace()
struct StableMatchingRanks {
    std::vector<std::size_t> proposer_ranks{}; ///< Assigned reviewer rank for each proposer,1..n.
    std::vector<std::size_t> reviewer_ranks{}; ///< Assigned proposer rank for each reviewer,1..n.
    std::size_t proposer_first_choices{}; ///< Number of proposers receiving rank1.
    std::size_t reviewer_first_choices{}; ///< Number of reviewers receiving rank1.
    bool has_agents{}; ///< Whether means describe a nonempty population.
    double proposer_mean{}; ///< Mean proposer rank;0 placeholder when empty.
    double reviewer_mean{}; ///< Mean reviewer rank;0 placeholder when empty.
};
/// @brief Owned stable perfect candidates from bounded exhaustive search.
/// Candidates are tested in lexicographic proposer-map order. Their proposal metadata is0 because enumeration is not a proposal execution. For n0 there is one empty permutation and one empty stable matching. Shared blocking checks are not an independent correctness oracle.
/// @par Example
/// @code
/// dm::algorithms::StableMatchingEnumeration e;
/// std::cout << e.candidates_tested; // 0
/// @endcode
/// @see gale_shapley(), gale_shapley_trace()
struct StableMatchingEnumeration {
    std::size_t candidates_tested{}; ///< Actual perfect permutations tested, equal to n!.
    std::size_t pairs_examined{}; ///< Actual pair tests, stopping at first block for rejected candidates.
    std::vector<StableMatchingResult> matchings{}; ///< All stable perfect candidates in lexicographic order.
};
/// @brief Match equal strict complete sides by preserved FIFO deferred acceptance.
/// Validate all side/row shapes, ranges and permutations before matching, even for empty inputs. Validation may allocate O(n) duplicate-detection scratch. FIFO starts0..n-1; rejections/displaced incumbents append at the tail. Each proposer advances monotonically and reviewers retain their best proposal so far. Returns the proposer-optimal/reviewer-pessimal stable perfect matching among stable matchings of these submitted strict complete preferences. Empty sides return empty maps and0 proposals. No input mutation or shared state. Including validation/rank preparation, ordinary time/storage are O(n²), even when only n proposals occur. Supports no capacities, incomplete lists, ties or couples.
/// @param proposer_preferences n strict complete favorite-first proposer rows.
/// @param reviewer_preferences n strict complete favorite-first reviewer rows.
/// @return Owned mutually inverse perfect maps and actual proposal count.
/// @throws std::invalid_argument Unequal side sizes, wrong row lengths, out-of-range IDs or repeated IDs.
/// @throws std::length_error Count or container bound overflow.
/// @throws std::bad_alloc Allocation of owned or working storage fails.
/// @par Example
/// @code
/// const auto m=dm::algorithms::gale_shapley({{0,1},{0,1}},{{1,0},{0,1}});
/// std::cout << m.proposals; // 3
/// @endcode
/// @par References
/// - D. Gale and L.S. Shapley (1962), College Admissions and the Stability of Marriage, American Mathematical Monthly69(1),9-15. doi:10.1080/00029890.1962.11989827. Theorems1-2; this implementation uses sequential FIFO proposals, not simultaneous stages.
/// @see gale_shapley(), is_stable(), gale_shapley_trace()
StableMatchingResult gale_shapley(const StablePreferences& proposer_preferences, const StablePreferences& reviewer_preferences);
/// @brief Match and count actual rank writes and proposal decisions.
/// Validate all side/row shapes, ranges and permutations before matching, even for empty inputs. Validation may allocate O(n) duplicate-detection scratch. FIFO starts0..n-1; rejections/displaced incumbents append at the tail. Each proposer advances monotonically and reviewers retain their best proposal so far. Returns the proposer-optimal/reviewer-pessimal stable perfect matching among stable matchings of these submitted strict complete preferences. Empty sides return empty maps and0 proposals. No input mutation or shared state. Including validation/rank preparation, ordinary time/storage are O(n²), even when only n proposals occur. Supports no capacities, incomplete lists, ties or couples. No replay or history retained.
/// @param proposer_preferences n strict complete favorite-first proposer rows.
/// @param reviewer_preferences n strict complete favorite-first reviewer rows.
/// @return Owned matching and actual kernel work.
/// @throws std::invalid_argument Unequal side sizes, wrong row lengths, out-of-range IDs or repeated IDs.
/// @throws std::length_error Count or container bound overflow.
/// @throws std::bad_alloc Allocation of owned or working storage fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::gale_shapley_work({{0,1},{0,1}},{{1,0},{0,1}});
/// std::cout << r.work.replacements; // 1
/// @endcode
/// @par References
/// - D. Gale and L.S. Shapley (1962), College Admissions and the Stability of Marriage, American Mathematical Monthly69(1),9-15. doi:10.1080/00029890.1962.11989827. Theorems1-2; this implementation uses sequential FIFO proposals, not simultaneous stages.
/// @see gale_shapley(), is_stable(), gale_shapley_trace()
GaleShapleyRun gale_shapley_work(const StablePreferences& proposer_preferences, const StablePreferences& reviewer_preferences);
/// @brief Bound individual sequential proposals for complete equal sides.
/// For n>0 return n*(n-1)+1: the last newly engaged reviewer receives one proposal, all others at most n. Empty input gives0. This counts individual proposals, not simultaneous stages in the paper; no allocation.
/// @param n Side size, including0.
/// @return Checked upper bound.
/// @throws std::length_error Count or container bound overflow.
/// @par Example
/// @code
/// std::cout << dm::algorithms::gale_shapley_proposal_bound(5); // 21
/// @endcode
/// @par References
/// - D. Gale and L.S. Shapley (1962), College Admissions and the Stability of Marriage, American Mathematical Monthly69(1),9-15. doi:10.1080/00029890.1962.11989827. Theorems1-2; this implementation uses sequential FIFO proposals, not simultaneous stages.
/// @see gale_shapley(), is_stable(), gale_shapley_trace()
std::size_t gale_shapley_proposal_bound(std::size_t n);
/// @brief Forecast conservative whole FIFO trace entries from a virtual side size.
/// Let Q=gale_shapley_proposal_bound(n). Input preferences and retained reviewer ranks cost3n+3n² rows/cells. Each event costs one metadata record plus next-choice and two partner maps and up to n entries in each queue:1+5n. Return3n+3n²+Q*(1+5n), not bytes or peak RAM. Excludes ordinary final maps/work/root metadata/working state/allocator bytes. Bounds all requested history before traversal, not actual observed proposals; checks count/container bounds without allocation.
/// @param n Virtual side size, including0.
/// @return Checked conservative retained-entry bound.
/// @throws std::length_error Count or container bound overflow.
/// @par Example
/// @code
/// std::cout << dm::algorithms::gale_shapley_trace_entries(2); // 51
/// @endcode
/// @par References
/// - D. Gale and L.S. Shapley (1962), College Admissions and the Stability of Marriage, American Mathematical Monthly69(1),9-15. doi:10.1080/00029890.1962.11989827. Theorems1-2; this implementation uses sequential FIFO proposals, not simultaneous stages.
/// @see gale_shapley(), is_stable(), gale_shapley_trace()
std::size_t gale_shapley_trace_entries(std::size_t n);
/// @brief Capture actual FIFO proposal decisions after whole-history admission.
/// Validate all side/row shapes, ranges and permutations before matching, even for empty inputs. Validation may allocate O(n) duplicate-detection scratch. FIFO starts0..n-1; rejections/displaced incumbents append at the tail. Each proposer advances monotonically and reviewers retain their best proposal so far. Returns the proposer-optimal/reviewer-pessimal stable perfect matching among stable matchings of these submitted strict complete preferences. Empty sides return empty maps and0 proposals. No input mutation or shared state. Including validation/rank preparation, ordinary time/storage are O(n²), even when only n proposals occur. Supports no capacities, incomplete lists, ties or couples. Full preference validation, including O(n) scratch, precedes quota. Admission via gale_shapley_trace_entries() precedes quadratic rank tables, matching state and input/history copies. Zero disables only optional quota; checked bounds remain. Captures one actual post-decision event per proposal; no replay, partial output or truncation. Full snapshots add O(n*Q) time/storage in the worst case.
/// @param proposer_preferences n strict complete favorite-first proposer rows.
/// @param reviewer_preferences n strict complete favorite-first reviewer rows.
/// @param max_entries Conservative whole retained-entry cap;0 disables only optional quota. Default 1000000.
/// @return Owned raw preferences/rank table/events, matching and actual work.
/// @throws std::invalid_argument Unequal side sizes, wrong row lengths, out-of-range IDs or repeated IDs.
/// @throws std::length_error Count or container bound overflow. Optional whole-history quota exceeded.
/// @throws std::bad_alloc Allocation of owned or working storage fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::gale_shapley_trace({{0,1},{0,1}},{{1,0},{0,1}},51);
/// std::cout << t.events.size(); // 3
/// @endcode
/// @par References
/// - D. Gale and L.S. Shapley (1962), College Admissions and the Stability of Marriage, American Mathematical Monthly69(1),9-15. doi:10.1080/00029890.1962.11989827. Theorems1-2; this implementation uses sequential FIFO proposals, not simultaneous stages.
/// @see gale_shapley(), is_stable(), gale_shapley_trace()
GaleShapleyTrace gale_shapley_trace(const StablePreferences& proposer_preferences, const StablePreferences& reviewer_preferences, std::size_t max_entries = 1000000);
/// @brief Check feasibility and absence of blocking pairs without unsafe indexing.
/// Validate all preferences first. Malformed map dimensions/ranges/duplicates/noninverse assignments return false; proposals metadata is ignored. Consistent partial assignments may use kUnmatched, treated as worse than every acceptable partner. Under complete equal-side preferences any nonperfect feasible matching is unstable. Empty valid matching is stable. Builds ranks only after feasibility checks and stops at the first actual blocking pair, O(n²) total time/storage. No mutation.
/// @param matching Candidate in proposer orientation; proposal metadata ignored.
/// @param proposer_preferences n strict complete favorite-first proposer rows.
/// @param reviewer_preferences n strict complete favorite-first reviewer rows.
/// @return True iff maps are feasible and no mutually preferred pair exists.
/// @throws std::invalid_argument Unequal side sizes, wrong row lengths, out-of-range IDs or repeated IDs.
/// @throws std::length_error Count or container bound overflow.
/// @throws std::bad_alloc Allocation of owned or working storage fails.
/// @par Example
/// @code
/// dm::algorithms::StableMatchingResult m{{0,0},{0,0},0};
/// std::cout << dm::algorithms::is_stable(m,{{0,1},{0,1}},{{0,1},{0,1}}); // 0
/// @endcode
/// @par References
/// - D. Gale and L.S. Shapley (1962), College Admissions and the Stability of Marriage, American Mathematical Monthly69(1),9-15. doi:10.1080/00029890.1962.11989827. Theorems1-2; this implementation uses sequential FIFO proposals, not simultaneous stages.
/// @see gale_shapley(), is_stable(), gale_shapley_trace()
[[nodiscard]] bool is_stable(const StableMatchingResult& matching, const StablePreferences& proposer_preferences, const StablePreferences& reviewer_preferences);
/// @brief Inspect feasibility and collect actual blocking-pair witnesses.
/// Validate full preferences, then candidate feasibility. Malformed candidates return all flags false with no pairs or rank-table construction. For a feasible candidate admit the conservative n² pair bound before rank tables and scanning, even if few/no blocks would occur. Default pair cap1m;0 disables only optional cap, not count/container checks. Scan all n² pairs in proposer-major order, retaining genuine mutually preferred pairs; no replay or partial output.
/// @param matching Candidate in proposer orientation; proposal metadata ignored.
/// @param proposer_preferences n strict complete favorite-first proposer rows.
/// @param reviewer_preferences n strict complete favorite-first reviewer rows.
/// @param max_pairs Whole worst-case witness count cap for feasible candidates;0 disables optional cap. Default 1000000.
/// @return Structural flags and all blocking pairs; no witnesses does not certify an infeasible candidate.
/// @throws std::invalid_argument Unequal side sizes, wrong row lengths, out-of-range IDs or repeated IDs.
/// @throws std::length_error Count or container bound overflow. Optional witness bound exceeded.
/// @throws std::bad_alloc Allocation of owned or working storage fails.
/// @par Example
/// @code
/// dm::algorithms::StableMatchingResult m{{1,0},{1,0},0};
/// const auto d=dm::algorithms::stable_matching_diagnostics(m,{{0,1},{0,1}},{{0,1},{0,1}});
/// std::cout << d.blocking_pairs.size(); // 1
/// @endcode
/// @par References
/// - D. Gale and L.S. Shapley (1962), College Admissions and the Stability of Marriage, American Mathematical Monthly69(1),9-15. doi:10.1080/00029890.1962.11989827. Theorems1-2; this implementation uses sequential FIFO proposals, not simultaneous stages.
/// @see gale_shapley(), is_stable(), gale_shapley_trace()
StabilityDiagnostics stable_matching_diagnostics(const StableMatchingResult& matching, const StablePreferences& proposer_preferences, const StablePreferences& reviewer_preferences, std::size_t max_pairs = 1000000);
/// @brief Exchange proposer/reviewer orientation with structural validation.
/// Require equal-length mutually inverse partial maps, allowing kUnmatched on both sides. Swap owned copies, preserving proposal metadata as the original execution count; this does not reverse a proposal history or certify stability. O(n) validation/copy, no mutation.
/// @param matching Candidate in proposer orientation; proposal metadata ignored.
/// @return Owned same pairing in opposite orientation.
/// @throws std::invalid_argument Malformed or inconsistent partial partner maps.
/// @throws std::bad_alloc Allocation of owned or working storage fails.
/// @par Example
/// @code
/// const auto m=dm::algorithms::reverse_stable_matching({{1,2,0},{2,0,1},4});
/// std::cout << m.proposer_match[0]; // 2
/// @endcode
/// @par References
/// - D. Gale and L.S. Shapley (1962), College Admissions and the Stability of Marriage, American Mathematical Monthly69(1),9-15. doi:10.1080/00029890.1962.11989827. Theorems1-2; this implementation uses sequential FIFO proposals, not simultaneous stages.
/// @see gale_shapley(), is_stable(), gale_shapley_trace()
StableMatchingResult reverse_stable_matching(const StableMatchingResult& matching);
/// @brief Summarize assigned-partner ranks for a structurally perfect candidate.
/// Validate complete preferences then mutual perfect maps. Build both inverse-rank tables once; return one-based ranks, first-choice counts and ordinal mean ranks. This does not certify stability. Empty valid inputs give has_agents=false and mean0 placeholders. O(n²) time/storage, no normalization or mutation.
/// @param matching Candidate in proposer orientation; proposal metadata ignored.
/// @param proposer_preferences n strict complete favorite-first proposer rows.
/// @param reviewer_preferences n strict complete favorite-first reviewer rows.
/// @return Owned per-agent ranks and descriptive summaries.
/// @throws std::invalid_argument Unequal side sizes, wrong row lengths, out-of-range IDs or repeated IDs. Matching is malformed or incomplete.
/// @throws std::length_error Count or container bound overflow.
/// @throws std::bad_alloc Allocation of owned or working storage fails.
/// @par Example
/// @code
/// const auto m=dm::algorithms::gale_shapley({{0}},{{0}});
/// std::cout << dm::algorithms::stable_matching_ranks(m,{{0}},{{0}}).proposer_mean; // 1
/// @endcode
/// @par References
/// - D. Gale and L.S. Shapley (1962), College Admissions and the Stability of Marriage, American Mathematical Monthly69(1),9-15. doi:10.1080/00029890.1962.11989827. Theorems1-2; this implementation uses sequential FIFO proposals, not simultaneous stages.
/// @see gale_shapley(), is_stable(), gale_shapley_trace()
StableMatchingRanks stable_matching_ranks(const StableMatchingResult& matching, const StablePreferences& proposer_preferences, const StablePreferences& reviewer_preferences);
/// @brief Count all perfect candidate permutations by checked factorial.
/// Return n!, including1 for n0. Count overflow throws promptly without allocation, even for SIZE_MAX input.
/// @param n Side size.
/// @return Representable factorial.
/// @throws std::length_error Count or container bound overflow.
/// @par Example
/// @code
/// std::cout << dm::algorithms::stable_matching_permutations(5); // 120
/// @endcode
/// @par References
/// - D. Gale and L.S. Shapley (1962), College Admissions and the Stability of Marriage, American Mathematical Monthly69(1),9-15. doi:10.1080/00029890.1962.11989827. Theorems1-2; this implementation uses sequential FIFO proposals, not simultaneous stages.
/// @see gale_shapley(), is_stable(), gale_shapley_trace()
std::size_t stable_matching_permutations(std::size_t n);
/// @brief Enumerate all stable perfect candidates under a whole-search bound.
/// Validate all preferences first, then check n!, pair-count and worst-case result payload/container bounds before search. max_permutations caps ALL tested candidates, not stable output count;0 disables only the optional cap. Build ranks once and use the shared blocking checker, stopping at the first block for each rejected candidate. Output maps are lexicographic and proposals metadata is0. n0 yields one empty stable matching. Time O(n!*n²), output can be factorial; no partial result or independent-oracle claim.
/// @param proposer_preferences n strict complete favorite-first proposer rows.
/// @param reviewer_preferences n strict complete favorite-first reviewer rows.
/// @param max_permutations Whole candidate-count quota;0 disables only optional cap. Default 1000000.
/// @return All stable perfect candidates and actual candidate/pair counts.
/// @throws std::invalid_argument Unequal side sizes, wrong row lengths, out-of-range IDs or repeated IDs.
/// @throws std::length_error Count or container bound overflow. Optional candidate quota exceeded.
/// @throws std::bad_alloc Allocation of owned or working storage fails.
/// @par Example
/// @code
/// const auto e=dm::algorithms::enumerate_stable_matchings({{0,1},{1,0}},{{1,0},{0,1}});
/// std::cout << e.matchings.size(); // 2
/// @endcode
/// @par References
/// - D. Gale and L.S. Shapley (1962), College Admissions and the Stability of Marriage, American Mathematical Monthly69(1),9-15. doi:10.1080/00029890.1962.11989827. Theorems1-2; this implementation uses sequential FIFO proposals, not simultaneous stages.
/// @see gale_shapley(), is_stable(), gale_shapley_trace()
StableMatchingEnumeration enumerate_stable_matchings(const StablePreferences& proposer_preferences, const StablePreferences& reviewer_preferences, std::size_t max_permutations = 1000000);
/// @brief Construct the original five-resident/five-reviewer synthetic fixture.
/// Use mt19937_64 seed20240927, shuffle iota rows for all proposers then all reviewers. Fixed-size owned data; standard-library shuffle mapping limits bit portability. This is not actual residency data.
/// @return Owned original two-side rankings.
/// @throws std::bad_alloc Allocation of owned or working storage fails.
/// @par Example
/// @code
/// std::cout << dm::algorithms::stable_matching_example_preferences().proposer_preferences.size(); // 5
/// @endcode
/// @par References
/// - D. Gale and L.S. Shapley (1962), College Admissions and the Stability of Marriage, American Mathematical Monthly69(1),9-15. doi:10.1080/00029890.1962.11989827. Theorems1-2; this implementation uses sequential FIFO proposals, not simultaneous stages.
/// @see gale_shapley(), is_stable(), gale_shapley_trace()
StableMatchingProfile stable_matching_example_preferences();
/// @brief Generate a deterministic complete strict preference family.
/// Each agent i ranks i first then other IDs ascending, on both sides. FIFO makes n proposals. Check2n+2n² row/cell entries and container bounds before allocation; default1m/0 disables only optional quota. Empty input produces two empty sides. O(n²) owned storage/work, no randomness or shared state.
/// @param n Side size, including0.
/// @param max_entries Whole generated row/cell entry cap;0 disables optional cap. Default 1000000.
/// @return Owned two-side rankings.
/// @throws std::length_error Count or container bound overflow. Optional generation cap exceeded.
/// @throws std::bad_alloc Allocation of owned or working storage fails.
/// @par Example
/// @code
/// const auto p=dm::algorithms::stable_matching_mutual_first_preferences(3);
/// std::cout << dm::algorithms::gale_shapley(p.proposer_preferences,p.reviewer_preferences).proposals; // 3
/// @endcode
/// @par References
/// - D. Gale and L.S. Shapley (1962), College Admissions and the Stability of Marriage, American Mathematical Monthly69(1),9-15. doi:10.1080/00029890.1962.11989827. Theorems1-2; this implementation uses sequential FIFO proposals, not simultaneous stages.
/// @see gale_shapley(), is_stable(), gale_shapley_trace()
StableMatchingProfile stable_matching_mutual_first_preferences(std::size_t n, std::size_t max_entries = 1000000);
/// @brief Generate a deterministic complete strict preference family.
/// Every row on both sides is0..n-1. FIFO makes n*(n+1)/2 proposals. Check2n+2n² row/cell entries and container bounds before allocation; default1m/0 disables only optional quota. Empty input produces two empty sides. O(n²) owned storage/work, no randomness or shared state.
/// @param n Side size, including0.
/// @param max_entries Whole generated row/cell entry cap;0 disables optional cap. Default 1000000.
/// @return Owned two-side rankings.
/// @throws std::length_error Count or container bound overflow. Optional generation cap exceeded.
/// @throws std::bad_alloc Allocation of owned or working storage fails.
/// @par Example
/// @code
/// const auto p=dm::algorithms::stable_matching_common_preferences(3);
/// std::cout << dm::algorithms::gale_shapley(p.proposer_preferences,p.reviewer_preferences).proposals; // 6
/// @endcode
/// @par References
/// - D. Gale and L.S. Shapley (1962), College Admissions and the Stability of Marriage, American Mathematical Monthly69(1),9-15. doi:10.1080/00029890.1962.11989827. Theorems1-2; this implementation uses sequential FIFO proposals, not simultaneous stages.
/// @see gale_shapley(), is_stable(), gale_shapley_trace()
StableMatchingProfile stable_matching_common_preferences(std::size_t n, std::size_t max_entries = 1000000);
} // namespace datamunge::algorithms
