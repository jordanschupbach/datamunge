#pragma once

/// @file general_problem_solver.hpp
/// @brief Bounded forward fact planning, actual search events and independent plan certificates.
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// @brief One grounded positive-precondition operator on a 64-bit fact state.
/// Applicable when all require bits hold. Transition is (state & ~remove) | add;
/// overlapping add/remove bits end present. No negative preconditions, variables,
/// theorem prover or external action is executed. All 64 bits are valid, including
/// bit 63. Names may be empty, duplicated or arbitrary byte strings; identity is
/// position in the operator vector. Legacy field order and types are preserved.
/// @par Example
/// @code
/// dm::algorithms::PlanningAction a{"make", 0, 1, 0};
/// std::cout << a.name << ' ' << a.add; // make 1
/// @endcode
/// @par References
/// - Fikes and Nilsson (1971), STRIPS, https://ijcai.org/proceedings/1971/63 .
///   Ground bit masks adapt precondition/add/delete effects, not first-order search.
/// @see general_problem_solver(), PlanningRun
struct PlanningAction {
    std::string name; ///< Display label; not a unique operator identifier.
    std::uint64_t require{0}; ///< All required facts must be present before the action.
    std::uint64_t add{0}; ///< Facts present after deletion, including overlapping effects.
    std::uint64_t remove{0}; ///< Facts cleared before applying add.
};

/// @brief Owned legacy planning outcome with operator names in application order.
/// Success contains an executable plan no longer than the requested budget, not
/// necessarily shortest. Initially satisfied goals give an empty successful plan
/// and final_state=initial. Failure has empty actions/final_state0/found=false;
/// its final_state is a placeholder, not a reached failure witness. Names alone
/// can be ambiguous; PlanningRun also retains operator indices. Stable layout.
/// @par Example
/// @code
/// const auto p = dm::algorithms::general_problem_solver(0, 1, {{"make",0,1,0}});
/// std::cout << p.found << ' ' << p.final_state << ' ' << p.actions[0]; // 1 1 make
/// @endcode
/// @see PlanningRun
struct PlanningResult {
    std::vector<std::string> actions; ///< Owned labels on success, empty on failure.
    std::uint64_t final_state{0}; ///< Successful terminal state, otherwise zero placeholder.
    bool found{false}; ///< Whether the positive goal conjunction was achieved within budget.
};

/// @brief Actual node, operator and pruning work from one ordered forward DFS.
/// Visits include root, goal, depth-cutoff and repeated-state entries. Expanded
/// states pass all three terminal/pruning checks and can be expanded again after
/// a shallower arrival. Goal/depth checks precede the best-depth map. Counters
/// exclude copies, allocation, hashing internals and trace capture; not timings.
/// @par Example
/// @code
/// const auto r = dm::algorithms::general_problem_solver_run(0,1,{{"make",0,1,0}});
/// std::cout << r.work.visits << ' ' << r.work.operator_checks; // 2 1
/// @endcode
/// @see PlanningRun
struct PlanningWork {
    std::size_t visits{}; ///< Entered frames, including all kinds of terminal visits.
    std::size_t expanded{}; ///< Nonterminal new or strictly shallower state expansions.
    std::size_t operator_checks{}; ///< Actual positive-precondition tests in input order.
    std::size_t applicable{}; ///< Tests that cause a successor to be entered, including self-loops.
    std::size_t repeat_prunes{}; ///< Non-goal non-cutoff arrivals no shallower than a prior expansion.
    std::size_t reopenings{}; ///< Prior expanded states reached at strictly smaller depth.
    std::size_t depth_cutoffs{}; ///< Non-goal visits at exactly the maximum allowed depth.
    std::size_t backtracks{}; ///< Failed applicable children whose selected operator is popped.
    std::size_t distinct_states{}; ///< Distinct keys admitted to the expansion-depth map.
    std::size_t peak_stack{}; ///< Largest actual heap-frame stack, including goal/cutoff frames.
};

/// @brief Owned ordinary planning result, unambiguous operator indices and actual work.
/// action_indices and result.actions describe the same successful sequence;
/// both are empty on failure. No replay or history is performed by this result.
/// Indices refer to the caller's original operator order; no shared state or
/// caller mutation. DFS frame storage is on the heap, not the C++ call stack.
/// @par Example
/// @code
/// const auto r = dm::algorithms::general_problem_solver_run(0,1,{{"make",0,1,0}});
/// std::cout << r.action_indices[0] << ' ' << r.result.final_state; // 0 1
/// @endcode
/// @see general_problem_solver_run(), PlanningTrace
struct PlanningRun {
    PlanningResult result; ///< Owned legacy-compatible labels, final state and success flag.
    std::vector<std::size_t> action_indices; ///< Exact input operator positions on the successful path.
    PlanningWork work; ///< Work counted in this execution, not reconstructed afterward.
};

/// @brief Primitive event kinds recorded at the actual search control-flow points.
/// test precedes apply; visit precedes goal/cutoff/repeat/reopen. finish follows
/// successful recursive-return-equivalent propagation, while backtrack follows
/// a failed child. exhausted means all operators of that frame were tested.
/// @par Example
/// @code
/// const auto k = dm::algorithms::PlanningEventKind::reopen;
/// std::cout << dm::algorithms::planning_event_name(k); // reopen
/// @endcode
/// @see PlanningEvent
enum class PlanningEventKind {
    visit, ///< Enter an actual heap frame before terminal checks.
    goal, ///< Accept a state satisfying every goal bit, even at the depth limit.
    cutoff, ///< Reject a non-goal frame at the complete-step budget.
    repeat, ///< Prune a state previously expanded at equal or smaller depth.
    reopen, ///< Expand a prior state with strictly more remaining budget.
    test, ///< Test the named input operator's positive preconditions.
    apply, ///< Select an applicable operator and enter its actual successor.
    backtrack, ///< Pop an operator after its child failed.
    finish, ///< Propagate a successful child's return along the selected path.
    exhausted ///< Reject a frame after testing every input operator.
};

/// @brief One actual fixed-schema search event; state masks are constant-width values.
/// state/depth identify the current frame. apply has the actual successor in
/// next_state; other kinds use next_state=state. action_index is meaningful for
/// test/apply/backtrack/finish, otherwise size_t maximum. applicable is meaningful
/// only for test. repeat/reopen expose the previous expansion depth and set
/// has_previous_depth; other events leave it false. No vector/path is replayed.
/// @par Example
/// @code
/// const auto t = dm::algorithms::general_problem_solver_trace(0,1,{{"make",0,1,0}});
/// std::cout << t.events[2].next_state << ' ' << t.events[2].action_index; // 1 0
/// @endcode
/// @see PlanningTrace
struct PlanningEvent {
    PlanningEventKind kind{PlanningEventKind::visit}; ///< Actual control-flow event.
    std::uint64_t state{}; ///< Current finite fact-state mask.
    std::uint64_t next_state{}; ///< Actual apply successor, or current state otherwise.
    std::size_t depth{}; ///< Operators already selected from the initial state.
    std::size_t action_index{std::numeric_limits<std::size_t>::max()}; ///< Operator position or no-operator sentinel.
    std::size_t previous_depth{}; ///< Previous expanded depth when has_previous_depth is true.
    bool applicable{}; ///< Result of this test event; false placeholder for other events.
    bool has_previous_depth{}; ///< Whether previous_depth is an actual map lookup result.
};

/// @brief Owned inputs, ordinary result and ordered actual search events from one execution.
/// Every event is one fixed-schema record, not one numeric entry or byte.
/// The online event quota excludes owned input operators/names, root and work
/// metadata, final ordinary output, working stack/map and allocator storage;
/// it is not a peak RAM limit. No partial or truncated trace is returned when
/// a quota is exceeded. Inputs, labels and snapshots are independent owned copies.
/// @par Example
/// @code
/// const auto t = dm::algorithms::general_problem_solver_trace(0,1,{{"make",0,1,0}},32,6);
/// std::cout << t.events.size() << ' ' << t.operators[0].name; // 6 make
/// @endcode
/// @see general_problem_solver_trace()
struct PlanningTrace {
    std::uint64_t initial{}; ///< Owned initial fact mask.
    std::uint64_t goals{}; ///< Owned positive goal mask.
    std::vector<PlanningAction> operators; ///< Owned operator order, effects and labels; excluded from event quota.
    PlanningRun run; ///< Ordinary result from the same execution; excluded from event quota.
    std::vector<PlanningEvent> events; ///< Actual chronological fixed-schema records.
};

/// @brief Owned finite certificate from independently executing operator indices.
/// Validates the entire index vector before any transition. If a precondition
/// fails, stops before that action: applicable=false, applied is its zero-based
/// failing position, states retains the valid prefix including initial. A goal
/// miss after a fully applicable sequence is reported by reaches_goals=false.
/// Empty sequences are applicable; goals may already hold. O(p) transition work
/// and O(p) owned states for p indices; no search, shortest-path or budget claim.
/// @par Example
/// @code
/// const auto c = dm::algorithms::check_planning_plan(0,1,{{"make",0,1,0}},{0});
/// std::cout << c.applicable << ' ' << c.reaches_goals << ' ' << c.states.back(); // 1 1 1
/// @endcode
/// @see check_planning_plan()
struct PlanningCertificate {
    bool applicable{true}; ///< Whether every selected operator's preconditions were met.
    bool reaches_goals{}; ///< Whether the actual terminal prefix state satisfies all goal bits.
    std::size_t applied{}; ///< Count of successful applications, or failing position when not applicable.
    std::uint64_t final_state{}; ///< Last actually reached prefix state, including on failure.
    std::vector<std::uint64_t> states; ///< Initial state plus every successfully reached successor.
};

/// @brief Return the stable lower-case label for a primitive planning event.
/// No global state or input mutation. O(1) fixed-length output.
/// @param kind A declared PlanningEventKind value.
/// @return Owned label visit/goal/cutoff/repeat/reopen/test/apply/backtrack/finish/exhausted.
/// @throws std::invalid_argument The enum value is not a declared kind.
/// @par Example
/// @code
/// std::cout << dm::algorithms::planning_event_name(dm::algorithms::PlanningEventKind::goal); // goal
/// @endcode
/// @see PlanningEvent
std::string planning_event_name(PlanningEventKind kind);

/// @brief Find a bounded positive-fact plan using ordered DFS with shallower-state reopening.
/// Preserves legacy signature/default 32/layouts, operator order, add-wins effects,
/// initial-goal success and failure placeholders. Corrects old depth-insensitive
/// visited-state false negatives. Goal testing precedes the depth limit; states
/// are pruned only if expanded at equal/smaller depth. Returns the first DFS plan,
/// not necessarily shortest; it is complete within max_steps for this finite
/// grounded model if execution returns normally. Heap frames remove recursive
/// call-stack depth limits. Names may be empty or duplicated; all 64 bits valid.
/// No external action, negative preconditions, theorem prover or GPS means-ends
/// heuristic. Expected O(m*S*min(K+1,S)) checks for m operators, S reachable states
/// and budget K, O(S+min(K,S)) working records plus output/name bytes; hashing is
/// expected constant time. No allocation for an entire theoretical 2^64 space.
/// @param initial Finite 64-bit set of initially true facts.
/// @param goals Positive conjunction of required terminal facts; zero succeeds initially.
/// @param actions Grounded operators in deterministic input order; empty is allowed.
/// @param max_steps Maximum operator applications in a plan, default 32; zero only tests initial.
/// @return Owned labels/final state on success; empty labels and final_state0 on failure.
/// @throws std::length_error Actual work counters cannot fit size_t.
/// @par Example
/// @code
/// const auto p = dm::algorithms::general_problem_solver(0,1,{{"make",0,1,0}});
/// std::cout << p.found << ' ' << p.actions.size(); // 1 1
/// @endcode
/// @par References
/// - Fikes and Nilsson (1971), https://ijcai.org/proceedings/1971/63 .
///   Uses grounded precondition/add/delete effects; search here is forward DFS,
///   not their first-order theorem prover and means-ends strategy.
/// @see general_problem_solver_run(), check_planning_plan()
PlanningResult general_problem_solver(std::uint64_t initial, std::uint64_t goals,
    const std::vector<PlanningAction>& actions, std::size_t max_steps=32);

/// @brief Execute the same bounded planner while returning actual work and operator indices.
/// Same search, effects, order, correctness and complexity as general_problem_solver.
/// No trace capture or hidden plan replay. Safe with arbitrary operator labels;
/// inputs are never changed and no global state is used.
/// @param initial Initially true fact mask.
/// @param goals Positive required terminal mask; zero succeeds initially.
/// @param actions Operators in input order; any 64-bit effects/labels are allowed.
/// @param max_steps Complete plan-step budget, default 32.
/// @return Owned result, exact successful operator indices and actual search counters.
/// @throws std::length_error Actual work counters cannot fit size_t.
/// @par Example
/// @code
/// const auto r = dm::algorithms::general_problem_solver_run(0,1,{{"make",0,1,0}});
/// std::cout << r.work.expanded << ' ' << r.work.peak_stack; // 1 2
/// @endcode
/// @see general_problem_solver(), PlanningRun
PlanningRun general_problem_solver_run(std::uint64_t initial, std::uint64_t goals,
    const std::vector<PlanningAction>& actions, std::size_t max_steps=32);

/// @brief Capture primitive events directly from the same executing forward planner.
/// Online capture checks each next actual record before retaining it; it does
/// not reserve an exponential worst-case tree or discount/truncate future events.
/// The ordinary search/arithmetic is unchanged. Root/input/final/working storage
/// exclusions are specified in PlanningTrace; the quota is not bytes or peakRAM.
/// Initially successful and zero-budget searches still produce actual events.
/// O(E) extra event storage for E captured records, in addition to ordinary search
/// state and owned input/name bytes. No partial return, replay or caller mutation.
/// @param initial Initially true facts.
/// @param goals Positive required terminal facts.
/// @param actions Operators copied with original ordering/effects/arbitrary labels.
/// @param max_steps Complete plan-step budget, default 32.
/// @param max_events Positive actual fixed-schema event-record cap, default 1000000; zero disables quota.
/// @return Owned input operators, chronological events and ordinary run from one execution.
/// @throws std::length_error Actual next event exceeds quota/vector capacity or work counters overflow.
/// @par Example
/// @code
/// const auto t = dm::algorithms::general_problem_solver_trace(0,1,{{"make",0,1,0}},32,6);
/// std::cout << t.events.size() << ' ' << t.run.result.found; // 6 1
/// @endcode
/// @see general_problem_solver_run(), PlanningTrace
PlanningTrace general_problem_solver_trace(std::uint64_t initial, std::uint64_t goals,
    const std::vector<PlanningAction>& actions, std::size_t max_steps=32,
    std::size_t max_events=1000000);

/// @brief Independently execute a selected operator-index sequence without searching.
/// Full index validation and complete p+1 state capacity/quota checks precede
/// transitions/allocation, even if an early precondition would fail. Positive
/// preconditions/add-wins semantics match the planner, but this is an independent
/// checker. Captures every successful prefix; no caller mutation or shared state.
/// This does not validate a claimed found flag or prove shortestness/budget;
/// inspect applicable/reaches_goals/final_state/length for those comparisons.
/// @param initial Initial state mask.
/// @param goals Positive required terminal mask.
/// @param actions Operators in the original order; names need not be unique.
/// @param action_indices Complete proposed sequence of operator positions.
/// @param max_states Positive whole prefix-state cap including initial, default 1000000; zero disables quota.
/// @return Owned applicability/goal diagnostics and actual valid prefix states.
/// @throws std::out_of_range Any proposed operator index is invalid, checked before quota/transitions.
/// @throws std::length_error Complete p+1 states exceed capacity/quota or size_t.
/// @par Example
/// @code
/// const auto c = dm::algorithms::check_planning_plan(0,2,{{"key",0,1,0},{"open",1,2,1}},{0,1},3);
/// std::cout << c.states.size() << ' ' << c.final_state << ' ' << c.reaches_goals; // 3 2 1
/// @endcode
/// @see PlanningCertificate, general_problem_solver_run()
PlanningCertificate check_planning_plan(std::uint64_t initial, std::uint64_t goals,
    const std::vector<PlanningAction>& actions, const std::vector<std::size_t>& action_indices,
    std::size_t max_states=1000000);

/// @brief Construct low-bit-first carry operators for a finite binary-counter planning family.
/// Operator k requires every lower bit, clears those lower bits and sets bit k.
/// No negative preconditions: operations with bit k already set may move backward;
/// best-depth pruning handles those cycles. From zero to bit(bits-1), this input
/// ordering has a successful DFS plan of 2^(bits-1) applications when admitted by
/// the step budget; no shortestness claim is made for arbitrary domains.
/// This family allocates only bits operators; executing it can be exponential.
/// @param bits Number of used fact bits, from 1 through 64; lower operator positions come first.
/// @return Owned operators named carry-0 through carry-(bits-1), with correct high-bit masks.
/// @throws std::invalid_argument bits is outside 1..64.
/// @par Example
/// @code
/// const auto a = dm::algorithms::planning_binary_counter(3);
/// const auto r = dm::algorithms::general_problem_solver_run(0,4,a,4);
/// std::cout << a.size() << ' ' << r.action_indices.size(); // 3 4
/// @endcode
/// @see general_problem_solver_run()
std::vector<PlanningAction> planning_binary_counter(std::size_t bits);

} // namespace datamunge::algorithms
