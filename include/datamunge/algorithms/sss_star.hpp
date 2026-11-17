#pragma once
/// @file sss_star.hpp
/// @brief Checked actual MTD upper-bound search with transposition memory and explicit preprocessing work.
#include <cstddef>
#include <cstdint>
#include <datamunge/algorithms/game_search.hpp>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own the SSSOptions contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::SSSOptions r{};
/// (void)r;
/// @endcode
/// @see sss_star_run()
struct SSSOptions {
  std::size_t max_nodes{
      1000000}; ///< Complete reachable input-node validation bound; zero disables policy.
  std::size_t max_depth{
      256}; ///< Complete input depth bound; zero disables policy, intrinsic1024 remains.
  std::size_t max_probes{
      100000}; ///< Maximum actual root probes; zero returns unresolved initial bound.
  std::size_t max_search_visits{
      20000000}; ///< Actual recursive search entries including table hits; zero disables policy.
  std::size_t max_records{
      1000000}; ///< Actual captured bound updates plus root probes; zero disables policy.
};
/// @brief Own the SSSProbe contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::SSSProbe r{};
/// (void)r;
/// @endcode
/// @see sss_star_run()
struct SSSProbe {
  std::size_t index{};           ///< Actual zero-based root probe.
  double      alpha{};           ///< Actual initial lower search window endpoint.
  double      beta{};            ///< Actual initial upper search window endpoint.
  double      value{};           ///< Actual fail-soft returned bound/value.
  double      lower{};           ///< Actual driver lower bound after this probe.
  double      upper{};           ///< Actual driver upper bound after this probe.
  long long search_leaf_reads{}; ///< Actual cumulative search leaf reads, excluding preprocessing.
  long long search_visits{};     ///< Actual cumulative recursive entries including cached cutoffs.
  std::size_t table_entries{};   ///< Actual transposition table size after this probe.
};
/// @brief Own the SSSBoundUpdate contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::SSSBoundUpdate r{};
/// (void)r;
/// @endcode
/// @see sss_star_run()
struct SSSBoundUpdate {
  std::size_t node{};  ///< Actual preflight preorder identity, no borrowed pointer retained.
  double      lower{}; ///< Actual merged transposition lower bound.
  double      upper{}; ///< Actual merged transposition upper bound.
};
/// @brief Own the SSSRun contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::SSSRun r{};
/// (void)r;
/// @endcode
/// @see sss_star_run()
struct SSSRun {
  GameResult
      result; ///< Actual current value/search counters; value is minimax only when converged. best_child remains-1.
  bool   converged{}; ///< Actual driver lower/upper interval closed.
  double lower{};     ///< Actual final lower bound, possibly negative infinity before evidence.
  double upper{};     ///< Actual final upper bound, possibly positive infinity before evidence.
  std::size_t
      preflight_nodes{}; ///< Actual reachable nodes visited to validate and obtain a finite initial bound.
  std::size_t
      preflight_leaf_reads{}; ///< Actual leaf values read by validation/bound initialization; not search evaluations.
  std::size_t table_hits{}; ///< Actual recursive entries finding prior transposition bounds.
  std::size_t probes{};     ///< Actual completed root probes.
  std::vector<SSSProbe>       history; ///< Optional actual root windows and bound updates.
  std::vector<SSSBoundUpdate> bounds;  ///< Optional actual transposition writes in execution order.
};
/// @brief Run checked fail-soft MTD upper-bound game search.
/// Immutable finite materialized tree: leaves have no children and finite values; internal nodes have nonempty children. Resolution finite positive and no greater than the smallest distinct leaf-value gap; duplicates allowed. Preflight reads every reachable leaf and reports this work separately from actual search leaf reads. The standard SSS*/MTD dominance comparison concerns a particular ordered-tree search model, not an unconditional advantage for total validation+search work. Root windows descend from a finite upper guess, reuse merged transposition bounds, and explicitly report probe exhaustion. Bounds are reacquired by node key after recursive insertions, avoiding invalidated unordered-map iterators. Nonrepresentable adjacent arithmetic uses nextafter endpoints; internal infinities denote bounds only. Depth capped intrinsically1024 before search recursion. Time depends on actual repeated probes/visits; preprocessing O(nodes+leaves log leaves), memory O(nodes), optional histories O(actual writes/probes). Borrowed tree is not retained in outputs; independent calls own all memory.
/// @param root Complete immutable finite valued game tree.
/// @param maximizing Whether root chooses maximum rather than minimum.
/// @param resolution Finite positive null-window width, default1; distinct leaves separated by at least this width.
/// @param options Complete input and actual probe/visit/capture policies.
/// @return Owned current result, converged flag, actual work, bounds and optional histories.
/// @throws std::invalid_argument Malformed tree, nonfinite values or invalid/incompatible resolution.
/// @throws std::length_error Input depth/size, actual search/record or counter capacity exceeded.
/// @throws std::logic_error Inconsistent merged search bounds.
/// @par Example
/// @code
/// dm::algorithms::GameNode tree{false,0,{{true,2,{}},{true,5,{}}}};
/// auto r=dm::algorithms::sss_star_run(tree,true);
/// std::print("{} {}\n",r.result.value,r.converged); // 5 true
/// @endcode
/// @par References
/// - G. C. Stockman, A minimax algorithm better than alpha-beta?, Artificial Intelligence12(2),1979; A. Plaat et al., Best-first fixed-depth minimax algorithms, Artificial Intelligence87,1996. This materialized-tree implementation includes an explicit all-leaf bound/validation scan.
/// @see sss_star()
SSSRun sss_star_run(
    const GameNode&   root,
    bool              maximizing,
    double            resolution = 1,
    const SSSOptions& options    = {});
/// @brief Run checked fail-soft MTD upper-bound game search with actual memory/probe states.
/// Immutable finite materialized tree: leaves have no children and finite values; internal nodes have nonempty children. Resolution finite positive and no greater than the smallest distinct leaf-value gap; duplicates allowed. Preflight reads every reachable leaf and reports this work separately from actual search leaf reads. The standard SSS*/MTD dominance comparison concerns a particular ordered-tree search model, not an unconditional advantage for total validation+search work. Root windows descend from a finite upper guess, reuse merged transposition bounds, and explicitly report probe exhaustion. Bounds are reacquired by node key after recursive insertions, avoiding invalidated unordered-map iterators. Nonrepresentable adjacent arithmetic uses nextafter endpoints; internal infinities denote bounds only. Depth capped intrinsically1024 before search recursion. Time depends on actual repeated probes/visits; preprocessing O(nodes+leaves log leaves), memory O(nodes), optional histories O(actual writes/probes). Borrowed tree is not retained in outputs; independent calls own all memory.
/// @param root Complete immutable finite valued game tree.
/// @param maximizing Whether root chooses maximum rather than minimum.
/// @param resolution Finite positive null-window width, default1; distinct leaves separated by at least this width.
/// @param options Complete input and actual probe/visit/capture policies.
/// @return Owned current result, converged flag, actual work, bounds and optional histories.
/// @throws std::invalid_argument Malformed tree, nonfinite values or invalid/incompatible resolution.
/// @throws std::length_error Input depth/size, actual search/record or counter capacity exceeded.
/// @throws std::logic_error Inconsistent merged search bounds.
/// @par Example
/// @code
/// dm::algorithms::GameNode tree{false,0,{{true,2,{}},{true,5,{}}}};
/// auto r=dm::algorithms::sss_star_trace(tree,true);
/// std::print("{} {}\n",r.result.value,r.converged); // 5 true
/// @endcode
/// @par References
/// - G. C. Stockman, A minimax algorithm better than alpha-beta?, Artificial Intelligence12(2),1979; A. Plaat et al., Best-first fixed-depth minimax algorithms, Artificial Intelligence87,1996. This materialized-tree implementation includes an explicit all-leaf bound/validation scan.
/// @see sss_star()
SSSRun sss_star_trace(
    const GameNode&   root,
    bool              maximizing,
    double            resolution = 1,
    const SSSOptions& options    = {});
/// @brief Return the preserved checked SSS*/MTD search result.
/// Same actual producer/default policies; throws if driver probes are exhausted instead of returning an unproved minimax value. leaves_evaluated/nodes_visited/cutoffs count search work only; validation/bound-scan work is exposed by sss_star_run. best_child remains-1 because no extra principal-variation extraction is performed.
/// @param root Complete finite materialized tree.
/// @param maximizing Root max/min choice.
/// @param resolution Finite positive width compatible with distinct leaf gaps, default1.
/// @return Owned proven minimax value and actual selected search counters.
/// @throws std::invalid_argument Malformed tree/resolution.
/// @throws std::length_error Default input/work/capacity bound exceeded.
/// @throws std::logic_error Inconsistent memory bounds.
/// @throws std::runtime_error Probe cap reached before bound convergence.
/// @par Example
/// @code
/// dm::algorithms::GameNode leaf{true,3,{}};
/// std::print("{}\n",dm::algorithms::sss_star(leaf,true).value); // 3
/// @endcode
/// @see sss_star_run()
GameResult sss_star(const GameNode& root, bool maximizing, double resolution = 1);
/// @brief Generate the original bounded integer-valued variable-branching tree fixture.
/// One local mt19937 seed. Nonterminal branching=2+rng()%3; depth-zero leaf value=rng()%21-10. Preserves original recursive engine draw order. Complete actual node count bounded during generation; no search or hidden minimax occurs.
/// @param depth Nonnegative depth at most10.
/// @param seed Complete mt19937 seed.
/// @param max_nodes Actual generated-node policy, default one million; zero disables policy.
/// @return Owned original structural tree fixture.
/// @throws std::invalid_argument Depth outside0..10.
/// @throws std::length_error Actual node/counter capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::sss_example_tree(0,1).leaf); // true
/// @endcode
/// @see sss_star_run()
GameNode sss_example_tree(int depth, std::uint32_t seed, std::size_t max_nodes = 1000000);
} // namespace datamunge::algorithms
