#pragma once
/// @file todd_coxeter.hpp
/// @brief Checked actual HLT coset definitions, congruence merges and explicit unresolved caps.
#include <cstddef>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own ToddOptions values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::ToddOptions r{};
/// (void)r;
/// @endcode
struct ToddOptions {
  std::size_t max_cosets{
      200000}; ///< Positive total ever-defined coset cap, not live count; exhaustion is unresolved.
  std::size_t max_table_cells{2000000}; ///< Actual allocated rows times2*ngen; zero disables.
  std::size_t max_operations{
      20000000}; ///< Actual selected rep/scan/link/definition/merge operations; zero disables.
  std::size_t max_word_symbols{
      1000000}; ///< Complete total supplied relator/subgroup symbols; zero disables.
  std::size_t max_history_values{
      1000000}; ///< Actual event records plus copied scan-word symbols; zero disables.
};
/// @brief Own ToddStep values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::ToddStep r{};
/// (void)r;
/// @endcode
struct ToddStep {
  std::string action;     ///< Actual define, link, merge, relator or subgroup scan event.
  int         coset{};    ///< Actual involved root or newly defined row.
  int         other{-1};  ///< Actual linked/merged partner, -1 without one.
  int         symbol{-1}; ///< Actual internal symbol2g or2g+1, -1 without one.
  std::size_t defined{};  ///< Actual ever-defined row count after this event.
  std::size_t live{};     ///< Actual current union-find live coset count.
  std::vector<int>
      word; ///< Actual internal scan symbols for relator/subgroup events; empty for other events.
};
/// @brief Own ToddRun values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::ToddRun r{};
/// (void)r;
/// @endcode
struct ToddRun {
  long long index{
      -1}; ///< Actual final index only when closed; -1 denotes unresolved resource exhaustion.
  bool
      closed{}; ///< Actual full scan/fill pass made no definitions/merges/changed links and every root row is complete.
  std::string stop; ///< Actual complete, coset limit, table limit or operation limit.
  std::vector<std::vector<int>>
      table; ///< Actual final raw row links, normalized to current representative IDs; -1 undefined on unresolved runs.
  std::vector<int> representatives; ///< Actual current representative for every ever-defined row.
  std::vector<int> live;            ///< Actual surviving row IDs in ascending order.
  std::vector<std::vector<int>>
      compact_table; ///< Actual closed quotient action on compact live IDs; empty while unresolved.
  std::size_t definitions{};  ///< Actual ever-defined cosets, including subgroup row0.
  std::size_t coincidences{}; ///< Actual successful distinct-root merges.
  std::size_t word_scans{};   ///< Actual relator/subgroup scans, including repeated closure passes.
  std::size_t symbol_visits{}; ///< Actual defined forward/backward scan transitions traversed.
  std::size_t
      operations{}; ///< Actual selected capped kernel events, not all primitive operations/FLOPs.
  std::vector<ToddStep>
      steps; ///< Optional actual definition/link/merge/scan histories; no enumeration replay.
};
/// @brief Enumerate actual subgroup cosets with closure validation and explicit resource stops.
/// Words use signed1-based generator IDs; positive/inverse symbols map to2g/2g+1. Require ngen>=0, all nonzero word symbols within range, positive max_cosets fitting int and complete word geometry. Empty generator presentation has one coset when all words are empty. HLT scans forward/backward, deduces one-gap links, defines rows for larger gaps, and merges coincidences by an iterative congruence queue over both generator directions. Repeated complete scan/fill passes continue until no actual link/definition/merge changes remain, avoiding unverified closure after earlier-root coincidences. Total ever-defined rows, table cells and selected operations have separate caps. Exhaustion returns an explicit unresolved status with index-1; it is not evidence that the group/index is infinite. A closed complete action table can be independently checked against every relation and inverse. No general polynomial-time/termination guarantee; owns all output and no shared state.
/// @param ngen Complete nonnegative generator count, each with an inverse column.
/// @param relators Borrowed complete words equal to the group identity.
/// @param subgroup_gens Borrowed complete words fixing subgroup coset0.
/// @param options Positive definition cap and complete/actual geometry/work/history policies.
/// @return Owned actual closed index/action or unresolved state, executed counts and optional histories.
/// @throws std::invalid_argument Malformed generator count, signed word symbol or zero coset cap.
/// @throws std::length_error Int/container geometry, complete word/history or intrinsic counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::todd_coxeter_run(1,{{1,1,1}},{});
/// std::print("{} {}\n",r.closed,r.index); // true 3
/// @endcode
/// @par References
/// - Todd and Coxeter, A Practical Method for Enumerating Cosets of a Finite Abstract Group, Proceedings of the Edinburgh Mathematical Society5 (1936),26-34.
/// @see todd_coxeter_index()
ToddRun todd_coxeter_run(
    int                                  ngen,
    const std::vector<std::vector<int>>& relators,
    const std::vector<std::vector<int>>& subgroup_gens,
    const ToddOptions&                   options = {});
/// @brief Enumerate actual subgroup cosets with closure validation and explicit resource stops.
/// Words use signed1-based generator IDs; positive/inverse symbols map to2g/2g+1. Require ngen>=0, all nonzero word symbols within range, positive max_cosets fitting int and complete word geometry. Empty generator presentation has one coset when all words are empty. HLT scans forward/backward, deduces one-gap links, defines rows for larger gaps, and merges coincidences by an iterative congruence queue over both generator directions. Repeated complete scan/fill passes continue until no actual link/definition/merge changes remain, avoiding unverified closure after earlier-root coincidences. Total ever-defined rows, table cells and selected operations have separate caps. Exhaustion returns an explicit unresolved status with index-1; it is not evidence that the group/index is infinite. A closed complete action table can be independently checked against every relation and inverse. No general polynomial-time/termination guarantee; owns all output and no shared state.
/// @param ngen Complete nonnegative generator count, each with an inverse column.
/// @param relators Borrowed complete words equal to the group identity.
/// @param subgroup_gens Borrowed complete words fixing subgroup coset0.
/// @param options Positive definition cap and complete/actual geometry/work/history policies.
/// @return Owned actual closed index/action or unresolved state, executed counts and optional histories.
/// @throws std::invalid_argument Malformed generator count, signed word symbol or zero coset cap.
/// @throws std::length_error Int/container geometry, complete word/history or intrinsic counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::todd_coxeter_trace(1,{{1,1,1}},{});
/// std::print("{} {}\n",r.closed,r.index); // true 3
/// @endcode
/// @par References
/// - Todd and Coxeter, A Practical Method for Enumerating Cosets of a Finite Abstract Group, Proceedings of the Edinburgh Mathematical Society5 (1936),26-34.
/// @see todd_coxeter_index()
ToddRun todd_coxeter_trace(
    int                                  ngen,
    const std::vector<std::vector<int>>& relators,
    const std::vector<std::vector<int>>& subgroup_gens,
    const ToddOptions&                   options = {});
/// @brief Preserve a closed subgroup index or-1 for unresolved coset enumeration.
/// Words use signed1-based generator IDs; positive/inverse symbols map to2g/2g+1. Require ngen>=0, all nonzero word symbols within range, positive max_cosets fitting int and complete word geometry. Empty generator presentation has one coset when all words are empty. HLT scans forward/backward, deduces one-gap links, defines rows for larger gaps, and merges coincidences by an iterative congruence queue over both generator directions. Repeated complete scan/fill passes continue until no actual link/definition/merge changes remain, avoiding unverified closure after earlier-root coincidences. Total ever-defined rows, table cells and selected operations have separate caps. Exhaustion returns an explicit unresolved status with index-1; it is not evidence that the group/index is infinite. A closed complete action table can be independently checked against every relation and inverse. No general polynomial-time/termination guarantee; owns all output and no shared state.
/// @param ngen Complete nonnegative generator count, each with an inverse column.
/// @param relators Borrowed complete words equal to the group identity.
/// @param subgroup_gens Borrowed complete words fixing subgroup coset0.
/// @param max_cosets Positive total ever-defined row cap, default200000, at most int max.
/// @return Actual closed index, or-1 for definition/table/operation exhaustion.
/// @throws std::invalid_argument Malformed complete presentation or nonpositive coset cap.
/// @throws std::length_error Int geometry/default complete word or capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::todd_coxeter_index(1,{{1,1,1}},{})); // 3
/// @endcode
/// @see todd_coxeter_run()
long long todd_coxeter_index(
    int                                  ngen,
    const std::vector<std::vector<int>>& relators,
    const std::vector<std::vector<int>>& subgroup_gens,
    long long                            max_cosets = 200000);
} // namespace datamunge::algorithms
