#pragma once
/// @file petrick.hpp
/// @brief Exact minimum-cardinality set covering by distributive expansion and absorption.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned minimum-cardinality covers, each a sorted set of nonnegative ids.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::petrick({{1}}).minimal_covers); // [[1]]
/// @endcode
struct PetrickResult {
  std::vector<std::vector<int>>
      minimal_covers; ///< All minimum-size covers; empty when unsatisfiable; {{}} for no clauses.
};
/// @brief Actual expansion and absorption outcome for one clause.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::petrick_trace({{1}}).stages[0].generated); // 1
/// @endcode
struct PetrickStage {
  std::size_t clause;    ///< Zero-based clause index.
  std::size_t generated; ///< Actual candidate products, including duplicates.
  std::size_t absorbed;  ///< Actual candidates discarded by subset/duplicate absorption.
  std::vector<std::vector<int>> terms; ///< Complete surviving irredundant terms after this clause.
};
/// @brief Finite expansion and retained-history policies, independently disabled by zero.
/// @par Example
/// @code
/// const dm::algorithms::PetrickOptions o;std::print("{}\n",o.max_candidates); // 1000000
/// @endcode
struct PetrickOptions {
  std::size_t max_candidates{
      1000000}; ///< Cumulative actual generated-product cap, not peak RAM or comparison count.
  std::size_t max_trace_entries{
      1000000}; ///< Stage/term records plus retained term ids; excludes final result.
};
/// @brief Owned exact result and actual intermediate irredundant terms.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::petrick_trace({{1}}).generated); // 1
/// @endcode
struct PetrickTrace {
  PetrickResult result;       ///< Actual minimum-cardinality output.
  std::size_t   generated{0}; ///< Cumulative actual generated candidate products.
  std::vector<PetrickStage>
      stages; ///< Actual completed clauses, stopping at unsatisfiable empty clause.
};
/// @brief Enumerate all minimum-cardinality covers of a nonnegative-id clause chart.
/// Each clause requires at least one id. Duplicate ids are allowed and absorbed.
/// An empty clause is unsatisfiable, so returns no covers; no clauses has exactly
/// one empty cover. All ids are validated before expansion, even beyond an empty
/// clause. Sorts candidates by cardinality, absorbs supersets, and selects the
/// smallest final terms. Exponential worst-case time/storage; default cumulative
/// generated-product cap is one million. Does not minimise literal cost or break
/// ties lexicographically. Inputs are borrowed unchanged; no shared mutable state.
/// @param covers_of_minterm Each inner vector lists alternative covering ids.
/// @return Owned sorted-id covers, all of minimum cardinality.
/// @throws std::invalid_argument If any id is negative.
/// @throws std::length_error If default expansion policy or storage ranges are exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::petrick({{0,1},{1,2}}).minimal_covers); // [[1]]
/// @endcode
/// @par References
/// - S. R. Petrick (1956), A direct determination of the irredundant forms of a
///   Boolean function from the set of prime implicants, AFCRC-TR-56-110.
PetrickResult petrick(const std::vector<std::vector<int>>& covers_of_minterm);
/// @brief Record actual generated/absorbed counts and terms from petrick's producer.
/// No second expansion populates history. Additional storage is the sum of
/// retained irredundant terms and ids over completed stages.
/// @param covers_of_minterm Complete nonnegative-id clause chart, unchanged.
/// @param options Cumulative generated-product and retained-history policies.
/// @return Owned exact result and every actual completed clause's terms.
/// @throws std::invalid_argument If any id is negative.
/// @throws std::length_error If expansion/history policies or storage ranges are exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::petrick_trace({{0,1},{1,2}}).stages.size()); // 2
/// @endcode
PetrickTrace petrick_trace(
    const std::vector<std::vector<int>>& covers_of_minterm,
    PetrickOptions                       options = {});
} // namespace datamunge::algorithms
