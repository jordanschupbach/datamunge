#pragma once
/// @file schreier_sims.hpp
/// @brief Checked deterministic stabilizer chains and actual orbit/sifting histories.
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>
/// @brief Canonical checked native algorithm producers and owned outputs.
namespace datamunge::algorithms {
/// @brief Owned base and strong-generating-chain transversals for a finite permutation group.
/// @par Example
/// @code
/// const dm::algorithms::BSGS value{};
/// std::print("{}\n",value.order); // 1
/// @endcode
struct BSGS {
  int              n{0}; ///< Degree; permutations are zero-based image vectors.
  std::vector<int> base; ///< Ordered distinct stabilizer base points.
  std::vector<std::unordered_map<int, std::vector<int>>>
                     transversal; ///< Level maps from orbit points to actual representatives.
  unsigned long long order{1};    ///< Checked product of orbit sizes, exact when representable.
};
/// @brief Complete input and actual work/history bounds; zero disables a size policy.
/// @par Example
/// @code
/// const dm::algorithms::SchreierSimsOptions value{};
/// std::print("{}\n",value.max_degree); // 64
/// @endcode
struct SchreierSimsOptions {
  std::size_t max_degree{64};                 ///< Complete permutation degree.
  std::size_t max_generators{10000};          ///< Complete supplied generator count.
  std::size_t max_point_operations{10000000}; ///< Actual counted permutation-loop point operations.
  std::size_t max_trace_entries{1000000}; ///< Orbit/sift headers and copied permutation entries.
};
/// @brief One actual first orbit discovery during a real transversal build.
/// @par Example
/// @code
/// const dm::algorithms::SchreierOrbitStep value{};
/// std::print("{}\n",value.level); // 0
/// @endcode
struct SchreierOrbitStep {
  std::size_t level{}; ///< Actual chain level and local generator index.
  std::size_t generator_index{}; ///< Actual chain level and local generator index.
  int base{}; ///< Actual base and generator edge.
  int source{}; ///< Actual base and generator edge.
  int target{}; ///< Actual base and generator edge.
  std::vector<int> representative;             ///< Actual newly stored coset representative.
};
/// @brief One actual completed sift, including an early failed level.
/// @par Example
/// @code
/// const dm::algorithms::SchreierSiftStep value{};
/// std::print("{}\n",value.failed_level); // 0
/// @endcode
struct SchreierSiftStep {
  std::size_t start_level{}; ///< Sift start and first missing level, or chain size on completion.
  std::size_t failed_level{}; ///< Sift start and first missing level, or chain size on completion.
  std::vector<int> input; ///< Owned actual before and after permutations.
  std::vector<int> residue; ///< Owned actual before and after permutations.
};
/// @brief Owned actual group chain and selected work/history from one producing execution.
/// @par Example
/// @code
/// const dm::algorithms::SchreierSimsRun value{};
/// std::print("{}\n",value.sifts); // 0
/// @endcode
struct SchreierSimsRun {
  BSGS        group; ///< Actual completed group chain.
  std::size_t point_operations{}; ///< Actual selected operations.
  std::size_t compositions{}; ///< Actual selected operations.
  std::size_t inversions{}; ///< Actual selected operations.
  std::size_t orbit_builds{}; ///< Actual selected operations.
  std::size_t schreier_generators{}; ///< Actual selected operations.
  std::size_t sifts{}; ///< Actual selected operations.
  std::vector<SchreierOrbitStep>
      discoveries; ///< Actual optional orbit discoveries, including rebuilds.
  std::vector<SchreierSiftStep> sift_steps; ///< Actual optional complete sift records.
};
/// @brief Construct a checked stabilizer chain.
/// Permutations are zero-based bijective image vectors; composition a[b[i]] applies b then a. Builds orbit transversals by breadth first search, sifts generators, then closes actual Schreier generators until all residues are identity. Orbit snapshots are sorted for deterministic traversal; supplied generator order is preserved. Empty generators give the identity group, including degree zero. Order is the exact checked product of stabilizer orbit sizes; overflow throws. Work depends on generated strong generators and actual restarts; counters count real permutation-loop visits, compositions, inversions, orbit rebuilds, sifts and Schreier generators, not predicted complexity. Trace owns actual records including repeated rebuild discoveries. Input is borrowed unchanged; calls share no mutable state.
/// @param n Nonnegative complete degree.
/// @param generators Borrowed complete bijective image vectors, each of length n.
/// @param options Input, actual work and retained-history limits.
/// @return Owned actual BSGS, work and optional histories.
/// @throws std::invalid_argument Negative degree or malformed generator.
/// @throws std::length_error Complete input or actual work/history bound exceeded.
/// @throws std::overflow_error Group order cannot fit unsigned long long.
/// @par Example
/// @code
/// auto r=dm::algorithms::schreier_sims_run(3,{{1,0,2},{1,2,0}});
/// std::print("{}\n",r.group.order); // 6
/// @endcode
/// @par References
/// - Sims (1970), Computational methods in the study of permutation groups. https://www.sciencedirect.com/science/article/pii/B9780080129754500205
/// @see bsgs_contains()
SchreierSimsRun schreier_sims_run(
    int                                  n,
    const std::vector<std::vector<int>>& generators,
    const SchreierSimsOptions&           options = {});
/// @brief Construct a checked stabilizer chain with actual orbit and sift histories.
/// Permutations are zero-based bijective image vectors; composition a[b[i]] applies b then a. Builds orbit transversals by breadth first search, sifts generators, then closes actual Schreier generators until all residues are identity. Orbit snapshots are sorted for deterministic traversal; supplied generator order is preserved. Empty generators give the identity group, including degree zero. Order is the exact checked product of stabilizer orbit sizes; overflow throws. Work depends on generated strong generators and actual restarts; counters count real permutation-loop visits, compositions, inversions, orbit rebuilds, sifts and Schreier generators, not predicted complexity. Trace owns actual records including repeated rebuild discoveries. Input is borrowed unchanged; calls share no mutable state.
/// @param n Nonnegative complete degree.
/// @param generators Borrowed complete bijective image vectors, each of length n.
/// @param options Input, actual work and retained-history limits.
/// @return Owned actual BSGS, work and optional histories.
/// @throws std::invalid_argument Negative degree or malformed generator.
/// @throws std::length_error Complete input or actual work/history bound exceeded.
/// @throws std::overflow_error Group order cannot fit unsigned long long.
/// @par Example
/// @code
/// auto r=dm::algorithms::schreier_sims_trace(3,{{1,0,2},{1,2,0}});
/// std::print("{}\n",r.group.order); // 6
/// @endcode
/// @par References
/// - Sims (1970), Computational methods in the study of permutation groups. https://www.sciencedirect.com/science/article/pii/B9780080129754500205
/// @see bsgs_contains()
SchreierSimsRun schreier_sims_trace(
    int                                  n,
    const std::vector<std::vector<int>>& generators,
    const SchreierSimsOptions&           options = {});
/// @brief Construct the checked group through the preserved signature.
/// Uses the same actual producer and default bounds without optional history.
/// @param n Nonnegative degree.
/// @param generators Borrowed degree-n bijections.
/// @return Owned exact checked group chain.
/// @throws std::invalid_argument Malformed degree or generator.
/// @throws std::length_error Default input/work bounds exceeded.
/// @throws std::overflow_error Group order overflow.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::schreier_sims(3,{{1,0,2},{1,2,0}}).order); // 6
/// @endcode
/// @par References
/// - Sims (1970), Computational methods in the study of permutation groups. https://www.sciencedirect.com/science/article/pii/B9780080129754500205
/// @see schreier_sims_run()
BSGS schreier_sims(int n, const std::vector<std::vector<int>>& generators);
/// @brief Sift a candidate through a structurally checked owned group chain.
/// Wrong-size or nonbijective candidates return false. Malformed chain shapes, base points, representatives or orbit-order metadata throw before indexing. Structural validation cannot establish that arbitrarily edited transversals are an authentic closed group; use a canonically produced BSGS. Input borrowed unchanged. Time O(degree*sum orbit sizes + degree*levels) including validation; no mutable state.
/// @param g Borrowed canonically produced group chain.
/// @param perm Borrowed zero-based candidate image vector.
/// @return True exactly for members of a canonically produced chain.
/// @throws std::invalid_argument Malformed group representation or metadata.
/// @par Example
/// @code
/// auto g=dm::algorithms::schreier_sims(3,{{1,2,0}});
/// std::print("{}\n",dm::algorithms::bsgs_contains(g,{1,0,2})); // false
/// @endcode
/// @par References
/// - Sims (1970), Computational methods in the study of permutation groups. https://www.sciencedirect.com/science/article/pii/B9780080129754500205
/// @see schreier_sims()
bool bsgs_contains(const BSGS& g, const std::vector<int>& perm);
/// @brief Construct the original transposition-and-cycle symmetric-group fixture.
/// Returns the transposition (0 1) followed by the full forward n-cycle; together they generate S_n.
/// @param n Degree in [2,64].
/// @return Two owned image-vector generators.
/// @throws std::invalid_argument Degree outside [2,64].
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::symmetric_group_generators(3)); // [[1, 0, 2], [1, 2, 0]]
/// @endcode
/// @par References
/// - Sims (1970), Computational methods in the study of permutation groups. https://www.sciencedirect.com/science/article/pii/B9780080129754500205
/// @see schreier_sims()
std::vector<std::vector<int>> symmetric_group_generators(int n);
} // namespace datamunge::algorithms
