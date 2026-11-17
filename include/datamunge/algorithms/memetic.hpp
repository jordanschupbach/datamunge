#pragma once

/// @file memetic.hpp
/// @brief Canonical memetic API, producing histories and checked native contracts.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned weighted adjacency lists for the checked native MAX-CUT model.
/// @details Every directed neighbour/finite weight entry must have a matching reverse entry of equal weight and multiplicity; parallel edges are allowed, self loops are rejected. Original row order is retained for floating sums.
/// @par Example
/// @code
/// dm::algorithms::MemeticGraph g{{{1,2}},{{0,2}}};
/// std::print("{}\n",g.size()); // 2
/// @endcode
/// @see memetic_max_cut, memetic_max_cut_analyze, memetic_max_cut_trace
/// @par Reference
/// Moscato (1989), On Evolution, Search, Optimization, Genetic Algorithms and Martial Arts: Towards Memetic Algorithms, C3P Report826, provides historical memetic context. The retained native MAX-CUT recipe and exact population/local conventions are specified explicitly here.
using MemeticGraph = std::vector<std::vector<std::pair<int, double>>>;

/// @brief Legacy weighted cut objective and one actual binary partition.
/// @details Field layout is retained. A heuristic result is a valid witness, not a global optimality certificate.
/// @par Example
/// @code
/// auto r=dm::algorithms::memetic_max_cut(2,{{{1,2}},{{0,2}}},4,2,1);
/// std::print("{}\n",r.value); // 2
/// @endcode
/// @see memetic_max_cut, memetic_max_cut_analyze, memetic_max_cut_trace
/// @par Reference
/// Moscato (1989), On Evolution, Search, Optimization, Genetic Algorithms and Martial Arts: Towards Memetic Algorithms, C3P Report826, provides historical memetic context. The retained native MAX-CUT recipe and exact population/local conventions are specified explicitly here.
struct MemeticMaxCutResult {
  double           value{0}; ///< Actual row-order sum of u<v crossing-edge weights.
  std::vector<int> side;     ///< Actual binary0/1 side per original vertex.
};

/// @brief Actual selected operation counts of the canonical population/local producer.
/// @details One local check computes one vertex delta; a sweep can be partial under a check cap. Random draws, fitness evaluation and validation are separate work.
/// @par Example
/// @code
/// auto a=dm::algorithms::memetic_max_cut_analyze(1,{{}},1,0,1);
/// std::print("{}\n",a.work.individuals); // 1
/// @endcode
/// @see memetic_max_cut, memetic_max_cut_analyze, memetic_max_cut_trace
/// @par Reference
/// Moscato (1989), On Evolution, Search, Optimization, Genetic Algorithms and Martial Arts: Towards Memetic Algorithms, C3P Report826, provides historical memetic context. The retained native MAX-CUT recipe and exact population/local conventions are specified explicitly here.
struct MemeticWork {
  std::size_t rng_draws{0}; ///< Actual xorshift64 state advances.
  std::size_t local_sweeps{
      0}; ///< Actual started sequential vertex sweeps, including terminal/partial ones.
  std::size_t local_checks{0};     ///< Actual evaluated single-vertex deltas.
  std::size_t delta_references{0}; ///< Actual listed neighbour reads in delta sums.
  std::size_t flips{0};            ///< Actual delta>1e-12 bit flips.
  std::size_t individuals{
      0}; ///< Actual candidate fitness evaluations, including one unfinished-refinement candidate on a work stop.
  std::size_t fitness_references{
      0}; ///< Actual listed neighbour inspections in candidate objective sums.
  std::size_t best_comparisons{0}; ///< Actual strict population best-index comparisons.
  std::size_t improvements{0};     ///< Actual first/strict global best candidate replacements.
};

/// @brief Actual best candidate and request-completion/refinement status.
/// @details complete means the requested population/generations finished; it does not mean globally optimal. locally_refined means the winning candidate finished a no-flip sweep under computed delta>1e-12 tests, not an exact-real global/local certificate.
/// @par Example
/// @code
/// auto a=dm::algorithms::memetic_max_cut_analyze(1,{{}},1,0,1);
/// std::print("{} {}\n",a.complete,a.locally_refined); // true true
/// @endcode
/// @see memetic_max_cut, memetic_max_cut_analyze, memetic_max_cut_trace
/// @par Reference
/// Moscato (1989), On Evolution, Search, Optimization, Genetic Algorithms and Martial Arts: Towards Memetic Algorithms, C3P Report826, provides historical memetic context. The retained native MAX-CUT recipe and exact population/local conventions are specified explicitly here.
struct MemeticAnalysis {
  MemeticMaxCutResult result; ///< Actual best evaluated valid binary partition and objective.
  bool                complete{
      true}; ///< False only when the optional actual local-check cap stops the requested run.
  bool locally_refined{
      false}; ///< Whether the actual winning candidate completed native local refinement.
  bool initialized{
      false}; ///< Whether the entire original population finished refinement/evaluation.
  int generations_completed{
      0};           ///< Actual fully installed offspring generations, excluding initialization.
  MemeticWork work; ///< Actual selected producer counts.
};

/// @brief One actual RNG state advance and its native use.
/// @details Kinds0 initialization bit,1 tournament modulo index,2 crossover bit,3 mutation uniform. Mutation uses top53 bits times2^-53; other choices use raw modulo bound, retaining native modulo bias.
/// @par Example
/// @code
/// auto t=dm::algorithms::memetic_max_cut_trace(1,{{}},1,0,1);
/// std::print("{}\n",t.draws[0].kind); // 0
/// @endcode
/// @see memetic_max_cut, memetic_max_cut_analyze, memetic_max_cut_trace
/// @par Reference
/// Moscato (1989), On Evolution, Search, Optimization, Genetic Algorithms and Martial Arts: Towards Memetic Algorithms, C3P Report826, provides historical memetic context. The retained native MAX-CUT recipe and exact population/local conventions are specified explicitly here.
struct MemeticDraw {
  unsigned    kind{0};        ///< Actual draw purpose0..3.
  int         generation{-1}; ///< -1 for initialization, otherwise zero-based offspring generation.
  std::size_t individual{0};  ///< Actual output population slot.
  std::size_t slot{0};        ///< Actual vertex or tournament draw position.
  std::uint64_t before{0};    ///< Actual RNG state before the13/7/17 xorshift.
  std::uint64_t after{0};     ///< Actual RNG state/output after the xorshift.
  std::size_t   bound{0};     ///< Actual modulo bound, zero for mutation uniform.
  std::size_t   choice{0};    ///< Actual raw modulo choice, zero for mutation kind.
  double        unit{0};      ///< Actual top53-bit unit value only for mutation, zero otherwise.
};

/// @brief One actual sequential local-search vertex delta and optional flip.
/// @details Input rows are summed in original order. The delta is the computed gain for flipping this vertex at its actual current partition; subsequent vertices see any accepted flip.
/// @par Example
/// @code
/// auto t=dm::algorithms::memetic_max_cut_trace(1,{{}},1,0,1);
/// std::print("{}\n",t.local_steps[0].delta); // 0
/// @endcode
/// @see memetic_max_cut, memetic_max_cut_analyze, memetic_max_cut_trace
/// @par Reference
/// Moscato (1989), On Evolution, Search, Optimization, Genetic Algorithms and Martial Arts: Towards Memetic Algorithms, C3P Report826, provides historical memetic context. The retained native MAX-CUT recipe and exact population/local conventions are specified explicitly here.
struct MemeticLocalStep {
  int         generation{-1}; ///< Actual initialization/offspring generation label.
  std::size_t individual{0};  ///< Actual candidate output slot.
  std::size_t sweep{0};       ///< Actual global zero-based started sweep id.
  int         vertex{0};      ///< Actual evaluated vertex.
  int         before{0};      ///< Actual bit before testing.
  double      delta{0};       ///< Actual computed row-order flip gain.
  bool        flipped{false}; ///< Actual strict delta>1e-12 decision.
  int         after{0};       ///< Actual bit after decision.
};

/// @brief One actual initialized or offspring candidate before/after native local refinement.
/// @details Draw spans include real initialization bits or tournaments/crossover/mutation. Local spans include only evaluated vertex steps, so an explicit work stop can retain an unfinished candidate with refined=false.
/// @par Example
/// @code
/// auto t=dm::algorithms::memetic_max_cut_trace(1,{{}},1,0,1);
/// std::print("{}\n",t.individuals[0].refined); // true
/// @endcode
/// @see memetic_max_cut, memetic_max_cut_analyze, memetic_max_cut_trace
/// @par Reference
/// Moscato (1989), On Evolution, Search, Optimization, Genetic Algorithms and Martial Arts: Towards Memetic Algorithms, C3P Report826, provides historical memetic context. The retained native MAX-CUT recipe and exact population/local conventions are specified explicitly here.
struct MemeticIndividual {
  int         generation{-1}; ///< Actual initialization/offspring generation label.
  std::size_t slot{0};        ///< Actual output population position.
  int         parent_a{-1};   ///< Actual tournament winner index, -1 during initialization.
  int         parent_b{-1};   ///< Actual second tournament winner, -1 during initialization.
  std::vector<int>
      before_local; ///< Actual candidate after random creation/crossover/mutation, before refinement.
  std::vector<int>
              after_local; ///< Actual candidate after completed or explicitly stopped refinement.
  double      fitness{0};  ///< Actual objective evaluated on after_local.
  bool        refined{false}; ///< Whether this candidate completed a native no-flip terminal sweep.
  std::size_t first_draw{0};  ///< First actual RNG record index for this candidate.
  std::size_t past_draw{0};   ///< Past-last actual RNG record index.
  std::size_t first_local{0}; ///< First actual local-step record index.
  std::size_t past_local{0};  ///< Past-last actual local-step record index.
};

/// @brief One actual fully installed population snapshot and strict best-index selection.
/// @details Index0 is the fully initialized population; index1..G are actual completed offspring populations. An unfinished generation never appears here.
/// @par Example
/// @code
/// auto t=dm::algorithms::memetic_max_cut_trace(1,{{}},1,1,1);
/// std::print("{}\n",t.generations.size()); // 2
/// @endcode
/// @see memetic_max_cut, memetic_max_cut_analyze, memetic_max_cut_trace
/// @par Reference
/// Moscato (1989), On Evolution, Search, Optimization, Genetic Algorithms and Martial Arts: Towards Memetic Algorithms, C3P Report826, provides historical memetic context. The retained native MAX-CUT recipe and exact population/local conventions are specified explicitly here.
struct MemeticGeneration {
  int index{0}; ///< Actual installed population generation, initialization zero.
  std::vector<std::vector<int>> population;    ///< Actual complete installed binary population.
  std::vector<double>           fitness;       ///< Actual corresponding evaluated objectives.
  std::size_t                   best_index{0}; ///< Actual first strictly best member index.
  double                        best_value{0}; ///< Actual selected population-best objective.
  double mean_fitness{0}; ///< Actual long-double accumulated population mean, returned as double.
  std::size_t distinct_cuts{
      0}; ///< Actual distinct binary partitions after identifying complements, measured from the installed population.
};

/// @brief Owned original graph and complete actual bounded population/RNG/local history.
/// @details Base logical quota5+2N+2E_list counts five scalar run inputs, N graph vertex slots, neighbour-id/weight scalars and N returned best bits. Add each RNG/local record, each candidate record plus its2N bits, and each installed generation record plus population bits/fitness values. Fields, row headers, capacities, runtime/returned duplicate copies and bytes are excluded.
/// @par Example
/// @code
/// auto t=dm::algorithms::memetic_max_cut_trace(0,{},1,0,1);
/// std::print("{}\n",t.captured_entries); // 8
/// @endcode
/// @see memetic_max_cut, memetic_max_cut_analyze, memetic_max_cut_trace
/// @par Reference
/// Moscato (1989), On Evolution, Search, Optimization, Genetic Algorithms and Martial Arts: Towards Memetic Algorithms, C3P Report826, provides historical memetic context. The retained native MAX-CUT recipe and exact population/local conventions are specified explicitly here.
struct MemeticTrace {
  int             n{0};                     ///< Owned original vertex count.
  MemeticGraph    graph;                    ///< Owned original ordered weighted lists.
  int             population_size{0};       ///< Owned requested positive population size.
  int             requested_generations{0}; ///< Owned requested nonnegative offspring generations.
  std::uint64_t   seed{0};                  ///< Owned original seed, before zero fallback.
  std::size_t     max_local_checks{0};      ///< Owned optional actual local-check cap.
  MemeticAnalysis analysis; ///< Actual best partition, completion/refinement flags and work.
  std::vector<MemeticDraw>       draws;       ///< Actual RNG advances and native uses.
  std::vector<MemeticLocalStep>  local_steps; ///< Actual sequential delta/flip decisions.
  std::vector<MemeticIndividual> individuals; ///< Actual candidate lifecycle snapshots.
  std::vector<MemeticGeneration> generations; ///< Actual fully installed population snapshots.
  std::size_t                    captured_entries{0}; ///< Complete logical capture count.
};

/// @brief Return the native memetic MAX-CUT heuristic using completed unrestricted local refinement.
/// @details Validate all ordered lists and exact reverse weight multiplicities before work; finite signed weights/parallel edges allowed, self loops rejected. Random binary initialization, three-draw tournaments, uniform per-bit crossover, per-bit mutation probability1/N, sequential delta>1e-12 flips to a no-flip sweep, and one-member elitism retain legacy draw/fitness/tie order. Seed0 uses the legacy nonzero fallback. No global optimum guarantee, and local refinement can require many sweeps. Population storage O(PN), plus graph/validation and candidate state; time depends on actual local checks and graph reads.
/// @param n Nonnegative vertex count.
/// @param adj Exactly n ordered symmetric weighted lists.
/// @param pop_size Positive population size, default30.
/// @param generations Nonnegative offspring generations, default40.
/// @param seed Native uint64 seed, default1.
/// @return Actual first strictly best evaluated partition and its objective.
/// @throws std::invalid_argument Before evaluation on malformed graph, nonbinary side, nonpositive population or negative generations as applicable.
/// @throws std::length_error On nonrepresentable count/allocation/capture geometry.
/// @throws std::overflow_error If absolute edge-weight sum or an actual computed objective/delta cannot remain finite.
/// @par Example
/// @code
/// auto r=dm::algorithms::memetic_max_cut(2,{{{1,2}},{{0,2}}},4,2,1);
/// std::print("{}\n",r.value); // 2
/// @endcode
/// @see memetic_max_cut, memetic_max_cut_analyze, memetic_max_cut_trace
/// @par Reference
/// Moscato (1989), On Evolution, Search, Optimization, Genetic Algorithms and Martial Arts: Towards Memetic Algorithms, C3P Report826, provides historical memetic context. The retained native MAX-CUT recipe and exact population/local conventions are specified explicitly here.
MemeticMaxCutResult memetic_max_cut(
    int                 n,
    const MemeticGraph& adj,
    int                 pop_size    = 30,
    int                 generations = 40,
    std::uint64_t       seed        = 1);

/// @brief Return actual best witness and request/refinement status under an optional local-check cap.
/// @details Zero disables the cap. A positive exhausted cap stops before the next vertex delta, evaluates that actual partially refined candidate and returns complete=false with the best actually evaluated partition. Such a winner may have locally_refined=false. The cap does not bound validation, RNG draws, fitness work or wall time. Complete=true still does not prove a global optimum.
/// @param n Nonnegative vertex count.
/// @param adj Exactly n complete checked weighted lists.
/// @param pop_size Positive requested population size.
/// @param generations Nonnegative requested generations.
/// @param seed Native uint64 seed, zero uses fallback.
/// @param max_local_checks Optional global actual vertex-delta cap, zero unlimited.
/// @return Actual best result, initialized/completion/refinement flags and operation counts.
/// @throws std::invalid_argument Before evaluation on malformed graph, nonbinary side, nonpositive population or negative generations as applicable.
/// @throws std::length_error On nonrepresentable count/allocation/capture geometry.
/// @throws std::overflow_error If absolute edge-weight sum or an actual computed objective/delta cannot remain finite.
/// @par Example
/// @code
/// auto a=dm::algorithms::memetic_max_cut_analyze(1,{{}},1,0,1);
/// std::print("{}\n",a.initialized); // true
/// @endcode
/// @see memetic_max_cut, memetic_max_cut_analyze, memetic_max_cut_trace
/// @par Reference
/// Moscato (1989), On Evolution, Search, Optimization, Genetic Algorithms and Martial Arts: Towards Memetic Algorithms, C3P Report826, provides historical memetic context. The retained native MAX-CUT recipe and exact population/local conventions are specified explicitly here.
MemeticAnalysis memetic_max_cut_analyze(
    int                 n,
    const MemeticGraph& adj,
    int                 pop_size         = 30,
    int                 generations      = 40,
    std::uint64_t       seed             = 1,
    std::size_t         max_local_checks = 0);

/// @brief Capture actual RNG draws, local flips, candidate states and installed populations from one run.
/// @details Same canonical producer and separate work cap as memetic_max_cut_analyze. Complete base input/output geometry is admitted before owned copies; every actual record/vector snapshot is admitted before capture. Default logical cap one million; zero disables it. Work-cap stop returns a complete actual bounded history, while capture failure throws without a prefix.
/// @param n Nonnegative vertex count.
/// @param adj Complete ordered weighted lists.
/// @param pop_size Positive population size.
/// @param generations Nonnegative requested offspring generations.
/// @param seed Native uint64 seed.
/// @param max_local_checks Optional actual delta-check cap, zero unlimited.
/// @param max_entries Complete logical capture cap, zero off.
/// @return Owned input and actual bounded lifecycle/history plus same-run best result.
/// @throws std::invalid_argument Before evaluation on malformed graph, nonbinary side, nonpositive population or negative generations as applicable.
/// @throws std::length_error On nonrepresentable count/allocation/capture geometry.
/// @throws std::overflow_error If absolute edge-weight sum or an actual computed objective/delta cannot remain finite.
/// @par Example
/// @code
/// auto t=dm::algorithms::memetic_max_cut_trace(0,{},1,0,1);
/// std::print("{}\n",t.analysis.result.value); // 0
/// @endcode
/// @see memetic_max_cut, memetic_max_cut_analyze, memetic_max_cut_trace
/// @par Reference
/// Moscato (1989), On Evolution, Search, Optimization, Genetic Algorithms and Martial Arts: Towards Memetic Algorithms, C3P Report826, provides historical memetic context. The retained native MAX-CUT recipe and exact population/local conventions are specified explicitly here.
MemeticTrace memetic_max_cut_trace(
    int                 n,
    const MemeticGraph& adj,
    int                 pop_size         = 30,
    int                 generations      = 40,
    std::uint64_t       seed             = 1,
    std::size_t         max_local_checks = 0,
    std::size_t         max_entries      = 1000000);

/// @brief Evaluate an actual binary partition under the same checked row-order MAX-CUT objective.
/// @details Each u<v listed edge contributes its weight iff endpoints differ; parallel entries contribute separately. Complete shape/weight/symmetry and every side bit are validated first.
/// @param n Nonnegative vertex count.
/// @param adj Complete checked weighted lists.
/// @param side Exactly n binary values.
/// @return Actual finite row-order objective.
/// @throws std::invalid_argument Before evaluation on malformed graph, nonbinary side, nonpositive population or negative generations as applicable.
/// @throws std::length_error On nonrepresentable count/allocation/capture geometry.
/// @throws std::overflow_error If absolute edge-weight sum or an actual computed objective/delta cannot remain finite.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::memetic_cut_value(2,{{{1,2}},{{0,2}}},{0,1})); // 2
/// @endcode
/// @see memetic_max_cut, memetic_max_cut_analyze, memetic_max_cut_trace
/// @par Reference
/// Moscato (1989), On Evolution, Search, Optimization, Genetic Algorithms and Martial Arts: Towards Memetic Algorithms, C3P Report826, provides historical memetic context. The retained native MAX-CUT recipe and exact population/local conventions are specified explicitly here.
double memetic_cut_value(int n, const MemeticGraph& adj, const std::vector<int>& side);

/// @brief Enumerate all distinct-by-complement partitions as an independent native exact comparator.
/// @details Fix vertex0 side0 and enumerate remaining bits in increasing mask order; score with the canonical checked objective. Complete enumeration is exponential, with2^(N-1) assignments for N>=1, one for an empty graph. Default optional vertex cap24; zero disables it, but N must still be at most63 for safe uint64 mask geometry. This is not the memetic heuristic and is practical only on small graphs.
/// @param n Nonnegative vertex count, at most63 and within optional cap.
/// @param adj Complete checked weighted graph.
/// @param max_vertices Optional complete enumeration vertex cap, default24; zero off.
/// @return First strictly maximum actual partition/objective under full enumeration.
/// @throws std::invalid_argument Before evaluation on malformed graph, nonbinary side, nonpositive population or negative generations as applicable.
/// @throws std::length_error On nonrepresentable count/allocation/capture geometry.
/// @throws std::overflow_error If absolute edge-weight sum or an actual computed objective/delta cannot remain finite.
/// @throws std::length_error Before enumeration if vertex/mask cap is exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::memetic_max_cut_exact(2,{{{1,2}},{{0,2}}});
/// std::print("{}\n",r.value); // 2
/// @endcode
/// @see memetic_max_cut, memetic_max_cut_analyze, memetic_max_cut_trace
/// @par Reference
/// Moscato (1989), On Evolution, Search, Optimization, Genetic Algorithms and Martial Arts: Towards Memetic Algorithms, C3P Report826, provides historical memetic context. The retained native MAX-CUT recipe and exact population/local conventions are specified explicitly here.
MemeticMaxCutResult
memetic_max_cut_exact(int n, const MemeticGraph& adj, std::size_t max_vertices = 24);

/// @brief Return the original deterministic sixteen-vertex integer-weight fixture.
/// @details Continue xorshift64 seed999 over upper-triangle pairs; next()%2 chooses inclusion, then next()%9+1 chooses weight. Both directions retain original insertion order.
/// @return Owned original weighted graph.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::memetic_demo_graph().size()); // 16
/// @endcode
/// @see memetic_max_cut, memetic_max_cut_analyze, memetic_max_cut_trace
/// @par Reference
/// Moscato (1989), On Evolution, Search, Optimization, Genetic Algorithms and Martial Arts: Towards Memetic Algorithms, C3P Report826, provides historical memetic context. The retained native MAX-CUT recipe and exact population/local conventions are specified explicitly here.
MemeticGraph memetic_demo_graph();

} // namespace datamunge::algorithms
