#pragma once
/// @file tournament_selection.hpp
/// @brief Actual with-replacement contestant draws, first-tie winners and historical modulo RNG.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own TournamentOptions values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TournamentOptions r{};
/// (void)r;
/// @endcode
struct TournamentOptions {
  std::size_t max_population{1000000}; ///< Complete finite fitness geometry; zero disables.
  std::size_t max_picks{1000000};      ///< Complete selected-parent count; zero disables.
  std::size_t max_contestants{
      100000000}; ///< Complete n*k raw draws, checked before sampling; zero disables.
  std::size_t max_records{1000000}; ///< Actual captured contestant events; zero disables.
};
/// @brief Own TournamentStep values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TournamentStep r{};
/// (void)r;
/// @endcode
struct TournamentStep {
  std::size_t   pick{};      ///< Actual zero-based tournament/output slot.
  int           round{};     ///< Actual contestant position0..k-1.
  std::uint64_t raw_word{};  ///< Actual historical xorshift word before modulo reduction.
  std::size_t   candidate{}; ///< Actual raw_word modulo population size, with replacement.
  std::size_t
      before{}; ///< Actual current best before comparison; candidate itself for first contestant.
  std::size_t after{}; ///< Actual best after this contestant; first sampled tie retained.
};
/// @brief Own TournamentRun values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TournamentRun r{};
/// (void)r;
/// @endcode
struct TournamentRun {
  std::vector<std::size_t> indices; ///< Actual selected winner per tournament in output order.
  std::vector<std::size_t> counts;  ///< Actual winner count per population row.
  std::vector<std::size_t>
      contestant_counts; ///< Actual participation counts, including repeats within one tournament.
  std::size_t random_words{};        ///< Actual raw xorshift draws, exactly n*k for positive picks.
  std::size_t comparisons{};         ///< Actual strict fitness comparisons, n*(k-1).
  std::vector<TournamentStep> steps; ///< Optional actual producing draw/winner events.
};
/// @brief Select actual tournament winners with producer draw diagnostics.
/// Finite fitness values may be negative because only ordering matters. Require n>=0 and k>0; positive picks require nonempty population. Zero picks validates all fitness and k, returns zero row counts and consumes no RNG word. Each tournament samples k rows independently with replacement using the historical xorshift13/7/17 word modulo population; this finite modulo mapping has bias and is not an exactly uniform bounded draw. Keep the first sampled contender on equal fitness. Under ideal independent uniform draws and strictly ranked population m, rank-r winner probability is((r+1)/m)^k-(r/m)^k for ascending zero-based rank; qualify finite-RNG/modulo effects and ties. Complete n*k geometry checked before work, O(population+n*k) work and owned outputs/optional event history; no shared state.
/// @param fitness Complete borrowed finite population fitness, any sign.
/// @param n Nonnegative requested parent slots.
/// @param k Strictly positive contestants per tournament, with replacement.
/// @param seed Local historical xorshift seed, default1; zero uses nonzero fallback.
/// @param options Complete population/pick/draw and actual capture policies; zero disables.
/// @return Owned actual winners/counts, selected work and optional raw draw events.
/// @throws std::invalid_argument Nonfinite fitness, negative picks, nonpositive k or empty positive-pick population.
/// @throws std::length_error Complete geometry/work or actual capture/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::tournament_run({2},3,4);
/// std::print("{} {}\n",r.counts,r.random_words); // [3] 12
/// @endcode
/// @see tournament_sample()
TournamentRun tournament_run(
    const std::vector<double>& fitness,
    int                        n,
    int                        k,
    std::uint64_t              seed    = 1,
    const TournamentOptions&   options = {});
/// @brief Select actual tournament winners with producer draw diagnostics.
/// Finite fitness values may be negative because only ordering matters. Require n>=0 and k>0; positive picks require nonempty population. Zero picks validates all fitness and k, returns zero row counts and consumes no RNG word. Each tournament samples k rows independently with replacement using the historical xorshift13/7/17 word modulo population; this finite modulo mapping has bias and is not an exactly uniform bounded draw. Keep the first sampled contender on equal fitness. Under ideal independent uniform draws and strictly ranked population m, rank-r winner probability is((r+1)/m)^k-(r/m)^k for ascending zero-based rank; qualify finite-RNG/modulo effects and ties. Complete n*k geometry checked before work, O(population+n*k) work and owned outputs/optional event history; no shared state.
/// @param fitness Complete borrowed finite population fitness, any sign.
/// @param n Nonnegative requested parent slots.
/// @param k Strictly positive contestants per tournament, with replacement.
/// @param seed Local historical xorshift seed, default1; zero uses nonzero fallback.
/// @param options Complete population/pick/draw and actual capture policies; zero disables.
/// @return Owned actual winners/counts, selected work and optional raw draw events.
/// @throws std::invalid_argument Nonfinite fitness, negative picks, nonpositive k or empty positive-pick population.
/// @throws std::length_error Complete geometry/work or actual capture/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::tournament_trace({2},3,4);
/// std::print("{} {}\n",r.counts,r.random_words); // [3] 12
/// @endcode
/// @see tournament_sample()
TournamentRun tournament_trace(
    const std::vector<double>& fitness,
    int                        n,
    int                        k,
    std::uint64_t              seed    = 1,
    const TournamentOptions&   options = {});
/// @brief Preserve historical tournament winner streams with checked complete inputs.
/// Finite fitness values may be negative because only ordering matters. Require n>=0 and k>0; positive picks require nonempty population. Zero picks validates all fitness and k, returns zero row counts and consumes no RNG word. Each tournament samples k rows independently with replacement using the historical xorshift13/7/17 word modulo population; this finite modulo mapping has bias and is not an exactly uniform bounded draw. Keep the first sampled contender on equal fitness. Under ideal independent uniform draws and strictly ranked population m, rank-r winner probability is((r+1)/m)^k-(r/m)^k for ascending zero-based rank; qualify finite-RNG/modulo effects and ties. Complete n*k geometry checked before work, O(population+n*k) work and owned outputs/optional event history; no shared state.
/// @param fitness Complete borrowed finite population fitness, any sign.
/// @param n Nonnegative requested parent slots.
/// @param k Strictly positive contestants per tournament, with replacement.
/// @param seed Local historical xorshift seed, default1; zero uses nonzero fallback.
/// @return Owned actual winning row indices.
/// @throws std::invalid_argument Malformed complete fitness/counts.
/// @throws std::length_error Default complete work/geometry exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::tournament_sample({2},3,4)); // [0, 0, 0]
/// @endcode
/// @see tournament_run()
std::vector<std::size_t>
tournament_sample(const std::vector<double>& fitness, int n, int k, std::uint64_t seed = 1);
} // namespace datamunge::algorithms
