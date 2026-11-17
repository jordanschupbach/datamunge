#pragma once
/// @file roulette_wheel.hpp
/// @brief Checked fitness-proportionate native sampling and actual bucket decisions.
#include <cstddef>
#include <cstdint>
#include <vector>
/// @brief Canonical evolutionary selection and owned stream records.
namespace datamunge::algorithms {
/// @brief Complete population, draw and actual-history bounds; zero disables a cap.
/// @par Example
/// @code
/// const dm::algorithms::RouletteOptions value{};
/// std::print("{}\n",value.max_draws); // 1000000
/// @endcode
struct RouletteOptions {
  std::size_t max_population{1000000};    ///< Complete fitness entry count.
  std::size_t max_draws{1000000};         ///< Actual requested sample count.
  std::size_t max_trace_entries{1000000}; ///< Actual retained draw records.
};
/// @brief One actual uniform threshold and strict-upper-CDF bucket search.
/// @par Example
/// @code
/// const dm::algorithms::RouletteDraw value{};
/// std::print("{}\n",value.comparisons); // 0
/// @endcode
struct RouletteDraw {
  std::uint64_t raw{};     ///< Actual xorshift output before its top53-bit conversion.
  double        uniform{}; ///< Actual represented uniform in [0,1).
  double        target{};  ///< Actual uniform times represented CDF total, clamped below the total.
  std::size_t   index{};   ///< Actual selected first strict-upper cumulative bucket.
  double        lower{};   ///< Actual cumulative mass preceding that bucket.
  double        upper{};   ///< Actual cumulative mass ending that bucket.
  std::size_t   comparisons{}; ///< Actual binary-search midpoint comparisons.
};
/// @brief Owned actual selection stream, represented model and selected search work.
/// @par Example
/// @code
/// const dm::algorithms::RouletteRun value{};
/// std::print("{}\n",value.counts.size()); // 0
/// @endcode
struct RouletteRun {
  std::vector<std::size_t> picks;      ///< Actual sampled original population ids.
  std::vector<std::size_t> counts;     ///< Actual per-id selection totals.
  std::vector<double>      cumulative; ///< Actual cumulative weights used by the producer.
  double                   total{};    ///< Actual represented positive CDF total.
  bool                     scaled{};   ///< Whether overflow-safe max-weight scaling was required.
  std::size_t comparisons{};           ///< Actual binary-search midpoint comparisons across draws.
  std::vector<RouletteDraw> draws;     ///< Actual optional raw/threshold/bucket records.
};
/// @brief Draw fitness-proportionate indices from the checked native stream.
/// Requires a nonempty finite nonnegative population with at least one positive weight, even for zero draws. Zeros have no selectable CDF interval. Preserves the historical unsigned xorshift13/7/17 stream, seed0 mapping and top53-bit uniform conversion. Uses original double cumulative addition when finite; if the positive sum would overflow, max-weight scaling precedes CDF formation. Consequently extreme subnormal masses can round away and receive no represented interval. Clamp rounded targets below the total to preserve a valid strict-upper bucket; no modulo selection is substituted. Selected comparisons count actual binary-search branches only. O(P+N log P) time, O(P+N) output plus optional N records. Inputs unchanged and records owned, independent calls share no state.
/// @param fitness Complete finite nonnegative weights.
/// @param n Requested actual draws, including zero.
/// @param seed Raw stream seed, default1; zero maps to historical constant.
/// @param options Complete population/draw/history policies.
/// @return Owned actual picks, counts, represented CDF and optional raw draw records.
/// @throws std::invalid_argument Empty, all-zero, negative or nonfinite model.
/// @throws std::length_error Input/draw/history or work-counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::roulette_wheel_run({0,2,0},4,1);
/// std::print("{}\n",r.counts[1]); // 4
/// @endcode
/// @par References
/// - Holland, Adaptation in Natural and Artificial Systems, https://mitpress.mit.edu/9780262581110/adaptation-in-natural-and-artificial-systems/.
/// @see roulette_wheel_sample()
RouletteRun roulette_wheel_run(
    const std::vector<double>& fitness,
    std::size_t                n,
    std::uint64_t              seed    = 1,
    const RouletteOptions&     options = {});
/// @brief Draw fitness-proportionate indices with actual bucket histories.
/// Requires a nonempty finite nonnegative population with at least one positive weight, even for zero draws. Zeros have no selectable CDF interval. Preserves the historical unsigned xorshift13/7/17 stream, seed0 mapping and top53-bit uniform conversion. Uses original double cumulative addition when finite; if the positive sum would overflow, max-weight scaling precedes CDF formation. Consequently extreme subnormal masses can round away and receive no represented interval. Clamp rounded targets below the total to preserve a valid strict-upper bucket; no modulo selection is substituted. Selected comparisons count actual binary-search branches only. O(P+N log P) time, O(P+N) output plus optional N records. Inputs unchanged and records owned, independent calls share no state.
/// @param fitness Complete finite nonnegative weights.
/// @param n Requested actual draws, including zero.
/// @param seed Raw stream seed, default1; zero maps to historical constant.
/// @param options Complete population/draw/history policies.
/// @return Owned actual picks, counts, represented CDF and optional raw draw records.
/// @throws std::invalid_argument Empty, all-zero, negative or nonfinite model.
/// @throws std::length_error Input/draw/history or work-counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::roulette_wheel_trace({0,2,0},4,1);
/// std::print("{}\n",r.counts[1]); // 4
/// @endcode
/// @par References
/// - Holland, Adaptation in Natural and Artificial Systems, https://mitpress.mit.edu/9780262581110/adaptation-in-natural-and-artificial-systems/.
/// @see roulette_wheel_sample()
RouletteRun roulette_wheel_trace(
    const std::vector<double>& fitness,
    std::size_t                n,
    std::uint64_t              seed    = 1,
    const RouletteOptions&     options = {});
/// @brief Return canonical roulette picks through the preserved signed-count signature.
/// Uses the same native stream and default million-entry/draw bounds. Negative counts now reject before allocation; zero draws still validate the complete nonempty model. Ordinary valid stream outputs are preserved.
/// @param fitness Complete nonnegative finite weights with positive sum.
/// @param n Nonnegative requested draw count.
/// @param seed Raw stream seed, default1.
/// @return Owned actual sampled original ids.
/// @throws std::invalid_argument Invalid model or negative count.
/// @throws std::length_error Default population/draw policy exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::roulette_wheel_sample({0,2},3).size()); // 3
/// @endcode
/// @par References
/// - Holland, Adaptation in Natural and Artificial Systems, https://mitpress.mit.edu/9780262581110/adaptation-in-natural-and-artificial-systems/.
/// @see roulette_wheel_run()
std::vector<std::size_t>
roulette_wheel_sample(const std::vector<double>& fitness, int n, std::uint64_t seed = 1);
} // namespace datamunge::algorithms
