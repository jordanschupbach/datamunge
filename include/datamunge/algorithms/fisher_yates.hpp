/// @file
/// @brief Seeded Fisher–Yates, owned actual swap evidence and bounded exact shuffle censuses.
#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

/// @brief Canonical numerical and combinatorial algorithms.
namespace datamunge::algorithms {
/// @brief Owned seeded shuffle output with actual swap-stage and raw-generator work counts.
/// stages counts n-1 swap attempts (zero for empty/singleton), including self-swaps;
/// nontrivial_swaps counts i!=j, even when the two values are equal. random_words
/// includes rejected range-sampler words. Copying owns the values; no shared state.
/// @par Example
/// @code
/// const auto r=dm::algorithms::fisher_yates_shuffle_work({7,7},42);
/// std::cout << r.values.size() << ' ' << r.stages; // 2 1
/// @endcode
/// @see fisher_yates_shuffle_work(), ShuffleTrace
struct ShuffleOutcome {
    std::vector<int> values; ///< Owned permuted values; duplicates and signed extremes preserved.
    std::size_t stages=0; ///< Actual swap attempts, including i==j.
    std::size_t nontrivial_swaps=0; ///< Attempts with different positions, irrespective of value equality.
    std::uint64_t random_words=0; ///< Actual next_u64 calls, including range rejections.
};

/// @brief One actual descending Fisher–Yates swap, with values read before mutation.
/// Stores no whole-array snapshot or post-hoc replay; i is finalized by this stage.
/// @par Example
/// @code
/// const auto t=dm::algorithms::fisher_yates_shuffle_trace({1,2,3},42);
/// std::cout << t.steps[0].i << ' ' << t.steps.size(); // 2 2
/// @endcode
/// @see fisher_yates_shuffle_trace()
struct ShuffleStep {
    std::size_t i=0; ///< Position finalized now, descending from n-1 through 1.
    std::size_t j=0; ///< Actual accepted inclusive [0,i] draw.
    int at_i=0; ///< Value at i immediately before this swap.
    int at_j=0; ///< Value at j immediately before this swap.
    std::uint64_t random_words=0; ///< Raw words consumed for this stage, including rejections.
};

/// @brief Owned initial input, seeded output and actual ordered swap records.
/// Complete capture units are C=2*n+max(n-1,0): two value arrays plus one entry
/// per stage. This counts entries, not bytes or peak RAM. No replay or shared state.
/// @par Example
/// @code
/// const auto t=dm::algorithms::fisher_yates_shuffle_trace({1,2,3},42,8);
/// std::cout << t.input.size() << ' ' << t.steps.size(); // 3 2
/// @endcode
/// @see fisher_yates_shuffle_trace(), ShuffleOutcome, ShuffleStep
struct ShuffleTrace {
    std::vector<int> input; ///< Owned original value order.
    ShuffleOutcome outcome; ///< Owned final values and actual work counters.
    std::vector<ShuffleStep> steps; ///< Actual stages in descending i order.
};

/// @brief Exact choice-tree models for permutation-law comparisons, without a random generator.
/// FisherYates is the descending shrinking-range algorithm. AllPositions is the
/// deliberately biased model: visit i=0..n-1 and independently choose j=0..n-1.
/// CyclicShift enumerates only the n rotations (one empty output when n=0).
/// @par Example
/// @code
/// const auto c=dm::algorithms::shuffle_choice_census(3,dm::algorithms::ShuffleCensusRule::CyclicShift);
/// std::cout << c.outcomes << ' ' << c.permutation_counts.size(); // 3 3
/// @endcode
/// @see shuffle_choice_census()
enum class ShuffleCensusRule {
    FisherYates, ///< Every shrinking-range choice sequence, with n! outcomes.
    AllPositions, ///< Every n-step full-range choice sequence, with n^n outcomes (one at zero).
    CyclicShift ///< Every rotation, with max(n,1) outcomes; uniform position marginals for n>0.
};

/// @brief Owned complete integer counts of labelled permutations and all position marginals.
/// Labels are the original positions 0..n-1, so repeated caller values cannot
/// merge distinguishable labels. by_position[p][v] counts original label v at
/// output position p. permutation_counts contains only observed outputs in
/// lexicographic order. Empty trials yield an empty map and zero marginal counts.
/// @par Example
/// @code
/// const auto c=dm::algorithms::shuffle_choice_census(3);
/// std::cout << c.outcomes << ' ' << c.by_position[0][0]; // 6 2
/// @endcode
/// @see fisher_yates_seed_census(), shuffle_choice_census()
struct ShuffleCensus {
    std::uint64_t outcomes=0; ///< Number of complete model choices or seeded executions counted.
    std::size_t work_units=0; ///< Checked n*n+outcomes*(n+1) census budget units.
    std::vector<std::vector<std::uint64_t>> by_position; ///< Owned n-by-n output-position/label counts.
    std::map<std::vector<int>,std::uint64_t> permutation_counts; ///< Owned observed labelled permutations and multiplicities.
};

/// @brief Return an owned permutation using descending Fisher–Yates and the canonical SplitMix64 sampler.
/// Preserves the seeded output convention: rng.uniform_u64(0,i) for i=n-1..1.
/// The sampler uses multiply-and-reject, not modulo reduction. The ideal uniform
/// permutation theorem assumes independent uniform bounded draws; a fixed uint64
/// seed gives a deterministic finite-state experiment, not that theorem's premise.
/// No cryptographic unpredictability is promised. Duplicates are preserved; empty
/// and singleton inputs consume no random words. Input is not mutated; separate
/// calls share no state. O(n) copying/swaps and O(n) owned storage, plus rejection
/// sampler retries (expected O(n) under ideal raw words, no fixed retry bound).
/// @param items Input values in any order; signed extremes, duplicates and empty input are valid.
/// @param seed Any uint64 seed; equal input and seed reproduce the same output.
/// @return Owned shuffled values, including the empty or singleton input unchanged.
/// @throws std::overflow_error If the actual raw-word counter exceeds uint64 storage.
/// @throws std::bad_alloc If owned numeric storage cannot be allocated.
/// @par Example
/// @code
/// const auto a=dm::algorithms::fisher_yates_shuffle({1,2,3,4,5,6},42);
/// std::cout << a[0] << ' ' << a[5]; // 3 5
/// @endcode
/// @par References
/// - R. Durstenfeld (1964), Algorithm 235: Random permutation, CACM 7(7),420,
///   https://doi.org/10.1145/364520.364540 .
/// - P. E. Black, Fisher-Yates shuffle, NIST DADS,
///   https://xlinux.nist.gov/dads/HTML/fisherYatesShuffle.html .
/// @see fisher_yates_shuffle_work(), fisher_yates_shuffle_trace()
std::vector<int> fisher_yates_shuffle(const std::vector<int>& items,std::uint64_t seed);

/// @brief Run the same seeded shuffle and count actual swap attempts and raw generator calls.
/// Shares the ordinary kernel and canonical range sampler, without history or replay.
/// O(n) output storage, O(n) copying/swaps plus retries; source values are unmodified.
/// Counts are algorithmic work, not runtime measurements or a uniformity certificate.
/// @param items Any input values, including empty/duplicates.
/// @param seed Any uint64 seed.
/// @return Owned shuffled values and actual work counters.
/// @throws std::overflow_error If the raw-word counter would exceed uint64 storage.
/// @throws std::bad_alloc If owned numeric storage cannot be allocated.
/// @par Example
/// @code
/// const auto r=dm::algorithms::fisher_yates_shuffle_work({1,2,3},42);
/// std::cout << r.stages << ' ' << r.random_words; // 2 2
/// @endcode
/// @par References
/// - R. Durstenfeld (1964), Algorithm 235: Random permutation, CACM 7(7),420,
///   https://doi.org/10.1145/364520.364540 .
/// - P. E. Black, Fisher-Yates shuffle, NIST DADS,
///   https://xlinux.nist.gov/dads/HTML/fisherYatesShuffle.html .
/// @see fisher_yates_shuffle(), ShuffleOutcome
ShuffleOutcome fisher_yates_shuffle_work(const std::vector<int>& items,std::uint64_t seed);

/// @brief Capture one seeded shuffle's actual input, output, pre-swap values and accepted choices.
/// Checks the whole C=2*n+max(n-1,0) capture before copying input or drawing randomness.
/// O(n) stored entries and copying/swaps plus sampler retries; owned records are
/// independent of later caller edits. No partial trace is returned on failure.
/// @param items Any input values, including empty/duplicates.
/// @param seed Any uint64 seed.
/// @param max_entries Whole capture cap, default 1000000; zero disables only this optional cap.
/// @return Owned actual swap trace with the same output and work as an ordinary run.
/// @throws std::length_error If capture accounting/capacities overflow or the complete cap is exceeded.
/// @throws std::overflow_error If the raw-word counter would overflow.
/// @throws std::bad_alloc If owned numeric storage cannot be allocated.
/// @par Example
/// @code
/// const auto t=dm::algorithms::fisher_yates_shuffle_trace({1,2,3},42);
/// std::cout << t.steps.size() << ' ' << t.outcome.stages; // 2 2
/// @endcode
/// @par References
/// - R. Durstenfeld (1964), Algorithm 235: Random permutation, CACM 7(7),420,
///   https://doi.org/10.1145/364520.364540 .
/// - P. E. Black, Fisher-Yates shuffle, NIST DADS,
///   https://xlinux.nist.gov/dads/HTML/fisherYatesShuffle.html .
/// @see ShuffleTrace, fisher_yates_shuffle_work()
ShuffleTrace fisher_yates_shuffle_trace(const std::vector<int>& items,std::uint64_t seed,std::size_t max_entries=1'000'000);

/// @brief Apply one completely validated descending Fisher–Yates choice sequence without a generator.
/// The choices correspond in order to i=n-1..1 and each must lie in [0,i].
/// Validates all choices before copying/mutation. Shares the seeded swap kernel;
/// O(n) work/storage, caller inputs unchanged and no shared state.
/// @param items Any values, including empty/duplicates.
/// @param choices Exactly max(n-1,0) valid indices, in descending-i stage order.
/// @return Owned permutation for these explicit choices.
/// @throws std::invalid_argument If the choice count or any accepted index is invalid.
/// @throws std::bad_alloc If owned numeric storage cannot be allocated.
/// @par Example
/// @code
/// const auto a=dm::algorithms::fisher_yates_from_choices({1,2,3},{0,0});
/// std::cout << a[0] << ' ' << a[1] << ' ' << a[2]; // 2 3 1
/// @endcode
/// @par References
/// - R. Durstenfeld (1964), Algorithm 235: Random permutation, CACM 7(7),420,
///   https://doi.org/10.1145/364520.364540 .
/// - P. E. Black, Fisher-Yates shuffle, NIST DADS,
///   https://xlinux.nist.gov/dads/HTML/fisherYatesShuffle.html .
/// @see shuffle_choice_census(), fisher_yates_shuffle()
std::vector<int> fisher_yates_from_choices(const std::vector<int>& items,const std::vector<std::size_t>& choices);

/// @brief Count whole labelled permutations and every position marginal over a contiguous seed interval.
/// Uses the same seeded kernel, once per seed; this is deterministic enumeration,
/// not independent random sampling. Zero trials are allowed, with empty permutation
/// counts; n=0 still counts one empty output per trial. No seed wrapping is allowed.
/// Checks C=n*n+trials*(n+1) before allocation or any shuffle. Matrix setup is O(n²);
/// map updates add O(log distinct_outputs) comparisons of O(n)-length keys per run,
/// plus O(n) shuffle work and actual rejection retries. Storage is bounded by C
/// entries up to map/container overhead; the cap is not a byte or peak-RAM bound.
/// @param n Number of distinct original-position labels, from zero through INT_MAX.
/// @param first_seed First uint64 seed in the nonwrapping interval.
/// @param trials Number of consecutive seeds; zero means no executions.
/// @param max_work Complete census entry/work cap, default 1000000; zero disables only this cap.
/// @return Owned exact seeded whole-output and marginal counts with admitted work units.
/// @throws std::invalid_argument If the seed interval wraps.
/// @throws std::length_error If labels exceed INT_MAX, complete accounting/capacities overflow or the cap is exceeded.
/// @throws std::bad_alloc If owned numeric storage cannot be allocated.
/// @throws std::overflow_error If a per-shuffle raw-word counter would overflow.
/// @par Example
/// @code
/// const auto c=dm::algorithms::fisher_yates_seed_census(2,0,10);
/// std::cout << c.outcomes << ' ' << c.work_units; // 10 34
/// @endcode
/// @par References
/// - R. Durstenfeld (1964), Algorithm 235: Random permutation, CACM 7(7),420,
///   https://doi.org/10.1145/364520.364540 .
/// - P. E. Black, Fisher-Yates shuffle, NIST DADS,
///   https://xlinux.nist.gov/dads/HTML/fisherYatesShuffle.html .
/// @see ShuffleCensus, shuffle_choice_census()
ShuffleCensus fisher_yates_seed_census(std::size_t n,std::uint64_t first_seed,std::uint64_t trials,std::size_t max_work=1'000'000);

/// @brief Enumerate every ideal finite choice sequence for a shuffle model, retaining exact counts.
/// No pseudorandom generator is used. FisherYates has n! equally weighted choices;
/// AllPositions has n^n ascending full-range choices; CyclicShift has n rotations.
/// At n=0 every model has one empty outcome. Models do not assert actual seeded
/// uniformity. Checks the whole C=n*n+outcomes*(n+1) before allocation/enumeration.
/// Map/copy/storage costs follow fisher_yates_seed_census(), without random retries;
/// complete count overflow and label/container limits still apply with the cap off.
/// @param n Number of distinct labels 0..n-1, at most INT_MAX.
/// @param rule Exact model, default FisherYates; invalid enum values are rejected.
/// @param max_work Complete census cap, default 1000000; zero disables only this optional cap.
/// @return Owned exact whole-permutation and position-marginal multiplicities.
/// @throws std::invalid_argument If the rule value is invalid.
/// @throws std::length_error If labels exceed INT_MAX, complete accounting/capacities overflow or the cap is exceeded.
/// @throws std::bad_alloc If owned numeric storage cannot be allocated.
/// @par Example
/// @code
/// const auto c=dm::algorithms::shuffle_choice_census(3,dm::algorithms::ShuffleCensusRule::AllPositions);
/// std::cout << c.outcomes << ' ' << c.permutation_counts.size(); // 27 6
/// @endcode
/// @par References
/// - R. Durstenfeld (1964), Algorithm 235: Random permutation, CACM 7(7),420,
///   https://doi.org/10.1145/364520.364540 .
/// - P. E. Black, Fisher-Yates shuffle, NIST DADS,
///   https://xlinux.nist.gov/dads/HTML/fisherYatesShuffle.html .
/// @see ShuffleCensusRule, fisher_yates_from_choices(), fisher_yates_seed_census()
ShuffleCensus shuffle_choice_census(std::size_t n,ShuffleCensusRule rule=ShuffleCensusRule::FisherYates,std::size_t max_work=1'000'000);

} // namespace datamunge::algorithms
