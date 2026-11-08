/// @file
/// @brief Shared actual Fisher–Yates swaps, canonical sampler accounting and complete bounded
/// censuses.
#include <algorithm>
#include <climits>
#include <datamunge/algorithms/fisher_yates.hpp>
#include <datamunge/random/random.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

/// @brief Canonical numerical and combinatorial algorithms.
namespace datamunge::algorithms {
/// @brief Checked accounting and shared actual shuffle transitions.
namespace {
/// @brief Canonical SplitMix64 stream with counted virtual raw-word requests.
class ShuffleEngine : public datamunge::random::SplitMix64 {
 public:
  /// @brief Initialise the canonical stream without consuming a word.
  /// @param seed Initial uint64 state.
  explicit ShuffleEngine(std::uint64_t seed) : SplitMix64(seed) {}
  /// @brief Advance exactly once, rejecting raw-counter overflow before stream mutation.
  /// @return The same next raw word as SplitMix64.
  /// @throws std::overflow_error If the next count cannot fit uint64.
  std::uint64_t next_u64() override {
    if (words == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("shuffle raw-word count overflow");
    ++words;
    return SplitMix64::next_u64();
  }
  std::uint64_t words = 0; ///< Actual raw words consumed, including rejection retries.
};
/// @brief One accepted position and actual raw sampler work used to select it.
struct ShuffleSelection {
  std::size_t   j;     ///< Accepted index in the current prefix.
  std::uint64_t words; ///< Actual raw draws, zero for explicit deterministic choices.
};
/// @brief Shared copy-and-swap loop with optional actual stage recording.
/// @tparam Select Callable returning a valid ShuffleSelection for the supplied final position.
/// @param items Original values, never modified.
/// @param select Actual accepted-position source, called once per descending stage.
/// @param steps Optional pre-admitted owned record destination.
/// @return Owned output and actual stage/raw-word counts.
/// @throws std::bad_alloc If values or records cannot be allocated.
/// @throws std::overflow_error If a source counter exceeds its format.
template<class Select>
ShuffleOutcome
shuffle_kernel(const std::vector<int>& items, Select& select, std::vector<ShuffleStep>* steps) {
  ShuffleOutcome out;
  out.values = items;
  for (std::size_t remaining = items.size(); remaining > 1; --remaining) {
    const auto i      = remaining - 1;
    const auto choice = select(i);
    if (steps)
      steps->push_back({i, choice.j, out.values[i], out.values[choice.j], choice.words});
    std::swap(out.values[i], out.values[choice.j]);
    ++out.stages;
    if (i != choice.j)
      ++out.nontrivial_swaps;
    if (choice.words > std::numeric_limits<std::uint64_t>::max() - out.random_words)
      throw std::overflow_error("shuffle word count overflow");
    out.random_words += choice.words;
  }
  return out;
}
/// @brief Run the actual seeded loop with the existing canonical rejection sampler.
/// @param items Input values.
/// @param seed Canonical SplitMix64 initial state.
/// @param steps Optional admitted actual-stage output.
/// @return Owned seeded output and measured primitive work.
/// @throws std::bad_alloc If owned arrays cannot be allocated.
/// @throws std::overflow_error If raw generator work cannot fit uint64.
ShuffleOutcome seeded_shuffle(
    const std::vector<int>&   items,
    std::uint64_t             seed,
    std::vector<ShuffleStep>* steps = nullptr) {
  ShuffleEngine engine(seed);
  /// @brief Obtain a canonical accepted bounded draw and its actual raw-word delta.
  auto select = [&](std::size_t i) {
    const auto before = engine.words;
    const auto j      = engine.uniform_u64(0, static_cast<std::uint64_t>(i));
    return ShuffleSelection{static_cast<std::size_t>(j), engine.words - before};
  };
  return shuffle_kernel(items, select, steps);
}
/// @brief Check a whole entry/work count and optional cap without allocating numeric output.
/// @param count Exact wide count.
/// @param cap Optional maximum, zero disables it.
/// @return Count representable as size_t.
/// @throws std::length_error For size_t overflow or exceeded nonzero cap.
std::size_t shuffle_budget(unsigned __int128 count, std::size_t cap) {
  if (count > std::numeric_limits<std::size_t>::max() || (cap && count > cap))
    throw std::length_error("complete shuffle budget exceeded");
  return static_cast<std::size_t>(count);
}
/// @brief Require a label domain representable by the legacy integer value API.
/// @param n Number of original-position labels.
/// @throws std::length_error If labels or vector capacities cannot fit.
void census_domain(std::size_t n) {
  if (n > static_cast<std::size_t>(INT_MAX) || n > std::vector<int>{}.max_size() ||
      n > std::vector<std::uint64_t>{}.max_size() ||
      n > std::vector<std::vector<std::uint64_t>>{}.max_size())
    throw std::length_error("shuffle census label capacity");
}
/// @brief Admit the complete matrix plus execution accounting before allocation.
/// @param n Number of distinguishable labels.
/// @param outcomes Complete execution count.
/// @param cap Optional whole C=n²+outcomes*(n+1) budget.
/// @return Owned zero counts with admitted metadata.
/// @throws std::length_error If labels, accounting or cap cannot fit.
/// @throws std::bad_alloc If the count matrix cannot be allocated.
ShuffleCensus census_start(std::size_t n, std::uint64_t outcomes, std::size_t cap) {
  census_domain(n);
  const auto units = shuffle_budget(
      static_cast<unsigned __int128>(n) * n + static_cast<unsigned __int128>(outcomes) * (n + 1),
      cap);
  ShuffleCensus out;
  out.outcomes   = outcomes;
  out.work_units = units;
  out.by_position.assign(n, std::vector<std::uint64_t>(n));
  return out;
}
/// @brief Record one complete labelled output in a pre-admitted census.
/// @param census Mutable owned aggregate, hidden until all executions succeed.
/// @param values Complete labelled permutation, moved into a new map key when needed.
/// @throws std::bad_alloc If a new output map key cannot be allocated.
void census_record(ShuffleCensus& census, std::vector<int> values) {
  for (std::size_t p = 0; p < values.size(); ++p)
    ++census.by_position[p][static_cast<std::size_t>(values[p])];
  ++census.permutation_counts[std::move(values)];
}
/// @brief Compute a checked complete finite model outcome count without enumeration.
/// @param n Label count already in the integer domain.
/// @param rule Valid model selector, checked here.
/// @return n!, n^n or max(n,1) for the selected exact choice model.
/// @throws std::invalid_argument If rule is not a defined enum value.
/// @throws std::length_error If the complete count cannot fit uint64.
std::uint64_t choice_outcomes(std::size_t n, ShuffleCensusRule rule) {
  if (rule == ShuffleCensusRule::CyclicShift)
    return n ? n : 1;
  if (rule != ShuffleCensusRule::FisherYates && rule != ShuffleCensusRule::AllPositions)
    throw std::invalid_argument("unknown shuffle census rule");
  std::uint64_t count = 1;
  for (std::size_t i = 0; i < n; ++i) {
    const auto base = rule == ShuffleCensusRule::FisherYates ? i + 1 : n;
    const auto next = static_cast<unsigned __int128>(count) * base;
    if (next > std::numeric_limits<std::uint64_t>::max())
      throw std::length_error("shuffle choice count overflow");
    count = static_cast<std::uint64_t>(next);
  }
  return count;
}
} // namespace

std::vector<int> fisher_yates_shuffle(const std::vector<int>& items, std::uint64_t seed) {
  return seeded_shuffle(items, seed).values;
}
ShuffleOutcome fisher_yates_shuffle_work(const std::vector<int>& items, std::uint64_t seed) {
  return seeded_shuffle(items, seed);
}
ShuffleTrace fisher_yates_shuffle_trace(
    const std::vector<int>& items,
    std::uint64_t           seed,
    std::size_t             max_entries) {
  const auto n = items.size(), stages = n ? n - 1 : 0;
  shuffle_budget(static_cast<unsigned __int128>(n) * 2 + stages, max_entries);
  if (stages > std::vector<ShuffleStep>{}.max_size())
    throw std::length_error("shuffle stage capacity");
  ShuffleTrace out;
  out.input = items;
  out.steps.reserve(stages);
  out.outcome = seeded_shuffle(items, seed, &out.steps);
  return out;
}
std::vector<int> fisher_yates_from_choices(
    const std::vector<int>&         items,
    const std::vector<std::size_t>& choices) {
  const auto n = items.size();
  if (choices.size() != (n ? n - 1 : 0))
    throw std::invalid_argument("shuffle choice count");
  for (std::size_t k = 0; k < choices.size(); ++k)
    if (choices[k] >= n - k)
      throw std::invalid_argument("shuffle choice outside prefix");
  std::size_t step = 0;
  /// @brief Return each already validated deterministic choice exactly once.
  auto select = [&](std::size_t) { return ShuffleSelection{choices[step++], 0}; };
  return shuffle_kernel(items, select, nullptr).values;
}
ShuffleCensus fisher_yates_seed_census(
    std::size_t   n,
    std::uint64_t first_seed,
    std::uint64_t trials,
    std::size_t   max_work) {
  if (trials && first_seed > std::numeric_limits<std::uint64_t>::max() - (trials - 1))
    throw std::invalid_argument("shuffle seed interval wraps");
  auto             out = census_start(n, trials, max_work);
  std::vector<int> labels(n);
  std::iota(labels.begin(), labels.end(), 0);
  for (std::uint64_t k = 0; k < trials; ++k)
    census_record(out, seeded_shuffle(labels, first_seed + k).values);
  return out;
}
ShuffleCensus shuffle_choice_census(std::size_t n, ShuffleCensusRule rule, std::size_t max_work) {
  census_domain(n);
  const auto       outcomes = choice_outcomes(n, rule);
  auto             out      = census_start(n, outcomes, max_work);
  std::vector<int> labels(n);
  std::iota(labels.begin(), labels.end(), 0);
  for (std::uint64_t rank = 0; rank < outcomes; ++rank) {
    if (rule == ShuffleCensusRule::FisherYates) {
      auto digits = rank;
      /// @brief Decode each exact mixed-radix accepted index for the current complete choice.
      auto select = [&](std::size_t i) {
        const auto j = digits % (i + 1);
        digits /= i + 1;
        return ShuffleSelection{static_cast<std::size_t>(j), 0};
      };
      census_record(out, shuffle_kernel(labels, select, nullptr).values);
    } else {
      auto values = labels;
      if (rule == ShuffleCensusRule::AllPositions) {
        auto digits = rank;
        for (std::size_t i = 0; i < n; ++i) {
          const auto j = digits % n;
          digits /= n;
          std::swap(values[i], values[static_cast<std::size_t>(j)]);
        }
      } else if (n)
        std::rotate(
            values.begin(), values.begin() + static_cast<std::ptrdiff_t>(rank), values.end());
      census_record(out, std::move(values));
    }
  }
  return out;
}
} // namespace datamunge::algorithms
