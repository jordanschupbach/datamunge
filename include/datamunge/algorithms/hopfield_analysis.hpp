#pragma once
/// @file hopfield_analysis.hpp
/// @brief Bounded original Hopfield storage/noise experiment and native bipolar state views.
#include <datamunge/algorithms/hopfield_network.hpp>
#include <datamunge/plot/plot.hpp>
#include <string>
namespace datamunge::algorithms {
/// @brief Owned actual corruption/recall trial from the original fixed-size noise study.
/// The noise parameter is a per-bit selection percentage, not an exact Hamming
/// distance. Full input/output states and the actual seeded run remain available.
/// @par Example
/// @code
/// const dm::algorithms::HopfieldNoiseTrial t;std::print("{}\n",t.initial_distance); // 0
/// @endcode
struct HopfieldNoiseTrial {
  int           noise_percent{0}; ///< Per-position selection threshold, from0 through50 in steps5.
  std::uint64_t recall_seed{0};   ///< Actual MT19937-64 recall shuffle seed, repeat index+1.
  std::vector<int>  probe;        ///< Owned actual independently corrupted pattern0.
  HopfieldRecallRun recall;       ///< Actual producing recall with owned final state/stop/work.
  std::size_t       initial_distance{0}; ///< Actual Hamming distance from pattern0 before recall.
  std::size_t       final_distance{0};   ///< Actual Hamming distance from pattern0 after recall.
};
/// @brief Finite-trial observed noise summary, with bit agreement distinct from exact recovery.
/// No population accuracy or confidence claim. Each trial has100 neurons; zero
/// trials is a default-construction sentinel, not produced by the study helper.
/// @par Example
/// @code
/// const dm::algorithms::HopfieldNoiseSummary s;std::print("{}\n",s.trials); // 0
/// @endcode
struct HopfieldNoiseSummary {
  int         noise_percent{0};         ///< Nominal independent per-position flip percentage.
  std::size_t trials{0};                ///< Actual complete recall calls at this level.
  std::size_t exact_recoveries{0};      ///< Actual outputs exactly equal to pattern0.
  std::size_t no_change_stops{0};       ///< Actual runs ending with an unchanged sweep.
  double      mean_initial_distance{0}; ///< Mean actual input Hamming distance, in neurons.
  double      bit_agreement{
      0}; ///< Mean (100-final_distance)/100, preserving original accumulation order.
  double exact_recovery_fraction{0}; ///< exact_recoveries/trials, distinct from bit agreement.
};
/// @brief Owned public original100-neuron/three-memory fixture and actual complete noise study.
/// Retains three extra random states before ten with-replacement probe toggles,
/// preserving the original data RNG stream. Those ten draws need not change ten
/// distinct positions. Stored weights, actual trace/trials and summaries are owned.
/// @par Example
/// @code
/// const dm::algorithms::HopfieldExperiment e;std::print("{}\n",e.memories.size()); // 0
/// @endcode
struct HopfieldExperiment {
  std::vector<std::vector<int>> memories; ///< Three actual100-neuron Bernoulli bipolar memories.
  HopfieldNetwork network; ///< Actual checked normalized Hebbian matrix for those memories.
  std::vector<std::vector<int>>
      random_states;      ///< Three actual independent comparison states drawn before the probe.
  std::vector<int> probe; ///< Actual pattern0 after ten with-replacement uniform-position toggles.
  HopfieldRecallTrace probe_trace; ///< Actual seed1/100-sweep recall of that probe, no replay.
  std::vector<HopfieldNoiseTrial>
      trials; ///< Actual level/repeat order, preserving each probe and result.
  std::vector<HopfieldNoiseSummary>
      summaries; ///< Eleven actual level summaries in ascending noise order.
};
/// @brief Run the original seeded100-neuron/three-memory experiment with complete trial admission.
/// Uses MT19937-64(data_seed), std::bernoulli_distribution(.5) for three memories
/// then three comparison states, then ten uniform0..99 toggle draws. Each noise
/// level0,5,..50 creates new probes by testing uniform0..99 modulo100 at every
/// neuron. Recall uses100 sweeps and seeds1..trials_per_level at every level.
/// Standard-library distribution/shuffle mappings need not agree across vendors.
/// Validates/admit full11*T calls before RNG work, training or result allocation.
/// Fixed N=100; O(T*N*N*S+N*N) work and O(T*N+N*N+S*N) owned/work storage,
/// where actual S<=100. Original storage arithmetic and noise accumulation retained.
/// @param trials_per_level Positive complete repeats per noise level, default30.
/// @param data_seed Dataset/probe RNG seed, default20240730.
/// @param max_trials Complete11*T trial cap, default10000; zero disables policy.
/// @return Owned complete fixed-size experiment, actual probes/runs/trace and summaries.
/// @throws std::invalid_argument If trials_per_level is zero.
/// @throws std::length_error On complete trial/count/container limits.
/// @throws std::overflow_error On nonfinite producing arithmetic.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto e=dm::algorithms::hopfield_original_experiment(1);
/// std::print("{} {} {}\n",e.network.size(),e.trials.size(),e.summaries.size()); // 100 11 11
/// @endcode
/// @see hopfield_train_checked(), hopfield_recall_trace(), HopfieldExperiment
inline HopfieldExperiment hopfield_original_experiment(
    std::size_t   trials_per_level = 30,
    std::uint64_t data_seed        = 20240730,
    std::size_t   max_trials       = 10000) {
  if (trials_per_level == 0)
    throw std::invalid_argument("Hopfield study requires positive repeats");
  if (trials_per_level > std::numeric_limits<std::size_t>::max() / 11)
    throw std::length_error("Hopfield study count overflow");
  const auto total = 11 * trials_per_level;
  if ((max_trials && total > max_trials) || total > std::vector<HopfieldNoiseTrial>{}.max_size())
    throw std::length_error("Hopfield study complete trial cap");
  HopfieldExperiment          e;
  std::mt19937_64             rng(data_seed);
  std::bernoulli_distribution bit(.5);
  e.memories.assign(3, std::vector<int>(100));
  for (auto& p : e.memories)
    for (auto& v : p)
      v = bit(rng) ? 1 : -1;
  e.network = hopfield_train_checked(e.memories);
  e.random_states.assign(3, std::vector<int>(100));
  for (auto& p : e.random_states)
    for (auto& v : p)
      v = bit(rng) ? 1 : -1;
  std::uniform_int_distribution<std::size_t> position(0, 99);
  e.probe = e.memories[0];
  for (unsigned i = 0; i < 10; ++i)
    e.probe[position(rng)] *= -1;
  e.probe_trace = hopfield_recall_trace(e.network, e.probe, 100, 1);
  e.trials.reserve(total);
  e.summaries.reserve(11);
  for (int pct = 0; pct <= 50; pct += 5) {
    HopfieldNoiseSummary summary;
    summary.noise_percent = pct;
    summary.trials        = trials_per_level;
    double correct = 0, initial_sum = 0;
    for (std::size_t t = 0; t < trials_per_level; ++t) {
      HopfieldNoiseTrial trial;
      trial.noise_percent = pct;
      trial.recall_seed   = static_cast<std::uint64_t>(t + 1);
      trial.probe         = e.memories[0];
      for (std::size_t i = 0; i < 100; ++i)
        if (static_cast<int>(position(rng) % 100) < pct)
          trial.probe[i] *= -1;
      trial.initial_distance = hopfield_hamming(trial.probe, e.memories[0]);
      trial.recall           = hopfield_recall_run(e.network, trial.probe, 100, trial.recall_seed);
      trial.final_distance   = hopfield_hamming(trial.recall.state, e.memories[0]);
      initial_sum += static_cast<double>(trial.initial_distance);
      correct += static_cast<double>(100 - trial.final_distance) / 100.;
      summary.exact_recoveries += trial.final_distance == 0;
      summary.no_change_stops += trial.recall.stop == HopfieldStop::NoChange;
      e.trials.push_back(std::move(trial));
    }
    summary.mean_initial_distance = initial_sum / static_cast<double>(trials_per_level);
    summary.bit_agreement         = correct / static_cast<double>(trials_per_level);
    summary.exact_recovery_fraction =
        static_cast<double>(summary.exact_recoveries) / static_cast<double>(trials_per_level);
    e.summaries.push_back(summary);
  }
  return e;
}
/// @brief View complete bipolar rows as native fixed-palette cells, first row at the top.
/// Dark grey denotes-1, blue denotes+1; no independent autoscaling. Requires at
/// least two rows and two columns, complete rectangular bipolar validation before
/// display policy/grid allocation. Cells follow panel geometry, not physical
/// neuron positions. O(rows*columns+rows+columns) preparation/owned render storage.
/// @param states Borrowed complete equal-length bipolar rows, at least2-by2.
/// @param title Owned display title; callers label the row order in title/caption.
/// @param max_cells Complete display-cell cap, default16384; zero disables policy.
/// @return Owned native RPlot with one polygon series per actual sign value.
/// @throws std::invalid_argument On dimensions/row lengths/signs before display policy.
/// @throws std::length_error On count/display/container limits.
/// @throws std::bad_alloc On plot/grid allocation failure.
/// @par Example
/// @code
/// const auto p=dm::algorithms::hopfield_states_plot({{1,-1},{-1,1}},"bipolar");
/// std::print("{}\n",p.series().size()); // 4
/// @endcode
/// @see graphics::RPlot::image(), hopfield_original_experiment()
inline graphics::RPlot hopfield_states_plot(
    const std::vector<std::vector<int>>& states,
    std::string                          title,
    std::size_t                          max_cells = 16384) {
  const auto rows = states.size(), cols = rows ? states[0].size() : 0;
  for (const auto& row : states) {
    if (row.size() != cols)
      throw std::invalid_argument("Hopfield state view row lengths");
    hopfield_detail::signs(row);
  }
  if (rows < 2 || cols < 2)
    throw std::invalid_argument("Hopfield state view needs at least2x2");
  if (cols > std::numeric_limits<std::size_t>::max() / rows)
    throw std::length_error("Hopfield state view count overflow");
  if (max_cells && rows * cols > max_cells)
    throw std::length_error("Hopfield state view complete cap");
  std::vector<double> x(cols), y(rows);
  std::iota(x.begin(), x.end(), 0.);
  std::iota(y.begin(), y.end(), 0.);
  std::vector<std::vector<double>> z(cols, std::vector<double>(rows));
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < cols; ++j)
      z[j][rows - 1 - i] = states[i][j];
  auto p = graphics::RPlot::create();
  p.image(x, y, z, {{51, 65, 85}, {37, 99, 235}}, -1, 1).title(std::move(title)).hide_axes(true);
  return p;
}
} // namespace datamunge::algorithms
