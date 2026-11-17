#pragma once
/// @file hopfield_network.hpp
/// @brief Checked Hebbian bipolar memory, preserved asynchronous recall and actual update histories.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned symmetric zero-diagonal Hopfield weight matrix with its historical aggregate layout.
/// Nonempty models require square finite weights, exact symmetry and zero diagonal
/// when used by energy/recall. Hebbian storage does not guarantee every supplied
/// pattern is stable, nor recall of the nearest stored pattern. No references or
/// shared state are retained; do not mutate weights concurrently with a call.
/// @par Example
/// @code
/// const dm::algorithms::HopfieldNetwork net{{{0.,1.},{1.,0.}}};
/// std::print("{}\n",net.size()); // 2
/// @endcode
struct HopfieldNetwork {
  std::vector<std::vector<double>>
      weights; ///< Owned complete N-by-N finite symmetric weights, diagonal zero.

  /// @brief Return the stored row count without validating the matrix.
  /// @return Number of represented neurons; O(1), no mutation or allocation.
  std::size_t size() const { return weights.size(); }
};
/// @brief Complete neuron/weight and online logical-history caps, each zero disabling policy.
/// Input validation precedes these policies and internal working copies. History
/// units count records/state/index values, not bytes/capacity/peak memory.
/// @par Example
/// @code
/// const dm::algorithms::HopfieldOptions o;std::print("{}\n",o.max_weights); // 1000000
/// @endcode
struct HopfieldOptions {
  std::size_t max_neurons{10000};   ///< Complete matrix row-count cap.
  std::size_t max_weights{1000000}; ///< Complete N*N weight-entry cap, including diagonal.
  std::size_t max_trace_entries{
      1000000}; ///< Online logical history cap; ignored without capture or during storage.
};
/// @brief Distinguish an actual unchanged sweep from exhaustion of the requested budget.
/// A budget stop is not a fixed-point certificate; zero budget performs no updates.
/// @par Example
/// @code
/// std::print("{}\n",int(dm::algorithms::HopfieldStop::NoChange)); // 0
/// @endcode
enum class HopfieldStop {
  NoChange,  ///< A complete producing sweep changed no neuron.
  SweepLimit ///< Sweep budget exhausted, including a zero budget.
};
/// @brief Selected actual recall and separate diagnostic-energy visits.
/// Excludes validation, copies, RNG/shuffle, allocation and history bookkeeping.
/// Fields include all diagonal zero terms. Counts are not timing or total traffic.
/// @par Example
/// @code
/// const dm::algorithms::HopfieldWork w;std::print("{}\n",w.field_terms); // 0
/// @endcode
struct HopfieldWork {
  std::size_t field_terms{0};    ///< Actual weighted-state terms added to producing local fields.
  std::size_t neuron_updates{0}; ///< Actual neuron decisions, including unchanged signs.
  std::size_t flips{0};          ///< Actual decisions that changed a state sign.
  std::size_t diagnostic_energy_terms{
      0}; ///< Extra full-energy terms for initial/end-sweep trace observations; zero without trace.
};
/// @brief Owned actual final state, completed sweeps, stop branch and selected work.
/// No nearest-memory or stored-pattern guarantee. NoChange certifies only the
/// computed local-field update rule, not an exact-arithmetic energy certificate.
/// @par Example
/// @code
/// const dm::algorithms::HopfieldRecallRun r;std::print("{}\n",r.sweeps); // 0
/// @endcode
struct HopfieldRecallRun {
  std::vector<int>
      state; ///< Owned actual bipolar final state, unchanged initialization for zero budget.
  std::size_t  sweeps{0}; ///< Actual complete shuffled asynchronous sweeps applied.
  HopfieldStop stop{HopfieldStop::SweepLimit}; ///< Actual terminal branch.
  HopfieldWork work; ///< Selected producing visits and separate diagnostic visits.
};
/// @brief Actual local field and sign decision at its producing asynchronous update.
/// Later updates in the same sweep see this decision immediately. Exact zero,
/// including negative zero, chooses+1, preserving historical code behavior.
/// @par Example
/// @code
/// const dm::algorithms::HopfieldUpdate u;std::print("{}\n",u.field); // 0
/// @endcode
struct HopfieldUpdate {
  std::size_t neuron{0}; ///< Actual updated zero-based neuron label.
  double      field{0};  ///< Actual sequential double sum using the current in-place state.
  int         before{1}; ///< Actual sign immediately before this decision.
  int         after{1};  ///< Actual sign selected by field>=0 ? +1 : -1.
};
/// @brief Owned actual shuffled sweep and its end-state/energy observations.
/// Energy observations are extra sequential full-matrix calculations at actual
/// states, not replayed dynamics or exact/compensated arithmetic. They can round.
/// @par Example
/// @code
/// const dm::algorithms::HopfieldSweep s;std::print("{}\n",s.updates.size()); // 0
/// @endcode
struct HopfieldSweep {
  std::size_t              sweep{0}; ///< One-based actual complete sweep number.
  std::vector<std::size_t> order; ///< Actual shuffled order, a complete permutation of neuron IDs.
  std::vector<HopfieldUpdate> updates;     ///< Actual chronological decisions, one per neuron.
  std::vector<int>            state_after; ///< Owned actual state at sweep end.
  std::size_t                 flips{0};    ///< Actual sign changes in this sweep.
  double                      energy_before{
      0}; ///< Actual initial/previous sweep-end diagnostic reused without recomputation.
  double energy_after{0}; ///< Extra full-energy observation after this producing sweep.
};
/// @brief Owned actual initialization and producing sweep/update history, without replay.
/// Initial1+N units count one record and its state. Each sweep admits1+3N before
/// shuffle/copies: one sweep record, N order IDs, N end-state values and N update
/// records. Scalar record fields/metadata/capacity/working/output/byte peak memory
/// excluded. Histories add O(S*N+1) storage; diagnostic energies add O((S+1)N*N)
/// work beyond ordinary recall. No truncation/partial result/input references.
/// @par Example
/// @code
/// const dm::algorithms::HopfieldRecallTrace t;std::print("{}\n",t.captured_entries); // 0
/// @endcode
struct HopfieldRecallTrace {
  HopfieldRecallRun run;               ///< Actual producing result, stop and selected work.
  std::vector<int>  initial_state;     ///< Owned supplied bipolar initialization.
  double            initial_energy{0}; ///< Extra full-energy observation at initialization.
  std::vector<HopfieldSweep> sweeps;   ///< Actual chronological complete sweeps and decisions.
  std::size_t captured_entries{0};     ///< Retained logical record/value units, not bytes.
};
/// @brief Internal domain checking, bounded admission and shared producing kernels.
namespace hopfield_detail {
/// @brief Add logical visit/history counts without overflow.
/// @param a Existing count.
/// @param b Additional units.
/// @return Exact sum.
/// @throws std::length_error On unrepresentable sum.
inline std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("Hopfield count overflow");
  return a + b;
}
/// @brief Check a complete square-entry count without allocating.
/// @param n Complete neuron count.
/// @return Exact N*N entry count.
/// @throws std::length_error On count/container representation limits.
inline std::size_t square(std::size_t n) {
  if (n && n > std::numeric_limits<std::size_t>::max() / n)
    throw std::length_error("Hopfield weight count");
  if (n > std::vector<double>{}.max_size() || n > std::vector<std::vector<double>>{}.max_size())
    throw std::length_error("Hopfield matrix size");
  return n * n;
}
/// @brief Validate every value of one bipolar state without allocation.
/// @param state Borrowed signs, empty permitted.
/// @throws std::invalid_argument On any value other than-1 or+1.
inline void signs(const std::vector<int>& state) {
  for (int value : state)
    if (value != -1 && value != 1)
      throw std::invalid_argument("Hopfield signs must be -1 or +1");
}
/// @brief Validate all matrix rows before any indexed square access, then finite symmetry/diagonal.
/// @param net Borrowed possibly malformed network.
/// @throws std::invalid_argument On nonsquare/nonfinite/asymmetric/nonzero-diagonal input.
inline void network(const HopfieldNetwork& net) {
  const auto n = net.size();
  for (const auto& row : net.weights)
    if (row.size() != n)
      throw std::invalid_argument("Hopfield square weights required");
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      if (!std::isfinite(net.weights[i][j]) || net.weights[i][j] != net.weights[j][i] ||
          (i == j && net.weights[i][j] != 0))
        throw std::invalid_argument("Hopfield finite symmetric zero-diagonal weights required");
}
/// @brief Validate the complete state dimension and signs before policies/copies.
/// @param net Borrowed already shape-validated network.
/// @param state Borrowed candidate state.
/// @throws std::invalid_argument On dimension or sign failure.
inline void state(const HopfieldNetwork& net, const std::vector<int>& state) {
  if (state.size() != net.size())
    throw std::invalid_argument("Hopfield state length");
  signs(state);
}
/// @brief Admit the complete matrix against optional neuron/entry policies.
/// @param n Complete row count.
/// @param options Matrix policies; zero disables each cap.
/// @throws std::length_error On count/container or complete policy excess.
inline void admit(std::size_t n, HopfieldOptions options) {
  const auto count = square(n);
  if ((options.max_neurons && n > options.max_neurons) ||
      (options.max_weights && count > options.max_weights))
    throw std::length_error("Hopfield matrix policy");
}
/// @brief Admit a complete known capture record before shuffle/copy work.
/// @param trace Optional owned history destination.
/// @param units Additional known complete logical units.
/// @param options History cap.
/// @throws std::length_error On count or history-policy excess.
inline void capture(HopfieldRecallTrace* trace, std::size_t units, HopfieldOptions options) {
  if (trace) {
    const auto next = add(trace->captured_entries, units);
    if (options.max_trace_entries && next > options.max_trace_entries)
      throw std::length_error("Hopfield history policy");
    trace->captured_entries = next;
  }
}
/// @brief Compute energy in the preserved row-major double arithmetic order.
/// @param net Borrowed validated network.
/// @param state Borrowed validated state.
/// @param diagnostic_terms Optional actual term counter for extra observations.
/// @return Actual finite computed energy.
/// @throws std::length_error On counter overflow.
/// @throws std::overflow_error On nonfinite arithmetic.
inline double energy(
    const HopfieldNetwork&  net,
    const std::vector<int>& state,
    std::size_t*            diagnostic_terms = nullptr) {
  double e = 0;
  for (std::size_t i = 0; i < net.size(); ++i)
    for (std::size_t j = 0; j < net.size(); ++j) {
      e -= net.weights[i][j] * state[i] * state[j];
      if (diagnostic_terms)
        *diagnostic_terms = add(*diagnostic_terms, 1);
    }
  e *= .5;
  if (!std::isfinite(e))
    throw std::overflow_error("Hopfield nonfinite energy");
  return e;
}
/// @brief Execute the shared preserving asynchronous kernel and optional actual capture.
/// @param net Borrowed finite symmetric zero-diagonal network.
/// @param initial Borrowed bipolar initialization, never modified or retained.
/// @param budget Maximum complete shuffled sweeps, zero permitted.
/// @param seed MT19937-64 seed for shuffle of the evolving order vector.
/// @param options Complete matrix/history policies.
/// @param trace Optional owned capture destination.
/// @return Owned actual state, stop branch, completed sweeps and selected work.
/// @throws std::invalid_argument On complete matrix/state domain failure before policies/internal copies.
/// @throws std::length_error On policy/count/container limits.
/// @throws std::overflow_error On nonfinite fields or extra diagnostic energy.
/// @throws std::bad_alloc On allocation failure.
inline HopfieldRecallRun execute(
    const HopfieldNetwork&  net,
    const std::vector<int>& initial,
    std::size_t             budget,
    std::uint64_t           seed,
    HopfieldOptions         options,
    HopfieldRecallTrace*    trace) {
  network(net);
  state(net, initial);
  admit(net.size(), options);
  const auto n = net.size();
  if (trace)
    capture(trace, add(1, n), options);
  HopfieldRecallRun run;
  run.state = initial;
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::mt19937_64 rng(seed);
  double          observed_energy = 0;
  if (trace) {
    trace->initial_state  = run.state;
    observed_energy       = energy(net, run.state, &run.work.diagnostic_energy_terms);
    trace->initial_energy = observed_energy;
  }
  for (std::size_t k = 0; k < budget; ++k) {
    HopfieldSweep record;
    if (trace) {
      capture(trace, add(1, add(n, add(n, n))), options);
      record.updates.reserve(n);
      record.energy_before = observed_energy;
    }
    std::shuffle(order.begin(), order.end(), rng);
    if (trace)
      record.order = order;
    bool        changed = false;
    std::size_t flips   = 0;
    for (const auto idx : order) {
      double field = 0;
      for (std::size_t j = 0; j < n; ++j) {
        field += net.weights[idx][j] * run.state[j];
        run.work.field_terms = add(run.work.field_terms, 1);
      }
      if (!std::isfinite(field))
        throw std::overflow_error("Hopfield nonfinite field");
      const int before = run.state[idx], after = field >= 0 ? 1 : -1;
      run.work.neuron_updates = add(run.work.neuron_updates, 1);
      if (after != before) {
        run.state[idx] = after;
        changed        = true;
        run.work.flips = add(run.work.flips, 1);
        ++flips;
      }
      if (trace)
        record.updates.push_back({idx, field, before, after});
    }
    run.sweeps = k + 1;
    if (trace) {
      record.sweep        = k + 1;
      record.flips        = flips;
      record.state_after  = run.state;
      observed_energy     = energy(net, run.state, &run.work.diagnostic_energy_terms);
      record.energy_after = observed_energy;
      trace->sweeps.push_back(std::move(record));
    }
    if (!changed) {
      run.stop = HopfieldStop::NoChange;
      break;
    }
  }
  return run;
}
} // namespace hopfield_detail
/// @brief Store complete bipolar patterns with checked Hebbian outer-product weights.
/// W_ij=sum_p(pattern[p][i]*pattern[p][j])/N for i!=j; diagonal zero. Validates
/// every row/sign before matrix admission/allocation; empty list or zero-length
/// rectangular patterns produce an empty network. Pattern/order arithmetic is
/// preserved; supplied memories need not be fixed points. O(P*N*N+P*N+1) work,
/// O(N*N+1) owned output, borrowed input unchanged and not retained.
/// @param patterns Borrowed rectangular bipolar patterns, including empty dimensions.
/// @param options Neuron/weight caps, default10k/1m; zero disables each; trace cap ignored.
/// @return Owned actual finite symmetric zero-diagonal network.
/// @throws std::invalid_argument On row/sign failure before caps/allocation.
/// @throws std::length_error On complete matrix or count/container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto n=dm::algorithms::hopfield_train_checked({{1,-1}});
/// std::print("{}\n",n.weights[0][1]); // -0.5
/// @endcode
/// @par References
/// - J. J. Hopfield (1982), Neural networks and physical systems with emergent
///   collective computational abilities, doi:10.1073/pnas.79.8.2554 . This
///   implementation uses normalized Hebbian weights and zero-field sign+1.
/// @see hopfield_train(), hopfield_recall_trace()
inline HopfieldNetwork hopfield_train_checked(
    const std::vector<std::vector<int>>& patterns,
    HopfieldOptions                      options = {}) {
  const auto n = patterns.empty() ? 0 : patterns.front().size();
  for (const auto& p : patterns) {
    if (p.size() != n)
      throw std::invalid_argument("Hopfield pattern row length");
    hopfield_detail::signs(p);
  }
  hopfield_detail::admit(n, options);
  HopfieldNetwork net;
  if (n == 0)
    return net;
  net.weights.assign(n, std::vector<double>(n, 0.));
  for (const auto& p : patterns)
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        if (i != j)
          net.weights[i][j] += static_cast<double>(p[i] * p[j]);
  const double inv = 1. / static_cast<double>(n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      net.weights[i][j] *= inv;
  return net;
}
/// @brief Store bipolar patterns with the historical interface and no optional matrix policies.
/// Same complete validation, arithmetic, ownership and storage semantics as
/// hopfield_train_checked(), with neuron/weight/capture caps disabled.
/// @param patterns Borrowed rectangular complete bipolar patterns, empty allowed.
/// @return Owned actual Hebbian network; no guarantee each supplied pattern is stable.
/// @throws std::invalid_argument On row/sign failure before allocation.
/// @throws std::length_error On count/container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::hopfield_train({{1,-1}}).size()); // 2
/// @endcode
/// @see hopfield_train_checked()
inline HopfieldNetwork hopfield_train(const std::vector<std::vector<int>>& patterns) {
  return hopfield_train_checked(patterns, {0, 0, 0});
}
/// @brief Evaluate E=-1/2 sum_ij W_ij*s_i*s_j with full matrix/state validation.
/// Requires finite exact-symmetric square weights and zero diagonal, bipolar
/// equal-size state. Empty gives zero. Preserves sequential double arithmetic;
/// not an exact energy/sign certificate. O(N*N+1) work, no allocation on success
/// or retained references/input mutation; nonfinite producing arithmetic throws.
/// @param net Borrowed possibly malformed network to validate completely.
/// @param state Borrowed complete bipolar state.
/// @return Actual finite computed energy.
/// @throws std::invalid_argument On matrix/state domain failure before indexed use.
/// @throws std::overflow_error On nonfinite producing energy arithmetic.
/// @par Example
/// @code
/// const auto net=dm::algorithms::hopfield_train({{1,-1}});
/// std::print("{}\n",dm::algorithms::hopfield_energy(net,{1,-1})); // -0.5
/// @endcode
/// @see hopfield_recall_trace()
inline double hopfield_energy(const HopfieldNetwork& net, const std::vector<int>& state) {
  hopfield_detail::network(net);
  hopfield_detail::state(net, state);
  return hopfield_detail::energy(net, state);
}
/// @brief Recall with the historical by-value state interface and preserved sign/shuffle arithmetic.
/// Zero local field selects+1 (it does not keep a negative sign). Initializes one
/// index order, shuffles it in place each sweep with MT19937-64(seed), and updates
/// asynchronously in that order until no change or budget exhaustion. Zero
/// budget returns initialization; empty state takes one empty unchanged sweep
/// when budget permits. Different standard-library shuffle mappings can differ.
/// Complete matrix/state validation precedes internal copies; the historical
/// by-value parameter itself may already copy at call entry. No optional caps.
/// O(N*N+S*N*N+S*N+1) ordinary work and O(N+1) extra output/working storage,
/// excluding the borrowed N*N matrix. No nearest-memory or universal convergence
/// certificate for floating arithmetic; finite budget always bounds execution.
/// @param net Borrowed finite symmetric zero-diagonal network.
/// @param state Owned initial bipolar state, returned relaxed; original caller vector unchanged.
/// @param max_sweeps Maximum complete sweeps, default100; zero allowed.
/// @param seed Shuffle seed, default0; repeatability depends on standard-library mapping.
/// @return Owned actual final state, possibly budget-limited rather than fixed.
/// @throws std::invalid_argument On matrix/state domain failure.
/// @throws std::length_error On count/container representation limits.
/// @throws std::overflow_error On nonfinite local fields.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto net=dm::algorithms::hopfield_train({{-1}});
/// std::print("{}\n",dm::algorithms::hopfield_recall(net,{-1},1,0)[0]); // 1
/// @endcode
/// @see hopfield_recall_run(), hopfield_recall_trace()
inline std::vector<int> hopfield_recall(
    const HopfieldNetwork& net,
    std::vector<int>       state,
    std::size_t            max_sweeps = 100,
    std::uint64_t          seed       = 0) {
  return hopfield_detail::execute(net, state, max_sweeps, seed, {0, 0, 0}, nullptr).state;
}
/// @brief Run shared recall with complete matrix policies and actual stop/work diagnostics.
/// Same state/shuffle/arithmetic/budget semantics as hopfield_recall(). Validates
/// borrowed input before internal copies or caps. Capture cap ignored.
/// @param net Borrowed complete network, validated before policies.
/// @param state Borrowed complete bipolar initialization, not mutated or retained.
/// @param max_sweeps Complete sweep budget, default100; zero allowed.
/// @param seed Shuffle seed, default0.
/// @param options Neuron/weight caps, default10k/1m, zero disabling each.
/// @return Owned actual state, sweep count, terminal branch and selected work.
/// @throws std::invalid_argument On complete matrix/state domain failure.
/// @throws std::length_error On matrix/count/container limits.
/// @throws std::overflow_error On nonfinite local fields.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hopfield_recall_run(dm::algorithms::HopfieldNetwork{},{});
/// std::print("{} {}\n",r.sweeps,r.stop==dm::algorithms::HopfieldStop::NoChange); // 1 true
/// @endcode
/// @see HopfieldRecallRun, HopfieldOptions
inline HopfieldRecallRun hopfield_recall_run(
    const HopfieldNetwork&  net,
    const std::vector<int>& state,
    std::size_t             max_sweeps = 100,
    std::uint64_t           seed       = 0,
    HopfieldOptions         options    = {}) {
  return hopfield_detail::execute(net, state, max_sweeps, seed, options, nullptr);
}
/// @brief Capture actual producing asynchronous decisions and extra sweep-boundary energies.
/// Same contracts as hopfield_recall_run(), with online complete initial/sweep
/// admission and owned histories, no replay. Diagnostic energies add work and
/// can overflow even when untraced local fields would remain finite.
/// @param net Borrowed complete network, never retained or mutated.
/// @param state Borrowed complete bipolar initialization.
/// @param max_sweeps Complete sweep budget, default100; zero allowed.
/// @param seed Shuffle seed, default0.
/// @param options Complete matrix/online history caps; defaults10k/1m/1m, zerooff each.
/// @return Owned actual initialization/sweeps and producing run with separate diagnostic counts.
/// @throws std::invalid_argument On complete matrix/state domain failure before policies.
/// @throws std::length_error On matrix/online history/count/container limits.
/// @throws std::overflow_error On nonfinite local fields or diagnostic energy.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::hopfield_recall_trace(dm::algorithms::hopfield_train({{-1}}),{-1},1);
/// std::print("{} {}\n",t.sweeps[0].updates[0].field,t.captured_entries); // 0 6
/// @endcode
/// @see HopfieldRecallTrace
inline HopfieldRecallTrace hopfield_recall_trace(
    const HopfieldNetwork&  net,
    const std::vector<int>& state,
    std::size_t             max_sweeps = 100,
    std::uint64_t           seed       = 0,
    HopfieldOptions         options    = {}) {
  HopfieldRecallTrace t;
  t.run = hopfield_detail::execute(net, state, max_sweeps, seed, options, &t);
  return t;
}
/// @brief Count differing positions of two complete equal-length bipolar states.
/// Rejects unequal lengths instead of silently ignoring a suffix. Empty equals
/// empty at distance0. O(N+1) time, O(1) storage; no allocation on success, input
/// mutation, references retained or shared state.
/// @param a Borrowed complete bipolar state.
/// @param b Borrowed complete bipolar state of the same length.
/// @return Exact Hamming distance across all positions.
/// @throws std::invalid_argument On length/sign failure before counting.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::hopfield_hamming({1,-1},{-1,-1})); // 1
/// @endcode
/// @see hopfield_recall_run()
inline std::size_t hopfield_hamming(const std::vector<int>& a, const std::vector<int>& b) {
  if (a.size() != b.size())
    throw std::invalid_argument("Hopfield Hamming lengths differ");
  hopfield_detail::signs(a);
  hopfield_detail::signs(b);
  std::size_t d = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    d += a[i] != b[i];
  return d;
}
} // namespace datamunge::algorithms
