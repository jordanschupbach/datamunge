/// @file cycle_detection.cpp
/// @brief Canonical checked Brent/Floyd kernels and reusable orbit/digit-map helpers.
#include <datamunge/algorithms/cycle_detection.hpp>
#include <limits>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private checked transition evaluation and shared detector kernels.
namespace {
/// @brief Increment an index/counter without unsigned wrap.
/// @param value Mutable size counter.
/// @throws std::overflow_error If the increment is unrepresentable.
void increment(std::size_t& value) {
  if (value == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("cycle counter exceeds size_t");
  ++value;
}

/// @brief Borrowed callback and checked counters shared by both canonical detectors.
class CycleEvaluator {
  const CycleFunction& function_; ///< Transition borrowed for this invocation.
  CycleOptions         options_;  ///< Global completed-call limit.
  CycleWork&           work_;     ///< Owned result being populated by the caller.
 public:
  /// @brief Validate a borrowed transition and attach counters.
  /// @param function Nonempty deterministic transition.
  /// @param options Global completed-call budget.
  /// @param work Mutable owned work summary.
  /// @throws std::bad_function_call If function is empty.
  CycleEvaluator(const CycleFunction& function, CycleOptions options, CycleWork& work)
      : function_(function), options_(options), work_(work) {
    if (!function_)
      throw std::bad_function_call();
  }
  /// @brief Apply one transition and charge its completed call to the actual phase.
  /// @param value Input state.
  /// @param phase Actual algorithm phase.
  /// @return Callback output, without replay or memoisation.
  /// @throws std::runtime_error Before exceeding a positive budget.
  /// @throws std::overflow_error Before overflowing the total counter.
  std::uint64_t apply(std::uint64_t value, CyclePhase phase) {
    if (options_.max_evaluations && work_.evaluations >= options_.max_evaluations)
      throw std::runtime_error("cycle evaluation budget exhausted");
    if (work_.evaluations == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("cycle evaluations exceed size_t");
    const auto next = function_(value);
    ++work_.evaluations;
    switch (phase) {
      case CyclePhase::detect:
        ++work_.detection_evaluations;
        break;
      case CyclePhase::align:
        ++work_.alignment_evaluations;
        break;
      case CyclePhase::locate:
        ++work_.entry_evaluations;
        break;
      case CyclePhase::measure:
        ++work_.period_evaluations;
        break;
    }
    return next;
  }
  /// @brief Count one actual state-equality comparison, including successful meetings.
  /// @param a First state.
  /// @param b Second state.
  /// @return Whether the states are equal.
  /// @throws std::overflow_error If the comparison count cannot be represented.
  bool equal(std::uint64_t a, std::uint64_t b) {
    increment(work_.comparisons);
    return a == b;
  }
};

/// @brief Record a real Brent state only for the recording specialisation.
/// @tparam Record Whether to allocate chronological records.
/// @param trace Owned result under construction.
/// @param phase Actual phase.
/// @param reference Reference/rear state.
/// @param runner Moving/front state.
/// @param reference_index Reference's original-orbit index.
/// @param runner_index Runner's original-orbit index.
/// @param power Detection block length, zero after detection.
/// @param offset Progress within this phase.
/// @param checkpoint Whether this is a no-call copy and doubling event.
template<bool Record>
void record(
    BrentCycleTrace& trace,
    CyclePhase       phase,
    std::uint64_t    reference,
    std::uint64_t    runner,
    std::size_t      reference_index,
    std::size_t      runner_index,
    std::size_t      power,
    std::size_t      offset,
    bool             checkpoint = false) {
  if constexpr (Record)
    trace.steps.push_back(
        {phase,
         reference,
         runner,
         reference_index,
         runner_index,
         power,
         offset,
         trace.work.evaluations,
         checkpoint});
}

/// @brief Canonical Brent algorithm; diagnostics never alter transition-call order.
/// @tparam Record Whether to retain actual chronological states.
/// @param x0 Starting state.
/// @param f Borrowed deterministic transition.
/// @param options Optional completed-call budget.
/// @return Owned trace (empty records when Record is false) and work summary.
/// @throws std::runtime_error If a positive budget is exhausted.
/// @throws std::overflow_error Before arithmetic wrap; callback/allocation errors propagate.
template<bool Record>
BrentCycleTrace brent_kernel(std::uint64_t x0, const CycleFunction& f, CycleOptions options) {
  BrentCycleTrace trace;
  CycleEvaluator  evaluate(f, options, trace.work);
  std::size_t     power = 1, lambda = 1, reference_index = 0, runner_index = 1;
  auto            reference = x0;
  auto            runner    = evaluate.apply(x0, CyclePhase::detect);
  record<Record>(
      trace, CyclePhase::detect, reference, runner, reference_index, runner_index, power, lambda);
  while (!evaluate.equal(reference, runner)) {
    if (power == lambda) {
      if (power > std::numeric_limits<std::size_t>::max() / 2)
        throw std::overflow_error("Brent block length exceeds size_t");
      reference       = runner;
      reference_index = runner_index;
      power *= 2;
      lambda = 0;
      record<Record>(
          trace,
          CyclePhase::detect,
          reference,
          runner,
          reference_index,
          runner_index,
          power,
          lambda,
          true);
    }
    runner = evaluate.apply(runner, CyclePhase::detect);
    increment(runner_index);
    increment(lambda);
    record<Record>(
        trace, CyclePhase::detect, reference, runner, reference_index, runner_index, power, lambda);
  }
  reference = runner = x0;
  reference_index = runner_index = 0;
  record<Record>(trace, CyclePhase::align, reference, runner, 0, 0, 0, 0);
  for (std::size_t i = 0; i < lambda; ++i) {
    runner = evaluate.apply(runner, CyclePhase::align);
    increment(runner_index);
    record<Record>(trace, CyclePhase::align, reference, runner, 0, runner_index, 0, runner_index);
  }
  std::size_t mu = 0;
  record<Record>(trace, CyclePhase::locate, reference, runner, 0, runner_index, 0, 0);
  while (!evaluate.equal(reference, runner)) {
    reference = evaluate.apply(reference, CyclePhase::locate);
    runner    = evaluate.apply(runner, CyclePhase::locate);
    increment(reference_index);
    increment(runner_index);
    increment(mu);
    record<Record>(
        trace, CyclePhase::locate, reference, runner, reference_index, runner_index, 0, mu);
  }
  trace.work.result = {lambda, mu};
  trace.work.entry  = reference;
  return trace;
}

/// @brief Admit the next actual Floyd pair before its callbacks or reset.
/// @tparam Record Whether rows are being retained.
/// @param trace Owned result being populated.
/// @param cap Online row cap; zero unlimited.
/// @throws std::length_error Row cap or vector capacity exhausted.
template<bool Record>
void admit_floyd(const FloydCycleTrace& trace, std::size_t cap) {
  if constexpr (Record) {
    if ((cap && trace.steps.size() >= cap) || trace.steps.size() == trace.steps.max_size())
      throw std::length_error("Floyd trace row cap exhausted");
  }
}
/// @brief Append an admitted actual pair without evaluating the callback.
/// @tparam Record Whether to allocate chronological records.
/// @param trace Owned result under construction.
/// @param step Actual state pair; its evaluation total is set by this function.
/// @throws std::bad_alloc Row allocation fails.
template<bool Record>
void record_floyd(FloydCycleTrace& trace, FloydCycleStep step) {
  if constexpr (Record) {
    step.evaluations = trace.work.evaluations;
    trace.steps.push_back(step);
  }
}
/// @brief Shared Floyd kernel preserving the historical repeated initial f(x0).
/// @tparam Record Whether to record actual complete pairs.
/// @param x0 Initial state.
/// @param f Borrowed deterministic transition.
/// @param options Global successful-call budget.
/// @param cap Online trace row cap, ignored without recording.
/// @return Actual work and optional owned states without replay.
/// @throws std::runtime_error Positive evaluation budget exhausted.
/// @throws std::length_error Positive row cap/container capacity exhausted.
/// @throws std::overflow_error Checked indices or counters cannot be represented.
/// @throws std::bad_function_call Empty callback; other callback errors propagate.
/// @throws std::bad_alloc Record allocation fails; callback effects remain.
template<bool Record>
FloydCycleTrace
floyd_kernel(std::uint64_t x0, const CycleFunction& f, CycleOptions options, std::size_t cap = 0) {
  FloydCycleTrace trace;
  auto&           work = trace.work;
  CycleEvaluator  evaluate(f, options, work);
  admit_floyd<Record>(trace, cap);
  auto        reference       = evaluate.apply(x0, CyclePhase::detect);
  auto        intermediate    = evaluate.apply(x0, CyclePhase::detect);
  auto        runner          = evaluate.apply(intermediate, CyclePhase::detect);
  std::size_t reference_index = 1, runner_index = 2;
  record_floyd<Record>(
      trace,
      {CyclePhase::detect,
       reference,
       runner,
       reference_index,
       runner_index,
       reference_index,
       0,
       true});
  while (!evaluate.equal(reference, runner)) {
    admit_floyd<Record>(trace, cap);
    increment(reference_index);
    increment(runner_index);
    increment(runner_index);
    reference    = evaluate.apply(reference, CyclePhase::detect);
    intermediate = evaluate.apply(runner, CyclePhase::detect);
    runner       = evaluate.apply(intermediate, CyclePhase::detect);
    record_floyd<Record>(
        trace,
        {CyclePhase::detect,
         reference,
         runner,
         reference_index,
         runner_index,
         reference_index,
         0,
         false});
  }
  admit_floyd<Record>(trace, cap);
  reference       = x0;
  reference_index = 0;
  std::size_t mu  = 0;
  record_floyd<Record>(
      trace, {CyclePhase::locate, reference, runner, reference_index, runner_index, mu, 0, true});
  while (!evaluate.equal(reference, runner)) {
    admit_floyd<Record>(trace, cap);
    increment(reference_index);
    increment(runner_index);
    increment(mu);
    reference = evaluate.apply(reference, CyclePhase::locate);
    runner    = evaluate.apply(runner, CyclePhase::locate);
    record_floyd<Record>(
        trace,
        {CyclePhase::locate, reference, runner, reference_index, runner_index, mu, 0, false});
  }
  work.entry         = reference;
  std::size_t lambda = 1;
  admit_floyd<Record>(trace, cap);
  runner_index = reference_index;
  increment(runner_index);
  runner = evaluate.apply(reference, CyclePhase::measure);
  record_floyd<Record>(
      trace,
      {CyclePhase::measure, reference, runner, reference_index, runner_index, lambda, 0, true});
  while (!evaluate.equal(reference, runner)) {
    admit_floyd<Record>(trace, cap);
    increment(runner_index);
    increment(lambda);
    runner = evaluate.apply(runner, CyclePhase::measure);
    record_floyd<Record>(
        trace,
        {CyclePhase::measure, reference, runner, reference_index, runner_index, lambda, 0, false});
  }
  work.result = {lambda, mu};
  return trace;
}
} // namespace

CycleResult brent_cycle_detection(std::uint64_t x0, const CycleFunction& f) {
  return brent_kernel<false>(x0, f, {}).work.result;
}
CycleResult floyd_cycle_detection(std::uint64_t x0, const CycleFunction& f) {
  return floyd_kernel<false>(x0, f, {}).work.result;
}
CycleWork brent_cycle_work(std::uint64_t x0, const CycleFunction& f, CycleOptions options) {
  return brent_kernel<false>(x0, f, options).work;
}
CycleWork floyd_cycle_work(std::uint64_t x0, const CycleFunction& f, CycleOptions options) {
  return floyd_kernel<false>(x0, f, options).work;
}
BrentCycleTrace brent_cycle_trace(std::uint64_t x0, const CycleFunction& f, CycleOptions options) {
  return brent_kernel<true>(x0, f, options);
}
FloydCycleTrace floyd_cycle_trace(
    std::uint64_t        x0,
    const CycleFunction& f,
    CycleOptions         options,
    std::size_t          max_records) {
  return floyd_kernel<true>(x0, f, options, max_records);
}
std::uint64_t
quadratic_cycle_step(std::uint64_t state, std::uint64_t modulus, std::uint64_t constant) {
  if (!modulus)
    throw std::invalid_argument("quadratic cycle step: modulus must be positive");
  return static_cast<std::uint64_t>(
      (static_cast<unsigned __int128>(state) * state + constant) % modulus);
}
std::vector<std::uint64_t>
iterate_states(std::uint64_t x0, const CycleFunction& f, std::size_t count) {
  std::vector<std::uint64_t> states;
  states.reserve(count);
  if (!count)
    return states;
  states.push_back(x0);
  while (states.size() < count)
    states.push_back(f(states.back()));
  return states;
}
std::vector<std::uint64_t> rho_successors(std::size_t mu, std::size_t lambda) {
  if (!lambda)
    throw std::invalid_argument("rho period must be positive");
  if (mu > std::numeric_limits<std::size_t>::max() - lambda)
    throw std::overflow_error("rho state count exceeds size_t");
  const auto                 count = mu + lambda;
  std::vector<std::uint64_t> next(count);
  for (std::size_t i = 0; i < count - 1; ++i)
    next[i] = i + 1;
  next.back() = mu;
  return next;
}
std::uint64_t sum_squared_decimal_digits(std::uint64_t value) noexcept {
  std::uint64_t sum = 0;
  do {
    const auto digit = value % 10;
    sum += digit * digit;
    value /= 10;
  } while (value);
  return sum;
}
bool is_happy_number(std::uint64_t value) {
  if (!value)
    return false;
  const auto work = brent_cycle_work(value, sum_squared_decimal_digits);
  return work.entry == 1 && work.result.lambda == 1;
}
} // namespace datamunge::algorithms
