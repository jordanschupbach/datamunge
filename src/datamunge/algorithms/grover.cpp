/// @file grover.cpp
/// @brief Shared checked Grover simulation with actual full-phase capture and selected operator
/// work.
#include <datamunge/algorithms/grover.hpp>
#include <limits>
#include <numbers>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private checked dimensions, exact operation order and optional capture sinks.
namespace grover_detail {
/// @brief Checked logical count addition.
/// @param a Current count.
/// @param b Increment.
/// @return Nonwrapping sum.
/// @throws std::length_error On size_t overflow.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > SIZE_MAX - a)
    throw std::length_error("Grover count overflow");
  return a + b;
}
/// @brief Checked logical count multiplication.
/// @param a First count.
/// @param b Second count.
/// @return Nonwrapping product.
/// @throws std::length_error On size_t overflow.
std::size_t multiply(std::size_t a, std::size_t b) {
  if (a && b > SIZE_MAX / a)
    throw std::length_error("Grover count product overflow");
  return a * b;
}
/// @brief Validate shift, target and step domains before policy caps or state allocation.
/// @param n Nonnegative representable qubit count.
/// @param target Marked basis index.
/// @param iterations Nonnegative step count.
/// @param cap Optional complete state cap, zero disables it.
/// @return Safe state size 2^n.
/// @throws std::invalid_argument On n/target/iterations.
/// @throws std::length_error On state cap/container size.
std::size_t geometry(int n, std::size_t target, int iterations, std::size_t cap) {
  if (n < 0 || n >= std::numeric_limits<std::size_t>::digits || iterations < 0)
    throw std::invalid_argument("Grover invalid n/iterations");
  const auto N = std::size_t{1} << n;
  if (target >= N)
    throw std::invalid_argument("Grover target outside state");
  if ((cap && N > cap) || N > QuantumState{}.max_size())
    throw std::length_error("Grover complete state limit");
  return N;
}
/// @brief Reject nonfinite complex arithmetic.
/// @param z Candidate amplitude/mean.
/// @throws std::overflow_error If either component is nonfinite.
void finite(Complex z) {
  if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
    throw std::overflow_error("Grover arithmetic became nonfinite");
}
/// @brief Apply the actual original oracle/mean/diffusion order with optional phase snapshots.
/// @param state Valid nonempty finite mutable state.
/// @param target Valid marked index.
/// @param capture Optional borrowed step record under construction, already admitted.
/// @param work Optional borrowed selected operation counters.
/// @throws std::overflow_error On nonfinite arithmetic, with possible partial mutation.
/// @throws std::length_error On checked work/container exhaustion.
/// @throws std::bad_alloc On snapshot allocation failure.
void step(QuantumState& state, std::size_t target, GroverStep* capture, GroverWork* work) {
  if (work) {
    work->oracle_calls      = add(work->oracle_calls, 1);
    work->mean_terms        = add(work->mean_terms, state.size());
    work->diffusion_updates = add(work->diffusion_updates, state.size());
  }
  state[target] = -state[target];
  if (capture)
    capture->after_oracle = state;
  Complex mean{0, 0};
  for (const auto& z : state)
    mean += z;
  mean /= static_cast<double>(state.size());
  finite(mean);
  if (capture)
    capture->oracle_mean = mean;
  for (auto& z : state) {
    z = 2. * mean - z;
    finite(z);
  }
  if (capture) {
    capture->after_diffusion    = state;
    capture->marked_probability = std::norm(state[target]);
  }
}
/// @brief Execute Hadamard preparation and actual complete iterations once.
/// @param n Qubit count, validated here.
/// @param target Marked index, validated here.
/// @param iterations Complete nonnegative step count.
/// @param o Complete state/capture limits.
/// @param trace Optional borrowed initially empty full-history sink.
/// @param probabilities Optional borrowed initially empty scalar-history sink.
/// @return Owned final state, deterministic summary and selected actual work.
/// @throws std::invalid_argument On invalid domains before caps.
/// @throws std::length_error On complete geometry/caps/counts/containers.
/// @throws std::overflow_error On nonfinite arithmetic.
/// @throws std::bad_alloc On allocation failure.
GroverRun execute(
    int                  n,
    std::size_t          target,
    int                  iterations,
    GroverOptions        o,
    GroverTrace*         trace,
    std::vector<double>* probabilities) {
  const auto N = geometry(n, target, iterations, o.max_amplitudes);
  if (trace) {
    const auto units =
        add(N, multiply(static_cast<std::size_t>(iterations), add(multiply(2, N), 1)));
    if ((o.max_trace_entries && units > o.max_trace_entries) ||
        static_cast<std::size_t>(iterations) > trace->steps.max_size())
      throw std::length_error("Grover complete capture limit");
    trace->captured_entries = units;
    trace->steps.reserve(iterations);
  }
  if (probabilities) {
    const auto count = add(static_cast<std::size_t>(iterations), 1);
    if (count > probabilities->max_size())
      throw std::length_error("Grover probability capacity");
    probabilities->reserve(count);
  }
  GroverRun r;
  r.state                  = qstate_zero(n);
  r.work.initialized_slots = N;
  for (int q = 0; q < n; ++q) {
    h_gate(r.state, q);
    r.work.hadamard_gates = add(r.work.hadamard_gates, 1);
    r.work.hadamard_pairs = add(r.work.hadamard_pairs, N / 2);
  }
  if (trace)
    trace->initial = r.state;
  if (probabilities)
    probabilities->push_back(std::norm(r.state[target]));
  for (int k = 0; k < iterations; ++k) {
    GroverStep record;
    record.iteration = k + 1;
    step(r.state, target, trace ? &record : nullptr, &r.work);
    if (trace)
      trace->steps.push_back(std::move(record));
    if (probabilities)
      probabilities->push_back(std::norm(r.state[target]));
  }
  r.result            = {most_probable(r.state), std::norm(r.state[target]), iterations};
  r.work.argmax_terms = N;
  return r;
}
/// @brief Convert the retained approximate default schedule after safe state sizing.
/// @param n Validated by this helper.
/// @return Original round(pi/4*sqrt(N)) count.
/// @throws std::invalid_argument On invalid n.
/// @throws std::length_error On state or int-count representability.
int legacy_iterations(int n) {
  const auto   N = geometry(n, 0, 0, 0);
  const double k = std::round(std::numbers::pi / 4 * std::sqrt(static_cast<double>(N)));
  if (k > std::numeric_limits<int>::max())
    throw std::length_error("Grover iteration count limit");
  return static_cast<int>(k);
}
} // namespace grover_detail
namespace detail {
void grover_step(QuantumState& s, std::size_t target) {
  if (s.empty() || target >= s.size())
    throw std::invalid_argument("Grover step target/state invalid");
  for (auto z : s)
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
      throw std::invalid_argument("Grover step state must be finite");
  grover_detail::step(s, target, nullptr, nullptr);
}
} // namespace detail
GroverResult grover(int n, std::size_t target) {
  return grover_detail::execute(
             n, target, grover_detail::legacy_iterations(n), {0, 0}, nullptr, nullptr)
      .result;
}
GroverRun grover_run(int n, std::size_t target, int iterations, GroverOptions options) {
  return grover_detail::execute(n, target, iterations, options, nullptr, nullptr);
}
GroverTrace grover_trace(int n, std::size_t target, int iterations, GroverOptions options) {
  GroverTrace t;
  t.run = grover_detail::execute(n, target, iterations, options, &t, nullptr);
  return t;
}
std::vector<double> grover_success_trace(int n, std::size_t target, int max_iters) {
  std::vector<double> p;
  (void)grover_detail::execute(n, target, max_iters, {0, 0}, nullptr, &p);
  return p;
}
int grover_first_peak_iterations(int n) {
  const auto N = grover_detail::geometry(n, 0, 0, 0);
  if (n <= 1)
    return 0;
  const double k =
      std::floor(std::numbers::pi / (4 * std::asin(1 / std::sqrt(static_cast<double>(N)))));
  if (k > std::numeric_limits<int>::max())
    throw std::length_error("Grover first peak count limit");
  return static_cast<int>(k);
}
} // namespace datamunge::algorithms
