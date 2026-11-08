/// @file goertzel.cpp
/// @brief Shared checked ordered real recurrence, online states and native synthetic tone helpers.
#include <cmath>
#include <datamunge/algorithms/goertzel.hpp>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private input checks, frequency plans, complete-record capture and shared tone constants.
namespace {
/// @brief Double pi constant equal to the original M_PI value.
constexpr double pi = std::numbers::pi;
/// @brief Q.23 low-group frequencies in Hz, ordered by keypad row.
constexpr std::array<double, 4> rows{697, 770, 852, 941};
/// @brief Q.23 high-group frequencies in Hz, ordered by keypad column.
constexpr std::array<double, 4> columns{1209, 1336, 1477, 1633};
/// @brief Q.23 four-by-four key layout, with uppercase reserve letters.
constexpr std::array<std::array<char, 4>, 4> keypad{
    {{{'1', '2', '3', 'A'}},
     {{'4', '5', '6', 'B'}},
     {{'7', '8', '9', 'C'}},
     {{'*', '0', '#', 'D'}}}};
/// @brief Validate every borrowed sample without retaining aliases or changing values.
/// @param x Real sample vector, including empty input.
/// @throws std::invalid_argument Any sample is nonfinite.
void samples_input(const std::vector<double>& x) {
  for (double v : x)
    if (!std::isfinite(v))
      throw std::invalid_argument("Goertzel: finite samples required");
}
/// @brief Validate a sampling rate before empty handling or allocation.
/// @param fs Sampling rate in Hz.
/// @throws std::invalid_argument The rate is not finite and strictly positive.
void rate_input(double fs) {
  if (!std::isfinite(fs) || fs <= 0)
    throw std::invalid_argument("signal: finite positive sample rate required");
}
/// @brief Validate a public frequency mode, including for empty input.
/// @param mode Requested selector.
/// @throws std::invalid_argument The enum value is unrecognized.
void mode_input(GoertzelFrequencyMode mode) {
  if (mode != GoertzelFrequencyMode::legacy_bin && mode != GoertzelFrequencyMode::unrounded)
    throw std::invalid_argument("Goertzel: unknown frequency mode");
}
/// @brief Reject a nonfinite actual evaluated selection, state, phase or output.
/// Does not split/reorder the original recurrence or terminal expressions.
/// @param value Actual double result, including finite negatives and signed zero.
/// @return Unchanged finite value.
/// @throws std::overflow_error The result is nonfinite.
double finite_value(double value) {
  if (!std::isfinite(value))
    throw std::overflow_error("Goertzel/signal: nonfinite evaluated value");
  return value;
}
/// @brief Safely perform the original offset-truncated signed integer conversion.
/// @param expression Actual finite 0.5+double(N)*f/fs value.
/// @return The original truncation-to-int result, including fractional values just beyond integer
/// limits.
/// @throws std::overflow_error The expression or its truncated integer is unrepresentable.
int legacy_integer(double expression) {
  finite_value(expression);
  const double t     = std::trunc(expression);
  const double lower = static_cast<double>(std::numeric_limits<int>::min());
  if (t < lower || t >= -lower)
    throw std::overflow_error("Goertzel: selected bin cannot fit int");
  return static_cast<int>(expression);
}
/// @brief Append a complete actual-state record when history is enabled.
/// @param events Optional owned history, or null for ordinary processing.
/// @param maximum Optional complete-record quota, zero disables it.
/// @param event Actual state or metadata to copy.
/// @throws std::length_error The next whole record exceeds quota/container capacity.
/// @throws std::bad_alloc History allocation fails.
void append(std::vector<GoertzelEvent>* events, std::size_t maximum, const GoertzelEvent& event) {
  if (!events)
    return;
  if ((maximum && events->size() >= maximum) || events->size() == events->max_size())
    throw std::length_error("Goertzel: complete record quota/capacity exceeded");
  events->push_back(event);
}
/// @brief Execute the single ordered recurrence using an already validated selection plan.
/// @param x Completely validated finite samples.
/// @param result Actual scalar plan; coefficient is active only for nonempty input.
/// @param events Optional owned history.
/// @param maximum Optional complete-record quota, zero disables it.
/// @return Owned actual states, raw power and update count from this same execution.
/// @throws std::overflow_error An actual recurrence state or terminal raw power is nonfinite.
/// @throws std::length_error The next whole record exceeds quota/container capacity.
/// @throws std::bad_alloc History allocation fails.
GoertzelRun execute(
    const std::vector<double>&  x,
    GoertzelRun                 result,
    std::vector<GoertzelEvent>* events,
    std::size_t                 maximum) {
  const auto   n     = x.size();
  const double coeff = result.coefficient;
  append(events, maximum, {GoertzelEventKind::initial, 0, 0, 0, 0, coeff});
  double s1 = 0, s2 = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double s0 = x[i] + coeff * s1 - s2;
    finite_value(s0);
    append(events, maximum, {GoertzelEventKind::sample, i, x[i], s1, s2, s0});
    s2 = s1;
    s1 = s0;
    ++result.sample_updates;
  }
  // Keep the original expression intact, rather than separately rounding terms.
  const double power = n ? s1 * s1 + s2 * s2 - coeff * s1 * s2 : 0.;
  finite_value(power);
  append(events, maximum, {GoertzelEventKind::stop, n, 0, s1, s2, power});
  result.s1    = s1;
  result.s2    = s2;
  result.power = power;
  return result;
}
/// @brief Build the actual frequency plan after complete validation and empty handling.
/// @param x Borrowed samples to validate.
/// @param f Finite signed target in Hz.
/// @param fs Finite positive rate in Hz.
/// @param mode Recognized frequency selector.
/// @return Actual scalar plan, with an inactive coefficient for empty input.
/// @throws std::invalid_argument A sample, parameter or mode is invalid.
/// @throws std::overflow_error Selected bin or evaluated angle/coefficient is
/// unrepresentable/nonfinite.
GoertzelRun
frequency_plan(const std::vector<double>& x, double f, double fs, GoertzelFrequencyMode mode) {
  samples_input(x);
  rate_input(fs);
  mode_input(mode);
  if (!std::isfinite(f))
    throw std::invalid_argument("Goertzel: finite target frequency required");
  GoertzelRun result;
  result.sample_count   = x.size();
  result.frequency_mode = mode;
  if (x.empty())
    return result;
  if (mode == GoertzelFrequencyMode::legacy_bin) {
    result.selected_bin = legacy_integer(0.5 + static_cast<double>(x.size()) * f / fs);
    result.omega        = finite_value(
        2 * pi * static_cast<double>(result.selected_bin) / static_cast<double>(x.size()));
  } else
    result.omega = finite_value(2 * pi * f / fs);
  result.coefficient     = finite_value(2 * std::cos(result.omega));
  result.has_coefficient = true;
  return result;
}
/// @brief Build the original unreduced unsigned-bin plan after complete sample validation.
/// @param x Borrowed finite samples to validate.
/// @param k Requested size_t bin, retained without modulo reduction or narrowing.
/// @return Actual scalar plan, with inactive coefficient for empty input.
/// @throws std::invalid_argument A sample is nonfinite.
/// @throws std::overflow_error Actual angle or coefficient is nonfinite.
GoertzelRun bin_plan(const std::vector<double>& x, std::size_t k) {
  samples_input(x);
  GoertzelRun result;
  result.sample_count = x.size();
  result.bin_indexed  = true;
  result.bin          = k;
  if (x.empty())
    return result;
  result.omega = finite_value(2 * pi * static_cast<double>(k) / static_cast<double>(x.size()));
  result.coefficient     = finite_value(2 * std::cos(result.omega));
  result.has_coefficient = true;
  return result;
}
} // namespace

double goertzel_power(const std::vector<double>& x, double target_freq, double sample_rate) {
  return goertzel_run(x, target_freq, sample_rate).power;
}
double goertzel_power_bin(const std::vector<double>& x, std::size_t k) {
  return goertzel_bin_run(x, k).power;
}
double
goertzel_power_at_frequency(const std::vector<double>& x, double target_freq, double sample_rate) {
  return goertzel_run(x, target_freq, sample_rate, GoertzelFrequencyMode::unrounded).power;
}
GoertzelRun goertzel_run(
    const std::vector<double>& x,
    double                     target_freq,
    double                     sample_rate,
    GoertzelFrequencyMode      mode) {
  return execute(x, frequency_plan(x, target_freq, sample_rate, mode), nullptr, 0);
}
GoertzelRun goertzel_bin_run(const std::vector<double>& x, std::size_t k) {
  return execute(x, bin_plan(x, k), nullptr, 0);
}
GoertzelTrace goertzel_trace(
    const std::vector<double>& x,
    double                     target_freq,
    double                     sample_rate,
    GoertzelFrequencyMode      mode,
    std::size_t                max_records) {
  GoertzelTrace result;
  result.run =
      execute(x, frequency_plan(x, target_freq, sample_rate, mode), &result.events, max_records);
  return result;
}
GoertzelTrace
goertzel_bin_trace(const std::vector<double>& x, std::size_t k, std::size_t max_records) {
  GoertzelTrace result;
  result.run = execute(x, bin_plan(x, k), &result.events, max_records);
  return result;
}
std::vector<double> sinusoid_signal(
    std::size_t                  n,
    double                       sample_rate,
    const std::vector<Sinusoid>& terms,
    std::size_t                  max_samples) {
  rate_input(sample_rate);
  for (const auto& tone : terms)
    if (!std::isfinite(tone.frequency_hz) || !std::isfinite(tone.amplitude) ||
        !std::isfinite(tone.phase_radians))
      throw std::invalid_argument("signal: finite sinusoid descriptors required");
  std::vector<double> result;
  if ((max_samples && n > max_samples) || n > result.max_size())
    throw std::length_error("signal: whole sample count exceeds quota/capacity");
  result.resize(n, 0.);
  for (std::size_t i = 0; i < n; ++i)
    for (const auto& tone : terms) {
      const double phase = finite_value(
          2 * pi * tone.frequency_hz * static_cast<double>(i) / sample_rate + tone.phase_radians);
      result[i] = finite_value(result[i] + tone.amplitude * std::sin(phase));
    }
  return result;
}
std::vector<double>
dtmf_signal(char key, std::size_t n, double sample_rate, std::size_t max_samples) {
  for (std::size_t r = 0; r < 4; ++r)
    for (std::size_t c = 0; c < 4; ++c)
      if (keypad[r][c] == key)
        return sinusoid_signal(n, sample_rate, {{rows[r], 1, 0}, {columns[c], 1, 0}}, max_samples);
  throw std::invalid_argument("DTMF: expected one of 0123456789*#ABCD");
}
DtmfPeaks dtmf_peaks(const std::vector<double>& x, double sample_rate, GoertzelFrequencyMode mode) {
  samples_input(x);
  rate_input(sample_rate);
  mode_input(mode);
  DtmfPeaks result;
  for (std::size_t i = 0; i < 8; ++i) {
    result.target_hz[i] = i < 4 ? rows[i] : columns[i - 4];
    result.responses[i] = goertzel_run(x, result.target_hz[i], sample_rate, mode);
    const auto count    = result.responses[i].sample_updates;
    if (count > std::numeric_limits<std::size_t>::max() - result.sample_updates)
      throw std::length_error("DTMF: aggregate update count overflow");
    result.sample_updates += count;
  }
  for (std::size_t i = 1; i < 4; ++i) {
    if (result.responses[i].power > result.responses[result.row].power)
      result.row = i;
    if (result.responses[i + 4].power > result.responses[result.column + 4].power)
      result.column = i;
  }
  result.candidate = keypad[result.row][result.column];
  return result;
}
} // namespace datamunge::algorithms
