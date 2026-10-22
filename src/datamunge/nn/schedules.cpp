#include "datamunge/nn/schedules.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::nn::schedule {

namespace {

void validate(std::int64_t step, double rate) {
  if (step < 0 || !std::isfinite(rate) || rate < 0)
    throw std::invalid_argument("schedule needs a nonnegative step and finite nonnegative rate");
}

}  // namespace

double step_decay(double initial, double factor, std::int64_t interval, std::int64_t step) {
  validate(step, initial);
  if (!std::isfinite(factor) || factor <= 0 || factor > 1 || interval <= 0)
    throw std::invalid_argument("step decay needs factor in (0,1] and positive interval");
  return initial * std::pow(factor, static_cast<double>(step / interval));
}

double exponential(double initial, double factor, std::int64_t interval, std::int64_t step) {
  validate(step, initial);
  if (!std::isfinite(factor) || factor <= 0 || factor > 1 || interval <= 0)
    throw std::invalid_argument("exponential decay needs factor in (0,1] and positive interval");
  return initial * std::pow(factor, static_cast<double>(step) / static_cast<double>(interval));
}

double cosine(double initial, double final_rate, std::int64_t duration, std::int64_t step) {
  validate(step, initial);
  validate(step, final_rate);
  if (duration <= 0) throw std::invalid_argument("cosine duration must be positive");
  const double phase = static_cast<double>(std::min(step, duration)) / static_cast<double>(duration);
  return final_rate + (initial - final_rate) * (1 + std::cos(std::acos(-1.0) * phase)) / 2;
}

double warmup(double peak, std::int64_t duration, std::int64_t step) {
  validate(step, peak);
  if (duration < 0) throw std::invalid_argument("warmup duration must be nonnegative");
  if (!duration) return peak;
  return peak * static_cast<double>(std::min(step, duration)) / static_cast<double>(duration);
}

double warmup_cosine(double peak, double final_rate, std::int64_t warmup_steps, std::int64_t duration,
                     std::int64_t step) {
  validate(step, peak);
  validate(step, final_rate);
  if (warmup_steps < 0 || warmup_steps >= duration)
    throw std::invalid_argument("warmup must be shorter than total duration");
  if (step < warmup_steps) return warmup(peak, warmup_steps, step);
  return cosine(peak, final_rate, duration - warmup_steps, step - warmup_steps);
}

double one_cycle(double initial, double peak, double final_rate, std::int64_t rise, std::int64_t duration,
                 std::int64_t step) {
  validate(step, initial);
  validate(step, peak);
  validate(step, final_rate);
  if (rise <= 0 || rise >= duration || peak < initial || peak < final_rate)
    throw std::invalid_argument("one-cycle needs a peak and two positive phases");
  if (step < rise) return cosine(initial, peak, rise, step);
  return cosine(peak, final_rate, duration - rise, step - rise);
}

double cyclical(double floor_rate, double peak, std::int64_t half_period, std::int64_t step, double amplitude_decay) {
  validate(step, floor_rate);
  validate(step, peak);
  if (peak < floor_rate || half_period <= 0 || half_period > std::numeric_limits<std::int64_t>::max() / 2 ||
      !std::isfinite(amplitude_decay) || amplitude_decay <= 0 || amplitude_decay > 1)
    throw std::invalid_argument("invalid cyclical schedule range, period or amplitude decay");
  const std::int64_t period = 2 * half_period;
  const double       phase  = static_cast<double>(step % period) / static_cast<double>(half_period);
  return floor_rate +
         (peak - floor_rate) * (1 - std::abs(phase - 1)) * std::pow(amplitude_decay, static_cast<double>(step / period));
}

}  // namespace datamunge::nn::schedule
