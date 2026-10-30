#pragma once

/// @file schedules.hpp
/// @brief Learning-rate schedules as pure functions of the step.
///
/// The formulas match tflow's `nn::schedule`. Every function throws
/// std::invalid_argument for a negative step, negative or non-finite rates
/// and out-of-range shape parameters.

#include <cstdint>

/// @brief Validated learning-rate functions indexed by a nonnegative training step.
namespace datamunge::nn::schedule {

/// initial * factor^floor(step / interval), factor in (0, 1].
double step_decay(double initial, double factor, std::int64_t interval, std::int64_t step);
/// initial * factor^(step / interval), continuous.
double exponential(double initial, double factor, std::int64_t interval, std::int64_t step);
/// Half-cosine from initial to final over `duration` steps, then constant.
double cosine(double initial, double final_rate, std::int64_t duration, std::int64_t step);
/// Linear ramp from 0 to peak over `duration` steps, then constant (duration 0: always peak).
double warmup(double peak, std::int64_t duration, std::int64_t step);
/// Linear warmup for `warmup_steps`, then cosine to `final_rate` at `duration`.
double warmup_cosine(double peak, double final_rate, std::int64_t warmup_steps, std::int64_t duration,
                     std::int64_t step);
/// Cosine rise from initial to peak over `rise` steps, then cosine to final at `duration`.
double one_cycle(double initial, double peak, double final_rate, std::int64_t rise, std::int64_t duration,
                 std::int64_t step);
/// Triangular cycles between floor and peak with the given half period; the
/// amplitude shrinks by `amplitude_decay` each full cycle.
double cyclical(double floor_rate, double peak, std::int64_t half_period, std::int64_t step,
                double amplitude_decay = 1.0);

}  // namespace datamunge::nn::schedule
