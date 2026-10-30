/// @file clock.cpp
/// @brief Validated native fixed-step accumulation with bounded catch-up.
#include <algorithm>
#include <cmath>
#include <datamunge/game/clock.hpp>
#include <stdexcept>
namespace datamunge::game {
FixedClock::FixedClock(double step, std::size_t max_steps)
    : step_(step), max_steps_(max_steps) {
  if (!std::isfinite(step) || step < 1e-6 || step > 1 || !max_steps ||
      max_steps > 10000)
    throw std::invalid_argument("invalid fixed clock configuration");
}
TickBatch FixedClock::advance(double elapsed) {
  if (!std::isfinite(elapsed) || elapsed < 0)
    throw std::invalid_argument(
        "elapsed seconds must be finite and nonnegative");
  double accepted = std::min(elapsed, step_ * max_steps_),
         total = remainder_ + accepted;
  auto count = std::min(max_steps_, static_cast<std::size_t>(total / step_));
  remainder_ = total - count * step_;
  if (remainder_ >= step_) {
    if (count < max_steps_) {
      ++count;
      remainder_ -= step_;
    } else
      remainder_ = std::nextafter(step_, 0.);
  }
  return {count, remainder_ / step_, elapsed - accepted};
}
}  // namespace datamunge::game
