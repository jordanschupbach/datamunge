/// @file batch.cpp
/// @brief Reusable batched native stepping, transactional failures and independent random streams.
#include <algorithm>
#include <cmath>
#include <datamunge/sim/batch.hpp>
#include <stdexcept>
namespace datamunge::sim {
/// @brief Translation-unit-local validation and construction helpers.
namespace {
/// @brief SplitMix64 output permutation; unsigned overflow is intentional.
/// @param value Counter value to scramble.
/// @return Permuted 64-bit value.
std::uint64_t mix(std::uint64_t value) noexcept {
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}
} // namespace
BatchWorld::BatchWorld(
    const World&     prototype,
    std::size_t      count,
    compute::Context context,
    std::uint64_t    value)
    : context_(std::move(context)) {
  if (!count || context_.device().backend != compute::Backend::CPU)
    throw std::invalid_argument("batch physics requires positive count and CPU context");
  worlds_.reserve(count);
  initial_.reserve(count);
  steps_.resize(count);
  random_.resize(count);
  advanced_.resize(count);
  errors_.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    worlds_.push_back(prototype.fork());
    initial_.push_back(worlds_.back()->save_state());
  }
  seed(value);
}
World& BatchWorld::world(std::size_t index) {
  return *worlds_.at(index);
}
const World& BatchWorld::world(std::size_t index) const {
  return *worlds_.at(index);
}
void BatchWorld::capture_reset(std::size_t index) {
  const auto state = world(index).save_state();
  initial_[index]  = state;
  steps_[index]    = 0;
}
void BatchWorld::validate_mask(const std::vector<std::uint8_t>& mask) const {
  if ((!mask.empty() && mask.size() != size()) ||
      std::any_of(mask.begin(), mask.end(), [](auto v) { return v > 1; }))
    throw std::invalid_argument("batch mask must be empty or one binary byte per environment");
}
void BatchWorld::reset(const std::vector<std::uint8_t>& mask) {
  validate_mask(mask);
  for (std::size_t i = 0; i < size(); ++i)
    if (mask.empty() || mask[i])
      worlds_[i]->validate_state(initial_[i]);
  for (std::size_t i = 0; i < size(); ++i)
    if (mask.empty() || mask[i]) {
      worlds_[i]->restore_state(initial_[i]);
      steps_[i] = 0;
    }
}
void BatchWorld::step(double delta, const std::vector<std::uint8_t>& mask) {
  validate_mask(mask);
  if (!std::isfinite(delta) || delta <= 0 || delta > 1. / 30)
    throw std::invalid_argument("batch delta must be in (0,1/30]");
  std::fill(advanced_.begin(), advanced_.end(), 0);
  std::fill(errors_.begin(), errors_.end(), std::exception_ptr{});
  /// @brief Stack-owned job descriptor keeps the callback within std::function's small-object
  /// storage.
  struct Job {
    BatchWorld* owner; ///< Borrowed coordinator, alive through synchronous submission.
    const std::vector<std::uint8_t>* mask; ///< Borrowed validated row mask.
    double delta; ///< Validated positive outer-step seconds.
  } job{this, &mask, delta};
  context_.parallel_for(size(), [job = &job](std::size_t i) {
    auto&       batch = *job->owner;
    const auto& mask  = *job->mask;
    if (!mask.empty() && !mask[i])
      return;
    try {
      batch.worlds_[i]->step(job->delta);
      batch.advanced_[i] = 1;
    } catch (...) {
      batch.errors_[i] = std::current_exception();
    }
  });
  std::exception_ptr error;
  for (const auto& e : errors_)
    if (e) {
      error = e;
      break;
    }
  if (error) {
    for (std::size_t i = 0; i < size(); ++i)
      if (advanced_[i])
        worlds_[i]->rewind_step();
    std::rethrow_exception(error);
  }
  for (std::size_t i = 0; i < size(); ++i)
    steps_[i] += advanced_[i];
}
std::uint64_t BatchWorld::steps(std::size_t index) const {
  return steps_.at(index);
}
void BatchWorld::seed(std::uint64_t value) noexcept {
  for (std::size_t i = 0; i < size(); ++i)
    random_[i] = mix(value + UINT64_C(0x9e3779b97f4a7c15) * (i + 1));
}
double BatchWorld::uniform(std::size_t index) {
  auto& state = random_.at(index);
  state += UINT64_C(0x9e3779b97f4a7c15);
  return static_cast<double>(mix(state) >> 11) * 0x1.0p-53;
}
} // namespace datamunge::sim
