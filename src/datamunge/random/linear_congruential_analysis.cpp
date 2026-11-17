/// @file linear_congruential_analysis.cpp
/// @brief Actual raw affine-step kernel, preadmitted consuming trace, logarithmic jump and bounded low-bit recurrence.
#include <datamunge/random/linear_congruential_analysis.hpp>
#include <limits>
#include <stdexcept>
#include <unordered_map>
namespace datamunge::random {
LCG64Step
lcg64_step(std::uint64_t state, std::uint64_t multiplier, std::uint64_t increment) noexcept {
  LCG64Step out;
  out.before     = state;
  out.multiplier = multiplier;
  out.increment  = increment;
  out.multiplication_wrapped =
      multiplier && state > std::numeric_limits<std::uint64_t>::max() / multiplier;
  out.product          = multiplier * state;
  out.addition_wrapped = out.product > std::numeric_limits<std::uint64_t>::max() - increment;
  out.after            = out.product + increment;
  return out;
}
LCG64Trace
lcg64_trace(LinearCongruential64& generator, std::size_t count, std::size_t max_entries) {
  if (count == std::numeric_limits<std::size_t>::max() ||
      count > std::vector<LCG64Step>{}.max_size() || (max_entries && count + 1 > max_entries))
    throw std::length_error("LCG complete trace count/quota");
  LCG64Trace out;
  out.captured_entries = count + 1;
  out.steps.reserve(count);
  out.initial_state = generator.state();
  for (std::size_t i = 0; i < count; ++i)
    out.steps.push_back(generator.next_step());
  out.final_state = generator.state();
  return out;
}
LCG64Jump lcg64_jump(
    std::uint64_t state,
    std::uint64_t multiplier,
    std::uint64_t increment,
    std::uint64_t count) {
  LCG64Jump out;
  out.before     = state;
  out.multiplier = multiplier;
  out.increment  = increment;
  out.count      = count;
  out.stages.reserve(64);
  std::uint64_t a = multiplier, c = increment, ma = 1, mc = 0, remaining = count;
  while (remaining) {
    LCG64JumpStage stage;
    stage.remaining        = remaining;
    stage.applied          = remaining & 1;
    stage.power_multiplier = a;
    stage.power_increment  = c;
    if (stage.applied) {
      mc = a * mc + c;
      ma = a * ma;
    }
    stage.accumulated_multiplier = ma;
    stage.accumulated_increment  = mc;
    stage.squared_increment      = (a + 1) * c;
    stage.squared_multiplier     = a * a;
    out.stages.push_back(stage);
    a = stage.squared_multiplier;
    c = stage.squared_increment;
    remaining >>= 1;
  }
  out.after = ma * state + mc;
  return out;
}
LCG64LowBitOrbit lcg64_low_bits_orbit(
    std::uint64_t seed,
    std::uint64_t multiplier,
    std::uint64_t increment,
    unsigned      bits,
    std::size_t   max_steps) {
  if (bits < 1 || bits > 20 || max_steps > 1048576)
    throw std::invalid_argument("LCG reduced width/step budget");
  LCG64LowBitOrbit out;
  out.bits       = bits;
  out.multiplier = multiplier;
  out.increment  = increment;
  out.states.reserve(max_steps + 1);
  const std::uint64_t                            mask  = (std::uint64_t{1} << bits) - 1;
  auto                                           state = seed & mask;
  std::unordered_map<std::uint64_t, std::size_t> first;
  first.emplace(state, 0);
  out.states.push_back(state);
  for (std::size_t k = 0; k < max_steps; ++k) {
    state = lcg64_step(state, multiplier, increment).after & mask;
    out.states.push_back(state);
    const auto found = first.find(state);
    if (found != first.end()) {
      out.complete      = true;
      out.repeat_offset = found->second;
      out.period        = out.states.size() - 1 - found->second;
      break;
    }
    first.emplace(state, out.states.size() - 1);
  }
  return out;
}
} // namespace datamunge::random
