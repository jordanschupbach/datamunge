#pragma once
/// @file linear_congruential_analysis.hpp
/// @brief Actual consuming LCG traces, exact affine exponentiation evidence and bounded autonomous low-bit orbit controls.
#include <cstddef>
#include <cstdint>
#include <datamunge/random/random.hpp>
#include <vector>
namespace datamunge::random {
/// @brief Owned actual raw transitions consumed from the caller engine, preserving its inherited normal cache.
/// Complete count+1 quota/geometry and vector reservation precede caller mutation.
/// Afterwards trivial record appends do not allocate. Default million records,
///0 optional cap off. Excludes field/capacity/output/runtime copies and bytes/RAM.
/// @par Example
/// @code
/// dm::random::LinearCongruential64 g(1,5,3);
/// const auto t=dm::random::lcg64_trace(g,3);
/// std::print("{} {}\n",t.initial_state,t.final_state); // 1 218
/// @endcode
/// @par References
/// - Hull and Dobell(1962), Random Number Generators, SIAM Review4(3),230–254, https://doi.org/10.1137/1004061.
/// @see lcg64_trace(), lcg64_jump(), lcg64_low_bits_orbit()
struct LCG64Trace {
  std::uint64_t initial_state{0}; ///< Actual original raw word before any trace updates.
  std::uint64_t final_state{0};   ///< Actual raw word after every committed requested update.
  std::vector<LCG64Step>
              steps; ///< Actual chronological multiply/add records; no reseeding or replay.
  std::size_t captured_entries{
      0}; ///< Actual one initial word plus count step records; final-state copy/fields excluded.
};
/// @brief Actual one exponentiation-loop state after optional bit composition and power squaring.
/// @par Example
/// @code
/// const auto j=dm::random::lcg64_jump(1,5,3,3);
/// std::print("{}\n",j.stages.size()); // 2
/// @endcode
/// @par References
/// - Hull and Dobell(1962), Random Number Generators, SIAM Review4(3),230–254, https://doi.org/10.1137/1004061.
/// @see lcg64_trace(), lcg64_jump(), lcg64_low_bits_orbit()
struct LCG64JumpStage {
  std::uint64_t remaining{0};        ///< Actual nonzero remaining jump count before this iteration.
  bool          applied{false};      ///< Actual low count bit selected power composition.
  std::uint64_t power_multiplier{0}; ///< Actual affine power multiplier before squaring.
  std::uint64_t power_increment{0};  ///< Actual affine power offset before squaring.
  std::uint64_t accumulated_multiplier{
      1}; ///< Actual accumulated affine multiplier after optional selected-bit composition.
  std::uint64_t accumulated_increment{
      0}; ///< Actual accumulated affine offset after optional selected-bit composition.
  std::uint64_t squared_multiplier{0}; ///< Actual next power multiplier after squaring modulo2^64.
  std::uint64_t squared_increment{0};  ///< Actual next power offset after squaring modulo2^64.
};
/// @brief Owned actual logarithmic affine-power evaluation, separate from any engine mutation.
/// This evaluates affine composition, not count raw steps. Stages are actual
/// composition evidence, not a fabricated raw-word trajectory. All products/adds
/// use defined unsigned wrap; no modular inverse/division by a-1 is required.
/// @par Example
/// @code
/// const auto j=dm::random::lcg64_jump(1,5,3,3);
/// std::print("{}\n",j.after); // 218
/// @endcode
/// @par References
/// - Hull and Dobell(1962), Random Number Generators, SIAM Review4(3),230–254, https://doi.org/10.1137/1004061.
/// @see lcg64_trace(), lcg64_jump(), lcg64_low_bits_orbit()
struct LCG64Jump {
  std::uint64_t before{0};            ///< Original supplied unsigned64 word.
  std::uint64_t multiplier{0};        ///< Original arbitrary unsigned64 recurrence multiplier.
  std::uint64_t increment{0};         ///< Original arbitrary unsigned64 recurrence increment.
  std::uint64_t count{0};             ///< Actual requested forward jump, full uint64 domain.
  std::uint64_t after{0};             ///< Actual accumulated affine transform applied to before.
  std::vector<LCG64JumpStage> stages; ///< Actual loop records, at most64; empty for count0.
};
/// @brief Owned actual autonomous reduced-modulus orbit or explicitly incomplete finite prefix.
/// Projection obeys the same recurrence modulo2^bits, independently of hidden
/// high bits. This helper does not consume an engine or empirically traverse
/// its2^64-state period. Prefix exhaustion is not a period certificate.
/// @par Example
/// @code
/// const auto o=dm::random::lcg64_low_bits_orbit(1,5,3,4,16);
/// std::print("{} {}\n",o.complete,o.period); // true 16
/// @endcode
/// @par References
/// - Hull and Dobell(1962), Random Number Generators, SIAM Review4(3),230–254, https://doi.org/10.1137/1004061.
/// @see lcg64_trace(), lcg64_jump(), lcg64_low_bits_orbit()
struct LCG64LowBitOrbit {
  unsigned      bits{0}; ///< Actual projected low-bit width in[1,20].
  std::uint64_t multiplier{
      0}; ///< Original arbitrary unsigned64 recurrence multiplier, not full-period validated.
  std::uint64_t increment{
      0}; ///< Original arbitrary unsigned64 increment, not full-period validated.
  std::vector<std::uint64_t>
      states; ///< Actual initial low-bit residue then every computed next residue, including closing duplicate if complete.
  bool        complete{false}; ///< True only when a real repeated reduced residue was encountered.
  std::size_t repeat_offset{
      0}; ///< Actual first index of closing repeated state when complete;0 placeholder otherwise.
  std::size_t period{
      0}; ///< Actual closed reduced cycle length when complete;0 placeholder otherwise.
};
/// @brief Consume exactly count real raw engine steps with complete preadmitted owned transition evidence.
/// Quota/geometry and full trivial-record reservation precede all generator
/// mutation. No prefix reruns, input state reconstruction or normal-cache reset.
/// @param generator Mutable caller full-period engine; advanced exactly count raw updates on success.
/// @param count Requested complete real raw update count,0 valid.
/// @param max_entries Complete initial-word-plus-step cap, default1000000;0 optional cap off.
/// @return Owned actual initial/final raw words and chronological producing records.
/// @throws std::length_error On complete count/container/quota geometry, before engine mutation.
/// @throws std::bad_alloc On complete trace reservation, before engine mutation.
/// @par Example
/// @code
/// dm::random::LinearCongruential64 g(1,5,3);
/// const auto t=dm::random::lcg64_trace(g,3,4);
/// std::print("{}\n",g.state()); // 218
/// @endcode
/// @par References
/// - Hull and Dobell(1962), Random Number Generators, SIAM Review4(3),230–254, https://doi.org/10.1137/1004061.
/// @see LinearCongruential64, LCG64Step
LCG64Trace
lcg64_trace(LinearCongruential64& generator, std::size_t count, std::size_t max_entries = 1000000);
/// @brief Evaluate a forward affine jump modulo2^64 by actual binary composition.
/// O(log(count+1)) unsigned affine operations, no raw-step replay or mutation.
/// Squaring(a,c) gives(a*a,(a+1)*c) modulo2^64. Counts/stages are composition
/// work, never count raw updates. No inverse or cryptographic claim.
/// @param state Original unsigned64 word.
/// @param multiplier Arbitrary unsigned64 coefficient; invalid full-period values allowed for pure diagnostics.
/// @param increment Arbitrary unsigned64 offset.
/// @param count Exact forward update count,0 through UINT64_MAX.
/// @return Owned actual resulting word and at most64 composition-loop records.
/// @throws std::bad_alloc On bounded loop-record allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::random::lcg64_jump(1,5,3,3).after); // 218
/// @endcode
/// @par References
/// - Hull and Dobell(1962), Random Number Generators, SIAM Review4(3),230–254, https://doi.org/10.1137/1004061.
/// @see LinearCongruential64, LCG64Step
LCG64Jump lcg64_jump(
    std::uint64_t state,
    std::uint64_t multiplier,
    std::uint64_t increment,
    std::uint64_t count);
/// @brief Compute the exact autonomous low-bit recurrence until its first real repeat or an explicit step budget.
/// O(performed steps) expected map work/storage; no mutation or copies of an
/// engine and no claim of raw generator distribution/independence/quality.
/// Complete means a repeated reduced residue, not a2^64 period traversal.
/// @param seed Original full word; initial residue is seed modulo2^bits.
/// @param multiplier Arbitrary coefficient, including non-full-period failure controls.
/// @param increment Arbitrary increment, including even/zero failure controls.
/// @param bits Projected low-bit width in[1,20], completely checked.
/// @param max_steps Real projected-step budget in[0,1048576], default65536;0 returns initial incomplete prefix.
/// @return Owned real reduced states and honest complete/cycle or incomplete-prefix status.
/// @throws std::invalid_argument On width/budget outside explicit finite domains.
/// @throws std::bad_alloc On bounded state/map allocation failure.
/// @par Example
/// @code
/// const auto o=dm::random::lcg64_low_bits_orbit(0,3,1,4,16);
/// std::print("{}\n",o.period); // 8
/// @endcode
/// @par References
/// - Hull and Dobell(1962), Random Number Generators, SIAM Review4(3),230–254, https://doi.org/10.1137/1004061.
/// @see LinearCongruential64, LCG64Step
LCG64LowBitOrbit lcg64_low_bits_orbit(
    std::uint64_t seed,
    std::uint64_t multiplier,
    std::uint64_t increment,
    unsigned      bits,
    std::size_t   max_steps = 65536);
} // namespace datamunge::random
