#pragma once
/// @file ntt_multiplication.hpp
/// @brief Checked exact fixed-prime decimal convolution with actual radix-two stage/carry histories.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
/// @brief Canonical bounded decimal multiplication and owned transform evidence.
namespace datamunge::algorithms {
/// @brief Complete decimal, transform and optional state-history bounds; zero disables a policy.
/// @par Example
/// @code
/// const dm::algorithms::NTTMultiplicationOptions value{};
/// std::print("{}\n",value.max_digits); // 1000000
/// @endcode
struct NTTMultiplicationOptions {
  std::size_t max_digits{
      1000000}; ///< Complete original character count of either operand, before leading-zero removal.
  std::size_t max_transform_size{
      8388608}; ///< Complete padded transform length, also subject to the hard2^23 root limit.
  std::size_t max_trace_entries{
      1000000}; ///< Captured input/product/coefficient scalars, stage states and actual carry records.
};
/// @brief Identity of one actual transform in a decimal product.
/// @par Example
/// @code
/// const auto value=dm::algorithms::NTTTransform::First;
/// std::print("{}\n",static_cast<int>(value)); // 0
/// @endcode
enum class NTTTransform {
  First,  ///< Forward transform of the first operand.
  Second, ///< Forward transform of the second operand.
  Inverse ///< Inverse transform of the actual pointwise spectral product.
};
/// @brief Actual phase that produced a captured transform state.
/// @par Example
/// @code
/// const auto value=dm::algorithms::NTTPhase::BitReverse;
/// std::print("{}\n",static_cast<int>(value)); // 0
/// @endcode
enum class NTTPhase {
  BitReverse, ///< Actual completed bit-reversal permutation.
  Butterfly,  ///< Actual completed radix-two butterfly stage.
  Scale       ///< Actual inverse-N modular scaling.
};
/// @brief One actual completed transform phase, owning its current residues.
/// @par Example
/// @code
/// const dm::algorithms::NTTStage value{};
/// std::print("{}\n",value.length); // 0
/// @endcode
struct NTTStage {
  NTTTransform  transform{}; ///< Actual input/pointwise-product stream.
  NTTPhase      phase{};     ///< Actual bit reversal, butterfly or inverse scaling.
  std::size_t   length{};    ///< Actual butterfly block length; zero outside butterfly stages.
  std::uint64_t root{};      ///< Actual stage twiddle root, inverse-N on Scale, zero on BitReverse.
  std::size_t
      butterflies{}; ///< Actual butterflies executed in this phase, zero outside butterfly stages.
  std::vector<std::uint64_t>
      values; ///< Actual current residue vector, copied after this producing phase.
};
/// @brief One actual base-ten carry propagation through the recovered integer convolution.
/// @par Example
/// @code
/// const dm::algorithms::NTTCarry value{};
/// std::print("{}\n",value.digit); // 0
/// @endcode
struct NTTCarry {
  std::size_t   index{};       ///< Actual little-endian coefficient position.
  std::uint64_t coefficient{}; ///< Actual inverse-transformed exact integer coefficient.
  std::uint64_t before{};      ///< Actual incoming base-ten carry.
  unsigned      digit{};       ///< Actual emitted decimal digit0..9.
  std::uint64_t after{};       ///< Actual outgoing integer carry.
};
/// @brief Owned exact decimal product and actual selected transform work.
/// @par Example
/// @code
/// const dm::algorithms::NTTMultiplicationRun value{};
/// std::print("{}\n",value.butterflies); // 0
/// @endcode
struct NTTMultiplicationRun {
  std::string product; ///< Actual canonical decimal product, no leading zeros except zero itself.
  std::string first;   ///< Owned validated first operand after leading-zero removal.
  std::string second;  ///< Owned validated second operand after leading-zero removal.
  std::size_t
      transform_size{}; ///< Actual padded power-of-two length; zero for a zero-product shortcut.
  std::size_t bit_reversal_visits{}; ///< Actual index visits i=1..N-1 in each of three transforms.
  std::size_t bit_reversal_swaps{};  ///< Actual committed bit-reversal swaps.
  std::size_t butterflies{}; ///< Actual butterflies across both forward and one inverse transform.
  std::size_t
      modular_products{}; ///< Actual native modular multiplications, including delegated root/inverse powering.
  std::size_t
      root_products{}; ///< Actual modular products reported by canonical mod_pow_trace calls.
  std::size_t pointwise_products{}; ///< Actual spectral entry products, N on a nonzero run.
  std::size_t
      inverse_scale_products{}; ///< Actual final inverse scaling products, N on a nonzero run.
  std::size_t carry_updates{};  ///< Actual padded coefficient carry steps, N on a nonzero run.
  std::vector<std::uint64_t>
      first_padded; ///< Actual optional first little-endian decimal digit input, padded to N.
  std::vector<std::uint64_t> second_padded; ///< Actual optional second padded input.
  std::vector<std::uint64_t> pointwise;     ///< Actual optional multiplied forward spectra.
  std::vector<std::uint64_t>
      coefficients; ///< Actual optional inverse-scaled exact integer convolution.
  std::vector<NTTStage>
      stages; ///< Actual optional completed phase states, not independent transforms.
  std::vector<NTTCarry> carries; ///< Actual optional padded base-ten carry records.
};
/// @brief Multiply checked nonnegative decimal integers using one exact fixed-prime NTT convolution.
/// Requires nonempty ASCII digit strings; validates all original characters/counts before a zero shortcut, accepts leading zeros and normalizes them. Uses prime998244353 and primitive root3, with a hard transform capacity2^23. Pads to the smallest power of two at least len(a)+len(b), preserving the historical transform geometry. Checks81*min(normalized lengths)<prime so every base-ten convolution coefficient is recovered as an ordinary integer, then performs actual carry propagation. Zero products return0 without transforms. This is a bounded fixed-prime radix-two method, not an implementation of the original arbitrary-length Schoenhage-Strassen ring algorithm. Actual word-operation work is O(N log N), storage O(N), within the documented root/coefficient bounds. Records count actual bit-reversal visits/swaps, butterflies, root powering products, pointwise/scaling products and carry updates separately; they do not establish wall-clock speedup over another method. Capture geometry is known before allocation:5*N+(3*log2(N)+4)*(N+1) logical coefficient/phase/carry entries; normalized strings and output are intrinsic. Inputs unchanged and outputs owned, no shared state.
/// @param a Complete first unsigned ASCII decimal operand.
/// @param b Complete second unsigned ASCII decimal operand.
/// @param options Complete original-input, hard/policy transform and optional capture controls.
/// @return Owned exact decimal product, selected actual work and optional same-execution states.
/// @throws std::invalid_argument Empty or nondecimal operand.
/// @throws std::length_error Complete digit, root/coefficient, transform, capture or container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::ntt_multiply_run("123","45");
/// std::print("{}\n",r.product); // 5535
/// @endcode
/// @par References
/// - Cooley and Tukey (1965), An Algorithm for the Machine Calculation of Complex Fourier Series; fixed-prime variant here. https://web.stanford.edu/class/cme324/classics/cooley-tukey.pdf
/// @see ntt_mul(), mod_pow_trace()
NTTMultiplicationRun ntt_multiply_run(
    const std::string&              a,
    const std::string&              b,
    const NTTMultiplicationOptions& options = {});
/// @brief Multiply checked nonnegative decimal integers using one exact fixed-prime NTT convolution with actual phase/carry histories.
/// Requires nonempty ASCII digit strings; validates all original characters/counts before a zero shortcut, accepts leading zeros and normalizes them. Uses prime998244353 and primitive root3, with a hard transform capacity2^23. Pads to the smallest power of two at least len(a)+len(b), preserving the historical transform geometry. Checks81*min(normalized lengths)<prime so every base-ten convolution coefficient is recovered as an ordinary integer, then performs actual carry propagation. Zero products return0 without transforms. This is a bounded fixed-prime radix-two method, not an implementation of the original arbitrary-length Schoenhage-Strassen ring algorithm. Actual word-operation work is O(N log N), storage O(N), within the documented root/coefficient bounds. Records count actual bit-reversal visits/swaps, butterflies, root powering products, pointwise/scaling products and carry updates separately; they do not establish wall-clock speedup over another method. Capture geometry is known before allocation:5*N+(3*log2(N)+4)*(N+1) logical coefficient/phase/carry entries; normalized strings and output are intrinsic. Inputs unchanged and outputs owned, no shared state.
/// @param a Complete first unsigned ASCII decimal operand.
/// @param b Complete second unsigned ASCII decimal operand.
/// @param options Complete original-input, hard/policy transform and optional capture controls.
/// @return Owned exact decimal product, selected actual work and optional same-execution states.
/// @throws std::invalid_argument Empty or nondecimal operand.
/// @throws std::length_error Complete digit, root/coefficient, transform, capture or container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::ntt_multiply_trace("123","45");
/// std::print("{}\n",r.product); // 5535
/// @endcode
/// @par References
/// - Cooley and Tukey (1965), An Algorithm for the Machine Calculation of Complex Fourier Series; fixed-prime variant here. https://web.stanford.edu/class/cme324/classics/cooley-tukey.pdf
/// @see ntt_mul(), mod_pow_trace()
NTTMultiplicationRun ntt_multiply_trace(
    const std::string&              a,
    const std::string&              b,
    const NTTMultiplicationOptions& options = {});
/// @brief Return the checked exact NTT decimal product through the preserved signature.
/// Delegates to the same producer/default bounds. Ordinary valid decimal outputs are preserved. Empty/malformed strings, roots beyond the fixed-prime capacity and oversized requests now reject instead of producing unchecked residues.
/// @param a Complete first nonempty decimal operand.
/// @param b Complete second nonempty decimal operand.
/// @return Canonical exact nonnegative decimal product.
/// @throws std::invalid_argument Empty/nondecimal input.
/// @throws std::length_error Default digit, hard root/coefficient or container policy exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::ntt_mul("111","111")); // 12321
/// @endcode
/// @par References
/// - Cooley and Tukey (1965), An Algorithm for the Machine Calculation of Complex Fourier Series; fixed-prime variant here. https://web.stanford.edu/class/cme324/classics/cooley-tukey.pdf
/// @see ntt_multiply_run()
std::string ntt_mul(const std::string& a, const std::string& b);
} // namespace datamunge::algorithms
