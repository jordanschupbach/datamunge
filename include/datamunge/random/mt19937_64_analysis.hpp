#pragma once
/// @file mt19937_64_analysis.hpp
/// @brief Inspectable MT19937-64 word operations and standard-engine conformance checks.
#include <array>
#include <cstddef>
#include <cstdint>
#include <datamunge/dstruct/dataframe.hpp>
#include <vector>
namespace datamunge::random {
/// @brief Inspect one MT19937-64 twist word without mutating an engine.
/// All arithmetic is unsigned 64-bit. Combine the upper 33 bits of left with
/// the lower 31 bits of right, shift right once, conditionally XOR the twist
/// mask, and XOR lag. During a refill, arguments must follow the ordered
/// in-place recurrence: some later lag/right inputs are already updated words.
/// O(1) work/storage, no allocation; safe for concurrent independent calls.
/// @param left Word supplying its upper 33 bits.
/// @param right Word supplying its lower 31 bits.
/// @param lag Word 156 recurrence positions ahead of left.
/// @return Owned four words: joined, shifted, odd correction, final next word.
/// @par Example
/// @code
/// const auto t=dm::random::mt19937_64_twist_steps(0x8000000000000000ULL,1,0);
/// std::print("{:016x}\n",t[3]); // f5026f5aa96619e9
/// @endcode
/// @par References
/// - M. Matsumoto and T. Nishimura, authors' 64-bit reference parameters:
///   https://math.sci.hiroshima-u.ac.jp/m-mat/MT/emt64.html
/// - C++ working draft, https://eel.is/c++draft/rand.eng.mers
/// @see mt19937_64_temper_steps(), MersenneTwister64
std::array<std::uint64_t, 4>
mt19937_64_twist_steps(std::uint64_t left, std::uint64_t right, std::uint64_t lag) noexcept;
/// @brief Inspect the four invertible MT19937-64 output-tempering stages.
/// Shifts and masked XORs operate modulo 2^64. Tempering changes the output
/// map, not the engine state or period; it does not conceal state information.
/// O(1) work/storage without allocation; no shared mutation.
/// @param word Any raw state word, including zero.
/// @return Owned five words: input, after right-29, left-17, left-37, right-43.
/// @par Example
/// @code
/// const auto t=dm::random::mt19937_64_temper_steps(1);
/// std::print("{:016x}\n",t.back()); // 0040002000020801
/// @endcode
/// @par References
/// - Authors' MT19937-64 parameters, https://math.sci.hiroshima-u.ac.jp/m-mat/MT/emt64.html
/// - C++ working draft, https://eel.is/c++draft/rand.predef
/// @see mt19937_64_untemper()
std::array<std::uint64_t, 5> mt19937_64_temper_steps(std::uint64_t word) noexcept;
/// @brief Invert all four MT19937-64 tempering stages, recovering one raw word.
/// Reverses the stages with finite geometric-series inverses of nilpotent
/// masked-shift maps over F2. Every 64-bit input has exactly one preimage.
/// O(1) work/storage without allocation; no shared state. This recovers a word,
/// not a seed, a block alignment or the normal sampler's cache.
/// @param output Full 64-bit tempered output; truncated floating values lose information.
/// @return Raw state word whose tempering equals output, exactly.
/// @par Example
/// @code
/// const auto x=dm::random::mt19937_64_untemper(0x0040002000020801ULL);
/// std::print("{}\n",x); // 1
/// @endcode
/// @par References
/// - Inversion derived from the masked XOR shifts in the C++ engine specification:
///   https://eel.is/c++draft/rand.eng.mers
/// @see mt19937_64_temper_steps(), MersenneTwister64::from_output_block()
std::uint64_t mt19937_64_untemper(std::uint64_t output) noexcept;
/// @brief Compare raw MT19937-64 streams with std::mt19937_64 for declared seeds.
/// Each row constructs fresh engines and compares every word, including refills.
/// This is an implementation check against the standard library, not a randomness
/// test or proof of the period. Seeds and endpoint words are hexadecimal strings
/// to preserve all 64 bits. Duplicate seeds retain duplicate rows; empty seeds
/// returns an empty typed table. Inputs are borrowed and unchanged; O(seeds*draws)
/// work and O(seeds) result storage, with fixed-size engine temporaries.
/// @param seeds Scalar seeds in caller order.
/// @param draws Positive raw outputs compared per seed, default 10000.
/// @return Owned seed/draws/mismatches/native_first/standard_first/native_last/standard_last table.
/// @throws std::invalid_argument If draws is zero, even with empty seeds.
/// @par Example
/// @code
/// const auto t=dm::random::mt19937_64_conformance({5489},10000);
/// std::print("{:.0f}\n",t.double_at("mismatches",0)); // 0
/// @endcode
/// @par References
/// - C++ working draft, https://eel.is/c++draft/rand.predef
/// @see MersenneTwister64, quality_tests()
dstruct::DataFrame mt19937_64_conformance(
    const std::vector<std::uint64_t>& seeds,
    std::size_t                       draws = 10000);
} // namespace datamunge::random
