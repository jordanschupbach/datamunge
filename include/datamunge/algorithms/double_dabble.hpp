#pragma once
/// @file double_dabble.hpp
/// @brief Checked low-bit binary-to-BCD conversion, actual nibble traces and logical work.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// @brief Logical work of the shared fixed-capacity shift-and-add-three kernel.
/// Counts nibble-level operations, excluding validation, allocation, zero trimming,
/// output formatting and trace copies. With B requested bits and D capacity,
/// digit_checks=nibble_shifts=B*D. Corrections and decimal carries depend on the
/// selected value and are equal; these are not CPU timings or hardware cycles.
/// @par Example
/// @code
/// const auto w=dm::algorithms::double_dabble_work(13,4);
/// std::print("{} {} {}\n",w.bits_processed,w.digit_checks,w.add3_corrections); // 4 8 1
/// @endcode
/// @see double_dabble_work(), DoubleDabbleTrace
struct DoubleDabbleWork {
  std::size_t bits_processed=0; ///< Requested input bits consumed, including leading zeros.
  std::size_t digit_checks=0; ///< Before-shift comparisons against five, one per nibble per bit.
  std::size_t add3_corrections=0; ///< Nibbles receiving three before a shift.
  std::size_t nibble_shifts=0; ///< Nibble shifts with incoming carry, one per nibble per bit.
  std::size_t decimal_carries=0; ///< Carry-out ones across all nibbles, including the most significant.
};

/// @brief One owned actual correction-and-shift stage, with padded most-significant-first nibbles.
/// before and after contain decimal digits0..9. corrected can contain8..12 and
/// is an intermediate binary nibble field, not decimal text. carry_out[i] moves
/// from nibble i toward i-1; the rightmost nibble receives input_bit. All four
/// vectors have trace.capacity entries. Default fields are an uncomputed placeholder.
/// @par Example
/// @code
/// const auto t=dm::algorithms::double_dabble_trace(13,4);
/// std::print("{} {}\n",t.steps.back().corrected,t.steps.back().after); // [0, 9] [1, 3]
/// @endcode
/// @par References
/// - H. Zheng, Modeling of Combinational Circuits, Shift and Add-3
///   (Double-Dabble), University of South Florida,
///   https://cse.usf.edu/~haozheng/teach/cda4253/slides/vhdl-2.pdf
/// @see DoubleDabbleTrace, double_dabble_trace()
struct DoubleDabbleStep {
  int bit_index=0; ///< Original selected bit index, visited from num_bits-1 down to zero.
  int input_bit=0; ///< Actual zero/one bit shifted into the least-significant nibble.
  std::uint64_t prefix=0; ///< Selected binary prefix after this stage, equal to decoded after.
  std::size_t corrections=0; ///< Actual add-three corrections within this stage.
  std::vector<int> before; ///< Padded decimal digit field before correction.
  std::vector<int> corrected; ///< Padded nibble field immediately after adding three where needed.
  std::vector<int> after; ///< Padded valid BCD field after shifting and carry propagation.
  std::vector<int> carry_out; ///< Zero/one carry from each nibble, aligned most-significant first.
};

/// @brief Owned actual binary-to-BCD result, steps and logical work from the shared kernel.
/// Captures at most64 stages of at most20 nibbles, without replay. Four nibble
/// snapshots per stage use O(B*D) storage; ordinary conversion/work calls keep
/// O(D) scratch. This finite uint64_t domain needs no optional history cap.
/// Default fields form an uncomputed placeholder, not a computed zero-width result.
/// Caller inputs are unchanged, no borrowed references or shared mutable state.
/// @par Example
/// @code
/// const auto t=dm::algorithms::double_dabble_trace(243,4);
/// std::print("{} {} {}\n",t.retained_value,t.capacity,t.digits); // 3 2 [3]
/// @endcode
/// @see double_dabble_trace(), DoubleDabbleStep, DoubleDabbleWork
struct DoubleDabbleTrace {
  std::uint64_t value=0; ///< Original unmasked input value.
  int num_bits=0; ///< Validated requested width in [0,64].
  std::uint64_t retained_value=0; ///< Low num_bits bits, or zero for width zero.
  std::size_t capacity=0; ///< Fixed padded nibble capacity for the requested width, at least one.
  std::vector<int> digits; ///< Final decimal digits, stripped of leading zeros except singleton zero.
  std::vector<DoubleDabbleStep> steps; ///< Actual stages in execution order; empty at width zero.
  DoubleDabbleWork work; ///< Actual logical operations, excluding snapshot/formatting overhead.
};

/// @brief Minimum BCD digit capacity sufficient for every value of a requested binary width.
/// Exact fixed integer lookup for widths0..64; no floating logarithm or runtime
/// division. Capacity is one for width zero and20 for width64. O(1), no allocation.
/// @param num_bits Requested width in [0,64].
/// @return Decimal digit count of2^num_bits-1, with zero represented by one digit.
/// @throws std::invalid_argument If the width is outside [0,64].
/// @par Example
/// @code
/// std::print("{} {}\n",dm::algorithms::double_dabble_capacity(8),dm::algorithms::double_dabble_capacity(64)); // 3 20
/// @endcode
/// @see double_dabble(), double_dabble_work()
std::size_t double_dabble_capacity(int num_bits);

/// @brief Convert the low requested bits of a uint64_t into decimal BCD digits.
/// Preserves the original function/return ordering and deliberate high-bit
/// truncation. Width zero selects no bits and returns{0}. Width64 uses every
/// input bit without a shift by64. Leading output zeros are removed, retaining
/// one zero. The shared conversion loop uses comparisons, add-three corrections,
/// shifts, OR and masks; capacity is an exact fixed lookup. O(B*D) logical work
/// and O(D) scratch, B=num_bits and D=capacity. No mutation/shared state.
/// @param value Original binary input, unchanged; bits above the width are ignored.
/// @param num_bits Requested input width in [0,64], including leading-zero bits.
/// @return Owned most-significant-first digits in0..9, at most20; never empty.
/// @throws std::invalid_argument If the width is outside [0,64], before allocation/arithmetic.
/// @throws std::bad_alloc If output/scratch storage cannot be allocated.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::double_dabble(243,8)); // [2, 4, 3]
/// @endcode
/// @par References
/// - H. Zheng, Modeling of Combinational Circuits, Shift and Add-3
///   (Double-Dabble), University of South Florida,
///   https://cse.usf.edu/~haozheng/teach/cda4253/slides/vhdl-2.pdf
/// @see double_dabble_string(), double_dabble_trace(), bcd_to_uint64()
std::vector<int> double_dabble(std::uint64_t value,int num_bits);

/// @brief Decimal string of the selected low bits through the same Double Dabble kernel.
/// Returns ordinary decimal text, not a packed BCD byte string. Width-zero,
/// truncation, validation and complexity follow double_dabble(); no shared state.
/// @param value Input binary value, unchanged.
/// @param num_bits Requested width in [0,64].
/// @return Owned nonempty ASCII digits with no leading zero except "0".
/// @throws std::invalid_argument If the width is outside [0,64].
/// @throws std::bad_alloc If string/scratch storage cannot be allocated.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::double_dabble_string(243,4)); // 3
/// @endcode
/// @par References
/// - H. Zheng, Modeling of Combinational Circuits, Shift and Add-3
///   (Double-Dabble), University of South Florida,
///   https://cse.usf.edu/~haozheng/teach/cda4253/slides/vhdl-2.pdf
/// @see double_dabble(), double_dabble_trace()
std::string double_dabble_string(std::uint64_t value,int num_bits);

/// @brief Capture actual Double Dabble nibble states without replay.
/// The executing kernel is shared with ordinary conversion. Snapshots retain
/// requested-width padding, while final digits remove leading zeros. Every
/// after state decodes to the selected input prefix. Captured storage is bounded
/// by64 stages times four20-nibble vectors plus scalar fields; O(B*D) work/storage.
/// @param value Original input, unchanged; high bits outside the width are ignored.
/// @param num_bits Requested width in [0,64]; zero gives no stages and digits{0}.
/// @return Owned input metadata, actual snapshots, final digits and work counters.
/// @throws std::invalid_argument If width is outside [0,64], before capture.
/// @throws std::bad_alloc If trace/scratch storage cannot be allocated.
/// @par Example
/// @code
/// const auto t=dm::algorithms::double_dabble_trace(13,4);
/// std::print("{} {}\n",t.steps.size(),t.steps.back().prefix); // 4 13
/// @endcode
/// @par References
/// - H. Zheng, Modeling of Combinational Circuits, Shift and Add-3
///   (Double-Dabble), University of South Florida,
///   https://cse.usf.edu/~haozheng/teach/cda4253/slides/vhdl-2.pdf
/// @see double_dabble(), DoubleDabbleStep
DoubleDabbleTrace double_dabble_trace(std::uint64_t value,int num_bits);

/// @brief Measure actual nibble work while converting, without retaining stage snapshots.
/// Same conversion, masking and validation as double_dabble(); O(B*D) logical
/// work and O(D) scratch. Includes all allocated leading-zero nibbles.
/// @param value Input value, unchanged; high bits above the width are ignored.
/// @param num_bits Requested width in [0,64].
/// @return Owned logical counters, zero at width zero.
/// @throws std::invalid_argument If width is outside [0,64].
/// @throws std::bad_alloc If scratch storage cannot be allocated.
/// @par Example
/// @code
/// const auto w=dm::algorithms::double_dabble_work(0,64);
/// std::print("{} {}\n",w.digit_checks,w.add3_corrections); // 1280 0
/// @endcode
/// @see double_dabble(), DoubleDabbleWork
DoubleDabbleWork double_dabble_work(std::uint64_t value,int num_bits);

/// @brief Decode an unpacked most-significant-first BCD digit vector with checked uint64_t arithmetic.
/// This inverse is ordinary decimal Horner evaluation using multiply-ten/add,
/// not a reverse shift/subtract-three algorithm. Leading zeros are accepted.
/// Complete digit-domain validation precedes overflow checking. O(n) time,
/// O(1) storage, no mutation/allocation or shared state.
/// @param digits Borrowed nonempty vector of decimal digit values0..9.
/// @return Exact decoded integer, including zero for any nonempty all-zero vector.
/// @throws std::invalid_argument If empty or any element lies outside0..9.
/// @throws std::overflow_error If the valid decimal value exceeds UINT64_MAX.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::bcd_to_uint64({0,2,4,3})); // 243
/// @endcode
/// @see double_dabble()
std::uint64_t bcd_to_uint64(const std::vector<int>& digits);
} // namespace datamunge::algorithms
