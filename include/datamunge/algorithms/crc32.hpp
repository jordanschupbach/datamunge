#pragma once
/// @file crc32.hpp
/// @brief Reflected IEEE CRC-32 with actual byte folds and complete single-bit corruption scans.
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace datamunge::algorithms {
/// @brief One actual byte-at-a-time reflected CRC register update.
/// raw_before precedes byte injection; lookup_index=(raw_before XOR byte)&255;
/// raw_after=(raw_before>>8) XOR table_value. Raw registers exclude final XOR.
/// @par Example
/// @code
/// const auto t=dm::algorithms::crc32_trace("1");
/// std::print("{:02X} {:08X}\n",t.steps[0].lookup_index,t.steps[0].raw_after); // CE 7C231048
/// @endcode
/// @par References
/// - PNG specification, CRC algorithm: https://www.w3.org/TR/png/
/// @see Crc32Trace
struct Crc32ByteStep {
  std::size_t index=0; ///< Zero-based byte position in the supplied message.
  std::uint8_t byte=0; ///< Actual unsigned input byte, including embedded zeros.
  std::uint32_t raw_before=0; ///< Actual uncomplemented register before this byte.
  std::uint8_t lookup_index=0; ///< Actual low-byte table address after byte XOR.
  std::uint32_t table_value=0; ///< Actual reflected-polynomial table entry used.
  std::uint32_t raw_after=0; ///< Actual register after the lookup/shift XOR.
};

/// @brief Owned complete actual byte folds from the same kernel as crc32().
/// Input bytes are consumed in order; no view/reference survives. Initialization
/// is FFFFFFFF and final XOR is FFFFFFFF. Empty input has raw FFFFFFFF,
/// checksum zero and no events. O(n) time/storage for n bytes.
/// @par Example
/// @code
/// const auto t=dm::algorithms::crc32_trace("123456789");
/// std::print("{:08X} {}\n",t.checksum,t.steps.size()); // CBF43926 9
/// @endcode
/// @par References
/// - PNG specification, CRC algorithm: https://www.w3.org/TR/png/
/// @see crc32_trace(), Crc32ByteStep
struct Crc32Trace {
  std::uint32_t checksum=0; ///< Public finalized CRC, including the final all-ones XOR.
  std::uint32_t raw=0xFFFFFFFFu; ///< Last raw register, before final XOR.
  std::vector<Crc32ByteStep> steps; ///< Owned actual byte events in input order.
};

/// @brief One independent single-input-bit corruption and its finalized CRC difference.
/// Bit numbering is byte-major, least-significant bit first inside each byte:
/// bit b toggles byte b/8 with mask 1<<(b%8). syndrome is changed XOR baseline.
/// @par Example
/// @code
/// const auto a=dm::algorithms::crc32_flip_scan("A");
/// std::print("{} {}\n",a.flips[0].bit,a.flips[0].syndrome!=0); // 0 true
/// @endcode
/// @see Crc32FlipScan
struct Crc32Flip {
  std::size_t bit=0; ///< Zero-based input-bit address under the documented LSB-first convention.
  std::uint32_t checksum=0; ///< Finalized CRC of the independently corrupted complete message.
  std::uint32_t syndrome=0; ///< XOR difference against the uncorrupted finalized CRC.
  unsigned changed_bits=0; ///< Popcount of syndrome, in [0,32].
};

/// @brief Owned complete census of each independent one-bit corruption of one message.
/// For fixed length, CRC differences are linear in the error pattern and
/// independent of payload. Output-bit counts are not collision probabilities
/// or cryptographic avalanche guarantees. Empty input has no flips, false
/// has_flips and zero summary placeholders. Independent calls share no mutable state.
/// @par Example
/// @code
/// const auto a=dm::algorithms::crc32_flip_scan("The quick brown fox");
/// std::print("{} {} {:.4f}\n",a.flips.size(),a.detected,a.mean_changed_bits); // 152 152 15.4605
/// @endcode
/// @see crc32_flip_scan(), Crc32Flip
struct Crc32FlipScan {
  std::uint32_t checksum=0; ///< Original finalized CRC.
  std::size_t detected=0; ///< Number of nonzero single-bit syndromes.
  bool has_flips=false; ///< Whether any input bit exists; guards the extrema/mean placeholders.
  unsigned minimum_changed_bits=0; ///< Smallest syndrome popcount, or zero if no bits.
  unsigned maximum_changed_bits=0; ///< Largest syndrome popcount, or zero if no bits.
  double mean_changed_bits=0; ///< Arithmetic mean of syndrome popcounts, or zero if no bits.
  std::vector<Crc32Flip> flips; ///< Owned records in ascending original bit order.
};

/// @brief Private immutable byte constants and shared reflected CRC-folding kernel.
namespace detail {
/// @brief Compile-time 256-entry table for reflected polynomial 0xEDB88320.
/// Each entry is eight right-shift/conditional-XOR reductions starting from
/// its byte index. The full normal polynomial is 0x104C11DB7, degree 32.
inline constexpr auto crc32_table=[] {
  std::array<std::uint32_t,256> table{};
  for(std::uint32_t n=0;n<256;++n) {
    std::uint32_t c=n;
    for(int k=0;k<8;++k) c=(c>>1)^(0xEDB88320u&(~(c&1u)+1u));
    table[n]=c;
  }
  return table;
}();
/// @brief Fold bytes through the one actual CRC byte kernel, optionally retaining events.
/// @param raw Uncomplemented input register.
/// @param data Read-only valid byte view, including embedded zeros.
/// @param events Optional event vector; null avoids all allocation/trace writes.
/// @return Final raw register before the public final XOR.
/// @throws std::length_error If an event container exceeds capacity.
/// @throws std::bad_alloc If an event append requires unavailable storage.
inline std::uint32_t crc32_fold(std::uint32_t raw,std::string_view data,std::vector<Crc32ByteStep>* events=nullptr) {
  std::size_t index=0;
  for(const unsigned char byte:data) {
    const auto before=raw;
    const auto address=static_cast<std::uint8_t>((raw^byte)&0xFFu);
    const auto value=crc32_table[address];
    raw=(raw>>8)^value;
    if(events) events->push_back({index,byte,before,address,value,raw});
    ++index;
  }
  return raw;
}
} // namespace detail

/// @brief Compute reflected IEEE CRC-32, also called CRC-32/ISO-HDLC.
/// Polynomial 04C11DB7 (reflected EDB88320), initial register FFFFFFFF,
/// reflected input/output and final XOR FFFFFFFF. Accepts arbitrary bytes,
/// including zeros; empty input returns zero. O(n) time, O(1) storage, one
/// lookup per byte. No allocation, mutation, retained views or shared mutable
/// state. This is an accidental-error checksum, not an authentication tag.
/// @param data Valid read-only byte view; embedded zero bytes do not terminate it.
/// @return Finalized unsigned 32-bit CRC of every supplied byte.
/// @par Example
/// @code
/// std::print("{:08X}\n",dm::algorithms::crc32("123456789")); // CBF43926
/// @endcode
/// @par References
/// - PNG specification, CRC algorithm: https://www.w3.org/TR/png/
/// @see crc32_trace(), crc32_flip_scan()
[[nodiscard]] inline std::uint32_t crc32(std::string_view data) {
  return detail::crc32_fold(0xFFFFFFFFu,data)^0xFFFFFFFFu;
}

/// @brief Retain every actual lookup and raw register from the production CRC kernel.
/// Owned complete byte history in O(n) time/storage. Valid empty input returns
/// the conventional checksum zero and no events. Independent calls are thread-safe.
/// @param data Read-only valid byte view, including zeros and high-bit bytes.
/// @param max_bytes Complete event cap, default 1000000; zero disables this optional cap.
/// @return Owned actual byte trace and the same finalized CRC as crc32(data).
/// @throws std::length_error If the complete event cap or container capacity is exceeded.
/// @throws std::bad_alloc If owned event storage cannot be allocated.
/// @par Example
/// @code
/// const auto t=dm::algorithms::crc32_trace("123456789",9);
/// std::print("{:08X} {:08X}\n",t.raw,t.checksum); // 340BC6D9 CBF43926
/// @endcode
/// @par References
/// - PNG specification, sample CRC code: https://www.w3.org/TR/png/
/// @see Crc32Trace, crc32()
Crc32Trace crc32_trace(std::string_view data,std::size_t max_bytes=1'000'000);

/// @brief Flip specified input-bit addresses into an owned binary string.
/// Fully validates every address before copying or mutation. Repeated addresses
/// toggle repeatedly, so duplicates cancel in pairs. Empty bit lists simply
/// copy the message; the original input/view remains untouched. O(n+m) time
/// and O(n) storage for n bytes and m requested toggles; no references survive.
/// @param data Valid read-only binary byte view.
/// @param bits Bit addresses in byte-major LSB-first order, each below 8*data.size().
/// @return Owned binary message with precisely the requested XOR toggles.
/// @throws std::invalid_argument If any address lies outside the supplied message.
/// @throws std::length_error If bit-count arithmetic or string capacity exceeds size_t limits.
/// @throws std::bad_alloc If owned message storage cannot be allocated.
/// @par Example
/// @code
/// const auto s=dm::algorithms::crc32_flip_bits("A",{0,0,1});
/// std::print("{}\n",s); // C
/// @endcode
/// @see crc32_flip_scan()
std::string crc32_flip_bits(std::string_view data,const std::vector<std::size_t>& bits);

/// @brief Census all independent single-bit corruptions using the same full-message CRC kernel.
/// Computes one baseline and 8n corrupted CRCs, each over n bytes. Exact scan
/// work is n*(8n+1) byte folds; copies/toggles/popcounts are additional work.
/// O(n+8n²) time and O(n+8n) storage. One scratch message is toggled/restored
/// rather than copied for each bit. Whole-call budget/size checks precede
/// allocation and checksum evaluation; results never silently truncate.
/// @param data Valid read-only binary view; empty input yields an absent census.
/// @param max_bytes_processed Complete byte-fold budget, default 100000000; zero disables this optional cap.
/// @return Owned baseline, complete independent bit records and population-count summaries.
/// @throws std::length_error If bit/work arithmetic, complete budget or container limits are exceeded.
/// @throws std::bad_alloc If scratch or owned results cannot be allocated.
/// @par Example
/// @code
/// const auto a=dm::algorithms::crc32_flip_scan("The quick brown fox",2907);
/// std::print("{:08X} {}\n",a.checksum,a.detected); // B74574DE 152
/// @endcode
/// @par References
/// - PNG CRC parameters: https://www.w3.org/TR/png/
/// @see Crc32FlipScan, crc32_flip_bits(), crc32_trace()
Crc32FlipScan crc32_flip_scan(std::string_view data,std::size_t max_bytes_processed=100'000'000);
} // namespace datamunge::algorithms
