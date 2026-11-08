#include <datamunge/algorithms/crc32.hpp>
/// @file crc32.cpp
/// @brief Complete owned byte traces and bounded independent bit-corruption censuses.
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private checked sizes, unsigned binary toggles and syndrome population counts.
namespace {
/// @brief Compute a complete number of input bits without size_t overflow.
/// @param bytes Number of message bytes.
/// @return Exactly eight times bytes.
/// @throws std::length_error If the bit count is not representable.
std::size_t crc32_bits(std::size_t bytes) {
  if(bytes>std::numeric_limits<std::size_t>::max()/8) throw std::length_error("CRC-32: bit count overflow");
  return bytes*8;
}
/// @brief Toggle one already validated byte-major LSB-first bit in place.
/// @param data Owned scratch bytes, retained by the caller.
/// @param bit Valid bit address below eight times data.size().
void crc32_toggle(std::string& data,std::size_t bit) {
  data[bit/8]=static_cast<char>(static_cast<unsigned char>(data[bit/8])^(1u<<(bit%8)));
}
/// @brief Count the set bits of one unsigned 32-bit CRC difference.
/// @param value Syndrome, including zero.
/// @return Population count in [0,32].
unsigned crc32_population(std::uint32_t value) {
  unsigned count=0;while(value) { value&=value-1;++count; }return count;
}
} // namespace
Crc32Trace crc32_trace(std::string_view data,std::size_t max_bytes) {
  if(max_bytes && data.size()>max_bytes) throw std::length_error("CRC-32: complete byte trace cap exceeded");
  Crc32Trace result;result.steps.reserve(data.size());
  result.raw=detail::crc32_fold(0xFFFFFFFFu,data,&result.steps);result.checksum=result.raw^0xFFFFFFFFu;
  return result;
}
std::string crc32_flip_bits(std::string_view data,const std::vector<std::size_t>& bits) {
  const auto count=crc32_bits(data.size());
  for(const auto bit:bits)if(bit>=count)throw std::invalid_argument("CRC-32: bit outside message");
  std::string result(data);for(const auto bit:bits)crc32_toggle(result,bit);return result;
}
Crc32FlipScan crc32_flip_scan(std::string_view data,std::size_t max_bytes_processed) {
  const auto bits=crc32_bits(data.size());
  if(bits==std::numeric_limits<std::size_t>::max() ||
     (data.size() && bits+1>std::numeric_limits<std::size_t>::max()/data.size()))
    throw std::length_error("CRC-32: complete byte-fold count overflow");
  const auto work=data.size()*(bits+1);
  if(max_bytes_processed && work>max_bytes_processed) throw std::length_error("CRC-32: complete byte-fold budget exceeded");
  Crc32FlipScan result;result.flips.reserve(bits);std::string scratch(data);result.checksum=crc32(data);
  result.has_flips=bits!=0;
  if(bits)result.minimum_changed_bits=32;
  long double total=0;
  for(std::size_t bit=0;bit<bits;++bit) {
    crc32_toggle(scratch,bit);const auto changed=crc32(scratch);crc32_toggle(scratch,bit);
    const auto syndrome=changed^result.checksum;const auto count=crc32_population(syndrome);
    result.flips.push_back({bit,changed,syndrome,count});result.detected+=syndrome!=0;
    result.minimum_changed_bits=std::min(result.minimum_changed_bits,count);
    result.maximum_changed_bits=std::max(result.maximum_changed_bits,count);total+=count;
  }
  if(bits)result.mean_changed_bits=static_cast<double>(total/bits);
  return result;
}
} // namespace datamunge::algorithms
