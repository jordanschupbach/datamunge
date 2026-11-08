/// @file double_dabble.cpp
/// @brief Shared checked shift-and-add-three conversion, capture and exact capacity lookup.
#include <datamunge/algorithms/double_dabble.hpp>
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Internal fixed-domain helpers and one executing conversion kernel.
namespace {
/// @brief Reject unsupported widths before allocation, subtraction or shifts.
/// @param bits Requested width.
/// @throws std::invalid_argument If outside0..64.
void validate(int bits) {if(bits<0 || bits>64)throw std::invalid_argument("Double Dabble width must be in [0,64]");}
/// @brief Select exactly the low validated width, avoiding a shift by64.
/// @param value Input unsigned value.
/// @param bits Valid width0..64.
/// @return Selected value, zero for an empty bit field.
std::uint64_t select_low(std::uint64_t value,int bits) {
  if(bits==64)return value;
  if(bits==0)return 0;
  return value&((std::uint64_t{1}<<bits)-1);
}
/// @brief Execute the shared recurrence, optionally retaining actual stage snapshots.
/// @param value Original input value.
/// @param bits Requested input width, validated here.
/// @param history Whether to store the actual before/corrected/after/carry vectors.
/// @return Owned result metadata, final digits and counters; history empty if disabled.
/// @throws std::invalid_argument For unsupported width.
/// @throws std::bad_alloc On scratch/output/capture allocation failure.
DoubleDabbleTrace run(std::uint64_t value,int bits,bool history) {
  validate(bits);DoubleDabbleTrace t;t.value=value;t.num_bits=bits;
  t.retained_value=select_low(value,bits);t.capacity=double_dabble_capacity(bits);
  std::vector<int> bcd(t.capacity,0);
  std::uint64_t prefix=0;
  for(int bit=bits-1;bit>=0;--bit) {
    DoubleDabbleStep s;s.bit_index=bit;s.input_bit=static_cast<int>((t.retained_value>>bit)&1);
    if(history)s.before=bcd;
    for(auto& digit:bcd) {
      ++t.work.digit_checks;
      if(digit>=5){digit+=3;++s.corrections;++t.work.add3_corrections;}
    }
    if(history){s.corrected=bcd;s.carry_out.resize(t.capacity);}
    int carry=s.input_bit;
    for(std::size_t k=bcd.size();k-- >0;) {
      const int shifted=(bcd[k]<<1)|carry;
      bcd[k]=shifted&15;carry=(shifted>>4)&1;
      ++t.work.nibble_shifts;t.work.decimal_carries+=carry;
      if(history)s.carry_out[k]=carry;
    }
    prefix=(prefix<<1)|static_cast<std::uint64_t>(s.input_bit);
    ++t.work.bits_processed;
    if(history){s.prefix=prefix;s.after=bcd;t.steps.push_back(std::move(s));}
  }
  auto first=bcd.begin();while(first+1!=bcd.end() && *first==0)++first;
  t.digits.assign(first,bcd.end());return t;
}
} // namespace
std::size_t double_dabble_capacity(int num_bits) {
  validate(num_bits);
  /// @brief Exact decimal lengths of2^b-1 for every supported width b, with length(0)=1.
  static constexpr std::array<unsigned char,65> capacity{1,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,5,6,6,6,7,7,7,7,8,8,8,9,9,9,10,10,10,10,11,11,11,12,12,12,13,13,13,13,14,14,14,15,15,15,16,16,16,16,17,17,17,18,18,18,19,19,19,19,20};
  return capacity[static_cast<std::size_t>(num_bits)];
}
std::vector<int> double_dabble(std::uint64_t value,int num_bits) {return run(value,num_bits,false).digits;}
std::string double_dabble_string(std::uint64_t value,int num_bits) {
  const auto digits=double_dabble(value,num_bits);std::string text;
  for(int d:digits)text+=static_cast<char>('0'+d);
  return text;
}
DoubleDabbleTrace double_dabble_trace(std::uint64_t value,int num_bits) {return run(value,num_bits,true);}
DoubleDabbleWork double_dabble_work(std::uint64_t value,int num_bits) {return run(value,num_bits,false).work;}
std::uint64_t bcd_to_uint64(const std::vector<int>& digits) {
  if(digits.empty())throw std::invalid_argument("BCD digits must not be empty");
  for(int d:digits)if(d<0 || d>9)throw std::invalid_argument("BCD digit outside [0,9]");
  std::uint64_t value=0;
  for(int d:digits) {
    if(value>(std::numeric_limits<std::uint64_t>::max()-static_cast<unsigned>(d))/10)
      throw std::overflow_error("BCD value exceeds uint64_t");
    value=value*10+static_cast<unsigned>(d);
  }
  return value;
}
} // namespace datamunge::algorithms
