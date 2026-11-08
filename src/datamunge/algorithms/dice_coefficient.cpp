/// @file dice_coefficient.cpp
/// @brief Checked overlap normalization and shared sparse byte-pair consumption.
#include <datamunge/algorithms/dice_coefficient.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>
namespace datamunge::algorithms {
/// @brief Local byte packing and dictionary kernel; no shared mutable state.
namespace {
/// @brief Pack two adjacent bytes in unsigned lexicographic order.
/// @param s Byte string containing positions offset and offset+1.
/// @param offset Valid start offset, strictly below s.size()-1.
/// @return First byte in high eight bits and second in low eight bits.
std::uint16_t key_at(const std::string& s,std::size_t offset) {
  static_assert(std::numeric_limits<unsigned char>::digits==8,"byte-bigram keys require 8-bit bytes");
  return static_cast<std::uint16_t>((static_cast<unsigned char>(s[offset])<<8)|static_cast<unsigned char>(s[offset+1]));
}
/// @brief Consume the left occurrence dictionary in ascending right-offset order.
/// @param a Borrowed first byte string.
/// @param b Borrowed second byte string.
/// @param shape Prechecked occurrence sizes/total; initial shared count zero.
/// @param trace Null for scalar calls; otherwise records actual rows/events/counts.
/// @return Legacy-compatible checked scores with the actual intersection count.
/// @throws std::bad_alloc For allocation failure; container length errors propagate.
DiceOverlap consume(const std::string& a,const std::string& b,const DiceOverlap& shape,DiceTrace* trace) {
  std::unordered_map<std::uint16_t,DiceBigramCount> counts;
  counts.reserve(std::min<std::size_t>(shape.left_bigrams,65536));
  for(std::size_t i=0;i<shape.left_bigrams;++i) {
    const auto key=key_at(a,i);auto& row=counts.try_emplace(key,DiceBigramCount{key}).first->second;
    ++row.left_count;if(trace)++trace->work.left_updates;
  }
  std::size_t shared=0;
  for(std::size_t j=0;j<shape.right_bigrams;++j) {
    const auto key=key_at(b,j);auto it=counts.find(key);
    const std::size_t before=it==counts.end()?0:it->second.left_count-it->second.shared_count;
    if(before){++it->second.shared_count;++shared;}
    if(trace) {
      ++trace->work.right_lookups;
      if(before)++trace->work.matches;else ++trace->work.misses;
      trace->events.push_back({j,key,before,before?before-1:0,before!=0});
      if(it==counts.end())it=counts.try_emplace(key,DiceBigramCount{key}).first;
      ++it->second.right_count;
    }
  }
  if(trace) {
    trace->counts.reserve(counts.size());
    for(const auto& [key,row]:counts)trace->counts.push_back(row);
    /// @brief Sort owned distinct-key rows by unsigned packed byte order.
    /// @param x First bigram row.
    /// @param y Second bigram row.
    /// @return True if x precedes y in unsigned byte lexicographic order.
    const auto by_key=[](const DiceBigramCount& x,const DiceBigramCount& y){return x.key<y.key;};
    std::sort(trace->counts.begin(),trace->counts.end(),by_key);
  }
  return dice_overlap_counts(shape.left_bigrams,shape.right_bigrams,shared,shape.total==0 && a==b);
}
} // namespace
DiceOverlap dice_overlap_counts(std::size_t left,std::size_t right,std::size_t shared,bool empty_equal) {
  if(shared>std::min(left,right))throw std::invalid_argument("dice_overlap_counts: intersection exceeds a side");
  if(left>std::numeric_limits<std::size_t>::max()-right)throw std::length_error("dice_overlap_counts: complete sum overflow");
  const std::size_t total=left+right,united=total-shared;
  return {left,right,shared,total,united,total?2.0*static_cast<double>(shared)/static_cast<double>(total):double(empty_equal),
    united?static_cast<double>(shared)/static_cast<double>(united):double(empty_equal)};
}
double dice_coefficient(const std::string& a,const std::string& b) {
  if(a==b)return 1.0;
  if(a.size()<2 || b.size()<2)return 0.0;
  return consume(a,b,dice_overlap_counts(a.size()-1,b.size()-1,0),nullptr).coefficient;
}
DiceTrace dice_coefficient_trace(const std::string& a,const std::string& b,std::size_t max_bigrams) {
  const auto shape=dice_overlap_counts(a.empty()?0:a.size()-1,b.empty()?0:b.size()-1,0,false);
  if((max_bigrams && shape.total>max_bigrams) || shape.right_bigrams>std::vector<DiceConsumeEvent>{}.max_size())
    throw std::length_error("dice_coefficient_trace: complete occurrence cap or event capacity");
  DiceTrace trace;trace.left=a;trace.right=b;trace.events.reserve(shape.right_bigrams);
  trace.overlap=consume(a,b,shape,&trace);return trace;
}
} // namespace datamunge::algorithms
