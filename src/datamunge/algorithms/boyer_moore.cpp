/// @file boyer_moore.cpp
/// @brief Shared Boyer-Moore preprocessing and optional recording of actual search decisions.
#include <datamunge/algorithms/boyer_moore.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace datamunge::algorithms {
BoyerMooreTables boyer_moore_tables(const std::string& pattern) {
  if (pattern.size()>static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()-1))
    throw std::length_error("boyer_moore: pattern exceeds signed border-index range");
  const auto m=static_cast<std::ptrdiff_t>(pattern.size());
  BoyerMooreTables result;result.last.fill(-1);
  for (std::size_t i=0;i<pattern.size();++i) result.last[static_cast<unsigned char>(pattern[i])]=static_cast<std::ptrdiff_t>(i);
  auto& shift=result.good_suffix;shift.assign(pattern.size()+1,0);
  std::vector<std::ptrdiff_t> border(pattern.size()+1,0);
  std::ptrdiff_t i=m,j=m+1;border[static_cast<std::size_t>(i)]=j;
  while (i>0) {
    while (j<=m && pattern[static_cast<std::size_t>(i-1)]!=pattern[static_cast<std::size_t>(j-1)]) {
      if (shift[static_cast<std::size_t>(j)]==0) shift[static_cast<std::size_t>(j)]=j-i;
      j=border[static_cast<std::size_t>(j)];
    }
    --i;--j;border[static_cast<std::size_t>(i)]=j;
  }
  j=border[0];
  for (i=0;i<=m;++i) {
    if (shift[static_cast<std::size_t>(i)]==0) shift[static_cast<std::size_t>(i)]=j;
    if (i==j) j=border[static_cast<std::size_t>(j)];
  }
  return result;
}
/// @brief Private canonical search loop, specialised to avoid ordinary-search trace allocation.
namespace {
/// @brief Search with the shared preprocessing and optionally retain every actual alignment.
/// @tparam Record Whether tables for ineligible searches, records and counters are retained.
/// @param text Borrowed bytes.
/// @param pattern Borrowed pattern bytes.
/// @return Matches and optional trace from the same right-to-left comparisons.
/// @throws std::length_error If preprocessing cannot represent or allocate its tables.
/// @throws std::overflow_error If the recorded comparison total would wrap.
template<bool Record>
BoyerMooreTrace search(const std::string& text,const std::string& pattern) {
  BoyerMooreTrace out;out.text_size=text.size();out.pattern_size=pattern.size();
  if constexpr (Record) out.tables=boyer_moore_tables(pattern);
  const auto n=text.size(),m=pattern.size();if (m==0 || m>n) return out;
  if constexpr (!Record) out.tables=boyer_moore_tables(pattern);
  std::size_t position=0;
  while (position<=n-m) {
    BoyerMooreAlignment step;step.position=position;std::size_t j=m;
    while (j>0) {
      if constexpr (Record) {
        if (out.comparisons==std::numeric_limits<std::size_t>::max())
          throw std::overflow_error("boyer_moore: comparison counter overflow");
        ++out.comparisons;++step.comparisons;
      }
      if (pattern[j-1]!=text[position+j-1]) break;
      --j;
    }
    step.matched=j==0;step.matched_suffix_length=m-j;
    if (step.matched) {
      step.mismatch_index=m;step.good_suffix_shift=out.tables.good_suffix[0];
      step.shift=static_cast<std::size_t>(step.good_suffix_shift);out.matches.push_back(position);
    } else {
      step.mismatch_index=j-1;step.text_byte=static_cast<unsigned char>(text[position+j-1]);
      step.bad_character_shift=static_cast<std::ptrdiff_t>(j-1)-out.tables.last[step.text_byte];
      step.good_suffix_shift=out.tables.good_suffix[j];
      step.shift=static_cast<std::size_t>(std::max<std::ptrdiff_t>({1,step.bad_character_shift,step.good_suffix_shift}));
    }
    if constexpr (Record) out.alignments.push_back(step);
    // Keep advancement in the remaining candidate range, even for extreme size_t text lengths.
    if (step.shift>n-m-position) break;
    position+=step.shift;
  }
  return out;
}
} // namespace
std::vector<std::size_t> boyer_moore_search(const std::string& text,const std::string& pattern) {
  return search<false>(text,pattern).matches;
}
BoyerMooreTrace boyer_moore_trace(const std::string& text,const std::string& pattern) {
  return search<true>(text,pattern);
}
} // namespace datamunge::algorithms
