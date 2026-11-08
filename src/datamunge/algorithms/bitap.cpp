/// @file bitap.cpp
/// @brief Canonical coupled Shift-And loop for exact and substitution-only byte searches.
#include <datamunge/algorithms/bitap.hpp>
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private scan counters, bounded-word traversal and legacy offset conversion.
namespace {
/// @brief Actual work performed by a scan, excluding pattern-mask preprocessing.
struct ScanStats {
  std::size_t bytes{0}; ///< Bytes consumed before the optional early stop.
  std::size_t updates{0}; ///< Error-level word updates.
  int budget{0}; ///< Effective substitution budget.
};

/// @brief Shared positive-bit Shift-And scan, emitting matches and optionally retaining states.
/// @tparam Emit Callable taking BitapMatch and returning true to stop the scan.
/// @param text Borrowed byte string.
/// @param pattern Borrowed pattern of at most 63 bytes.
/// @param requested Requested substitution budget, clamped to the pattern length.
/// @param emit Match consumer; may stop on the first qualifying window.
/// @param trace Nullable borrowed step vector; no recording when null.
/// @return Actual scan counts and effective budget.
/// @throws std::invalid_argument If the pattern exceeds 63 bytes.
template<class Emit>
ScanStats scan(const std::string& text,const std::string& pattern,int requested,Emit emit,
               std::vector<BitapStep>* trace) {
  const auto m=pattern.size();
  if (m>63) throw std::invalid_argument("bitap: pattern exceeds 63 bytes");
  ScanStats stats;
  stats.budget=std::clamp(requested,0,static_cast<int>(m));
  if (m==0) {
    for (std::size_t i=0;i<=text.size();++i) if (emit(BitapMatch{i,i,0})) break;
    return stats;
  }
  std::array<std::uint64_t,256> masks{};
  for (std::size_t j=0;j<m;++j) masks[static_cast<unsigned char>(pattern[j])]|=std::uint64_t{1}<<j;
  std::array<std::uint64_t,64> state{};
  const std::uint64_t active=(std::uint64_t{1}<<m)-1,accept=std::uint64_t{1}<<(m-1);
  for (std::size_t i=0;i<text.size();++i) {
    const auto byte=static_cast<unsigned char>(text[i]);
    const auto mask=masks[byte];
    auto previous=state[0];
    state[0]=((state[0]<<1)|1)&mask;
    for (int d=1;d<=stats.budget;++d) {
      const auto old=state[d];
      state[d]=((((old<<1)|1)&mask)|((previous<<1)|1))&active;
      previous=old;
    }
    ++stats.bytes;
    stats.updates+=stats.budget+1;
    if (trace) trace->push_back({i,byte,{state.begin(),state.begin()+stats.budget+1}});
    if (i+1<m) continue;
    for (int d=0;d<=stats.budget;++d) if (state[d]&accept) {
      if (emit(BitapMatch{i+1-m,i+1,d})) return stats;
      break;
    }
  }
  return stats;
}

/// @brief Convert a found byte offset to the legacy signed result type.
/// @param value Nonnegative byte offset.
/// @return Offset represented as long.
/// @throws std::overflow_error If long cannot represent value.
long legacy_index(std::size_t value) {
  if (value>static_cast<std::size_t>(std::numeric_limits<long>::max()))
    throw std::overflow_error("bitap: matching offset exceeds long");
  return static_cast<long>(value);
}
} // namespace

long bitap_search(const std::string& text,const std::string& pattern) {
  if (pattern.empty()) return 0;
  if (pattern.size()>63) return -1;
  long result=-1;
  scan(text,pattern,0,[&](BitapMatch match){result=legacy_index(match.begin);return true;},nullptr);
  return result;
}

long bitap_fuzzy_search(const std::string& text,const std::string& pattern,int max_errors) {
  if (pattern.empty()) return 0;
  if (pattern.size()>62) return -1;
  long result=-1;
  scan(text,pattern,max_errors,[&](BitapMatch match){result=legacy_index(match.end-1);return true;},nullptr);
  return result;
}

std::vector<BitapMatch> bitap_matches(const std::string& text,const std::string& pattern,int max_substitutions) {
  std::vector<BitapMatch> result;
  scan(text,pattern,max_substitutions,[&](BitapMatch match){result.push_back(match);return false;},nullptr);
  return result;
}

BitapTrace bitap_trace(const std::string& text,const std::string& pattern,int max_substitutions) {
  BitapTrace result;
  const auto stats=scan(text,pattern,max_substitutions,
      [&](BitapMatch match){result.matches.push_back(match);return false;},&result.steps);
  result.bytes_scanned=stats.bytes;
  result.word_updates=stats.updates;
  result.effective_budget=stats.budget;
  return result;
}
} // namespace datamunge::algorithms
