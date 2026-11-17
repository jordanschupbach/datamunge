/// @file lzw.cpp
/// @brief Shared checked string-dictionary LZW producers and actual streaming/learning observations.
#include <datamunge/algorithms/lzw.hpp>
#include <limits>
#include <stdexcept>
#include <unordered_map>
namespace datamunge::algorithms {
/// @brief Private complete numeric validation, checked counts and shared producing loops.
namespace {
/// @brief Add actual counts without wraparound.
/// @param count Actual selected count.
/// @param extra Actual next increment.
void increase(std::size_t& count, std::size_t extra = 1) {
  if (extra > std::numeric_limits<std::size_t>::max() - count)
    throw std::length_error("LZW: count overflow");
  count += extra;
}
/// @brief Admit actual selected logical input/record/string-byte capture before copying it.
/// @param count Actual captured entries, updated only within the local result.
/// @param extra Actual additional entries.
/// @param quota Optional capture cap.
void admit(std::size_t& count, std::size_t extra, std::size_t quota) {
  increase(count, extra);
  if (quota && count > quota)
    throw std::length_error("LZW: capture quota");
}
/// @brief Produce exact historical greedy codes and optionally capture actual byte/flush decisions.
/// @param data Complete original arbitrary bytes.
/// @param trace Optional owned actual history output.
/// @param quota Optional selected capture cap.
/// @return Actual codes and selected dictionary/string work.
LZWEncodeAnalysis encode(const std::string& data, LZWEncodeTrace* trace, std::size_t quota) {
  if (trace) {
    admit(trace->captured_entries, data.size(), quota);
    trace->input = data;
  }
  std::unordered_map<std::string, int> dict;
  for (int i = 0; i < 256; ++i)
    dict.emplace(std::string(1, char(i)), i);
  std::size_t       next_code = 256;
  LZWEncodeAnalysis out;
  std::string       w;
  for (std::size_t position = 0; position < data.size(); ++position) {
    const char  c         = data[position];
    std::string candidate = w + c;
    increase(out.work.input_bytes);
    increase(out.work.lookups);
    increase(out.work.candidate_bytes, candidate.size());
    const bool known   = dict.find(candidate) != dict.end();
    int        emitted = -1, learned = -1;
    if (!known) {
      if (next_code > std::size_t(std::numeric_limits<int>::max()))
        throw std::length_error("LZW: integer dictionary code exhausted");
      emitted = dict.at(w);
      learned = int(next_code++);
      out.codes.push_back(emitted);
      increase(out.work.emissions);
      dict.emplace(candidate, learned);
      increase(out.work.insertions);
      increase(out.work.dictionary_bytes, candidate.size());
    }
    if (trace) {
      admit(trace->captured_entries, 1, quota);
      admit(trace->captured_entries, w.size(), quota);
      admit(trace->captured_entries, candidate.size(), quota);
      trace->steps.push_back(
          {position, static_cast<unsigned char>(c), false, w, candidate, known, emitted, learned});
    }
    w = known ? std::move(candidate) : std::string(1, c);
  }
  if (!w.empty()) {
    const int emitted = dict.at(w);
    increase(out.work.lookups);
    increase(out.work.emissions);
    out.codes.push_back(emitted);
    if (trace) {
      admit(trace->captured_entries, 1, quota);
      admit(trace->captured_entries, w.size(), quota);
      trace->steps.push_back({data.size(), 0, true, w, {}, false, emitted, -1});
    }
  }
  return out;
}
/// @brief Validate every numeric code before dictionary allocation or output/capture construction.
/// @param codes Complete original integer stream.
void validate(const std::vector<int>& codes) {
  if (codes.empty())
    return;
  if (codes.size() > std::size_t(std::numeric_limits<int>::max()) - 254)
    throw std::length_error("LZW: integer dictionary geometry");
  if (codes.front() < 0 || codes.front() > 255)
    throw std::invalid_argument("LZW: first code is not a seed byte");
  for (std::size_t k = 1; k < codes.size(); ++k)
    if (codes[k] < 0 || std::size_t(codes[k]) > 255 + k)
      throw std::invalid_argument("LZW: unavailable future dictionary code");
}
/// @brief Reconstruct actual words/learned entries once with incremental output/capture admission.
/// @param codes Completely validated original integer stream.
/// @param output_cap Optional actual decoded-byte cap.
/// @param trace Optional owned actual history output.
/// @param quota Optional selected capture cap.
/// @return Actual decoded output and dictionary/special-case work.
LZWDecodeAnalysis decode(
    const std::vector<int>& codes,
    std::size_t             output_cap,
    LZWDecodeTrace*         trace,
    std::size_t             quota) {
  validate(codes);
  if (trace) {
    admit(trace->captured_entries, codes.size(), quota);
    trace->input = codes;
  }
  LZWDecodeAnalysis out;
  if (codes.empty()) {
    out.work.dictionary_bytes = 0;
    return out;
  }
  std::vector<std::string> dict;
  dict.reserve(256 + codes.size());
  for (int i = 0; i < 256; ++i)
    dict.emplace_back(1, char(i));
  out.work.dictionary_bytes = 256;
  std::string w;
  for (std::size_t k = 0; k < codes.size(); ++k) {
    const int   code         = codes[k];
    const bool  special      = std::size_t(code) == dict.size();
    std::string entry        = special ? w + w.front() : dict[std::size_t(code)];
    std::string learned      = k ? w + entry.front() : std::string{};
    const int   learned_code = k ? int(dict.size()) : -1;
    const auto  begin        = out.data.size();
    auto        end          = begin;
    increase(end, entry.size());
    if (end > out.data.max_size() || (output_cap && end > output_cap))
      throw std::length_error("LZW: decoded output cap");
    if (trace) {
      admit(trace->captured_entries, 1, quota);
      admit(trace->captured_entries, entry.size(), quota);
      admit(trace->captured_entries, learned.size(), quota);
      trace->steps.push_back({k, code, special, entry, learned_code, learned, begin, end});
    }
    out.data += entry;
    increase(out.work.codes);
    increase(out.work.emitted_bytes, entry.size());
    out.work.special_cases += special;
    if (k) {
      increase(out.work.dictionary_bytes, learned.size());
      increase(out.work.insertions);
      dict.push_back(std::move(learned));
    }
    w = std::move(entry);
  }
  return out;
}
} // namespace
std::vector<int> lzw_encode(const std::string& data) {
  return encode(data, nullptr, 0).codes;
}
LZWEncodeAnalysis lzw_encode_analyze(const std::string& data) {
  return encode(data, nullptr, 0);
}
LZWEncodeTrace lzw_encode_trace(const std::string& data, std::size_t max_entries) {
  LZWEncodeTrace t;
  t.analysis = encode(data, &t, max_entries);
  return t;
}
std::string lzw_decode(const std::vector<int>& codes) {
  return decode(codes, 0, nullptr, 0).data;
}
LZWDecodeAnalysis lzw_decode_analyze(const std::vector<int>& codes, std::size_t max_output_bytes) {
  return decode(codes, max_output_bytes, nullptr, 0);
}
LZWDecodeTrace lzw_decode_trace(
    const std::vector<int>& codes,
    std::size_t             max_output_bytes,
    std::size_t             max_entries) {
  LZWDecodeTrace t;
  t.analysis = decode(codes, max_output_bytes, &t, max_entries);
  return t;
}
std::string lzw_classic_text() {
  return "TOBEORNOTTOBEORTOBEORNOT";
}
} // namespace datamunge::algorithms
