/// @file run_length_encoding.cpp
/// @brief Shared actual raw-byte run scanners and checked append producer.
#include <datamunge/algorithms/run_length_encoding.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
/// @brief Canonical raw-byte compression implementation.
namespace datamunge::algorithms {
/// @brief Private run producers and capture admission.
namespace {
/// @brief Admit one actual optional span before its allocation or append.
/// @param r Actual current run.
/// @param o Capture policy.
/// @param capture Whether history is requested.
void admit(const RLERun& r, const RLEOptions& o, bool capture) {
  if (capture && (r.steps.size() == r.steps.max_size() ||
                  (o.max_trace_entries && r.steps.size() >= o.max_trace_entries)))
    throw std::length_error("RLE: history policy");
}
/// @brief Scan actual maximal byte runs, counting real equality decisions.
/// @param text Borrowed complete bytes.
/// @param o Policies.
/// @param capture Retain actual spans.
/// @return Owned produced records.
RLERun encode(const std::string& text, const RLEOptions& o, bool capture) {
  if (o.max_bytes && text.size() > o.max_bytes)
    throw std::length_error("RLE: source policy");
  RLERun out;
  out.text = text;
  for (std::size_t i = 0; i < text.size();) {
    if (o.max_runs && out.runs.size() >= o.max_runs)
      throw std::length_error("RLE: run policy");
    admit(out, o, capture);
    std::size_t j = i, tests = 0;
    while (j < text.size()) {
      if (out.comparisons == std::numeric_limits<std::size_t>::max())
        throw std::length_error("RLE: comparison capacity");
      ++out.comparisons;
      ++tests;
      if (text[j] != text[i])
        break;
      ++j;
    }
    const auto id = out.runs.size();
    out.runs.push_back({text[i], j - i});
    if (capture)
      out.steps.push_back({id, i, j, text[i], tests});
    i = j;
  }
  return out;
}
/// @brief Validate complete decoder geometry before actual appends.
/// @param runs Borrowed records.
/// @param o Policies.
/// @param capture Retain append spans.
/// @return Owned actual decoded bytes.
RLERun decode(const std::vector<Run>& runs, const RLEOptions& o, bool capture) {
  if (o.max_runs && runs.size() > o.max_runs)
    throw std::length_error("RLE: input run policy");
  if (capture && o.max_trace_entries && runs.size() > o.max_trace_entries)
    throw std::length_error("RLE: history policy");
  std::size_t total = 0;
  for (const auto& r : runs) {
    if (r.count > std::numeric_limits<std::size_t>::max() - total)
      throw std::length_error("RLE: decoded count capacity");
    total += r.count;
    if ((o.max_bytes && total > o.max_bytes) || total > std::string{}.max_size())
      throw std::length_error("RLE: decoded byte policy");
  }
  RLERun out;
  out.decoding = true;
  out.runs     = runs;
  out.text.reserve(total);
  for (std::size_t i = 0; i < runs.size(); ++i) {
    admit(out, o, capture);
    const auto begin = out.text.size();
    out.text.append(runs[i].count, runs[i].symbol);
    out.emitted_bytes += runs[i].count;
    if (capture)
      out.steps.push_back({i, begin, out.text.size(), runs[i].symbol, 0});
  }
  return out;
}
} // namespace
RLERun rle_encode_run(const std::string& t, const RLEOptions& o) {
  return encode(t, o, false);
}
RLERun rle_encode_trace(const std::string& t, const RLEOptions& o) {
  return encode(t, o, true);
}
RLERun rle_decode_run(const std::vector<Run>& r, const RLEOptions& o) {
  return decode(r, o, false);
}
RLERun rle_decode_trace(const std::vector<Run>& r, const RLEOptions& o) {
  return decode(r, o, true);
}
std::vector<Run> rle_encode(const std::string& t) {
  return encode(t, {}, false).runs;
}
std::string rle_decode(const std::vector<Run>& r) {
  return decode(r, {}, false).text;
}
std::size_t rle_encoded_cells(const std::vector<Run>& r) {
  return r.size();
}
std::string rle_example_pattern(std::size_t size, bool alternating, std::size_t max_bytes) {
  if ((max_bytes && size > max_bytes) || size > std::string{}.max_size())
    throw std::length_error("RLE: fixture policy");
  std::string out(size, 'A');
  if (alternating)
    for (std::size_t i = 1; i < size; i += 2)
      out[i] = 'B';
  return out;
}
} // namespace datamunge::algorithms
