/// @file lz77.cpp
/// @brief Checked shared greedy byte encoder and validated overlap-capable decoder with actual capture.
#include <algorithm>
#include <datamunge/algorithms/lz77.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private checked counts, optional admission and shared producing loops.
namespace {
/// @brief Add a logical count without overflow.
/// @param value Actual current count.
/// @param extra Actual increment.
void increase(std::size_t& value, std::size_t extra = 1) {
  if (extra > std::numeric_limits<std::size_t>::max() - value)
    throw std::length_error("LZ77: count overflow");
  value += extra;
}
/// @brief Admit actual encoder records before their capture copy.
/// @param trace Optional actual output.
/// @param amount Actual new logical input/record count.
/// @param quota Optional capture cap.
void admit(LZ77EncodeTrace* trace, std::size_t amount, std::size_t quota) {
  if (!trace)
    return;
  increase(trace->captured_entries, amount);
  if (quota && trace->captured_entries > quota)
    throw std::length_error("LZ77: encoder trace quota");
}
/// @brief Run one actual oldest-start greedy window scan and optional record capture.
/// @param data Complete original input.
/// @param window Nonnegative offset cap.
/// @param max_match Nonnegative copied-length cap.
/// @param trace Optional actual encoder history.
/// @param quota Optional selected history cap.
/// @return Actual greedy tuple stream and work.
LZ77EncodeAnalysis encode(
    const std::string& data,
    int                window,
    int                max_match,
    LZ77EncodeTrace*   trace,
    std::size_t        quota) {
  if (window < 0 || max_match < 0)
    throw std::invalid_argument("LZ77: negative window or match cap");
  if (data.size() > std::size_t(std::numeric_limits<int>::max()))
    throw std::length_error("LZ77: input exceeds native int token geometry");
  if (trace) {
    admit(trace, data.size(), quota);
    trace->input = data;
  }
  LZ77EncodeAnalysis out;
  std::size_t        i = 0;
  while (i < data.size()) {
    const auto  start    = i > std::size_t(window) ? i - window : 0;
    std::size_t best_len = 0, best_off = 0;
    for (std::size_t j = start; j < i; ++j) {
      increase(out.work.candidates);
      std::size_t len = 0;
      while (len < std::size_t(max_match) && len < data.size() - i) {
        const auto a     = static_cast<unsigned char>(data[j + len]),
                   b     = static_cast<unsigned char>(data[i + len]);
        const bool equal = a == b;
        increase(out.work.comparisons);
        if (trace) {
          admit(trace, 1, quota);
          trace->comparisons.push_back({i, j, len, a, b, equal});
        }
        if (!equal)
          break;
        ++len;
      }
      const bool record = len > best_len;
      if (record) {
        best_len = len;
        best_off = i - j;
        increase(out.work.best_updates);
      }
      if (trace) {
        admit(trace, 1, quota);
        trace->candidates.push_back({i, j, len, record});
      }
    }
    const bool literal = best_len < data.size() - i;
    const auto next    = i + best_len + (literal ? 1 : 0);
    out.tokens.push_back(
        {int(best_off), int(best_len), literal ? data[i + best_len] : '\0', literal});
    increase(out.work.emissions);
    if (trace) {
      admit(trace, 1, quota);
      trace->emissions.push_back({i, next, out.tokens.size() - 1});
    }
    i = next;
  }
  return out;
}
/// @brief Validate the complete stream and exact produced size before any output indexing/allocation.
/// @param tokens Complete original tuple stream.
/// @param cap Optional complete decoded-byte cap.
/// @return Exact complete decoded byte count.
std::size_t decoded_size(const std::vector<LZ77Token>& tokens, std::size_t cap) {
  std::size_t size = 0;
  for (const auto& t : tokens) {
    if (t.offset < 0 || t.length < 0 || (!t.length && (t.offset || !t.literal)) ||
        (t.length && (t.offset == 0 || std::size_t(t.offset) > size)))
      throw std::invalid_argument("LZ77: invalid complete token stream");
    increase(size, std::size_t(t.length));
    increase(size, t.literal ? 1 : 0);
    if (size > std::string().max_size() || (cap && size > cap))
      throw std::length_error("LZ77: complete decoded output limit");
  }
  return size;
}
/// @brief Decode each actual byte once after complete structural/output/capture admission.
/// @param tokens Complete original tuples.
/// @param cap Optional complete decoded-byte limit.
/// @param trace Optional owned actual token/byte output.
/// @param quota Optional complete logical history cap.
/// @return Actual decoded string and producing copy/literal counts.
LZ77DecodeAnalysis decode(
    const std::vector<LZ77Token>& tokens,
    std::size_t                   cap,
    LZ77DecodeTrace*              trace,
    std::size_t                   quota) {
  const auto size = decoded_size(tokens, cap);
  if (trace) {
    auto entries = tokens.size();
    increase(entries, size);
    if (quota && entries > quota)
      throw std::length_error("LZ77: complete decoder trace quota");
    trace->captured_entries = entries;
    trace->input            = tokens;
    trace->bytes.reserve(size);
  }
  LZ77DecodeAnalysis out;
  out.data.reserve(size);
  for (std::size_t token = 0; token < tokens.size(); ++token) {
    const auto& t = tokens[token];
    ++out.tokens;
    const auto initial = out.data.size();
    if (t.length) {
      const auto start = initial - std::size_t(t.offset);
      for (int k = 0; k < t.length; ++k) {
        const auto     source   = start + std::size_t(k);
        const unsigned value    = static_cast<unsigned char>(out.data[source]);
        const auto     position = out.data.size();
        out.data.push_back(char(value));
        ++out.copied;
        out.overlapping_reads += source >= initial;
        if (trace)
          trace->bytes.push_back({token, position, source, value});
      }
    }
    if (t.literal) {
      const auto position = out.data.size();
      out.data.push_back(t.next);
      ++out.literals;
      if (trace)
        trace->bytes.push_back(
            {token,
             position,
             std::numeric_limits<std::size_t>::max(),
             static_cast<unsigned char>(t.next)});
    }
  }
  return out;
}
} // namespace
std::vector<LZ77Token> lz77_encode(const std::string& data, int window, int max_match) {
  return encode(data, window, max_match, nullptr, 0).tokens;
}
LZ77EncodeAnalysis lz77_encode_analyze(const std::string& data, int window, int max_match) {
  return encode(data, window, max_match, nullptr, 0);
}
LZ77EncodeTrace
lz77_encode_trace(const std::string& data, int window, int max_match, std::size_t max_entries) {
  LZ77EncodeTrace t;
  t.analysis = encode(data, window, max_match, &t, max_entries);
  return t;
}
std::string lz77_decode(const std::vector<LZ77Token>& tokens) {
  return decode(tokens, 0, nullptr, 0).data;
}
LZ77DecodeAnalysis lz77_decode_analyze(
    const std::vector<LZ77Token>& tokens,
    std::size_t                   max_output_bytes) {
  return decode(tokens, max_output_bytes, nullptr, 0);
}
LZ77DecodeTrace lz77_decode_trace(
    const std::vector<LZ77Token>& tokens,
    std::size_t                   max_output_bytes,
    std::size_t                   max_entries) {
  LZ77DecodeTrace t;
  t.analysis = decode(tokens, max_output_bytes, &t, max_entries);
  return t;
}
std::string lz77_demo_text() {
  return "the quick brown fox jumps over the lazy dog. the quick brown fox jumps over the lazy "
         "dog.";
}
} // namespace datamunge::algorithms
