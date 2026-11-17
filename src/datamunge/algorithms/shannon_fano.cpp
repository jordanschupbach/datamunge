/// @file shannon_fano.cpp
/// @brief One actual bounded top-down partition producer and shared prefix-code adapters.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/detail/prefix_codec.hpp>
#include <datamunge/algorithms/shannon_fano.hpp>
#include <functional>
#include <limits>
#include <map>
#include <utility>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked helpers.
namespace {
/// @brief Convert and validate every supplied ASCII byte codeword before using a codec.
/// @param c Borrowed complete model.
/// @param max_bits Complete sum of codeword bits admitted before copying.
/// @return Owned unsigned-byte-indexed bool words.
std::vector<std::vector<bool>> words(const ShannonFanoCode& c,std::size_t max_bits) {
  std::size_t complete=0;for(const auto& [symbol,word]:c.codes){prefix_detail::admit(complete,word.size(),max_bits);complete+=word.size();}
  std::vector<std::vector<bool>> out(256);
  for (const auto& [symbol, word] : c.codes) {
    if (word.empty())
      throw std::invalid_argument("Shannon-Fano: empty active word");
    auto& bits = out[static_cast<unsigned char>(symbol)];
    bits.reserve(word.size());
    for (char bit : word) {
      if (bit != '0' && bit != '1')
        throw std::invalid_argument("Shannon-Fano: nonbinary model");
      bits.push_back(bit == '1');
    }
  }
  return out;
}
/// @brief Build actual sorted partitions and codewords once.
/// @param text Borrowed raw input.
/// @param o Policies.
/// @param capture Retain actual choices.
/// @return Owned actual construction output.
ShannonFanoRun build(const std::string& text, const ShannonFanoOptions& o, bool capture) {
  prefix_detail::admit(0, text.size(), o.max_bytes);
  ShannonFanoRun              out;
  std::map<char, std::size_t> frequencies;
  for (unsigned char c : text) {
    ++out.frequencies[c];
    ++frequencies[static_cast<char>(c)];
  }
  std::vector<std::pair<char, std::size_t>> symbols(frequencies.begin(), frequencies.end());
  std::sort(symbols.begin(), symbols.end(), [](const auto& a, const auto& b) {
    return a.second != b.second ? a.second > b.second : a.first < b.first;
  });
  if (capture)
    prefix_detail::admit(
        2 * symbols.size(), symbols.empty() ? 0 : symbols.size() - 1, o.max_trace_entries);
  for (const auto& [c, f] : symbols) {
    out.code.codes[c] = "";
    if (capture) {
      out.sorted_symbols.push_back(static_cast<unsigned char>(c));
      out.sorted_frequencies.push_back(f);
    }
  }
  if (symbols.size() == 1) {
    prefix_detail::admit(0, 1, o.max_bits);
    out.code.codes[symbols[0].first] = "0";
    ++out.code_bit_appends;
  }
  /// @brief Perform one actual minimum-imbalance split and then visit its children.
  /// @param lo Inclusive actual sorted range address.
  /// @param hi Exclusive actual sorted range address.
  std::function<void(std::size_t, std::size_t)> split = [&](std::size_t lo, std::size_t hi) {
    if (hi - lo <= 1)
      return;
    std::size_t total = 0;
    for (std::size_t i = lo; i < hi; ++i)
      total += symbols[i].second;
    std::size_t accumulated = 0, best = lo + 1, best_left = 0,
                best_difference = std::numeric_limits<std::size_t>::max();
    for (std::size_t i = lo; i + 1 < hi; ++i) {
      prefix_detail::admit(out.split_comparisons, 1, o.max_split_comparisons);
      ++out.split_comparisons;
      accumulated += symbols[i].second;
      const auto right      = total - accumulated,
                 difference = accumulated >= right ? accumulated - right : right - accumulated;
      if (difference < best_difference) {
        best_difference = difference;
        best            = i + 1;
        best_left       = accumulated;
      }
    }
    if (capture)
      out.splits.push_back({lo, hi, best, best_left, total - best_left, best_difference});
    for (std::size_t i = lo; i < hi; ++i) {
      prefix_detail::admit(out.code_bit_appends, 1, o.max_bits);
      out.code.codes[symbols[i].first] += i < best ? '0' : '1';
      ++out.code_bit_appends;
    }
    split(lo, best);
    split(best, hi);
  };
  split(0, symbols.size());
  long double entropy = 0;
  for (const auto& [c, f] : symbols) {
    const auto length = out.code.codes[c].size();
    if (length && f > std::numeric_limits<std::size_t>::max() / length)
      throw std::length_error("Shannon-Fano: weighted length capacity");
    prefix_detail::admit(out.weighted_bits, f * length, 0);
    out.weighted_bits += f * length;
    const long double probability = static_cast<long double>(f) / text.size();
    entropy -= probability * std::log2(probability);
  }
  out.entropy      = static_cast<double>(entropy);
  out.average_bits = text.empty() ? 0 : static_cast<double>(out.weighted_bits) / text.size();
  return out;
}
/// @brief Validate complete source geometry and convert byte identities to shared codec ids.
/// @param text Borrowed source bytes.
/// @param o Complete source policy.
/// @return Owned unsigned-byte ids.
std::vector<int> ids(const std::string& text, const ShannonFanoOptions& o) {
  prefix_detail::admit(0, text.size(), o.max_bytes);
  std::vector<int> out;
  out.reserve(text.size());
  for (unsigned char c : text)
    out.push_back(c);
  return out;
}
/// @brief Encode once with the shared actual prefix producer and format its logical bits.
/// @param c Borrowed model.
/// @param text Borrowed raw source.
/// @param o Policies.
/// @param capture Retain actual emitted spans.
/// @return Owned actual encoding.
ShannonFanoCodecRun encode(
    const ShannonFanoCode&    c,
    const std::string&        text,
    const ShannonFanoOptions& o,
    bool                      capture) {
  auto r = prefix_detail::encode(
      words(c,o.max_bits), ids(text, o), o.max_bits, o.max_bytes, o.max_nodes, capture, o.max_trace_entries);
  ShannonFanoCodecRun out;
  out.bits.reserve(r.bits.size());
  for (bool bit : r.bits)
    out.bits += bit ? '1' : '0';
  out.bit_visits = r.bit_visits;
  out.trie_nodes = r.nodes;
  out.words      = std::move(r.spans);
  return out;
}
/// @brief Decode once through the shared trie and format actual unsigned-byte identities.
/// @param c Borrowed model.
/// @param bits Borrowed complete ASCII payload.
/// @param count Required exact complete output count.
/// @param o Policies.
/// @param capture Retain actual word emissions.
/// @return Owned actual decoding.
ShannonFanoCodecRun decode(
    const ShannonFanoCode&    c,
    const std::string&        bits,
    std::size_t               count,
    const ShannonFanoOptions& o,
    bool                      capture) {
  prefix_detail::admit(0, bits.size(), o.max_bits);
  std::vector<bool> payload;
  payload.reserve(bits.size());
  for (char bit : bits) {
    if (bit != '0' && bit != '1')
      throw std::invalid_argument("Shannon-Fano: nonbinary payload");
    payload.push_back(bit == '1');
  }
  auto r = prefix_detail::decode(
      words(c,o.max_bits), payload, count, o.max_bits, o.max_bytes, o.max_nodes, capture, o.max_trace_entries);
  ShannonFanoCodecRun out;
  out.text.reserve(r.symbols.size());
  for (int symbol : r.symbols)
    out.text += static_cast<char>(static_cast<unsigned char>(symbol));
  out.bit_visits = r.bit_visits;
  out.trie_nodes = r.nodes;
  out.words      = std::move(r.spans);
  return out;
}
} // namespace
ShannonFanoRun shannon_fano_build_run(const std::string& t, const ShannonFanoOptions& o) {
  return build(t, o, false);
}
ShannonFanoRun shannon_fano_build_trace(const std::string& t, const ShannonFanoOptions& o) {
  return build(t, o, true);
}
ShannonFanoCodecRun shannon_fano_encode_run(
    const ShannonFanoCode&    c,
    const std::string&        t,
    const ShannonFanoOptions& o) {
  return encode(c, t, o, false);
}
ShannonFanoCodecRun shannon_fano_encode_trace(
    const ShannonFanoCode&    c,
    const std::string&        t,
    const ShannonFanoOptions& o) {
  return encode(c, t, o, true);
}
ShannonFanoCodecRun shannon_fano_decode_run(
    const ShannonFanoCode&    c,
    const std::string&        b,
    std::size_t               n,
    const ShannonFanoOptions& o) {
  return decode(c, b, n, o, false);
}
ShannonFanoCodecRun shannon_fano_decode_trace(
    const ShannonFanoCode&    c,
    const std::string&        b,
    std::size_t               n,
    const ShannonFanoOptions& o) {
  return decode(c, b, n, o, true);
}
ShannonFanoCode shannon_fano_build(const std::string& t) {
  return shannon_fano_build_run(t).code;
}
std::string shannon_fano_encode(const ShannonFanoCode& c, const std::string& t) {
  return shannon_fano_encode_run(c, t).bits;
}
double shannon_fano_average_bits(const ShannonFanoCode& c, const std::string& t) {
  const ShannonFanoOptions o;
  prefix_detail::admit(0, t.size(), o.max_bytes);
  auto book = words(c,o.max_bits);
  prefix_detail::tree(book, o.max_nodes, o.max_bits);
  std::size_t size = 0;
  for (unsigned char symbol : t) {
    if (book[symbol].empty())
      throw std::invalid_argument("Shannon-Fano: absent source byte");
    prefix_detail::admit(size, book[symbol].size(), o.max_bits);
    size += book[symbol].size();
  }
  return t.empty() ? 0 : static_cast<double>(size) / t.size();
}
std::string shannon_fano_example_text() {
  return "the quick brown fox jumps over the lazy dog. pack my box with five dozen liquor jugs.";
}
std::string shannon_fano_frequency_text(const std::vector<std::size_t>& counts, std::size_t cap) {
  if (counts.size() > 256)
    throw std::invalid_argument("byte-frequency fixture: at most256 ids");
  std::size_t total = 0;
  for (auto count : counts) {
    prefix_detail::admit(total, count, cap);
    total += count;
  }
  if (total > std::string().max_size())
    throw std::length_error("byte-frequency fixture: container capacity");
  std::string out;
  out.reserve(total);
  for (std::size_t i = 0; i < counts.size(); ++i)
    out.append(counts[i], static_cast<char>(static_cast<unsigned char>(i)));
  return out;
}
} // namespace datamunge::algorithms
