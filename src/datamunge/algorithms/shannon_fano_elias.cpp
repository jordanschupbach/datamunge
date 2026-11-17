/// @file shannon_fano_elias.cpp
/// @brief Exact native integer-midpoint construction and shared complete-word codec delegation.
#include <cmath>
#include <datamunge/algorithms/detail/prefix_codec.hpp>
#include <datamunge/algorithms/shannon_fano_elias.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked helpers.
namespace {
/// @brief Produce each real integer-midpoint word once, optionally retaining actual records.
/// @param f Borrowed complete counts.
/// @param o Complete bounds.
/// @param capture Retain actual positive-count word records.
/// @return Owned actual codebook and work.
SFECodebookRun build(const std::vector<std::uint32_t>& f, const SFEOptions& o, bool capture) {
  prefix_detail::admit(0, f.size(), o.max_symbols);
  if (f.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("SFE: int alphabet capacity");
  std::uint64_t total    = 0;
  std::size_t   positive = 0;
  for (auto value : f) {
    if (value > std::numeric_limits<std::uint64_t>::max() - total)
      throw std::length_error("SFE: total overflow");
    total += value;
    if (value)
      ++positive;
  }
  if (!f.empty() && !total)
    throw std::invalid_argument("SFE: all-zero model");
  if (capture)
    prefix_detail::admit(0, positive, o.max_trace_entries);
  std::size_t complete_bits = 0;
  for (auto value : f)
    if (value) {
      unsigned __int128 scaled = value;
      std::size_t       length = 1;
      while (scaled < total) {
        scaled *= 2;
        ++length;
      }
      prefix_detail::admit(complete_bits, length, o.max_bits);
      complete_bits += length;
    }
  SFECodebookRun out;
  out.total = total;
  out.codebook.resize(f.size());
  std::uint64_t cumulative = 0;
  for (std::size_t symbol = 0; symbol < f.size(); ++symbol) {
    if (!f[symbol])
      continue;
    unsigned __int128 scaled = f[symbol];
    std::size_t       length = 1;
    while (scaled < total) {
      scaled *= 2;
      ++length;
    }
    const unsigned __int128 denominator = static_cast<unsigned __int128>(total) * 2;
    unsigned __int128       remainder = static_cast<unsigned __int128>(cumulative) * 2 + f[symbol];
    auto&                   word      = out.codebook[symbol];
    word.reserve(length);
    for (std::size_t bit = 0; bit < length; ++bit) {
      remainder *= 2;
      const bool value = remainder >= denominator;
      if (value)
        remainder -= denominator;
      word.push_back(value);
      ++out.emitted_bits;
    }
    if (capture)
      out.words.push_back({symbol, f[symbol], cumulative, length, word});
    cumulative += f[symbol];
  }
  long double entropy = 0, weighted = 0;
  for (std::size_t i = 0; i < f.size(); ++i)
    if (f[i]) {
      const long double probability = static_cast<long double>(f[i]) / total;
      entropy -= probability * std::log2(probability);
      weighted += probability * out.codebook[i].size();
    }
  out.entropy      = static_cast<double>(entropy);
  out.average_bits = static_cast<double>(weighted);
  return out;
}
/// @brief Project actual shared-prefix outputs without replaying any codec work.
/// @param r Owned actual private codec result.
/// @return Owned public midpoint codec result.
SFECodecRun project(prefix_detail::Result r) {
  return {std::move(r.bits), std::move(r.symbols), r.bit_visits, r.nodes, std::move(r.spans)};
}
} // namespace
SFECodebookRun shannon_fano_elias_run(const std::vector<std::uint32_t>& f, const SFEOptions& o) {
  return build(f, o, false);
}
SFECodebookRun shannon_fano_elias_trace(const std::vector<std::uint32_t>& f, const SFEOptions& o) {
  return build(f, o, true);
}
SFECodecRun shannon_fano_elias_encode_run(
    const std::vector<int>&           s,
    const std::vector<std::uint32_t>& f,
    const SFEOptions&                 o) {
  auto book = build(f, o, false);
  return project(
      prefix_detail::encode(
          book.codebook, s, o.max_bits, o.max_symbols, o.max_nodes, false, o.max_trace_entries));
}
SFECodecRun shannon_fano_elias_encode_trace(
    const std::vector<int>&           s,
    const std::vector<std::uint32_t>& f,
    const SFEOptions&                 o) {
  auto book = build(f, o, false);
  return project(
      prefix_detail::encode(
          book.codebook, s, o.max_bits, o.max_symbols, o.max_nodes, true, o.max_trace_entries));
}
SFECodecRun shannon_fano_elias_decode_run(
    const std::vector<bool>&          b,
    const std::vector<std::uint32_t>& f,
    std::size_t                       n,
    const SFEOptions&                 o) {
  auto book = build(f, o, false);
  return project(
      prefix_detail::decode(
          book.codebook, b, n, o.max_bits, o.max_symbols, o.max_nodes, false, o.max_trace_entries));
}
SFECodecRun shannon_fano_elias_decode_trace(
    const std::vector<bool>&          b,
    const std::vector<std::uint32_t>& f,
    std::size_t                       n,
    const SFEOptions&                 o) {
  auto book = build(f, o, false);
  return project(
      prefix_detail::decode(
          book.codebook, b, n, o.max_bits, o.max_symbols, o.max_nodes, true, o.max_trace_entries));
}
std::vector<std::vector<bool>> shannon_fano_elias_codebook(const std::vector<std::uint32_t>& f) {
  return shannon_fano_elias_run(f).codebook;
}
std::vector<bool> shannon_fano_elias_encode(
    const std::vector<int>&           s,
    const std::vector<std::uint32_t>& f) {
  return shannon_fano_elias_encode_run(s, f).bits;
}
std::vector<int>
shannon_fano_elias_decode(const std::vector<bool>& b, const std::vector<std::uint32_t>& f, int n) {
  if (n < 0)
    throw std::invalid_argument("SFE: negative decoded count");
  return shannon_fano_elias_decode_run(b, f, static_cast<std::size_t>(n)).symbols;
}
} // namespace datamunge::algorithms
