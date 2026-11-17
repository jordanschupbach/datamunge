#pragma once
/// @file fnv_analysis.hpp
/// @brief Owned byte traces, bucket censuses and controlled FNV bit-flip diagnostics.
#include <algorithm>
#include <bit>
#include <datamunge/algorithms/fnv_hash.hpp>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/plot/plot.hpp>
#include <format>
#include <limits>
#include <map>
#include <numeric>
#include <set>

namespace datamunge::algorithms {
/// @brief Counted bucket occupancy and distinct-key integer-hash collisions.
/// All fields own their values. Bucket pairs count unordered INPUT RECORD pairs
/// sharing a bucket, including duplicates. Hash pairs count unordered DISTINCT
/// byte-key pairs with equal full-width integer hashes. These are different events.
/// Numeric dataframe columns convert counters to double, exact up to 2^53.
/// @par Example
/// @code
/// const auto p = dm::algorithms::fnv_bucket_profile({"a", "a"}, 16);
/// std::print("{} {}\n", p.bucket_pairs, p.hash_pairs); // 1 0
/// @endcode
/// @see fnv_bucket_profile()
struct FNVBucketProfile {
  unsigned                 bits{64};         ///< Full hash width; completed profiles use 32 or 64.
  std::size_t              keys{0};          ///< Input record count, including duplicate strings.
  std::size_t              distinct_keys{0}; ///< Distinct byte strings, not distinct hashes.
  std::size_t              bytes{0};         ///< Total input bytes hashed, excluding terminators.
  std::uint64_t            bucket_pairs{0};  ///< Unordered record pairs sharing a reduced bucket.
  std::uint64_t            hash_pairs{0}; ///< Unordered distinct-key pairs sharing the full hash.
  std::vector<std::size_t> counts;        ///< Record counts for buckets 0..B-1.

  /// @brief Copy bucket IDs and record counts into a table.
  /// @return Owned frame with bucket and keys columns; empty for an unevaluated profile.
  /// @par Example
  /// @code
  /// const auto p = dm::algorithms::fnv_bucket_profile({"a"}, 4);
  /// const auto n = p.table().nrows(); // 4
  /// @endcode
  /// @see summary()
  dstruct::DataFrame table() const {
    std::vector<double> bucket(counts.size()), n;
    std::iota(bucket.begin(), bucket.end(), 0.0);
    for (auto count : counts)
      n.push_back(static_cast<double>(count));
    dstruct::DataFrame d;
    d.add_column("bucket", bucket);
    d.add_column("keys", n);
    return d;
  }
  /// @brief Summarize finite observed occupancy, without a randomness certificate.
  /// @return One owned row: bits, keys, distinct_keys, bytes, buckets, occupied,
  /// max_load, mean_load, bucket_pairs, hash_pairs and iid_expected_bucket_pairs.
  /// The last is n(n-1)/(2B) under an explicit independent-uniform null, not an FNV guarantee.
  /// Empty profiles use zero for undefined mean/reference quantities.
  /// @par Example
  /// @code
  /// const auto p = dm::algorithms::fnv_bucket_profile({"a", "b"}, 4);
  /// const auto n = p.summary().nrows(); // 1
  /// @endcode
  /// @see table(), plot()
  dstruct::DataFrame summary() const {
    const double b = static_cast<double>(counts.size()), n = static_cast<double>(keys);
    const auto   occupied =
        std::count_if(counts.begin(), counts.end(), [](auto x) { return x != 0; });
    const auto maximum = counts.empty() ? 0 : *std::max_element(counts.begin(), counts.end());
    dstruct::DataFrame d;
    d.add_column("bits", std::vector<double>{double(bits)});
    d.add_column("keys", std::vector<double>{n});
    d.add_column("distinct_keys", std::vector<double>{double(distinct_keys)});
    d.add_column("bytes", std::vector<double>{double(bytes)});
    d.add_column("buckets", std::vector<double>{b});
    d.add_column("occupied", std::vector<double>{double(occupied)});
    d.add_column("max_load", std::vector<double>{double(maximum)});
    d.add_column("mean_load", std::vector<double>{b ? n / b : 0});
    d.add_column("bucket_pairs", std::vector<double>{double(bucket_pairs)});
    d.add_column("hash_pairs", std::vector<double>{double(hash_pairs)});
    d.add_column(
        "iid_expected_bucket_pairs", std::vector<double>{b && n > 1 ? n * (n - 1) / (2 * b) : 0});
    return d;
  }
  /// @brief Plot observed record counts and their arithmetic mean on a numeric bucket axis.
  /// @return Owned editable native bar plot, with headroom for the automatic legend.
  /// @throws std::invalid_argument If no buckets are present.
  /// @par Example
  /// @code
  /// dm::algorithms::fnv_bucket_profile({"a", "b"}, 4).plot().save_svg("build/fnv.svg");
  /// @endcode
  /// @see summary()
  graphics::RPlot plot() const {
    if (counts.empty())
      throw std::invalid_argument("FNVBucketProfile::plot: no buckets");
    const auto t = table();
    auto       p = graphics::RPlot::create();
    p.bars(
        t.get_column<double>("bucket").values(),
        t.get_column<double>("keys").values(),
        "observed keys",
        {37, 99, 235});
    p.abline_h(double(keys) / counts.size(), {220, 38, 38}, 1.5);
    p.title(std::format("FNV-1a-{}: {} keys, {} buckets", bits, keys, counts.size()))
        .x_label("bucket (integer hash modulo B)")
        .y_label("records")
        .y_limits(0, double(*std::max_element(counts.begin(), counts.end())) * 1.35 + 1);
    return p;
  }
};

/// @brief Form an owned bucket census using canonical full-width FNV hashes modulo B.
/// Validates width and positive B before work/allocation. Duplicate strings stay
/// separate bucket records, but are not distinct-key hash collisions. Uses ordered
/// sets/maps, with O(B+n log n) dictionary operations plus byte hashing/comparisons
/// and O(B+distinct-key bytes) storage. Pure, thread-safe; no input changes.
/// @param keys Borrowed raw-byte keys, including empty strings and embedded NULs.
/// @param buckets Positive bucket count; default 16. Power-of-two B uses low log2(B) bits.
/// @param bits Full FNV width, 32 or 64; default 64.
/// @return Owned counts and checked scalar counters; no keys or views retained.
/// @throws std::invalid_argument If width/B are invalid.
/// @throws std::length_error If vector capacity, byte count or pair counts overflow.
/// @throws std::bad_alloc If storage cannot be allocated.
/// @par Example
/// @code
/// const auto p = dm::algorithms::fnv_bucket_profile({"a", "b", "a"}, 16);
/// std::print("{} {}\n", p.keys, p.distinct_keys); // 3 2
/// @endcode
/// @par References
/// - RFC 9923, Sections 2, 3 and 6: https://www.rfc-editor.org/rfc/rfc9923.html
/// @see FNVBucketProfile, fnv1a_32(), fnv1a_64()
inline FNVBucketProfile fnv_bucket_profile(
    const std::vector<std::string>& keys,
    std::size_t                     buckets = 16,
    unsigned                        bits    = 64) {
  if (!buckets || (bits != 32 && bits != 64))
    throw std::invalid_argument("fnv_bucket_profile: width or bucket count");
  FNVBucketProfile p;
  p.bits = bits;
  p.keys = keys.size();
  if (buckets > p.counts.max_size())
    throw std::length_error("fnv_bucket_profile: bucket capacity");
  p.counts.assign(buckets, 0);
  std::map<std::uint64_t, std::set<std::string>> hashes;
  for (const auto& key : keys) {
    if (key.size() > std::numeric_limits<std::size_t>::max() - p.bytes)
      throw std::length_error("fnv_bucket_profile: byte count");
    p.bytes += key.size();
    const auto h = bits == 32 ? std::uint64_t(fnv1a_32(key)) : fnv1a_64(key);
    auto&      n = p.counts[h % buckets];
    if (n > std::numeric_limits<std::uint64_t>::max() - p.bucket_pairs)
      throw std::length_error("fnv_bucket_profile: bucket pairs");
    p.bucket_pairs += n++;
    auto&      set      = hashes[h];
    const auto previous = set.size();
    if (set.insert(key).second) {
      if (previous > std::numeric_limits<std::uint64_t>::max() - p.hash_pairs)
        throw std::length_error("fnv_bucket_profile: hash pairs");
      p.hash_pairs += previous;
      ++p.distinct_keys;
    }
  }
  return p;
}

/// @brief Tabulate actual byte updates, keeping large hash words as exact hexadecimal strings.
/// @param trace Borrowed native trace, unchanged; bits must be 32 or 64.
/// @return Owned index/byte/before/xored/after table, one row per retained byte.
/// @throws std::invalid_argument If trace.bits is not 32 or 64.
/// @par Example
/// @code
/// const auto d = dm::algorithms::fnv_trace_table(dm::algorithms::fnv1a_trace("ab", 32));
/// const auto n = d.nrows(); // 2
/// @endcode
/// @see fnv1a_trace()
inline dstruct::DataFrame fnv_trace_table(const FNVTrace& trace) {
  if (trace.bits != 32 && trace.bits != 64)
    throw std::invalid_argument("fnv_trace_table: width");
  std::vector<double>      index, byte;
  std::vector<std::string> before, xored, after;
  for (const auto& e : trace.steps) {
    index.push_back(double(e.index));
    byte.push_back(e.byte);
    before.push_back(std::format("{:0{}X}", e.before, trace.bits / 4));
    xored.push_back(std::format("{:0{}X}", e.xored, trace.bits / 4));
    after.push_back(std::format("{:0{}X}", e.after, trace.bits / 4));
  }
  dstruct::DataFrame d;
  d.add_column("index", index);
  d.add_column("byte", byte);
  d.add_column("before", before);
  d.add_column("xored", xored);
  d.add_column("after", after);
  return d;
}

/// @brief Original report's 32-word fixture in unchanged order.
/// @return Owned lowercase ASCII words; not a random sample of language.
/// @par Example
/// @code
/// const auto n = dm::algorithms::fnv_example_words().size(); // 32
/// @endcode
/// @see fnv_bucket_profile()
inline std::vector<std::string> fnv_example_words() {
  return {"the",  "quick", "brown",     "fox",   "jumps",     "over",   "lazy",   "dog",
          "hash", "table", "algorithm", "data",  "structure", "vector", "matrix", "graph",
          "tree", "heap",  "queue",     "stack", "alpha",     "beta",   "gamma",  "delta",
          "one",  "two",   "three",     "four",  "five",      "six",    "seven",  "eight"};
}

/// @brief Generate all one-byte keys with a chosen low-bit residue, in byte order.
/// @param low_bits Fixed low bits, from 0 to 8; default 0 enumerates all 256 bytes.
/// @param residue Low-bit value, less than 2^low_bits; default zero.
/// @return Owned keys, each exactly one raw byte; count 256/2^low_bits.
/// @throws std::invalid_argument If bits or residue are out of range.
/// @par Example
/// @code
/// const auto n = dm::algorithms::fnv_byte_keys(4, 1).size(); // 16
/// @endcode
/// @see fnv_bucket_profile()
inline std::vector<std::string> fnv_byte_keys(unsigned low_bits = 0, unsigned residue = 0) {
  if (low_bits > 8 || residue >= (1u << low_bits))
    throw std::invalid_argument("fnv_byte_keys: residue or bit count");
  std::vector<std::string> keys;
  for (unsigned byte = 0; byte < 256; ++byte)
    if ((byte & ((1u << low_bits) - 1)) == residue)
      keys.emplace_back(1, static_cast<char>(byte));
  return keys;
}

/// @brief Flip each input bit separately and count changed output bits against the original hash.
/// Input bit IDs are byte-major, least-significant bit first. This is a finite
/// sensitivity experiment, not a cryptographic or distribution test. Cost O(n²)
/// hashed bytes and O(n) result rows; no input mutation or retained string copies.
/// @param data Borrowed raw bytes, unchanged.
/// @param bits Full FNV width, 32 or 64; default 64.
/// @param max_rows Whole-row limit, default 1000000; zero disables the policy limit.
/// @return Owned input_bit/byte/bit/changed_output_bits table; empty for empty input.
/// @throws std::invalid_argument If bits is not 32 or 64.
/// @throws std::length_error If 8*n overflows or exceeds max_rows/vector capacity.
/// @par Example
/// @code
/// const auto n = dm::algorithms::fnv_bit_flips("a").nrows(); // 8
/// @endcode
/// @par References
/// - RFC 9923, Section 1.4: https://www.rfc-editor.org/rfc/rfc9923.html#section-1.4
/// @see fnv1a_32(), fnv1a_64()
inline dstruct::DataFrame
fnv_bit_flips(const std::string& data, unsigned bits = 64, std::size_t max_rows = 1000000) {
  if (bits != 32 && bits != 64)
    throw std::invalid_argument("fnv_bit_flips: width");
  if (data.size() > std::numeric_limits<std::size_t>::max() / 8)
    throw std::length_error("fnv_bit_flips: row count");
  const auto rows = data.size() * 8;
  if ((max_rows && rows > max_rows) || rows > std::vector<double>().max_size())
    throw std::length_error("fnv_bit_flips: whole row quota");
  /// @brief Dispatch the requested fixed-width hash for this controlled bit-flip experiment.
  /// @param s Borrowed byte string.
  /// @return Zero-extended integer hash.
  const auto hash = [bits](const std::string& s) {
    return bits == 32 ? std::uint64_t(fnv1a_32(s)) : fnv1a_64(s);
  };
  const auto          baseline = hash(data);
  std::vector<double> input, byte, bit, changed;
  for (std::size_t i = 0; i < data.size(); ++i)
    for (unsigned j = 0; j < 8; ++j) {
      auto copy = data;
      copy[i]   = static_cast<char>(static_cast<unsigned char>(copy[i]) ^ (1u << j));
      input.push_back(double(i * 8 + j));
      byte.push_back(double(i));
      bit.push_back(j);
      changed.push_back(std::popcount(baseline ^ hash(copy)));
    }
  dstruct::DataFrame d;
  d.add_column("input_bit", input);
  d.add_column("byte", byte);
  d.add_column("bit", bit);
  d.add_column("changed_output_bits", changed);
  return d;
}
} // namespace datamunge::algorithms
