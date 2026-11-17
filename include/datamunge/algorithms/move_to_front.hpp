#pragma once
/// @file move_to_front.hpp
/// @brief Checked reversible byte move-to-front transforms with actual ranks and work.
#include <array>
#include <cstddef>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual linear-search comparisons and shifted alphabet entries.
/// Encode comparisons include the successful test; decoding indexes directly and
/// has zero search comparisons. Alphabet initialization's256 stores are excluded.
/// @par Example
/// @code
/// auto a=dm::algorithms::mtf_encode_analyze("ABBA");
/// std::print("{} {}\n",a.comparisons,a.shifts); // 136 132
/// @endcode
/// @see mtf_encode_analyze()
struct MTFAnalysis {
  std::vector<int> indices;        ///< Actual encoded ranks; empty for decoder analysis.
  std::string      bytes;          ///< Actual decoded bytes; empty for encoder analysis.
  std::size_t      comparisons{0}; ///< Actual successful/unsuccessful alphabet equality tests.
  std::size_t      shifts{0};      ///< Actual alphabet entries copied one position right.
};
/// @brief Actual selected byte/rank before moving it to the alphabet front.
/// Previous front and the selected byte are observed from the same evolving
///256-byte permutation as ordinary execution; no Unicode normalization occurs.
/// @par Example
/// @code
/// auto t=dm::algorithms::mtf_encode_trace("ABBA");
/// std::print("{}\n",t.steps.back().rank); // 1
/// @endcode
/// @see MTFTrace
struct MTFStep {
  std::size_t   index{0};        ///< Actual original stream position.
  unsigned char byte{0};         ///< Actual selected byte0..255, including NUL.
  int           rank{0};         ///< Actual current alphabet position0..255.
  unsigned char front_before{0}; ///< Actual alphabet-front byte before this move.
  std::size_t   comparisons{0};  ///< Actual encoder equality tests for this byte; zero for decode.
  std::size_t   shifts{0};       ///< Actual right shifts for this byte, exactly rank.
};
/// @brief Owned original stream and actual per-byte transform history.
/// Encode retains input_bytes and result.indices; decode retains input_indices
/// and result.bytes. Other two stream fields stay empty. Capture units are one
/// input element, one output element and one step record per byte, exactly3n.
/// @par Example
/// @code
/// auto t=dm::algorithms::mtf_encode_trace("ABBA");
/// std::print("{} {}\n",t.result.indices,t.captured_entries); // [65, 66, 0, 1] 12
/// @endcode
/// @see mtf_encode_trace(), mtf_decode_trace()
struct MTFTrace {
  bool                 decoding{false};     ///< True for actual inverse transform.
  std::string          input_bytes;         ///< Owned original bytes on encode.
  std::vector<int>     input_indices;       ///< Owned original ranks on decode.
  MTFAnalysis          result;              ///< Actual output plus observed work.
  std::vector<MTFStep> steps;               ///< Actual moves, one per original stream position.
  std::size_t          captured_entries{0}; ///< Exactly3n successful capture units.
};
/// @brief Encode arbitrary bytes to ranks in a moving unsigned-byte alphabet.
/// Initial alphabet is0..255. Search the current rank, emit it, shift preceding
/// entries right and put the byte at front. Exact lossless recoding, not entropy
/// compression; storing ranks as ints can expand the stream. Empty allowed.
/// O(256n) worst-case work, fixed256-byte driver alphabet plus owned n outputs.
/// @param data Arbitrary byte string, including embedded NUL/high bytes.
/// @return Owned ranks0..255 in original stream order.
/// @throws std::length_error If actual work counters exceed size_t.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::mtf_encode("ABBA")); // [65, 66, 0, 1]
/// @endcode
/// @par References
/// - J. L. Bentley et al. (1986), A locally adaptive data compression scheme, Communications of the ACM29,320–330, doi:10.1145/5684.5688.
/// @see mtf_decode(), mtf_encode_analyze()
std::vector<int> mtf_encode(const std::string& data);
/// @brief Decode valid move-to-front ranks with the same evolving alphabet.
/// Validates every rank before indexing/output work; this corrects the legacy
/// out-of-bounds behavior on negative or above255 indices. Empty input gives
/// an empty string. O(256n) work, fixed alphabet plus n owned output bytes.
/// @param indices Complete rank stream, each in[0,255].
/// @return Owned exact original bytes when given an encoded stream.
/// @throws std::invalid_argument If any rank is outside[0,255].
/// @throws std::length_error If actual shift counts exceed size_t.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::mtf_decode({65,66,0,1})); // ABBA
/// @endcode
/// @see mtf_encode(), mtf_decode_analyze()
std::string mtf_decode(const std::vector<int>& indices);
/// @brief Obtain actual encoded ranks and work from the ordinary producer.
/// Same semantics/complexity as mtf_encode(), without replay or trace storage.
/// @param data Arbitrary input bytes, empty allowed.
/// @return Owned actual output ranks and counts; bytes field empty.
/// @throws std::length_error If work counts exceed size_t.
/// @par Example
/// @code
/// auto a=dm::algorithms::mtf_encode_analyze("AA");
/// std::print("{} {}\n",a.indices,a.shifts); // [65, 0] 65
/// @endcode
/// @see MTFAnalysis
MTFAnalysis mtf_encode_analyze(const std::string& data);
/// @brief Obtain actual decoded bytes and work from the ordinary inverse producer.
/// Full rank validation precedes output indexing. Same complexity as mtf_decode().
/// @param indices Complete ranks0..255, empty allowed.
/// @return Owned actual decoded bytes/counts; indices field empty.
/// @throws std::invalid_argument If any rank is invalid.
/// @throws std::length_error If shift counters exceed size_t.
/// @par Example
/// @code
/// auto a=dm::algorithms::mtf_decode_analyze({65,0});
/// std::print("{} {}\n",a.bytes,a.shifts); // AA 65
/// @endcode
/// @see MTFAnalysis
MTFAnalysis mtf_decode_analyze(const std::vector<int>& indices);
/// @brief Capture actual byte/rank/front observations during ordinary encoding.
/// Preflights the complete3n logical-entry quota before copying or transforming.
/// Zero disables the optional quota. Trace owns all data; no replay or shared state.
/// @param data Arbitrary input bytes, empty allowed.
/// @param max_entries Complete capture cap, default1000000; zero disables.
/// @return Owned actual encoder history.
/// @throws std::length_error If quota, complete capture geometry or work counters exceed limits.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::mtf_encode_trace("ABBA",12).steps.size()); // 4
/// @endcode
/// @see MTFTrace
MTFTrace mtf_encode_trace(const std::string& data, std::size_t max_entries = 1000000);
/// @brief Capture actual byte/rank/front observations during ordinary decoding.
/// Validates all ranks before the complete3n quota preflight and inverse work.
/// Zero disables the optional quota. Same producer as mtf_decode(), no replay.
/// @param indices Complete original rank stream, each0..255.
/// @param max_entries Complete capture cap, default1000000; zero disables.
/// @return Owned actual inverse history.
/// @throws std::invalid_argument If any rank is invalid, before quota checks.
/// @throws std::length_error If capture geometry/quota or work counters exceed limits.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::mtf_decode_trace({65,66,0,1}).result.bytes); // ABBA
/// @endcode
/// @see MTFTrace
MTFTrace mtf_decode_trace(const std::vector<int>& indices, std::size_t max_entries = 1000000);
/// @brief Native observed rank distribution, including exact count-based median.
/// Median averages the two central ranks for even count. Empty means/median/
/// entropy are explicit zero placeholders. Entropy is empirical marginal bits per
/// rank, not compressed size or a claim that an entropy coder was run.
/// @par Example
/// @code
/// auto s=dm::algorithms::mtf_index_summary({65,66,0,1});
/// std::print("{} {}\n",s.zeros,s.median); // 1 33
/// @endcode
/// @see mtf_index_summary()
struct MTFIndexSummary {
  std::array<std::size_t, 256> histogram{};     ///< Actual frequency of each rank.
  std::size_t                  count{0};        ///< Actual total ranks.
  std::size_t                  zeros{0};        ///< Actual zero-rank count.
  std::size_t                  one_to_three{0}; ///< Actual count in1..3.
  std::size_t                  four_or_more{0}; ///< Actual count in4..255.
  double                       mean{0};   ///< Arithmetic rank mean, zero placeholder if empty.
  double                       median{0}; ///< Actual rank median, zero placeholder if empty.
  double entropy_bits{0};                 ///< Empirical marginal Shannon entropy, zero if empty.
};
/// @brief Summarize validated ranks using a native fixed-size histogram.
/// O(n+256) work, fixed256-count driver storage. Full validation precedes counts.
/// @param indices Complete rank stream0..255, empty allowed.
/// @return Owned counts, mean, exact histogram median and empirical entropy.
/// @throws std::invalid_argument If any rank is invalid.
/// @par Example
/// @code
/// auto s=dm::algorithms::mtf_index_summary({0,0,1,1});
/// std::print("{} {}\n",s.median,s.entropy_bits); // 0.5 1
/// @endcode
/// @see MTFIndexSummary
MTFIndexSummary mtf_index_summary(const std::vector<int>& indices);
/// @brief Return the original fourfold repetitive sentence as owned bytes.
/// @return Original text, without a sentinel or terminating NUL counted as data.
/// @par Example
/// @code
/// std::print("{}\n",!dm::algorithms::mtf_demo_text().empty()); // true
/// @endcode
/// @see mtf_encode()
std::string mtf_demo_text();
} // namespace datamunge::algorithms
