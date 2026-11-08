#pragma once
/// @file burrows_wheeler.hpp
/// @brief Byte-consistent cyclic BWT and checked LF inversion, with actual native provenance.

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief Last column and primary row of the sorted cyclic-rotation matrix.
/// No sentinel is appended; every byte, including zero and 128..255, is data.
/// Unsigned-byte lexicographic order is used. Equal rotations are ordered by start
/// offset, so the primary row is the first row equal to the input. Field order and
/// types retain the original aggregate layout; default construction is safely empty.
/// @par Example
/// @code
/// const auto r=dm::algorithms::bwt_transform("banana");
/// std::print("{} {}\n",r.transformed,r.index); // nnbaaa 3
/// @endcode
/// @par References
/// - M. Burrows and D. J. Wheeler (1994), A Block-sorting Lossless Data Compression
///   Algorithm, SRC Research Report 124, https://www.cs.jhu.edu/~langmea/resources/burrows_wheeler.pdf.
/// @see bwt_transform(), bwt_inverse()
struct BWTResult {
  std::string transformed; ///< Last byte of each sorted cyclic rotation, with unchanged length/frequencies.
  std::size_t index{0}; ///< Original rotation's row; zero for empty input.
};

/// @brief Optional complete-call byte-comparison limit for forward tracing.
/// Zero means unlimited. A positive budget counts actual byte-pair tests inside
/// rotation comparisons and throws before exceeding it; no partial transform is
/// returned. This bounds the deliberately naive sort's inspected-byte work.
/// @par Example
/// @code
/// dm::algorithms::BWTForwardOptions o;o.max_byte_comparisons=1;
/// bool rejected=false;
/// try {(void)dm::algorithms::bwt_transform_trace("banana",o);}
/// catch (const std::runtime_error&) {rejected=true;}
/// std::print("{}\n",rejected); // true
/// @endcode
/// @see bwt_transform_trace()
struct BWTForwardOptions {
  std::size_t max_byte_comparisons{0}; ///< Actual inspected-byte-pair budget, zero for unlimited.
};

/// @brief Owned input, actual sorted rotation starts and actual comparison counters.
/// No rotation matrix or comparator history is materialized. start offsets break
/// full-rotation ties. Counters reflect this toolchain's std::sort calls and are
/// not portable fixed formulas; overflow throws rather than wrapping. O(n) storage
/// in addition to the input/result, with worst-case O(n^2 log(n+1)) forward work.
/// @par Example
/// @code
/// const auto t=dm::algorithms::bwt_transform_trace("banana");
/// std::print("{}\n",t.rotation_starts); // [5, 3, 1, 0, 4, 2]
/// @endcode
/// @see bwt_transform_trace(), BWTForwardOptions
struct BWTForwardTrace {
  std::string input; ///< Owned original bytes.
  BWTResult result; ///< Same result produced by the ordinary transform.
  std::vector<std::size_t> rotation_starts; ///< Actual sorted start offsets, one per matrix row.
  std::size_t rotation_comparisons{0}; ///< Actual comparator invocations, including equal rotations.
  std::size_t byte_comparisons{0}; ///< Inspected byte pairs inside those invocations.
  std::size_t equal_rotation_comparisons{0}; ///< Invocations inspecting all n pairs and using the start-offset tie break.
};

/// @brief Owned stable last-to-first correspondence for any byte last column.
/// counts[c] is the frequency of byte c; starts[c] counts bytes strictly below c.
/// occurrence_rank[i] counts earlier equal bytes in L, so lf[i]=starts[L[i]]+rank[i]
/// and first_column[lf[i]]=L[i]. lf is always a permutation, but that alone does not
/// certify that an arbitrary last-column/primary-index pair is a valid cyclic BWT.
/// O(n+256) construction time/storage; arrays use size_t, not narrowing int counts.
/// @par Example
/// @code
/// const auto m=dm::algorithms::bwt_lf_mapping("nnbaaa");
/// std::print("{} {}\n",m.first_column,m.lf); // aaabnn [4, 5, 3, 0, 1, 2]
/// @endcode
/// @par References
/// - M. Burrows and D. J. Wheeler (1994), SRC Research Report 124, section 2,
///   https://www.cs.jhu.edu/~langmea/resources/burrows_wheeler.pdf.
/// @see bwt_lf_mapping(), BWTInverseTrace
struct BWTLFMapping {
  std::array<std::size_t,256> counts{}; ///< Byte frequencies in L, summing to n.
  std::array<std::size_t,256> starts{}; ///< Unsigned-byte first-column offsets.
  std::string first_column; ///< All L bytes in unsigned sorted order.
  std::vector<std::size_t> occurrence_rank; ///< Zero-based same-byte occurrence rank at each L position.
  std::vector<std::size_t> lf; ///< Stable mapping from L rows into first-column rows.
};

/// @brief One actual LF inversion step and its right-to-left output write.
/// Repeated rows are allowed: a periodic input can revisit a row before n steps
/// have been taken. Exactly n bytes are written, independent of LF cycle length.
/// @par Example
/// @code
/// const auto t=dm::algorithms::bwt_inverse_trace(dm::algorithms::bwt_transform("banana"));
/// std::print("{} {} {}\n",t.steps[0].row,t.steps[0].output_index,static_cast<unsigned int>(t.steps[0].value)); // 3 5 97
/// @endcode
/// @see BWTInverseTrace
struct BWTInverseStep {
  std::size_t row{0}; ///< Current row before following LF.
  std::size_t next_row{0}; ///< Actual lf[row] successor.
  std::size_t output_index{0}; ///< Written output position, decreasing from n-1 to zero.
  unsigned char value{0}; ///< Byte read from the current L row and written to output.
};

/// @brief Owned encoded input, LF provenance, decoded bytes and actual inversion steps.
/// Ordinary inversion shares the kernel but does not allocate step/rank/first-column
/// records or copy the encoded input. Tracing uses O(n+256) storage and time, and
/// never replays the inversion to fabricate a path. Calls share no mutable state.
/// @par Example
/// @code
/// const auto t=dm::algorithms::bwt_inverse_trace(dm::algorithms::bwt_transform("banana"));
/// std::print("{} {}\n",t.output,t.steps.size()); // banana 6
/// @endcode
/// @see bwt_inverse_trace(), BWTLFMapping
struct BWTInverseTrace {
  BWTResult input; ///< Owned last column and primary row.
  BWTLFMapping mapping; ///< Actual frequencies, ranks, first column and LF permutation.
  std::string output; ///< Exactly n reconstructed bytes.
  std::vector<BWTInverseStep> steps; ///< Actual walk/write order, including repeated rows.
};

/// @brief Shared unsigned-byte sort and LF kernels used by ordinary and traced APIs.
namespace bwt_detail {
/// @brief Increment an actual forward-work counter without wrapping.
/// @param count Counter to increment.
/// @throws std::overflow_error If the next count is not representable.
inline void increment(std::size_t& count) {
  if (count==std::numeric_limits<std::size_t>::max()) throw std::overflow_error("BWT trace counter overflow");
  ++count;
}

/// @brief Sort actual rotation indices once, optionally retaining provenance/counters.
/// @tparam Record Whether to own input and count the actual comparator work.
/// @param text Borrowed bytes, including embedded zero/high-bit bytes.
/// @param options Forward trace budget; ordinary calls use no limit.
/// @return Transform and optional actual order/counters, never a partial result.
/// @throws std::runtime_error Before a positive inspected-byte budget is exceeded.
/// @throws std::overflow_error Before a trace counter could wrap.
/// @throws std::bad_alloc If output/order/trace storage cannot be allocated.
/// @throws std::length_error If required storage exceeds vector/string capacity.
template<bool Record>
BWTForwardTrace transform(const std::string& text,BWTForwardOptions options) {
  const auto n=text.size();BWTForwardTrace trace;
  if constexpr (Record) trace.input=text;
  if (!n) return trace;
  std::vector<std::size_t> rotations(n);
  std::iota(rotations.begin(),rotations.end(),std::size_t{0});
  /// @brief Compare cyclic rotations in unsigned-byte order, with deterministic start ties.
  /// @param a First original start offset.
  /// @param b Second original start offset.
  /// @return Whether rotation a precedes rotation b.
  auto compare=[&](std::size_t a,std::size_t b) {
    if constexpr (Record) increment(trace.rotation_comparisons);
    auto left=a,right=b;
    for (std::size_t k=0;k<n;++k) {
      if constexpr (Record) {
        if (options.max_byte_comparisons && trace.byte_comparisons>=options.max_byte_comparisons)
          throw std::runtime_error("BWT byte-comparison budget exceeded");
        increment(trace.byte_comparisons);
      }
      const auto ca=static_cast<unsigned char>(text[left]),cb=static_cast<unsigned char>(text[right]);
      if (ca!=cb) return ca<cb;
      if (++left==n) left=0;
      if (++right==n) right=0;
    }
    if constexpr (Record) increment(trace.equal_rotation_comparisons);
    return a<b;
  };
  std::sort(rotations.begin(),rotations.end(),compare);
  trace.result.transformed.resize(n);
  for (std::size_t row=0;row<n;++row) {
    const auto start=rotations[row];
    trace.result.transformed[row]=text[start?start-1:n-1];
    if (!start) trace.result.index=row;
  }
  if constexpr (Record) trace.rotation_starts=std::move(rotations);
  return trace;
}

/// @brief Build the actual LF permutation, optionally retaining occurrence/first-column provenance.
/// @tparam Record Whether to store first-column and occurrence-rank vectors.
/// @param last Borrowed last-column bytes.
/// @return Actual mapping with checked-width frequency/rank arithmetic.
/// @throws std::bad_alloc If mapping storage cannot be allocated.
/// @throws std::length_error If required storage exceeds vector/string capacity.
template<bool Record>
BWTLFMapping map(const std::string& last) {
  BWTLFMapping result;
  for (const unsigned char c:last) ++result.counts[c];
  for (std::size_t c=1;c<256;++c) result.starts[c]=result.starts[c-1]+result.counts[c-1];
  result.lf.resize(last.size());
  if constexpr (Record) {result.first_column.resize(last.size());result.occurrence_rank.resize(last.size());}
  std::array<std::size_t,256> occurrence{};
  for (std::size_t i=0;i<last.size();++i) {
    const auto c=static_cast<unsigned char>(last[i]);
    result.lf[i]=result.starts[c]+occurrence[c];
    if constexpr (Record) {
      result.occurrence_rank[i]=occurrence[c];result.first_column[result.lf[i]]=last[i];
    }
    ++occurrence[c];
  }
  return result;
}

/// @brief Check the primary row before any LF/output allocation or indexing.
/// @param encoded Last column and primary row to inspect.
/// @throws std::invalid_argument If an empty input has nonzero index, or a nonempty index is out of range.
inline void validate(const BWTResult& encoded) {
  if (encoded.transformed.empty()?encoded.index!=0:encoded.index>=encoded.transformed.size())
    throw std::invalid_argument("BWT primary index is outside the last column");
}

/// @brief Decode once via actual stable LF steps, optionally owning their records.
/// @tparam Record Whether to own input, first-column/rank provenance and walk records.
/// @param encoded Borrowed valid transform, with primary range validated here.
/// @return Decoded bytes and optional provenance; periodic walks still perform n writes.
/// @throws std::invalid_argument For an invalid primary row.
/// @throws std::bad_alloc If mapping/output/trace storage cannot be allocated.
/// @throws std::length_error If required storage exceeds vector/string capacity.
template<bool Record>
BWTInverseTrace inverse(const BWTResult& encoded) {
  validate(encoded);BWTInverseTrace trace;
  const auto n=encoded.transformed.size();
  if constexpr (Record) trace.input=encoded;
  trace.mapping=map<Record>(encoded.transformed);
  trace.output.resize(n,'\0');
  if constexpr (Record) trace.steps.reserve(n);
  auto row=encoded.index;
  for (std::size_t k=0;k<n;++k) {
    const auto next=trace.mapping.lf[row];
    const auto value=static_cast<unsigned char>(encoded.transformed[row]);
    trace.output[n-1-k]=static_cast<char>(value);
    if constexpr (Record) trace.steps.push_back({row,next,n-1-k,value});
    row=next;
  }
  return trace;
}
} // namespace bwt_detail

/// @brief Transform a byte string by sorting cyclic rotations and retaining the last column.
/// Preserves ASCII results, including banana -> (nnbaaa,3), and fixes signed-char
/// ordering for arbitrary binary bytes by using unsigned order consistently with
/// inversion. No sentinel is appended. Equal rotations use ascending start offsets.
/// Sorts n indices, without allocating n full rotations. O(n^2 log(n+1)) worst-case
/// time, O(n) auxiliary storage; does not compress or mutate input. Empty gives ("",0).
/// @param s Borrowed byte string; embedded zero/high-bit bytes are supported.
/// @return Owned last column and zero-based original-row index.
/// @throws std::bad_alloc If order/output storage cannot be allocated.
/// @throws std::length_error If required storage exceeds vector/string capacity.
/// @par Example
/// @code
/// const auto r=dm::algorithms::bwt_transform("banana");
/// std::print("{} {}\n",r.transformed,r.index); // nnbaaa 3
/// @endcode
/// @par References
/// - M. Burrows and D. J. Wheeler (1994), A Block-sorting Lossless Data Compression
///   Algorithm, SRC Research Report 124, https://www.cs.jhu.edu/~langmea/resources/burrows_wheeler.pdf.
/// @see bwt_transform_trace(), bwt_inverse()
inline BWTResult bwt_transform(const std::string& s) {
  return bwt_detail::transform<false>(s,{}).result;
}

/// @brief Transform once and expose the actual sorted starts and comparator work.
/// Shares the ordinary transform kernel; no sort is replayed and no matrix is stored.
/// Counters include byte equality tests preceding a mismatch or a full-rotation tie.
/// std::sort call counts may differ between toolchains. A positive byte budget throws
/// without returning partial output. O(n) trace storage, O(n^2 log(n+1)) worst-case time.
/// @param s Borrowed binary or text bytes, possibly empty.
/// @param options Actual byte-comparison budget; zero by default means unlimited.
/// @return Owned input/result, sorted start indices and actual counters.
/// @throws std::runtime_error Before a positive byte-comparison cap is exceeded.
/// @throws std::overflow_error Before a trace counter could wrap.
/// @throws std::bad_alloc If order/output/trace storage cannot be allocated.
/// @throws std::length_error If required storage exceeds vector/string capacity.
/// @par Example
/// @code
/// const auto t=dm::algorithms::bwt_transform_trace("abab");
/// std::print("{} {}\n",t.result.transformed,t.rotation_starts); // bbaa [0, 2, 1, 3]
/// @endcode
/// @par References
/// - M. Burrows and D. J. Wheeler (1994), SRC Research Report 124, section 2,
///   https://www.cs.jhu.edu/~langmea/resources/burrows_wheeler.pdf.
/// @see bwt_transform(), BWTForwardOptions
inline BWTForwardTrace bwt_transform_trace(const std::string& s,BWTForwardOptions options={}) {
  return bwt_detail::transform<true>(s,options);
}

/// @brief Build a byte-consistent stable LF mapping and its actual occurrence provenance.
/// Accepts any last-column bytes, including empty input; no primary row is needed.
/// Every returned LF is a permutation, which does not by itself establish membership
/// in the cyclic-transform image. O(n+256) time/storage; input is not mutated.
/// @param last Borrowed last column, including embedded zero/high-bit bytes.
/// @return Owned frequencies, cumulative starts, first column, ranks and LF permutation.
/// @throws std::bad_alloc If mapping/provenance storage cannot be allocated.
/// @throws std::length_error If required storage exceeds vector/string capacity.
/// @par Example
/// @code
/// const auto m=dm::algorithms::bwt_lf_mapping("nnbaaa");
/// std::print("{}\n",m.occurrence_rank); // [0, 1, 0, 0, 1, 2]
/// @endcode
/// @par References
/// - M. Burrows and D. J. Wheeler (1994), SRC Research Report 124, section 2,
///   https://www.cs.jhu.edu/~langmea/resources/burrows_wheeler.pdf.
/// @see bwt_inverse(), BWTLFMapping
inline BWTLFMapping bwt_lf_mapping(const std::string& last) {return bwt_detail::map<true>(last);}

/// @brief Recover exactly n bytes from a valid cyclic BWT pair with a checked primary row.
/// Uses unsigned byte frequencies and size_t counters consistently with the forward
/// transform. A periodic LF walk may revisit rows before n writes; it is not stopped
/// at the first repeated row. O(n+256) time/storage; no trace records are allocated.
/// Range checks do not certify transform-image membership: arbitrary in-range pairs
/// decode deterministically but need not retransform to the supplied pair. This is
/// not corruption detection. Empty input requires index=0. Input is not mutated.
/// @param r Last column and primary index returned by bwt_transform(), or a same-format pair.
/// @return Exactly r.transformed.size() decoded bytes.
/// @throws std::invalid_argument If primary index is invalid, including empty/nonzero index.
/// @throws std::bad_alloc If mapping/output storage cannot be allocated.
/// @throws std::length_error If required storage exceeds vector/string capacity.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::bwt_inverse(dm::algorithms::bwt_transform("banana"))); // banana
/// @endcode
/// @par References
/// - M. Burrows and D. J. Wheeler (1994), SRC Research Report 124, section 2,
///   https://www.cs.jhu.edu/~langmea/resources/burrows_wheeler.pdf.
/// @see bwt_inverse_trace(), bwt_transform()
inline std::string bwt_inverse(const BWTResult& r) {return bwt_detail::inverse<false>(r).output;}

/// @brief Invert once and own the actual LF correspondence and right-to-left write path.
/// Shares the ordinary inversion kernel without replay. Exactly n steps are recorded,
/// including repeated LF rows for periodic inputs. Primary validation and arbitrary
/// in-range pair semantics match bwt_inverse(). O(n+256) time/storage, no shared state.
/// @param r Borrowed last-column/primary-row pair, normally produced by bwt_transform().
/// @return Owned encoded input, mapping, output and actual per-byte inversion steps.
/// @throws std::invalid_argument If the primary index is invalid.
/// @throws std::bad_alloc If input/mapping/output/trace storage cannot be allocated.
/// @throws std::length_error If required storage exceeds vector/string capacity.
/// @par Example
/// @code
/// const auto t=dm::algorithms::bwt_inverse_trace(dm::algorithms::bwt_transform("abab"));
/// std::print("{} {} {}\n",t.output,t.steps.size(),t.steps[2].row==t.steps[0].row); // abab 4 true
/// @endcode
/// @par References
/// - M. Burrows and D. J. Wheeler (1994), SRC Research Report 124, section 2,
///   https://www.cs.jhu.edu/~langmea/resources/burrows_wheeler.pdf.
/// @see bwt_inverse(), BWTInverseTrace
inline BWTInverseTrace bwt_inverse_trace(const BWTResult& r) {return bwt_detail::inverse<true>(r);}

} // namespace datamunge::algorithms
