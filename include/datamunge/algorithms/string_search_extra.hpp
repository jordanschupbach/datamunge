#pragma once

/// @file string_search_extra.hpp
/// @brief Horspool and reusable Zhu-Takaoka byte search, with actual scan-work diagnostics.
#include <datamunge/dstruct/dataframe.hpp>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// @brief Owned substring occurrences and actual work done by one search scan.
///
/// Only pattern/text equality checks count as comparisons; preprocessing,
/// table lookups and loop conditions are excluded. Positions are byte offsets.
/// @par Example
/// @code
/// const auto result = dm::algorithms::ZhuTakaoka("aa").scan("aaaa");
/// const auto count = result.positions.size(); // 3
/// @endcode
/// @see ZhuTakaoka::scan(), string_search_comparison()
struct StringSearchScan {
    std::vector<std::size_t> positions; ///< Sorted zero-based offsets, including overlapping matches.
    std::size_t comparisons = 0; ///< Equality tests between pattern and text bytes, including failures.
    std::size_t alignments = 0; ///< Pattern windows actually visited.
};

/// @brief Actual right-to-left comparison outcome and window-end shift for one Horspool alignment.
/// Ending byte is always text[position+m-1], even if a different text byte caused
/// a mismatch. Full matches use mismatch_index=m rather than a valid pattern index.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::boyer_moore_horspool_trace("AXCABCA","ABCA");
/// std::print("{} {}\n",trace.alignments[0].mismatch_index,trace.alignments[0].shift); // 1 3
/// @endcode
/// @see HorspoolTrace
struct HorspoolAlignment {
    std::size_t position{0}; ///< Pattern-window start in text bytes.
    std::size_t matched_suffix_length{0}; ///< Equal trailing bytes, or m on a full match.
    std::size_t mismatch_index{0}; ///< Failed pattern index, or m on a full match.
    std::size_t comparisons{0}; ///< Actual equality tests in this alignment, including a failed test.
    unsigned char ending_byte{0}; ///< Last text byte of this window; always the shift-table key.
    std::size_t shift{0}; ///< Selected positive table shift, including the departure after the final alignment.
    bool matched{false}; ///< Whether all m pattern bytes matched.
};

/// @brief Owned Horspool shifts, actual alignment records and the shared scan result.
/// Tables and records contain no borrowed bytes. Empty/oversized patterns have no
/// alignments; empty patterns have zero shifts. Metadata and all positions count bytes.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::boyer_moore_horspool_trace("AXCABCA","ABCA");
/// std::print("{} {}\n",trace.scan.positions,trace.scan.comparisons); // [3] 7
/// @endcode
/// @see boyer_moore_horspool_trace(), StringSearchScan
struct HorspoolTrace {
    StringSearchScan scan; ///< Same positions and counters as the canonical unrecorded scan.
    std::array<std::size_t,256> shifts{}; ///< Actual shift table, keyed by unsigned window-ending byte.
    std::vector<HorspoolAlignment> alignments; ///< Every actual visited window and its selected shift.
    std::size_t text_size{0}; ///< Original text byte length.
    std::size_t pattern_size{0}; ///< Original pattern byte length.
};

/// @brief Build Horspool shifts from every pattern byte except the final one.
/// Each entry is m by default, or m-1-i for its rightmost occurrence i<m-1.
/// Thus shifts lie in [1,m] for nonempty patterns; empty patterns yield all zeros.
/// O(m+256) work and O(256) owned output; no allocation, input mutation or shared state.
/// @param pattern Borrowed arbitrary byte sequence, possibly empty.
/// @return Owned 256-entry size_t shift array indexed by unsigned byte values.
/// @par Example
/// @code
/// const auto shifts=dm::algorithms::horspool_shifts("abra");
/// std::print("{} {} {} {}\n",shifts['a'],shifts['b'],shifts['r'],shifts['z']); // 3 2 1 4
/// @endcode
/// @par References
/// - R. N. Horspool (1980), Practical fast searching in strings, Software: Practice and Experience
///   10(6), 501-506, doi:10.1002/spe.4380100608.
/// @see boyer_moore_horspool_trace(), boyer_moore_horspool_search()
std::array<std::size_t,256> horspool_shifts(const std::string& pattern);

/// @brief Record actual Horspool alignments using the same loop as ordinary and corpus search.
/// A shift always uses the window's ending byte, including after a full match.
/// Always preprocesses the pattern, even for an empty/oversized search. Preprocessing
/// costs O(m+256); search comparisons are O((n-m+1)*m) in the worst eligible case.
/// Recording uses O(A+K+256) owned storage for A alignments and K overlapping matches.
/// Comparison counters exclude preprocessing and loop checks. Case-sensitive byte
/// semantics support NUL/high bytes, with no Unicode normalization. Calls share no
/// state, borrow inputs unchanged and retain no text/pattern copies.
/// @param text Borrowed arbitrary text bytes.
/// @param pattern Borrowed bytes; empty patterns match nowhere.
/// @return Owned actual shifts, records, sizes and scan result.
/// @throws std::overflow_error If the number of search equality tests exceeds size_t.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::boyer_moore_horspool_trace("aaaa","aa");
/// std::print("{} {}\n",trace.scan.positions,trace.scan.comparisons); // [0, 1, 2] 6
/// @endcode
/// @par References
/// - R. N. Horspool (1980), Practical fast searching in strings, doi:10.1002/spe.4380100608;
///   this implementation reports all overlapping occurrences.
/// @see horspool_shifts(), boyer_moore_horspool_search()
HorspoolTrace boyer_moore_horspool_trace(const std::string& text,const std::string& pattern);

/// @brief Compiled Zhu-Takaoka matcher with a two-byte bad-character shift and strong good suffix.
///
/// Preprocessing costs O(m+256^2) time/storage for pattern length m. The
/// 256^2-entry int table is retained for searches over many separate texts.
/// Scans compare right to left and cost O(n*m) in the worst case; periodic
/// matches can require full comparisons at every position. Empty patterns
/// match nowhere by this library's convention. Matching is case-sensitive
/// on bytes, with embedded NUL and high-bit bytes supported; there is no
/// Unicode normalisation. Const scans may run concurrently on this immutable
/// object; returned positions and tables own their storage.
/// @par Example
/// @code
/// const dm::algorithms::ZhuTakaoka matcher("ana");
/// const auto hits = matcher.search("banana"); // {1,3}
/// @endcode
/// @par References
/// - R. F. Zhu and T. Takaoka (1987), On improving the average case of the
///   Boyer-Moore string matching algorithm, Journal of Information Processing 10(3), 173-177.
///   https://ipsj.ixsq.nii.ac.jp/records/59825
/// @see zhu_takaoka_search(), boyer_moore_horspool_search()
class ZhuTakaoka {
public:
    /// @brief Copy a pattern and build its reusable shift tables.
    /// @param pattern Byte pattern; empty and single-byte patterns need no digram table.
    /// @throws std::length_error If pattern.size() exceeds INT_MAX, the shift-table index limit.
    /// @par Example
    /// @code
    /// const dm::algorithms::ZhuTakaoka matcher("Miss.");
    /// const auto size = matcher.pattern().size(); // 5
    /// @endcode
    /// @par References
    /// - R. F. Zhu and T. Takaoka (1987), Journal of Information Processing 10(3), 173-177.
    /// @see scan(), pattern()
    explicit ZhuTakaoka(std::string pattern);

    /// @brief Borrow the pattern owned by this matcher.
    /// @return Read-only pattern reference, valid for this object's lifetime.
    const std::string& pattern() const noexcept { return pattern_; }

    /// @brief Find every occurrence, including overlaps, using the compiled tables.
    /// @param text Case-sensitive byte string; borrowed and unchanged.
    /// @return Owned sorted zero-based byte offsets; empty for an empty pattern or short text.
    /// @par Example
    /// @code
    /// const auto hits = dm::algorithms::ZhuTakaoka("aa").search("aaaa"); // {0,1,2}
    /// @endcode
    /// @par References
    /// - R. F. Zhu and T. Takaoka (1987), Journal of Information Processing 10(3), 173-177.
    /// @see scan(), trace()
    std::vector<std::size_t> search(const std::string& text) const;

    /// @brief Search and count actual visited windows and byte comparisons.
    /// @param text Byte string to scan, borrowed and unchanged.
    /// @return Owned positions and counters from the same loop as search().
    /// @par Example
    /// @code
    /// const auto work = dm::algorithms::ZhuTakaoka("aa").scan("aaaa");
    /// const auto comparisons = work.comparisons; // 6
    /// @endcode
    /// @par References
    /// - R. F. Zhu and T. Takaoka (1987), Journal of Information Processing 10(3), 173-177.
    /// @see search(), string_search_comparison()
    StringSearchScan scan(const std::string& text) const;

    /// @brief Trace every alignment of the actual matcher, including the selected shift.
    ///
    /// Columns: offset, matched_suffix, comparisons, match (0/1), digram_shift,
    /// suffix_shift and advance. On full matches only the suffix shift is used,
    /// and digram_shift is zero. A single-byte pattern advances by one.
    /// @param text Borrowed byte string.
    /// @return Owned trace; numeric offsets are represented exactly up to 2^53.
    /// @par Example
    /// @code
    /// const auto trace = dm::algorithms::ZhuTakaoka("aa").trace("aaaa");
    /// const auto rows = trace.nrows(); // 3
    /// @endcode
    /// @par References
    /// - R. F. Zhu and T. Takaoka (1987), Journal of Information Processing 10(3), 173-177.
    /// @see scan(), digram_table()
    dstruct::DataFrame trace(const std::string& text) const;

    /// @brief Inspect the digram shifts for all ordered pairs in an alphabet.
    /// @param alphabet Bytes whose Cartesian square is listed, in supplied order; duplicates retained.
    /// @return Owned columns first_byte, last_byte, shift, in byte-code form (including NUL).
    /// @throws std::invalid_argument If the pattern has fewer than two bytes.
    /// @par Example
    /// @code
    /// const auto table = dm::algorithms::ZhuTakaoka("aba").digram_table("ab");
    /// const auto rows = table.nrows(); // 4
    /// @endcode
    /// @par References
    /// - R. F. Zhu and T. Takaoka (1987), Journal of Information Processing 10(3), 173-177.
    /// @see trace()
    dstruct::DataFrame digram_table(const std::string& alphabet) const;
private:
    /// @brief Run the shared search loop, optionally materialising its alignment trace.
    /// @param text Borrowed byte string.
    /// @param trace Optional output frame; null omits trace storage.
    /// @return Owned positions and actual counters.
    StringSearchScan scan_impl(const std::string& text, dstruct::DataFrame* trace) const;
    std::string pattern_; ///< Owned bytes, never modified after construction.
    std::vector<int> digram_; ///< 256x256 shifts, flattened by first_byte*256+last_byte.
    std::vector<int> suffix_; ///< Strong good-suffix shift for each mismatch index; index zero also handles full match.
};

/// @brief One-shot Boyer-Moore-Horspool byte search using a one-byte shift table.
///
/// Right-to-left comparisons, O(m+256) preprocessing, O(n*m) worst-case search;
/// empty patterns match nowhere. Embedded NUL and high-bit bytes are supported.
/// Shifts use the last window byte and omit the final pattern position during
/// preprocessing. O(256) auxiliary storage plus O(K) owned offsets; no alignment
/// records are allocated, inputs are unchanged and independent calls share no state.
/// @param text Borrowed text bytes.
/// @param pattern Borrowed pattern bytes.
/// @return Sorted zero-based offsets of all overlapping occurrences.
/// @throws std::overflow_error If the number of search equality tests exceeds size_t.
/// @par Example
/// @code
/// const auto hits = dm::algorithms::boyer_moore_horspool_search("banana", "ana"); // {1,3}
/// @endcode
/// @par References
/// - R. N. Horspool (1980), Practical fast searching in strings, Software: Practice and Experience
///   10(6), 501-506, https://doi.org/10.1002/spe.4380100608
/// @see horspool_shifts(), boyer_moore_horspool_trace(), zhu_takaoka_search()
std::vector<std::size_t> boyer_moore_horspool_search(const std::string& text, const std::string& pattern);

/// @brief One-shot Zhu-Takaoka search; compile a ZhuTakaoka object to reuse its tables.
/// @param text Borrowed bytes, unchanged.
/// @param pattern Borrowed pattern, copied into a temporary matcher.
/// @return All overlapping zero-based byte offsets; an empty pattern matches nowhere.
/// @throws std::length_error If pattern.size() exceeds INT_MAX.
/// @par Example
/// @code
/// const auto hits = dm::algorithms::zhu_takaoka_search("banana", "ana"); // {1,3}
/// @endcode
/// @par References
/// - R. F. Zhu and T. Takaoka (1987), Journal of Information Processing 10(3), 173-177.
///   https://ipsj.ixsq.nii.ac.jp/records/59825
/// @see ZhuTakaoka, boyer_moore_horspool_search()
std::vector<std::size_t> zhu_takaoka_search(const std::string& text, const std::string& pattern);

/// @brief Compare actual naive, Horspool and Zhu-Takaoka scans on a DataFrame text column.
///
/// Compiles each pattern once per method, then searches records separately so
/// matches cannot cross record boundaries. Counts exclude preprocessing and are
/// work metrics, not elapsed time. No case folding or other transformation occurs.
/// @param frame Borrowed dataset, unchanged.
/// @param column String column, no nulls.
/// @param patterns Byte patterns, including empty patterns (which match nowhere).
/// @return Owned rows per pattern/method: pattern, method, records, bytes, matches,
///         alignments, comparisons and agrees_with_naive (1 iff every row's positions match).
/// @throws std::invalid_argument If the column has the wrong type or a null value.
/// @throws std::out_of_range If the column is absent.
/// @throws std::length_error If a pattern exceeds INT_MAX.
/// @throws std::overflow_error If a Horspool scan's comparison counter exceeds size_t.
/// @par Example
/// @code
/// const auto comparison = dm::algorithms::string_search_comparison(
///     dm::datasets::titanic(), "Name", {"Miss.", "Mrs."});
/// const auto rows = comparison.nrows(); // 6
/// @endcode
/// @par References
/// - R. F. Zhu and T. Takaoka (1987), Journal of Information Processing 10(3), 173-177.
/// - R. N. Horspool (1980), https://doi.org/10.1002/spe.4380100608
/// @see ZhuTakaoka::scan()
dstruct::DataFrame string_search_comparison(const dstruct::DataFrame& frame, const std::string& column,
                                            const std::vector<std::string>& patterns);
} // namespace datamunge::algorithms
