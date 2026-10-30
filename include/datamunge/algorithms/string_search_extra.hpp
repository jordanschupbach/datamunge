#pragma once

/// @file string_search_extra.hpp
/// @brief Horspool and reusable Zhu-Takaoka byte search, with actual scan-work diagnostics.
#include <datamunge/dstruct/dataframe.hpp>

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
/// @param text Borrowed text bytes.
/// @param pattern Borrowed pattern bytes.
/// @return Sorted zero-based offsets of all overlapping occurrences.
/// @par Example
/// @code
/// const auto hits = dm::algorithms::boyer_moore_horspool_search("banana", "ana"); // {1,3}
/// @endcode
/// @par References
/// - R. N. Horspool (1980), Practical fast searching in strings, Software: Practice and Experience
///   10(6), 501-506, https://doi.org/10.1002/spe.4380100608
/// @see zhu_takaoka_search()
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
