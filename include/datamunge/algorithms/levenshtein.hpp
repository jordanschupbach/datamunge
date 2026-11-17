#pragma once
/// @file levenshtein.hpp
/// @brief Two-row byte edit distance with actual producing row capture and optional optimal traceback.
#include <cstddef>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual selected rolling-row operations; trace-only traceback is separate work.
/// @par Example
/// @code
/// const auto a=dm::algorithms::levenshtein_analyze("ab","a");
/// std::print("{}\n",a.work.cells); // 2
/// @endcode
/// @see LevenshteinAnalysis
struct LevenshteinWork {
  std::size_t rows{0};  ///< Actual nonboundary rows evaluated, longer byte length.
  std::size_t cells{0}; ///< Actual nonboundary cell/byte comparisons, n*m.
  std::size_t equalities{
      0}; ///< Actual equal byte comparisons among DP cells, not an alignment count.
  std::size_t traceback_moves{
      0}; ///< Additional trace-only optimal path moves;0 in ordinary/count-only calls.
};
/// @brief Owned actual byte distance and same-execution selected work.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::levenshtein_analyze("kitten","sitting").distance); // 3
/// @endcode
/// @see levenshtein_analyze()
struct LevenshteinAnalysis {
  std::size_t     distance{0}; ///< Actual minimum unit insertion/deletion/substitution cost.
  LevenshteinWork work; ///< Actual producing row/cell/equality and optional traceback counts.
};
/// @brief One actual optimal traceback move selected from the captured producing DP table.
/// Diagonal is preferred if optimal, then up, then left; costs are0 for equal
/// diagonals and1 otherwise. Coordinates are original actual row/column prefix
/// lengths before moving toward the origin, not necessarily a/b order.
/// @par Example
/// @code
/// const auto t=dm::algorithms::levenshtein_trace("a","b");
/// std::print("{}\n",t.path[0].cost); // 1
/// @endcode
/// @see LevenshteinTrace
struct LevenshteinEdit {
  std::size_t row{0};         ///< Actual prefix row before move.
  std::size_t column{0};      ///< Actual prefix column before move.
  std::size_t next_row{0};    ///< Actual chosen predecessor row.
  std::size_t next_column{0}; ///< Actual chosen predecessor column.
  std::size_t cost{0};        ///< Actual chosen edge cost,0 or1.
};
/// @brief Owned actual rolling-row snapshots and additional native optimal traceback, no DP rerun.
/// Rows hold the longer original byte string and columns the shorter; ties put
/// original b in rows. The full table is copied from each actually computed
/// rolling row, never independently rebuilt. Additional native traceback reads
/// this table once and records its actual optimal moves, without claiming that
/// the ordinary scalar API performs traceback.
/// Complete conservative admission counts input bytes, all table cells and at
/// most a.size+b.size path records before copying/evaluation. captured_entries
/// records actual input/table/path count. Excludes fields/capacities/runtime
/// rows/output copies/bytes/RAM. Default million logical records,0 cap off.
/// @par Example
/// @code
/// const auto t=dm::algorithms::levenshtein_trace("kitten","sitting");
/// std::print("{} {}\n",t.table.back().back(),t.row_bytes); // 3 sitting
/// @endcode
/// @see levenshtein_trace()
struct LevenshteinTrace {
  std::string row_bytes;         ///< Owned original longer byte string, b when equal lengths.
  std::string column_bytes;      ///< Owned original shorter byte string, a when equal lengths.
  bool        rows_are_a{false}; ///< True exactly when original a was strictly longer than b.
  std::vector<std::vector<std::size_t>>
      table; ///< Owned original boundary and every actual produced row.
  std::vector<LevenshteinEdit>
                      path;     ///< Additional actual optimal moves, full corner-to-origin order.
  LevenshteinAnalysis analysis; ///< Actual distance and producing/traceback selected work.
  std::size_t captured_entries{0}; ///< Actual original byte, table-cell and path-record count.
};
/// @brief Return unit insertion/deletion/substitution distance between complete byte strings.
/// Embedded NUL bytes are ordinary elements; no Unicode normalization/grapheme
/// decoding or transposition edit. Empty strings valid. The shorter input is
/// the column dimension of two rolling rows: O(n*m+n+m) time and O(min(n,m)+1)
/// runtime storage. Complete row/work and distance-arithmetic geometry checked
/// before allocation. No mutation, shared state or retained history.
/// @param a Borrowed complete original first byte string.
/// @param b Borrowed complete original second byte string.
/// @return Actual symmetric minimum unit edit cost.
/// @throws std::length_error On full row/work/safe increment geometry limits.
/// @throws std::bad_alloc On rolling-row allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::levenshtein_distance("kitten","sitting")); // 3
/// @endcode
/// @par References
/// - Wagner and Fischer(1974), The String-to-String Correction Problem, https://doi.org/10.1145/321796.321811.
/// @see levenshtein_trace(), levenshtein_analyze()
std::size_t levenshtein_distance(const std::string& a, const std::string& b);
/// @brief Return actual rolling-row byte distance and selected work without full-table capture.
/// Same complete byte/empty/geometry contracts as levenshtein_distance().
/// Equalities counts every equal DP cell, not only chosen optimal matches.
/// @param a Borrowed complete first byte string.
/// @param b Borrowed complete second byte string.
/// @return Owned actual distance and producing work; no traceback work.
/// @throws std::length_error On complete row/work/safe increment geometry limits.
/// @throws std::bad_alloc On rolling-row allocation failure.
/// @par Example
/// @code
/// const auto a=dm::algorithms::levenshtein_analyze("ab","a");
/// std::print("{} {}\n",a.distance,a.work.equalities); // 1 1
/// @endcode
/// @par References
/// - Wagner and Fischer(1974), https://doi.org/10.1145/321796.321811.
/// @see levenshtein_distance()
LevenshteinAnalysis levenshtein_analyze(const std::string& a, const std::string& b);
/// @brief Capture actual producing rows and perform a separate native optimal traceback on that table.
/// No DP reconstruction/prefix reruns. Full conservative input/table/path quota
/// is admitted before copies and evaluation; an eventual shorter path can still
/// be rejected by that conservative bound. Original orientations are explicit.
/// @param a Borrowed complete first byte string.
/// @param b Borrowed complete second byte string.
/// @param max_entries Conservative complete logical cap, default1000000;0 disables optional cap.
/// @return Owned actual rows, optimal moves and real producing/diagnostic work.
/// @throws std::length_error On complete geometry or conservative capture quota.
/// @throws std::bad_alloc On runtime/capture allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::levenshtein_trace("a","b",8);
/// std::print("{}\n",t.captured_entries); // 7
/// @endcode
/// @par References
/// - Wagner and Fischer(1974), https://doi.org/10.1145/321796.321811.
/// @see LevenshteinTrace
LevenshteinTrace
levenshtein_trace(const std::string& a, const std::string& b, std::size_t max_entries = 1000000);
} // namespace datamunge::algorithms
