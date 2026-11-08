#pragma once
/// @file
/// @brief Checked unrestricted byte-edit distance, actual Lowrance–Wagner tables and executable scripts.
#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::algorithms {
/// @brief Selected prefix-DP move, also identifying a literal byte-edit instruction.
/// Native scripts omit matches. Candidate ties prefer diagonal (match/substitute),
/// insert, erase, then transpose. A DP transpose can bridge deleted/inserted interior
/// bytes; a ByteEdit transpose always swaps exactly two adjacent current bytes.
/// @par Example
/// @code
/// const auto t=dm::algorithms::damerau_levenshtein_trace("ab","ba");
/// std::print("{}\n",t.cells.back().choice==dm::algorithms::DamerauMove::transpose); // true
/// @endcode
/// @see DamerauCell, ByteEdit
enum class DamerauMove {
  match, ///< Zero-cost equal-byte diagonal, omitted from native scripts.
  substitute, ///< Unit-cost replacement of one byte.
  insert, ///< Unit-cost insertion of one byte.
  erase, ///< Unit-cost deletion of one byte.
  transpose ///< DP bridge, or a literal adjacent swap in ByteEdit.
};

/// @brief Checked rectangular capacities and finite sentinel for Lowrance–Wagner.
/// rows=n+2, columns=m+2, cells=rows*columns, interior_cells=n*m, bound=n+m.
/// Empty inputs still have the sentinel/prefix borders. Values are capacities,
/// not operation counts or allocated bytes; no references or shared state.
/// @par Example
/// @code
/// const auto p=dm::algorithms::damerau_shape(2,3);
/// std::print("{} {} {} {}\n",p.rows,p.columns,p.cells,p.bound); // 4 5 20 5
/// @endcode
/// @see damerau_shape(), DamerauTrace
struct DamerauShape {
  std::size_t rows{0}; ///< Source length plus two sentinel/prefix rows.
  std::size_t columns{0}; ///< Target length plus two sentinel/prefix columns.
  std::size_t cells{0}; ///< Complete flat sentinel-table capacity.
  std::size_t interior_cells{0}; ///< Source length times target length.
  std::size_t bound{0}; ///< Checked n+m; infinity marker and candidate clipping bound.
};

/// @brief Actual checked operations of the shared unrestricted-distance kernel.
/// Byte matches count all equal-byte cells, not matches in one recovered script.
/// Eligible bridges have both earlier occurrence indices positive. Selected
/// bridges count winning DP cells, not necessarily edits on the final traceback.
/// Counts exclude initialization, allocation, input copying and script recovery.
/// @par Example
/// @code
/// const auto t=dm::algorithms::damerau_levenshtein_trace("CA","ABC");
/// std::print("{} {}\n",t.work.cells_evaluated,t.work.last_row_updates); // 6 2
/// @endcode
/// @see DamerauTrace
struct DamerauWork {
  std::size_t cells_evaluated{0}; ///< Actual interior cell updates, exactly n*m.
  std::size_t byte_comparisons{0}; ///< Actual equality tests, one per interior cell.
  std::size_t byte_matches{0}; ///< True equality tests that update the current row's last column.
  std::size_t last_row_updates{0}; ///< Completed source rows recorded in the 256-byte alphabet table.
  std::size_t eligible_transpositions{0}; ///< Cells with both previous cross-match occurrences available.
  std::size_t selected_transpositions{0}; ///< Cells choosing a bridge strictly below earlier candidates.
};

/// @brief All actual candidates and the selected move at one positive-prefix cell.
/// Prefixes i,j are one-based lengths. last_source is the previous source row
/// holding the current target byte; last_target is the earlier column of this
/// row matching its source byte, captured before the current equality update.
/// Candidate costs are clipped to bound=n+m; an unavailable bridge has bound.
/// This cannot change an interior minimum, which is at most max(i,j)<n+m.
/// @par Example
/// @code
/// const auto t=dm::algorithms::damerau_levenshtein_trace("CA","ABC");
/// const auto& c=t.cells.back();
/// std::print("{} {} {}\n",c.last_source,c.last_target,c.transposition); // 1 1 2
/// @endcode
/// @see DamerauTrace
struct DamerauCell {
  std::size_t source_prefix{0}; ///< Positive source prefix length i.
  std::size_t target_prefix{0}; ///< Positive target prefix length j.
  std::size_t last_source{0}; ///< Earlier cross-matching source position, zero if absent.
  std::size_t last_target{0}; ///< Earlier cross-matching target position, zero if absent.
  std::size_t diagonal{0}; ///< Match/substitution candidate.
  std::size_t insertion{0}; ///< Insertion candidate.
  std::size_t deletion{0}; ///< Deletion candidate.
  std::size_t transposition{0}; ///< Clipped delete/swap/insert bridge candidate, or bound if unavailable.
  std::size_t value{0}; ///< Actual stored minimum.
  DamerauMove choice{DamerauMove::match}; ///< First minimum in the documented deterministic tie order.
};

/// @brief Owned inputs, full sentinel table and actual Lowrance–Wagner cell records.
/// table is row-major with shape.columns columns. Ordinary prefix distance
/// D(i,j) is table[(i+1)*columns+j+1]. cells is row-major over positive prefixes,
/// at index (i-1)*target.size()+j-1. Every cell is retained; no replay or snapshots.
/// Strings are literal bytes, including NUL/high bytes, without Unicode decoding,
/// normalization or case folding. Public edits are possible; helper checks are
/// local consistency checks, not a certificate of edited-table optimality.
/// @par Example
/// @code
/// const auto t=dm::algorithms::damerau_levenshtein_trace("CA","ABC");
/// std::print("{} {} {}\n",t.distance,t.table.size(),t.cells.size()); // 2 20 6
/// @endcode
/// @see damerau_levenshtein_trace(), damerau_prefix_distance(), damerau_edit_script()
struct DamerauTrace {
  std::string source; ///< Owned original source bytes.
  std::string target; ///< Owned original target bytes.
  DamerauShape shape; ///< Checked table capacities and clipping bound.
  std::size_t distance{0}; ///< Unrestricted unit-cost distance of the complete inputs.
  std::vector<std::size_t> table; ///< Complete actual sentinel table, row-major.
  std::vector<DamerauCell> cells; ///< Actual candidates and choices for all n*m positive-prefix cells.
  DamerauWork work; ///< Actual checked kernel operations.
};

/// @brief One checked edit at a position in the current evolving byte string.
/// before/after own literal payloads: insert 0/1 bytes, erase 1/0, substitute
/// 1/1 different bytes, transpose 2/2 reversed bytes, match 1/1 equal bytes.
/// Native shortest scripts omit matches, so their vector length equals cost.
/// No references into the trace or source survive. Positions are current-state
/// offsets, not original source offsets.
/// @par Example
/// @code
/// const auto e=dm::algorithms::damerau_edit_script(dm::algorithms::damerau_levenshtein_trace("CA","ABC"));
/// std::print("{} {} {}\n",e.size(),e[0].position,e[1].position); // 2 0 1
/// @endcode
/// @see damerau_edit_script(), apply_byte_edits()
struct ByteEdit {
  DamerauMove kind{DamerauMove::match}; ///< Literal single-byte operation; transpose swaps two adjacent bytes.
  std::size_t position{0}; ///< Offset in the current string when this edit executes.
  std::string before; ///< Exact expected bytes to replace, empty for insertion.
  std::string after; ///< Exact replacement bytes, empty for deletion.
};

/// @brief Compute checked Lowrance–Wagner table dimensions without allocating.
/// O(1) time/storage, no caller mutation or shared state. All five counts must
/// fit size_t, including border dimensions and the finite n+m marker.
/// @param source_bytes Source length n, including zero.
/// @param target_bytes Target length m, including zero.
/// @return Exact rectangular capacities and finite sentinel.
/// @throws std::length_error A dimension, product or sentinel cannot fit size_t.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::damerau_shape(0,3).cells); // 10
/// @endcode
/// @see DamerauShape, damerau_levenshtein_trace()
DamerauShape damerau_shape(std::size_t source_bytes,std::size_t target_bytes);

/// @brief Compute unrestricted unit-cost Damerau–Levenshtein distance between byte strings.
/// Retains the original signature and Lowrance–Wagner semantics: insertion,
/// deletion, substitution and adjacent transposition cost one, equality zero;
/// characters/substrings can participate in further edits. Empty inputs give
/// the other byte length. The result is a metric on literal byte strings and
/// does not exceed Levenshtein distance. Uses the same executing kernel as the
/// trace, a full checked (n+2)*(m+2) table and 256 last-row entries. Time/storage
/// O((n+1)*(m+1)+256), including borders/empty cases; no optional size cap,
/// caller mutation, retained references, decoding, normalization or shared state.
/// Candidate arithmetic clips safely at checked n+m, without changing minima.
/// @param a Read-only source bytes, including embedded NULs.
/// @param b Read-only target bytes, including embedded NULs.
/// @return Exact nonnegative unit-cost byte distance, between |n-m| and max(n,m).
/// @throws std::length_error Checked dimensions/sentinel or container capacity exceeded.
/// @throws std::bad_alloc Distance-table allocation failure.
/// @par Example
/// @code
/// std::print("{} {}\n",dm::algorithms::damerau_levenshtein_distance("ab","ba"),dm::algorithms::damerau_levenshtein_distance("CA","ABC")); // 1 2
/// @endcode
/// @par References
/// - R. Lowrance and R. A. Wagner (1975), An Extension of the String-to-String Correction Problem, JACM 22(2), 177–183, doi:10.1145/321879.321880; unit costs satisfy its cost conditions.
/// - F. J. Damerau (1964), A Technique for Computer Detection and Correction of Spelling Errors, CACM 7(3), 171–176, doi:10.1145/363958.363994.
/// @see damerau_levenshtein_trace(), optimal_string_alignment_distance()
std::size_t damerau_levenshtein_distance(const std::string& a,const std::string& b);

/// @brief Capture every actual unrestricted-distance cell and first-minimum choice without replay.
/// Same byte/unit-cost semantics as damerau_levenshtein_distance(). The complete
/// border-inclusive cell cap is checked before input copying, table/record
/// allocation or evaluation; zero disables only this optional cap. Empty cases
/// still need their full border table and can exceed the cap. At most one cell
/// record per n*m update, with checked counters and no array snapshots. Time
/// and owned storage O((n+1)*(m+1)+256). Results never silently truncate.
/// @param a Read-only source bytes.
/// @param b Read-only target bytes.
/// @param max_cells Full (n+2)*(m+2) table cap, default 1000000; zero disables.
/// @return Owned full inputs, table, actual candidates/choices and work.
/// @throws std::length_error Geometry, complete cap or container capacity exceeded.
/// @throws std::overflow_error Actual operation counter cannot fit size_t.
/// @throws std::bad_alloc Input/table/record allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::damerau_levenshtein_trace("CA","ABC",20);
/// std::print("{} {}\n",t.distance,t.work.cells_evaluated); // 2 6
/// @endcode
/// @par References
/// - R. Lowrance and R. A. Wagner (1975), doi:10.1145/321879.321880.
/// @see DamerauTrace, damerau_edit_script()
DamerauTrace damerau_levenshtein_trace(const std::string& a,const std::string& b,std::size_t max_cells=1000000);

/// @brief Read one ordinary prefix distance from a checked trace table in O(1).
/// Validates prefix bounds and input-derived flat table size, not optimality or
/// every public record. No allocation, mutation, borrowed return or shared state.
/// @param trace Read-only owned trace.
/// @param source_prefix Prefix length in [0,source.size()].
/// @param target_prefix Prefix length in [0,target.size()].
/// @return Stored ordinary D(i,j), excluding the sentinel offset.
/// @throws std::out_of_range A prefix exceeds its input byte length.
/// @throws std::invalid_argument Flat table size is malformed.
/// @throws std::length_error Input-derived geometry cannot fit size_t.
/// @par Example
/// @code
/// const auto t=dm::algorithms::damerau_levenshtein_trace("CA","ABC");
/// std::print("{}\n",dm::algorithms::damerau_prefix_distance(t,2,3)); // 2
/// @endcode
/// @see DamerauTrace
std::size_t damerau_prefix_distance(const DamerauTrace& trace,std::size_t source_prefix,std::size_t target_prefix);

/// @brief Expand the stored native optimal traceback into literal executable unit edits.
/// Diagonal/insert/erase moves give direct operations. A transposition bridge
/// deletes source-interior bytes, swaps its now-adjacent endpoints and inserts
/// target-interior bytes between them. This yields current-state positions and
/// exactly distance instructions; matches are omitted. Checks input-derived
/// table/record shapes, endpoint distance and visited local coordinates, chosen
/// recurrences and crossed bytes. Does not certify arbitrary edited tables as
/// globally minimal. The distance cap is checked after basic shape/endpoint
/// validation and before traceback/output allocation; zero disables it. Local
/// validation may allocate O(n+m) temporary blocks. Native time/storage is
/// O(n+m+distance), without rerunning the DP; outputs own their byte payloads.
/// @param trace Read-only complete native or locally consistent public trace.
/// @param max_edits Complete instruction cap, default 100000; zero disables.
/// @return Owned source-to-target script in execution order, empty for equal strings.
/// @throws std::invalid_argument Malformed table/record or visited witness recurrence.
/// @throws std::length_error Geometry, complete distance cap or container capacity exceeded.
/// @throws std::bad_alloc Traceback/instruction allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::damerau_levenshtein_trace("CA","ABC");
/// const auto edits=dm::algorithms::damerau_edit_script(t,2);
/// std::print("{}\n",dm::algorithms::apply_byte_edits(t.source,edits)); // ABC
/// @endcode
/// @see ByteEdit, apply_byte_edits()
std::vector<ByteEdit> damerau_edit_script(const DamerauTrace& trace,std::size_t max_edits=100000);

/// @brief Replay literal byte edits against an owned evolving string, checking each operation.
/// Validates each kind's payload lengths/content, current position and exact
/// expected before bytes before that operation. Native scripts omit matches;
/// explicit equal one-byte match records are accepted as checked no-ops. A
/// failure discards this call's local state, leaving an lvalue caller unchanged.
/// No decoding/normalization or retained references. With k edits and maximum
/// evolving length L, std::string replacement costs O(k*L+n) worst-case time,
/// O(L) output storage; payload validation is constant per valid unit operation.
/// @param source By-value original bytes, including NULs.
/// @param edits Read-only current-state instructions, applied in supplied order.
/// @return Owned final string; empty edit list returns the input unchanged.
/// @throws std::out_of_range Current position or removed span lies outside the string.
/// @throws std::invalid_argument Invalid enum, payload shape, swap/replacement or expected bytes.
/// @throws std::length_error Evolving string exceeds container capacity.
/// @throws std::bad_alloc Input/output replacement allocation failure.
/// @par Example
/// @code
/// const dm::algorithms::ByteEdit e{dm::algorithms::DamerauMove::transpose,0,"CA","AC"};
/// std::print("{}\n",dm::algorithms::apply_byte_edits("CA",{e})); // AC
/// @endcode
/// @see damerau_edit_script(), ByteEdit
std::string apply_byte_edits(std::string source,const std::vector<ByteEdit>& edits);

/// @brief Compute the restricted optimal-string-alignment (OSA) byte distance.
/// Unit Levenshtein moves plus a local crossed-byte D(i-2,j-2)+1 candidate.
/// Unlike unrestricted distance this cannot reuse an aligned substring in a
/// later edit; CA->ABC costs 3 and triangle inequality can fail. Symmetric,
/// byte-exact and nonnegative, but not a metric. Three rolling rows give
/// O((n+1)*(m+1)) time and O(min(n,m)+1) storage, including empty inputs. No
/// size cap, caller mutation, retained references or shared state.
/// @param a Read-only source bytes, including NULs.
/// @param b Read-only target bytes, including NULs.
/// @return Restricted OSA distance, at least unrestricted DL and at most Levenshtein.
/// @throws std::length_error Checked row dimension/bound or container capacity exceeded.
/// @throws std::bad_alloc Rolling-row allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::optimal_string_alignment_distance("CA","ABC")); // 3
/// @endcode
/// @par References
/// - R. Lowrance and R. A. Wagner (1975), doi:10.1145/321879.321880, for the unrestricted comparison; OSA here is explicitly the local two-prefix transposition restriction.
/// @see damerau_levenshtein_distance()
std::size_t optimal_string_alignment_distance(const std::string& a,const std::string& b);
} // namespace datamunge::algorithms
