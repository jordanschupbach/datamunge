#pragma once
/// @file needleman_wunsch.hpp
/// @brief Checked global byte alignment with actual score-table and traceback observations.
#include <cstddef>
#include <datamunge/algorithms/alignment_types.hpp>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual winning recurrence/traceback move under diagonal/up/left tie priority.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::needleman_wunsch_move_name(dm::algorithms::NeedlemanWunschMove::Diagonal)); // diagonal
/// @endcode
/// @see needleman_wunsch_move_name()
enum class NeedlemanWunschMove {
  Diagonal, ///< Consume one byte from both inputs.
  Up,       ///< Consume first-input byte, inserting a gap in second output.
  Left      ///< Consume second-input byte, inserting a gap in first output.
};
/// @brief Stable display name for a global-alignment move.
/// @param move Actual move enum, unknown cast values allowed.
/// @return diagonal, up, left or unknown.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::needleman_wunsch_move_name(dm::algorithms::NeedlemanWunschMove::Up)); // up
/// @endcode
/// @see NeedlemanWunschMove
std::string_view needleman_wunsch_move_name(NeedlemanWunschMove move) noexcept;
/// @brief Complete score-table and actual capture policies, zero disabling either.
/// max_cells includes both boundary row and column before table allocation.
/// Ordinary needleman_wunsch() disables optional policies, retaining its entry point.
/// @par Example
/// @code
/// dm::algorithms::NeedlemanWunschOptions o;std::print("{}\n",o.max_cells); // 1000000
/// @endcode
/// @see needleman_wunsch_trace()
struct NeedlemanWunschOptions {
  std::size_t max_cells{1000000}; ///< Complete(n+1)*(m+1) score-value cap.
  std::size_t max_trace_entries{
      1000000}; ///< Complete base plus actual traceback/output capture cap.
};
/// @brief Actual global optimum, full native score table and selected work.
/// Default scoring2,−1,−2 uses linear gaps. Positive gap scores are allowed
/// and can prefer opposing inserted gaps. No affine-gap state or biological
/// probability model is implied. Rows index first-input prefixes, columns second.
/// @par Example
/// @code
/// auto r=dm::algorithms::needleman_wunsch_run("AAAC","AGC",{1,-1,-1});
/// std::print("{} {}\n",r.alignment.score,r.cells); // 0 20
/// @endcode
/// @see needleman_wunsch_run()
struct NeedlemanWunschRun {
  Alignment alignment; ///< Actual score and diagonal-first optimal aligned rows.
  std::vector<std::vector<int>>
              scores;   ///< Actual(n+1)-by-(m+1) table, including fully penalized borders.
  std::size_t cells{0}; ///< Complete allocated score-value count.
  std::size_t boundary_assignments{
      0};                         ///< Actual border writes, n+m+2 because origin is assigned twice.
  std::size_t interior_cells{0};  ///< Actual recurrence cells, n*m.
  std::size_t traceback_steps{0}; ///< Actual emitted alignment columns.
  std::size_t diagonal_steps{0};  ///< Actual traceback diagonal moves.
  std::size_t up_steps{0};        ///< Actual traceback first-input-only moves.
  std::size_t left_steps{0};      ///< Actual traceback second-input-only moves.
};
/// @brief Actual candidates, selected score and first-max direction of an interior cell.
/// Fill order is increasing first prefix then increasing second prefix. Ties
/// choose diagonal, then up, then left. The ordinary producer makes this same
/// first-maximum selection; no report-side score table is reconstructed.
/// @par Example
/// @code
/// auto t=dm::algorithms::needleman_wunsch_trace("AA","A",{1,-1,-1});
/// std::print("{} {}\n",t.cells.back().diagonal,t.cells.back().up); // 0 0
/// @endcode
/// @see NeedlemanWunschTrace
struct NeedlemanWunschCell {
  std::size_t         i{0};        ///< Actual first-prefix length, at least1.
  std::size_t         j{0};        ///< Actual second-prefix length, at least1.
  int                 diagonal{0}; ///< Actual previous-diagonal score plus byte-pair score.
  int                 up{0};       ///< Actual preceding-row score plus gap.
  int                 left{0};     ///< Actual preceding-column score plus gap.
  int                 score{0};    ///< Actual committed first maximum.
  NeedlemanWunschMove chosen{
      NeedlemanWunschMove::Diagonal}; ///< Actual first-max recurrence direction.
};
/// @brief Actual reverse-order traceback emission and score-table coordinates.
/// Records run from bottom-right to origin; final aligned rows reverse these
/// emitted bytes. Literal input hyphens are rejected so inserted gaps are unambiguous.
/// @par Example
/// @code
/// auto t=dm::algorithms::needleman_wunsch_trace("A","A");
/// std::print("{}\n",t.traceback.front().a); // A
/// @endcode
/// @see NeedlemanWunschTrace
struct NeedlemanWunschTraceback {
  std::size_t         i_before{0}; ///< Actual first-prefix coordinate before this move.
  std::size_t         j_before{0}; ///< Actual second-prefix coordinate before this move.
  std::size_t         i_after{0};  ///< Actual first-prefix coordinate after this move.
  std::size_t         j_after{0};  ///< Actual second-prefix coordinate after this move.
  NeedlemanWunschMove move{NeedlemanWunschMove::Diagonal}; ///< Actual selected traceback branch.
  char                a{0}; ///< Actual first output byte appended before final reversal.
  char                b{0}; ///< Actual second output byte appended before final reversal.
};
/// @brief Owned original byte inputs, actual table fills and actual traceback.
/// Capture units count3 scoring scalars, both input strings, row records, score
/// values, interior-cell records, traceback records and both aligned outputs.
/// For alignment length L:3+n+m+(n+1)+(n+1)*(m+1)+n*m+3L.
/// Table and fill counts are pre-admitted; actual traceback/output units follow
/// online. No replay, truncation or partial returned trace.
/// @par Example
/// @code
/// auto t=dm::algorithms::needleman_wunsch_trace("A","A");
/// std::print("{}\n",t.captured_entries); // 15
/// @endcode
/// @see needleman_wunsch_trace()
struct NeedlemanWunschTrace {
  std::string                           a;       ///< Owned original first byte sequence.
  std::string                           b;       ///< Owned original second byte sequence.
  AlignmentScoring                      scoring; ///< Actual copied linear scoring components.
  NeedlemanWunschRun                    run;   ///< Actual optimum, native table and selected work.
  std::vector<NeedlemanWunschCell>      cells; ///< Actual interior recurrence records.
  std::vector<NeedlemanWunschTraceback> traceback; ///< Actual bottom-right-to-origin emissions.
  std::size_t captured_entries{0};                 ///< Actual admitted logical history units.
};
/// @brief Compute an optimal end-to-end byte alignment with fully penalized gaps.
/// H(i,j)=max(diagonal+pair,up+gap,left+gap), borders i*gap and j*gap.
/// Actual traceback prefers diagonal, then up, then left. Accepts empty/NUL/high
/// bytes; reserves literal '-' for inserted gaps. No Unicode normalization or
/// affine gap penalty. Conservative complete score admission requires
/// (n+m)*max(abs(scoring))<=INT_MAX, excluding unrepresentable signed arithmetic.
/// O((n+1)*(m+1)) work/storage, no input mutation or retained references.
/// @param a First original bytes, excluding literal '-'.
/// @param b Second original bytes, excluding literal '-'.
/// @param scoring Linear integer scores, default2,−1,−2; positive gaps allowed.
/// @return Owned actual optimal score and aligned rows in original caller order.
/// @throws std::invalid_argument If either input contains reserved '-'.
/// @throws std::overflow_error If conservative complete int score geometry is exceeded.
/// @throws std::length_error If complete input/table/output geometry is unrepresentable.
/// @par Example
/// @code
/// auto r=dm::algorithms::needleman_wunsch("AAAC","AGC",{1,-1,-1});
/// std::print("{} {} {}\n",r.score,r.a_aligned,r.b_aligned); // 0 AAAC -AGC
/// @endcode
/// @par References
/// - S. B. Needleman and C. D. Wunsch (1970), A general method applicable to the search for similarities in the amino acid sequence of two proteins, Journal of Molecular Biology48,443–453, doi:10.1016/0022-2836(70)90057-4.
/// @see needleman_wunsch_run(), needleman_wunsch_trace(), hirschberg()
Alignment
needleman_wunsch(const std::string& a, const std::string& b, AlignmentScoring scoring = {});
/// @brief Return the actual full score table and selected work of ordinary alignment.
/// Same producer/tie policy as needleman_wunsch(), no replay. Complete cell cap
/// includes boundaries and is checked before table allocation; capture cap ignored.
/// @param a First original bytes, without '-'.
/// @param b Second original bytes, without '-'.
/// @param scoring Linear integer scores, default2,−1,−2.
/// @param options Complete table cap, default1000000 cells, zero disables.
/// @return Owned actual alignment, score table and selected work.
/// @throws std::invalid_argument If a reserved hyphen occurs.
/// @throws std::overflow_error If score geometry exceeds safe int bounds.
/// @throws std::length_error If geometry or complete cell cap is exceeded.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// auto r=dm::algorithms::needleman_wunsch_run("A","A");std::print("{}\n",r.scores[1][1]); // 2
/// @endcode
/// @see NeedlemanWunschRun
NeedlemanWunschRun needleman_wunsch_run(
    const std::string&     a,
    const std::string&     b,
    AlignmentScoring       scoring = {},
    NeedlemanWunschOptions options = {});
/// @brief Capture actual recurrence decisions and emitted traceback columns.
/// Complete input/score/cell validation precedes capture/table allocation. Base
/// history geometry is pre-admitted, then three units per actual alignment column
/// (one record, two output bytes). Optional policies use zero to disable; no partial
/// result on failure. Extra trace storage follows admitted units.
/// @param a Complete first bytes, retained in original order.
/// @param b Complete second bytes, retained in original order.
/// @param scoring Linear integer scoring, default2,−1,−2.
/// @param options Complete cells/actual history caps, defaults1000000 each.
/// @return Owned original strings, actual table/fill decisions, traceback and result.
/// @throws std::invalid_argument If a reserved hyphen occurs before policy checks.
/// @throws std::overflow_error If conservative score bounds are exceeded.
/// @throws std::length_error If geometry or cell/history quota exceeds limits.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// auto t=dm::algorithms::needleman_wunsch_trace("A","");std::print("{}\n",t.run.alignment.b_aligned); // -
/// @endcode
/// @see NeedlemanWunschTrace
NeedlemanWunschTrace needleman_wunsch_trace(
    const std::string&     a,
    const std::string&     b,
    AlignmentScoring       scoring = {},
    NeedlemanWunschOptions options = {});
/// @brief Return the original three additional default-scored sequence pairs.
/// @return Owned pairs GATTACA/GCATGCU, HELLO/HELP, ALIGN/ALIGNMENT, in that order.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::needleman_wunsch_demo_pairs().size()); // 3
/// @endcode
/// @see needleman_wunsch()
std::vector<std::pair<std::string, std::string>> needleman_wunsch_demo_pairs();
} // namespace datamunge::algorithms
