#pragma once
/// @file smith_waterman.hpp
/// @brief Checked optimal local byte alignment with actual DP and traceback.
#include <cstddef>
#include <datamunge/algorithms/alignment_types.hpp>
#include <string>
#include <vector>
/// @brief Canonical checked native producers and owned values.
namespace datamunge::algorithms {
/// @brief SmithWatermanOptions owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SmithWatermanOptions value{};
/// std::print("{}\n",value.max_cells); // 1000000
/// @endcode
struct SmithWatermanOptions {
  std::size_t max_cells{1000000};       ///< Complete bordered DP geometry; zero disables.
  std::size_t max_trace_steps{1000000}; ///< Captured actual traceback records; zero disables.
};
/// @brief SmithWatermanStep owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SmithWatermanStep value{};
/// std::print("{}\n",value.score); // 0
/// @endcode
struct SmithWatermanStep {
  std::size_t row{};    ///< Actual first-string prefix length before traceback move.
  std::size_t column{}; ///< Actual second-string prefix length before traceback move.
  int         score{};  ///< Actual DP value at this state.
  char        move{};   ///< D=diagonal, U=first byte against gap, L=gap against second byte.
};
/// @brief SmithWatermanRun owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SmithWatermanRun value{};
/// std::print("{}\n",value.evaluated_cells); // 0
/// @endcode
struct SmithWatermanRun {
  Alignment   alignment;         ///< Actual optimal local score and selected gapped substring rows.
  std::size_t begin_a{};         ///< Inclusive actual first input substring start.
  std::size_t end_a{};           ///< Exclusive actual first input substring end.
  std::size_t begin_b{};         ///< Inclusive actual second input substring start.
  std::size_t end_b{};           ///< Exclusive actual second input substring end.
  std::size_t evaluated_cells{}; ///< Actual interior recurrence evaluations.
  std::size_t traceback_steps{}; ///< Actual selected backtracking moves.
  std::vector<std::vector<int>>
      table; ///< Optional actual completed producing DP table, including zero border.
  std::vector<SmithWatermanStep>
      traceback; ///< Optional actual maximum-to-zero backward path, no replay.
};
/// @brief Find an optimal local linear-gap byte alignment.
/// H(i,j)=max(0,H(i-1,j-1)+pair_score,H(i-1,j)+gap,H(i,j-1)+gap). Gap must be nonpositive; match/mismatch are arbitrary int values. Empty inputs or no positive cells give score0, empty rows and zero spans. Strict maximum updates retain the first row-major maximum; traceback ties choose diagonal, then up, then left and stop at zero. No affine gaps or Unicode tokenization. Literal dash input bytes are allowed but ambiguous in gapped strings; coordinates/moves disambiguate. Complete bordered geometry checked before allocation; int64 candidates prevent int overflow and unrepresentable selected scores throw. Time/storage O(|a|*|b|), optional retained table uses the actual working table, not a rebuilt one. Borrowed inputs, owned outputs; no shared state.
/// @param a Complete first raw byte string.
/// @param b Complete second raw byte string.
/// @param scoring Integer match/mismatch/nonpositive gap, default2,-1,-2.
/// @param options Complete DP and optional actual traceback policies; zero disables policies.
/// @return Owned actual optimum, substring spans, work and optional producing table/path.
/// @throws std::invalid_argument Positive gap score.
/// @throws std::length_error DP/trace/counter/container capacity exceeded.
/// @throws std::overflow_error Selected optimal DP score exceeds int range.
/// @par Example
/// @code
/// auto r=dm::algorithms::smith_waterman_run("AAGT","AGT");
/// std::print("{} {}\n",r.alignment.score,r.alignment.a_aligned); // 6 AGT
/// @endcode
/// @par References
/// - T. F. Smith and M. S. Waterman, Identification of common molecular subsequences, J. Mol. Biol.147(1),1981, DOI10.1016/0022-2836(81)90087-5.
/// @see smith_waterman()
SmithWatermanRun smith_waterman_run(
    const std::string&          a,
    const std::string&          b,
    AlignmentScoring            scoring = {},
    const SmithWatermanOptions& options = {});
/// @brief Find an optimal local linear-gap byte alignment with actual DP/traceback values.
/// H(i,j)=max(0,H(i-1,j-1)+pair_score,H(i-1,j)+gap,H(i,j-1)+gap). Gap must be nonpositive; match/mismatch are arbitrary int values. Empty inputs or no positive cells give score0, empty rows and zero spans. Strict maximum updates retain the first row-major maximum; traceback ties choose diagonal, then up, then left and stop at zero. No affine gaps or Unicode tokenization. Literal dash input bytes are allowed but ambiguous in gapped strings; coordinates/moves disambiguate. Complete bordered geometry checked before allocation; int64 candidates prevent int overflow and unrepresentable selected scores throw. Time/storage O(|a|*|b|), optional retained table uses the actual working table, not a rebuilt one. Borrowed inputs, owned outputs; no shared state.
/// @param a Complete first raw byte string.
/// @param b Complete second raw byte string.
/// @param scoring Integer match/mismatch/nonpositive gap, default2,-1,-2.
/// @param options Complete DP and optional actual traceback policies; zero disables policies.
/// @return Owned actual optimum, substring spans, work and optional producing table/path.
/// @throws std::invalid_argument Positive gap score.
/// @throws std::length_error DP/trace/counter/container capacity exceeded.
/// @throws std::overflow_error Selected optimal DP score exceeds int range.
/// @par Example
/// @code
/// auto r=dm::algorithms::smith_waterman_trace("AAGT","AGT");
/// std::print("{} {}\n",r.alignment.score,r.alignment.a_aligned); // 6 AGT
/// @endcode
/// @par References
/// - T. F. Smith and M. S. Waterman, Identification of common molecular subsequences, J. Mol. Biol.147(1),1981, DOI10.1016/0022-2836(81)90087-5.
/// @see smith_waterman()
SmithWatermanRun smith_waterman_trace(
    const std::string&          a,
    const std::string&          b,
    AlignmentScoring            scoring = {},
    const SmithWatermanOptions& options = {});
/// @brief Return the preserved checked local alignment result.
/// Same actual native producer and default complete geometry policies; gap must be nonpositive.
/// @param a Complete first byte string.
/// @param b Complete second byte string.
/// @param scoring Default2,-1,-2 integer scores; nonpositive gap.
/// @return Owned selected optimal local score and gapped rows.
/// @throws std::invalid_argument Positive gap.
/// @throws std::length_error Default DP/counter/container bound exceeded.
/// @throws std::overflow_error Score exceeds int range.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::smith_waterman("AAGT","AGT").score); // 6
/// @endcode
/// @see smith_waterman_run()
Alignment smith_waterman(const std::string& a, const std::string& b, AlignmentScoring scoring = {});
} // namespace datamunge::algorithms
