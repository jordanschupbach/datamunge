#pragma once
/// @file lis.hpp
/// @brief Strict integer LIS with actual minimal-tail search/frontier evidence and original-index predecessor reconstruction.
#include <cstddef>
#include <limits>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual selected strict-LIS binary search, tail updates and witness reconstruction work.
/// Excludes guards/copies/allocation/capture, elapsed time and CPU instructions.
/// @par Example
/// @code
/// const auto a=dm::algorithms::lis_analyze({2,3,1});
/// std::print("{} {}\n",a.work.appends,a.work.replacements); // 2 1
/// @endcode
/// @par References
/// - Fredman(1975), On computing the length of longest increasing subsequences, Discrete Mathematics11(1),29–35, https://doi.org/10.1016/0012-365X(75)90103-X; minimal-tail binary search with explicit predecessor reconstruction here.
/// @see lis_analyze(), lis_trace()
struct LISWork {
  std::size_t comparisons{0}; ///< Actual frontier-value<incoming-value comparisons.
  std::size_t appends{0};     ///< Actual new minimal-tail length classes appended.
  std::size_t replacements{
      0}; ///< Actual existing tail classes overwritten, including equal-value ties.
  std::size_t backtrack{
      0}; ///< Actual selected source indices visited while reconstructing the final witness.
};
/// @brief Owned actual deterministic strict-LIS values, source indices and producing work.
/// @par Example
/// @code
/// const auto a=dm::algorithms::lis_analyze({2,3,1});
/// std::print("{} {}\n",a.indices[0],a.indices[1]); // 0 1
/// @endcode
/// @par References
/// - Fredman(1975), On computing the length of longest increasing subsequences, Discrete Mathematics11(1),29–35, https://doi.org/10.1016/0012-365X(75)90103-X; minimal-tail binary search with explicit predecessor reconstruction here.
/// @see lis_analyze(), lis_trace()
struct LISAnalysis {
  std::vector<int> sequence; ///< Actual selected optimal values in original input order.
  std::vector<std::size_t>
          indices; ///< Actual original indices corresponding one-to-one to sequence.
  LISWork work;    ///< Actual selected search/update/backtrack operations.
};
/// @brief One actual lower-bound comparison against the real current frontier.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lis_trace({2,3,1});
/// std::print("{}\n",t.probes[0].less); // true
/// @endcode
/// @par References
/// - Fredman(1975), On computing the length of longest increasing subsequences, Discrete Mathematics11(1),29–35, https://doi.org/10.1016/0012-365X(75)90103-X; minimal-tail binary search with explicit predecessor reconstruction here.
/// @see lis_analyze(), lis_trace()
struct LISProbe {
  std::size_t input_index{0}; ///< Actual original input identity being inserted.
  std::size_t low{0};         ///< Actual inclusive search bound before this probe.
  std::size_t high{0};        ///< Actual exclusive search bound before this probe.
  std::size_t middle{0};      ///< Actual current frontier class tested.
  std::size_t tail_index{0};  ///< Actual original identity stored in that tail class.
  int         tail_value{0};  ///< Actual original value at tail_index.
  bool        less{false};    ///< Actual tail_value<input[input_index] result.
};
/// @brief Owned real frontier indices before/after one original value and its actual predecessor link.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lis_trace({2,3,1});
/// std::print("{}\n",t.insertions.back().after[0]); // 2
/// @endcode
/// @par References
/// - Fredman(1975), On computing the length of longest increasing subsequences, Discrete Mathematics11(1),29–35, https://doi.org/10.1016/0012-365X(75)90103-X; minimal-tail binary search with explicit predecessor reconstruction here.
/// @see lis_analyze(), lis_trace()
struct LISInsertion {
  std::size_t input_index{0}; ///< Actual original value identity.
  std::size_t length_class{
      0}; ///< Actual first frontier class with value>=incoming value, or append class.
  std::size_t predecessor{
      std::numeric_limits<std::size_t>::
          max()}; ///< Actual preceding witness index, or SIZE_MAX sentinel when no predecessor.
  std::vector<std::size_t> before; ///< Actual original tail identities before replacing/appending.
  std::vector<std::size_t> after;  ///< Actual original tail identities after the committed update.
};
/// @brief Owned original integers, final actual predecessor/frontier arrays and every producing search/update record.
/// Input/predecessor2*n records admitted before copies; each actual probe and
/// insertion record plus before/after frontier members admitted before append,
/// final-tail records before ownership transfer. Complete result or exception.
/// Default million actual records,0 optional cap off; excludes scalar fields,
/// capacities/output/runtime copies and bytes/RAM. Frontier snapshots can take
/// quadratic capture storage despite O(n log n) ordinary producing work.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lis_trace({2,3,1});
/// std::print("{} {}\n",t.final_tails[0],t.final_tails[1]); // 2 1
/// @endcode
/// @par References
/// - Fredman(1975), On computing the length of longest increasing subsequences, Discrete Mathematics11(1),29–35, https://doi.org/10.1016/0012-365X(75)90103-X; minimal-tail binary search with explicit predecessor reconstruction here.
/// @see lis_analyze(), lis_trace()
struct LISTrace {
  std::vector<int> input; ///< Owned complete original arbitrary int values.
  std::vector<std::size_t>
      predecessor; ///< Actual original-index links, SIZE_MAX for absent parent.
  std::vector<std::size_t>
      final_tails; ///< Actual final minimal-tail original indices; generally not themselves a valid subsequence.
  std::vector<LISProbe> probes; ///< Actual chronological lower-bound comparisons.
  std::vector<LISInsertion>
      insertions; ///< Actual real frontier-before/after snapshots and chosen parent for each original element.
  LISAnalysis analysis; ///< Actual optimal witness/source indices and producing work.
  std::size_t captured_entries{
      0}; ///< Actual input/predecessor/final-tail/probe/insertion/frontier-member logical records.
};
/// @brief Return one strictly increasing longest integer subsequence with historical deterministic ties.
/// Minimal-tail frontier uses lower_bound on strict<: equal incoming values
/// replace the existing class and do not extend length. Each new parent points
/// to the preceding class's actual earlier identity. Reconstruct from the last
/// tail through those real links. Frontier values are not a witness by themselves.
/// O(n log(n+1)) time/O(n) runtime, no mutation or shared state; no claim of a
/// lexicographically smallest, earliest-index or unique optimal witness.
/// @param input Borrowed complete original arbitrary integer values; empty and duplicates valid.
/// @return Owned actual optimal values in original order, empty for empty input.
/// @throws std::length_error On complete vector/work geometry or selected actual count overflow.
/// @throws std::bad_alloc On runtime/output allocation failure.
/// @par Example
/// @code
/// const auto s=dm::algorithms::longest_increasing_subsequence({2,3,1});
/// std::print("{} {}\n",s[0],s[1]); // 2 3
/// @endcode
/// @par References
/// - Fredman(1975), On computing the length of longest increasing subsequences, Discrete Mathematics11(1),29–35, https://doi.org/10.1016/0012-365X(75)90103-X; minimal-tail binary search with explicit predecessor reconstruction here.
/// @see LISAnalysis, LISTrace
std::vector<int> longest_increasing_subsequence(const std::vector<int>& input);
/// @brief Return the actual optimal strict-LIS witness, original identities and selected producing work.
/// Same tie, strictness, reconstruction and complete input contracts as the
/// legacy values-only API; no optional frontier/probe history or replay.
/// @param input Borrowed complete original arbitrary integer values; empty and duplicates valid.
/// @return Owned actual values/source indices and real selected work.
/// @throws std::length_error On complete vector/work geometry or selected actual count overflow.
/// @throws std::bad_alloc On runtime/output allocation failure.
/// @par Example
/// @code
/// const auto a=dm::algorithms::lis_analyze({1,1});
/// std::print("{} {}\n",a.sequence.size(),a.indices[0]); // 1 1
/// @endcode
/// @par References
/// - Fredman(1975), On computing the length of longest increasing subsequences, Discrete Mathematics11(1),29–35, https://doi.org/10.1016/0012-365X(75)90103-X; minimal-tail binary search with explicit predecessor reconstruction here.
/// @see LISAnalysis, LISTrace
LISAnalysis lis_analyze(const std::vector<int>& input);
/// @brief Capture every real minimal-tail comparison/frontier update and actual predecessor state once.
/// Shared producer, no report-side search or index recovery. Actual record
/// quota is checked before copies/appends; failures return no partial result.
/// @param input Borrowed complete original arbitrary integers.
/// @param max_entries Actual logical record cap, default1000000;0 disables only optional cap.
/// @return Owned complete original, actual frontier/link/probe/update evidence and selected optimal witness.
/// @throws std::length_error On complete vector/work geometry or selected actual count overflow.
/// @throws std::bad_alloc On runtime/output allocation failure.
/// @throws std::length_error On actual logical capture count/quota overflow.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lis_trace({2,3,1});
/// std::print("{}\n",t.analysis.sequence.size()); // 2
/// @endcode
/// @par References
/// - Fredman(1975), On computing the length of longest increasing subsequences, Discrete Mathematics11(1),29–35, https://doi.org/10.1016/0012-365X(75)90103-X; minimal-tail binary search with explicit predecessor reconstruction here.
/// @see LISAnalysis, LISTrace
LISTrace lis_trace(const std::vector<int>& input, std::size_t max_entries = 1000000);
} // namespace datamunge::algorithms
