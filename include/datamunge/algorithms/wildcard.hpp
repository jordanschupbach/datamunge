#pragma once

/// @file wildcard.hpp
/// @brief Instrumented restricted-glob matching, prefix-state diagrams and DataFrame filtering.
#include <datamunge/algorithms/substrings.hpp>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/plot/plot.hpp>

namespace datamunge::algorithms {

/// @brief Result and actual work counters from the same last-star loop as wildcard_match().
///
/// steps counts text-loop iterations and trailing-star consumptions, excluding
/// the final accept/reject decision. literal_tests counts byte-equality tests
/// at literal pattern positions, whether successful or not. A failed literal
/// comparison and a star retry can belong to the same step. Counts measure
/// algorithmic work, not elapsed time or comparisons performed by unrelated code.
/// @par Example
/// @code
/// const auto result=dm::algorithms::wildcard_match_stats("abc","a*c");
/// const bool match=result.matched; // true
/// @endcode
/// @see wildcard_match_stats(), wildcard_match_trace()
struct WildcardMatchStats {
  bool matched{false}; ///< Whole-text acceptance result.
  std::size_t steps{0}; ///< Text-loop iterations plus trailing-star transitions.
  std::size_t literal_tests{0}; ///< Actual literal byte-equality comparisons.
  std::size_t question_matches{0}; ///< Consumed text bytes matched by a question mark.
  std::size_t star_steps{0}; ///< Pattern-star visits, including trailing-star transitions.
  std::size_t retries{0}; ///< Expansions of the most recently remembered star.
};

/// @brief Match the entire byte string and count work in the canonical last-star scan.
/// @param text Borrowed bytes, including possible NULs and metacharacter-looking text bytes.
/// @param pattern Borrowed restricted glob; exactly wildcard_match() semantics.
/// @return Owned acceptance flag and counters, without retaining either input.
/// O((n+1)*(m+1)) worst-case time and O(1) storage; no shared state.
/// @par Example
/// @code
/// const auto stats=dm::algorithms::wildcard_match_stats("aaaa","*aab");
/// const bool match=stats.matched; // false
/// @endcode
/// @see wildcard_match(), wildcard_match_trace()
WildcardMatchStats wildcard_match_stats(const std::string& text,const std::string& pattern);

/// @brief Trace actual state transitions of the same greedy last-star matcher.
///
/// Records literal/question matches, empty-star choices, retries, trailing stars
/// and final decisions. Early rejection is itself the terminal transition.
/// Indices are zero-based byte offsets; star_after=-1 means the text loop
/// remembers no star. mark_after is meaningful only when a star is remembered.
/// O((n+1)*(m+1)) matching time, with O(number of recorded events) result storage.
/// Inputs are borrowed and unchanged; calls share no mutable state.
/// @param text Borrowed byte string.
/// @param pattern Restricted byte glob.
/// @param max_events Positive maximum trace rows, default 100000; does not alter match semantics.
/// @return Owned event, action, text_before, pattern_before, text_after, pattern_after,
/// star_after, mark_after and literal_tests columns, the latter a cumulative count.
/// @throws std::invalid_argument If max_events is zero.
/// @throws std::length_error If retaining the complete trace would exceed max_events.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::wildcard_match_trace("xaylmz","x?y*z");
/// const bool nonempty=trace.nrows()>0; // true
/// @endcode
/// @see wildcard_match_stats(), wildcard_dp_table()
dstruct::DataFrame wildcard_match_trace(const std::string& text,const std::string& pattern,
                                         std::size_t max_events=100000);

/// @brief Build the full prefix dynamic program for exactly the same restricted glob language.
///
/// D(i,j) means pattern[0,j) accepts text[0,i). A star combines D(i,j-1)
/// (empty) and D(i-1,j) (consume one); literals/questions use D(i-1,j-1).
/// This independent recurrence is for explanation and validation, not the
/// production last-star decision path. O((n+1)*(m+1)) time/result storage.
/// @param text Borrowed bytes.
/// @param pattern Restricted byte glob.
/// @param max_cells Positive allocation limit including empty prefixes, default 1000000.
/// @return Owned text_prefix, pattern_prefix and reachable (zero/one) rows in text-major order.
/// @throws std::invalid_argument If max_cells is zero.
/// @throws std::length_error If the grid exceeds max_cells or representable dimensions.
/// @par Example
/// @code
/// const auto table=dm::algorithms::wildcard_dp_table("abc","a*c");
/// const auto cells=table.nrows(); // 16
/// @endcode
/// @see wildcard_dp_plot(), wildcard_match()
dstruct::DataFrame wildcard_dp_table(const std::string& text,const std::string& pattern,
                                      std::size_t max_cells=1000000);

/// @brief Plot reachable prefix states and the final accept/reject cell of the restricted glob DP.
///
/// Pattern prefixes run rightwards and text prefixes downwards. Byte labels
/// use visible ASCII or hexadecimal for controls/non-ASCII bytes; they do not
/// interpret UTF-8. The returned plot owns its points and labels. Time/storage
/// follow wildcard_dp_table(); callers choose a readable canvas for the grid.
/// @param text Borrowed byte string.
/// @param pattern Restricted byte glob.
/// @param max_cells Positive grid limit, default 10000.
/// @return Owned green/grey prefix-state plot with a gold/red terminal marker.
/// @throws std::invalid_argument If max_cells is zero.
/// @throws std::length_error If the prefix grid exceeds the limit.
/// @par Example
/// @code
/// auto plot=dm::algorithms::wildcard_dp_plot("abc","a*c");
/// plot.title("Restricted-glob prefix states");
/// @endcode
/// @see wildcard_dp_table(), wildcard_match_trace()
graphics::RPlot wildcard_dp_plot(const std::string& text,const std::string& pattern,
                                  std::size_t max_cells=10000);

/// @brief Summarise matches and actual greedy work across a string DataFrame column.
///
/// Each non-null cell is one independent whole-text match. Nulls are skipped
/// and counted; no concatenation, case conversion or Unicode decoding occurs.
/// Rows follow supplied pattern order, including duplicates. O(sum of matching
/// work) time and O(number of patterns) result storage, excluding input strings.
/// @param data Borrowed frame, unchanged.
/// @param column Required string column.
/// @param patterns Restricted glob patterns; empty gives a zero-row table with the same schema.
/// @return Owned pattern, input_rows, evaluated_rows, skipped_rows, matches, text_bytes,
/// steps, literal_tests and retries columns.
/// @throws std::out_of_range If column is absent.
/// @throws std::invalid_argument If column is not a string column.
/// @par Example
/// @code
/// const auto table=dm::algorithms::wildcard_summary(dm::datasets::titanic(),"Name",{"*Miss.*"});
/// const auto matches=table.double_at("matches",0); // 182
/// @endcode
/// @see wildcard_filter(), wildcard_match_stats()
dstruct::DataFrame wildcard_summary(const dstruct::DataFrame& data,const std::string& column,
                                     const std::vector<std::string>& patterns);

/// @brief Keep original rows whose non-null string cell matches the entire restricted glob.
/// @param data Borrowed frame, unchanged; all columns and original order are retained.
/// @param column Required string column; null cells never match.
/// @param pattern Restricted case-sensitive byte glob.
/// @return Owned row subset with the original schema, including an empty subset when no cells match.
/// @throws std::out_of_range If column is absent.
/// @throws std::invalid_argument If column is not a string column.
/// @par Example
/// @code
/// const auto rows=dm::algorithms::wildcard_filter(dm::datasets::titanic(),"Name","*Miss.*");
/// const auto count=rows.nrows(); // 182
/// @endcode
/// @see wildcard_summary(), wildcard_match()
dstruct::DataFrame wildcard_filter(const dstruct::DataFrame& data,const std::string& column,
                                    const std::string& pattern);

} // namespace datamunge::algorithms
