#pragma once
/// @file viterbi.hpp
/// @brief Checked native viterbi and actual producer state.
#include <cstddef>
#include <datamunge/algorithms/forward_backward.hpp>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual maximum-joint-path decoding result.
/// @par Example
/// @code
/// dm::algorithms::ViterbiResult r{};
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
struct ViterbiResult {
  std::vector<std::size_t>
         states;             ///< Actual best path; empty for impossible or empty observations.
  double log_probability{0}; ///< Best joint log mass; -infinity for impossible,0 for empty.
  bool   possible{false};    ///< Positive-probability path exists; true for the empty path.
};
/// @brief Actual Viterbi dynamic-programming tables and backpointers.
/// @par Example
/// @code
/// auto r=dm::algorithms::viterbi_work(dm::algorithms::forward_backward_example_model(),{0,1,2});
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
struct ViterbiWork {
  ViterbiResult result; ///< Actual decoded path and joint mass.
  std::vector<std::vector<double>>
      delta; ///< Actual maximum log joint prefix masses, time by state.
  std::vector<std::vector<std::size_t>>
              predecessor;         ///< Actual maximizing predecessor, first row unused.
  std::size_t transition_terms{0}; ///< Actual predecessor candidates evaluated, (T-1)*N*N.
};
/// @brief Decode the most likely joint hidden path using log-space dynamic programming.
/// Strict improvements retain the smallest predecessor/final-state index on ties. This local tie rule is not a lexicographic full-path promise. Impossible observations yield no path and -infinity. O(T*N^2) work, O(T*N) storage; table cells<=1000000, predecessor terms<=20000000. No shared state.
/// @param hmm Borrowed finite stochastic model, fully validated.
/// @param obs Borrowed valid symbols; empty has one empty path.
/// @return Actual best path, prefix tables, backpointers and selected candidate count.
/// @throws std::invalid_argument Invalid complete model.
/// @throws std::out_of_range Invalid symbol.
/// @throws std::length_error Table/work budget exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::viterbi_work(dm::algorithms::forward_backward_example_model(),{0,1,2});
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
ViterbiWork viterbi_work(const HiddenMarkovModel& hmm, const std::vector<std::size_t>& obs);
/// @brief Return the same checked Viterbi producer best path.
/// @param hmm Borrowed finite stochastic model.
/// @param obs Borrowed valid observations.
/// @return Actual joint-optimal path/status; empty path for impossible evidence.
/// @throws std::invalid_argument Invalid model.
/// @throws std::out_of_range Invalid symbol.
/// @throws std::length_error Table/work budget exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::viterbi(dm::algorithms::forward_backward_example_model(),{0,1,2});
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
ViterbiResult viterbi(const HiddenMarkovModel& hmm, const std::vector<std::size_t>& obs);

} // namespace datamunge::algorithms
