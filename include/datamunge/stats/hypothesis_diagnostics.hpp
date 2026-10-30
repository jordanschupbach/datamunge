#pragma once
/// @file hypothesis_diagnostics.hpp
/// @brief Result tables, null-reference plots and exact finite-support calibration of tests.
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/hypothesis_test_result.hpp>
#include <string>
#include <vector>
namespace datamunge::stats {
/// @brief Exposes pooled average ranks used by the Wilcoxon rank-sum statistic.
/// Finite samples are concatenated in x,y order; equal values receive average one-based
/// ranks. Inputs are unchanged and calls share no mutable state. O(N log N) work/O(N) space.
/// @param x Nonempty finite first sample.
/// @param y Nonempty finite second sample.
/// @return Owned group/sample_row/value/rank frame, with group labels x and y and zero-based
/// indices within each original sample.
/// @throws std::invalid_argument Empty or nonfinite samples.
/// @par Example
/// @code
/// const auto ranks=dm::stats::rank_sum_table({1,2,2},{3,4,5});
/// std::print("{:.1f}\n",ranks.double_at("rank",1)); // 2.5
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/wilcox.test.html
/// @see wilcoxon_rank_sum_test()
[[nodiscard]] dstruct::DataFrame rank_sum_table(
    const std::vector<double>& x,
    const std::vector<double>& y);
/// @brief Collects heterogeneous htest-style results into an owned dataframe.
/// Unused scalar fields retain their result defaults; read each test's contract rather
/// than interpreting a zero as an estimated quantity. Absent confidence bounds are null.
/// No p-value adjustment or hypothesis selection is performed. Inputs are unchanged;
/// calls have no shared mutable state. O(m) work/storage for m results.
/// @param results Results in display order; empty is allowed.
/// @param labels Optional row labels; empty uses each result's method, otherwise length m.
/// @return
/// label/method/statistic/parameter1/parameter2/p_value/estimate1/estimate2/ci_lower/ci_upper.
/// @throws std::invalid_argument Nonempty labels have a different length.
/// @par Example
/// @code
/// const auto t=dm::stats::t_test_one_sample({1,2,3,4});
/// const auto table=dm::stats::hypothesis_test_table({t},{"mean = 0"});
/// std::print("{}\n",table.nrows()); // 1
/// @endcode
/// @see HypothesisTestResult
[[nodiscard]] dstruct::DataFrame hypothesis_test_table(
    const std::vector<HypothesisTestResult>& results,
    const std::vector<std::string>&          labels = {});

/// @brief Enumerates exact-binomial and score-test rejection rules on every count 0..n.
/// Both tests use the stated alternative and reject when p_value<=alpha. Two-sided exact
/// p-values sum probabilities no larger than the observed PMF, with relative tolerance
/// 1e-7; this is not generally twice the smaller tail. Score p-values optionally use a
/// continuity correction. O(n^2) work and O(n) owned storage; no simulation is used.
/// @param n Positive trial count; n+1 must be representable.
/// @param p Null probability strictly between zero and one (score variance must be positive).
/// @param alpha Finite rejection threshold strictly between zero and one, default .05.
/// @param alternative TwoSided (default), Less or Greater, chosen before seeing the count.
/// @param correct Apply the score test continuity correction, default true.
/// @return successes/null_probability/exact_p/score_p/reject_exact/reject_score frame.
/// @throws std::invalid_argument Invalid n, p, alpha or alternative.
/// @par Example
/// @code
/// const auto region=dm::stats::binomial_rejection_table(10);
/// std::print("{}\n",region.nrows()); // 11
/// @endcode
/// @par References
/// - R binom.test: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/binom.test.html
/// @see binomial_test(), proportion_test_one_sample(), binomial_operating_characteristics()
[[nodiscard]] dstruct::DataFrame binomial_rejection_table(
    std::size_t n,
    double      p           = .5,
    double      alpha       = .05,
    Alternative alternative = Alternative::TwoSided,
    bool        correct     = true);

/// @brief Computes binomial-test size and power by finite enumeration, not Monte Carlo.
/// Uses binomial_rejection_table() at each n and sums rejection probabilities under p_null
/// (size) and p_true (power). The exact test can be conservative because counts are discrete;
/// the approximate score rule need not control size for every n,p_null. Inputs are unchanged;
/// calls use no mutable shared state. O(sum(n^2)) work and O(max(n)) workspace.
/// @param sample_sizes Nonempty positive trial counts; repeats/order are preserved.
/// @param p_null Fixed null success probability strictly in (0,1).
/// @param p_true Fixed true probability in [0,1], used only to evaluate power.
/// @param alpha Rejection threshold in (0,1), default .05.
/// @param alternative Prespecified alternative, default TwoSided.
/// @param correct Apply continuity correction to the score rule, default true.
/// @return n/exact_size/score_size/exact_power/score_power frame.
/// @throws std::invalid_argument Empty sizes, invalid probabilities/alpha/alternative or count.
/// @par Example
/// @code
/// const auto power=dm::stats::binomial_operating_characteristics({10},.5,.7);
/// std::print("{:.6f}\n",power.double_at("exact_size",0)); // 0.021484
/// @endcode
/// @par References
/// - Clopper, Pearson (1934), The Use of Confidence or Fiducial Limits Illustrated in the
///   Case of the Binomial. https://doi.org/10.1093/biomet/26.4.404
/// @see binomial_rejection_table()
[[nodiscard]] dstruct::DataFrame binomial_operating_characteristics(
    const std::vector<std::size_t>& sample_sizes,
    double                          p_null,
    double                          p_true,
    double                          alpha       = .05,
    Alternative                     alternative = Alternative::TwoSided,
    bool                            correct     = true);

/// @brief Exposes Fisher's conditional hypergeometric support and selected probability tail.
/// Margins are fixed from the observed row-major 2x2 table. TwoSided selects tables whose
/// conditional probability is no larger than the observed table, using relative tolerance
/// 1e-7. Less/Greater refer to the top-left count (equivalently the odds ratio). Empty
/// margins give a degenerate one-point support. O(s) work/storage for s feasible counts.
/// @param a Top-left observed count.
/// @param b Top-right observed count.
/// @param c Bottom-left observed count.
/// @param d Bottom-right observed count.
/// @param alternative Prespecified tail, default TwoSided.
/// @return top_left/probability/selected frame; probabilities can underflow in extreme tails.
/// @throws std::invalid_argument Count-total overflow or invalid alternative.
/// @par Example
/// @code
/// const auto support=dm::stats::fisher_exact_support(0,3,3,0);
/// std::print("{}\n",support.nrows()); // 4
/// @endcode
/// @par References
/// - R fisher.test: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/fisher.test.html
/// @see fisher_exact_test_2x2()
[[nodiscard]] dstruct::DataFrame fisher_exact_support(
    std::size_t a,
    std::size_t b,
    std::size_t c,
    std::size_t d,
    Alternative alternative = Alternative::TwoSided);

/// @brief Draws a Student null density and the p-value tail of a t-statistic result.
/// This recipe assumes statistic is t and parameter1 is its positive df (t-tests or
/// correlation t-approximations); it cannot validate that provenance. It uses alternative
/// to shade tails beyond the observed statistic. Density/tails are truncated to the drawn
/// finite range; the stored p-value, not polygon area, is the numerical answer.
/// @param result Finite t statistic, positive finite df and valid alternative.
/// @return Owned editable plot; no rendering or input mutation occurs.
/// @throws std::invalid_argument Nonfinite statistic/df, nonpositive df or invalid alternative.
/// @par Example
/// @code
/// const auto t=dm::stats::t_test_one_sample({1,2,3,4});
/// dm::stats::t_test_reference_plot(t).save_svg("hypothesis_t.svg");
/// @endcode
/// @par References
/// - Student (1908), The Probable Error of a Mean. https://doi.org/10.1093/biomet/6.1.1
/// @see t_test_one_sample(), t_test_two_sample()
[[nodiscard]] graphics::RPlot t_test_reference_plot(const HypothesisTestResult& result);
} // namespace datamunge::stats
