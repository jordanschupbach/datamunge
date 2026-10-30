/// @file hypothesis_diagnostics.cpp
/// @brief Owned hypothesis-test diagnostics and deterministic finite-support calibration.
#include <datamunge/stats/detail/ranking.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/fisher_exact_test.hpp>
#include <datamunge/stats/hypothesis_diagnostics.hpp>
#include <datamunge/stats/proportion_test.hpp>
#include <limits>
#include <optional>
namespace datamunge::stats {
dstruct::DataFrame rank_sum_table(const std::vector<double>& x, const std::vector<double>& y) {
  detail::check_sample(x);
  detail::check_sample(y);
  std::vector<double> values = x, indices;
  values.insert(values.end(), y.begin(), y.end());
  std::vector<std::string> groups;
  for (std::size_t i = 0; i < x.size(); ++i) {
    groups.push_back("x");
    indices.push_back(i);
  }
  for (std::size_t i = 0; i < y.size(); ++i) {
    groups.push_back("y");
    indices.push_back(i);
  }
  dstruct::DataFrame out;
  out.add_column("group", groups);
  out.add_column("sample_row", indices);
  out.add_column("value", values);
  out.add_column("rank", detail::rank_with_ties(values));
  return out;
}
dstruct::DataFrame hypothesis_test_table(
    const std::vector<HypothesisTestResult>& results,
    const std::vector<std::string>&          labels) {
  if (!labels.empty() && labels.size() != results.size())
    throw std::invalid_argument("hypothesis_test_table: label count mismatch");
  std::vector<std::string>           names, methods;
  std::vector<double>                statistic, df1, df2, p, e1, e2;
  std::vector<std::optional<double>> lower, upper;
  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto& r = results[i];
    names.push_back(labels.empty() ? r.method : labels[i]);
    methods.push_back(r.method);
    statistic.push_back(r.statistic);
    df1.push_back(r.parameter1);
    df2.push_back(r.parameter2);
    p.push_back(r.p_value);
    e1.push_back(r.estimate1);
    e2.push_back(r.estimate2);
    lower.push_back(r.has_conf_int ? std::optional<double>(r.conf_int_lower) : std::nullopt);
    upper.push_back(r.has_conf_int ? std::optional<double>(r.conf_int_upper) : std::nullopt);
  }
  dstruct::DataFrame out;
  out.add_column("label", names);
  out.add_column("method", methods);
  out.add_column("statistic", statistic);
  out.add_column("parameter1", df1);
  out.add_column("parameter2", df2);
  out.add_column("p_value", p);
  out.add_column("estimate1", e1);
  out.add_column("estimate2", e2);
  out.add_column("ci_lower", lower);
  out.add_column("ci_upper", upper);
  return out;
}
dstruct::DataFrame binomial_rejection_table(
    std::size_t n,
    double      p,
    double      alpha,
    Alternative alternative,
    bool        correct) {
  detail::check_alternative(alternative);
  detail::check_confidence(alpha);
  if (n == 0 || n == std::numeric_limits<std::size_t>::max() || !std::isfinite(p) || p <= 0 ||
      p >= 1)
    throw std::invalid_argument("binomial_rejection_table: positive n and 0<p<1 required");
  std::vector<double> count, mass, exact, score, reject_exact, reject_score;
  for (std::size_t x = 0; x <= n; ++x) {
    const double pe = detail::binomial_test_p(x, n, p, alternative);
    const double ps = proportion_test_one_sample(x, n, p, alternative, correct).p_value;
    count.push_back(x);
    mass.push_back(random::binomial_pdf(x, n, p));
    exact.push_back(pe);
    score.push_back(ps);
    reject_exact.push_back(pe <= alpha);
    reject_score.push_back(ps <= alpha);
  }
  dstruct::DataFrame out;
  out.add_column("successes", count);
  out.add_column("null_probability", mass);
  out.add_column("exact_p", exact);
  out.add_column("score_p", score);
  out.add_column("reject_exact", reject_exact);
  out.add_column("reject_score", reject_score);
  return out;
}
dstruct::DataFrame binomial_operating_characteristics(
    const std::vector<std::size_t>& sizes,
    double                          p_null,
    double                          p_true,
    double                          alpha,
    Alternative                     alternative,
    bool                            correct) {
  if (sizes.empty() || !std::isfinite(p_true) || p_true < 0 || p_true > 1)
    throw std::invalid_argument(
        "binomial_operating_characteristics: nonempty sizes and 0<=p_true<=1 required");
  std::vector<double> n, es, ss, ep, sp;
  for (auto size : sizes) {
    const auto  region     = binomial_rejection_table(size, p_null, alpha, alternative, correct);
    long double exact_size = 0, score_size = 0, exact_power = 0, score_power = 0;
    for (std::size_t x = 0; x <= size; ++x) {
      const double null_mass = region.double_at("null_probability", x),
                   true_mass = random::binomial_pdf(x, size, p_true);
      if (region.double_at("reject_exact", x)) {
        exact_size += null_mass;
        exact_power += true_mass;
      }
      if (region.double_at("reject_score", x)) {
        score_size += null_mass;
        score_power += true_mass;
      }
    }
    n.push_back(size);
    es.push_back(static_cast<double>(exact_size));
    ss.push_back(static_cast<double>(score_size));
    ep.push_back(static_cast<double>(exact_power));
    sp.push_back(static_cast<double>(score_power));
  }
  dstruct::DataFrame out;
  out.add_column("n", n);
  out.add_column("exact_size", es);
  out.add_column("score_size", ss);
  out.add_column("exact_power", ep);
  out.add_column("score_power", sp);
  return out;
}
dstruct::DataFrame fisher_exact_support(
    std::size_t a,
    std::size_t b,
    std::size_t c,
    std::size_t d,
    Alternative alternative) {
  (void)fisher_exact_test_2x2(a, b, c, d, alternative); // validates total and alternative
  const std::size_t total = a + b + c + d, row1 = a + b, col1 = a + c,
                    lo = row1 > total - col1 ? row1 - (total - col1) : 0, hi = std::min(row1, col1);
  /// @brief Conditional fixed-margin log probability.
  const auto log_pmf = [&](std::size_t k) {
    return detail::test_log_choose(col1, k) + detail::test_log_choose(total - col1, row1 - k) -
           detail::test_log_choose(total, row1);
  };
  const long double   threshold = log_pmf(a) + std::log1p(1e-7L);
  std::vector<double> count, mass, selected;
  for (std::size_t k = lo;; ++k) {
    const auto lp = log_pmf(k);
    count.push_back(k);
    mass.push_back(static_cast<double>(std::exp(lp)));
    selected.push_back(
        alternative == Alternative::Less      ? k <= a
        : alternative == Alternative::Greater ? k >= a
                                              : lp <= threshold);
    if (k == hi)
      break;
  }
  dstruct::DataFrame out;
  out.add_column("top_left", count);
  out.add_column("probability", mass);
  out.add_column("selected", selected);
  return out;
}
graphics::RPlot t_test_reference_plot(const HypothesisTestResult& result) {
  detail::check_alternative(result.alternative);
  if (!std::isfinite(result.statistic) || !std::isfinite(result.parameter1) ||
      result.parameter1 <= 0)
    throw std::invalid_argument(
        "t_test_reference_plot: finite statistic and positive finite df required");
  const double t = result.statistic, df = result.parameter1,
               limit = std::max(5.0, std::abs(t) * 1.3);
  std::vector<double> x, y;
  for (int i = 0; i <= 512; ++i) {
    const double v = -limit + 2 * limit * i / 512;
    x.push_back(v);
    y.push_back(random::student_t_pdf(v, df));
  }
  auto plot = graphics::RPlot::create();
  /// @brief Adds one truncated shaded tail under the Student density.
  const auto shade = [&](double left, double right) {
    std::vector<double> xx{left}, yy{0};
    for (int i = 0; i <= 128; ++i) {
      const double v = left + (right - left) * i / 128;
      xx.push_back(v);
      yy.push_back(random::student_t_pdf(v, df));
    }
    xx.push_back(right);
    yy.push_back(0);
    plot.polygon(xx, yy, {220, 38, 38}, true);
  };
  if (result.alternative == Alternative::TwoSided) {
    shade(-limit, -std::abs(t));
    shade(std::abs(t), limit);
  } else if (result.alternative == Alternative::Less)
    shade(-limit, t);
  else
    shade(t, limit);
  plot.lines(x, y, "Student null", {37, 99, 235});
  plot.segments({t}, {0}, {t}, {random::student_t_pdf(t, df)}, {0, 0, 0}, 2);
  return plot.title("Observed statistic and null tails")
      .x_label("t statistic")
      .y_label("density")
      .x_limits(-limit, limit)
      .y_limits(0, random::student_t_pdf(0, df) * 1.3);
}
} // namespace datamunge::stats
