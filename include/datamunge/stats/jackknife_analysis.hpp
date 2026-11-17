#pragma once
/// @file jackknife_analysis.hpp
/// @brief Native statistic bridges and actual leave-one-out jackknife graphics.
#include <cmath>
#include <datamunge/plot/diagram.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/jackknife.hpp>
#include <datamunge/vector.hpp>
#include <format>
#include <limits>
#include <stdexcept>
namespace datamunge::stats {
/// @brief Native univariate statistics used by the worked jackknife experiments.
/// Mean/median/sample SD delegate to canonical vector operations; population
/// variance is sample SD squared times (n-1)/n, with finite singleton variance0.
/// @par Example
/// @code
/// std::print("{:.0f}\n",dm::stats::jackknife_statistic(dm::stats::JackknifeStatistic::Median)({1,2,8})); // 2
/// @endcode
/// @see jackknife_statistic()
enum class JackknifeStatistic {
  Mean,               ///< Canonical arithmetic mean.
  PopulationVariance, ///< Variance with denominator n, singleton finite value gives zero.
  SampleVariance,     ///< Canonical sample SD squared, denominator n-1.
  Median              ///< Canonical median, midpoint of two middle values for even count.
};
/// @brief Return an owned callback bridging a named statistic to native vector operations.
/// Closure captures only the enum, never input references. Each call copies its
/// borrowed vector into an owned native vector; median may make another copy.
/// Mean/variance O(n), median average O(n), all O(n) temporary storage. Native
/// empty/insufficient-size/nonfinite rules apply (typically NaN); the checked
/// jackknife separately requires finite input/callback returns. Variance squares
/// can overflow double; such nonfinite callback results are rejected by jackknife.
/// Population singleton returns0 only for a finite value. Thread safety follows
/// native operations; there is no shared state or mutation of caller values.
/// @param kind One valid named statistic, rejected otherwise before callback creation.
/// @return Owned stateless numeric callback, usable with jackknife or Bootstrap.
/// @throws std::invalid_argument On an unknown enum value.
/// @throws std::bad_alloc On callback/native-vector allocation failure.
/// @par Example
/// @code
/// const auto f=dm::stats::jackknife_statistic(dm::stats::JackknifeStatistic::PopulationVariance);
/// std::print("{:.6f}\n",f({1,2,3})); // 0.666667
/// @endcode
/// @see JackknifeStatistic, jackknife()
inline SampleStatistic jackknife_statistic(JackknifeStatistic kind) {
  switch (kind) {
    case JackknifeStatistic::Mean:
    case JackknifeStatistic::PopulationVariance:
    case JackknifeStatistic::SampleVariance:
    case JackknifeStatistic::Median:
      break;
    default:
      throw std::invalid_argument("jackknife statistic kind");
  }
  return [kind](const std::vector<double>& x) {
    const auto v = datamunge::c(x);
    switch (kind) {
      case JackknifeStatistic::Mean:
        return datamunge::mean(v);
      case JackknifeStatistic::Median:
        return datamunge::median(v);
      case JackknifeStatistic::PopulationVariance:
        if (x.empty())
          return std::numeric_limits<double>::quiet_NaN();
        if (x.size() == 1)
          return std::isfinite(x[0]) ? 0. : std::numeric_limits<double>::quiet_NaN();
        [[fallthrough]];
      case JackknifeStatistic::SampleVariance: {
        const double s        = datamunge::sd(v);
        const double variance = s * s;
        return kind == JackknifeStatistic::SampleVariance
                   ? variance
                   : variance * ((static_cast<double>(x.size()) - 1) / x.size());
      }
    }
    return std::numeric_limits<double>::quiet_NaN();
  };
}
/// @brief Draw one actual leave-one-out result and its algebraic pseudo-value.
/// Shows only the selected original observation, full estimate, actual deletion
/// statistic and pseudo-value. Validates visited finite fields and result counts,
/// not callback provenance. No fit replay, sample mutation or retained references.
/// Constant graphic primitive/label count; complete input count admission precedes
/// visited value reads. This cap is not a byte/RAM quota.
/// @param sample Borrowed complete original observations.
/// @param result Borrowed actual delete-one result.
/// @param deleted Deleted observation index, default0.
/// @param max_elements Complete input display ceiling, default10000; zero disables.
/// @return Owned fitted native concept Diagram.
/// @throws std::out_of_range On absent deleted index.
/// @throws std::invalid_argument On size mismatch or nonfinite displayed fields.
/// @throws std::length_error On enabled complete input ceiling.
/// @throws std::bad_alloc On graphics allocation failure.
/// @par Example
/// @code
/// const std::vector<double> x{1,2,4};
/// const auto r=dm::stats::jackknife(x,dm::stats::jackknife_statistic(dm::stats::JackknifeStatistic::Mean));
/// std::print("{}\n",dm::stats::jackknife_deletion_diagram(x,r,2).to_svg().empty()); // false
/// @endcode
/// @see jackknife()
inline graphics::Diagram jackknife_deletion_diagram(
    const std::vector<double>& sample,
    const JackknifeResult&     result,
    std::size_t                deleted      = 0,
    std::size_t                max_elements = 10000) {
  if (max_elements && sample.size() > max_elements)
    throw std::length_error("jackknife diagram complete cap");
  if (deleted >= sample.size())
    throw std::out_of_range("jackknife diagram deleted index");
  if (result.leave_one_out.size() != sample.size() ||
      result.pseudo_values.size() != sample.size() || !std::isfinite(sample[deleted]) ||
      !std::isfinite(result.estimate) || !std::isfinite(result.leave_one_out[deleted]) ||
      !std::isfinite(result.pseudo_values[deleted]))
    throw std::invalid_argument("jackknife diagram actual fields");
  graphics::Diagram          d;
  graphics::DiagramNodeStyle style;
  style.width = 6;
  d.node(
       "full",
       {0, 0},
       std::format("Full sample: n={}\nactual statistic {:.6g}", sample.size(), result.estimate),
       style)
      .node(
          "delete",
          {8, 0},
          std::format(
              "Delete index {}: value {:.6g}\nactual statistic {:.6g}",
              deleted,
              sample[deleted],
              result.leave_one_out[deleted]),
          style)
      .node(
          "pseudo",
          {16, 0},
          std::format(
              "Algebraic pseudo-value {:.6g}\nnot an independent observation",
              result.pseudo_values[deleted]),
          style)
      .edge("full", "delete")
      .edge("delete", "pseudo");
  return d;
}
/// @brief Plot actual leave-one-out statistic against the deleted observation value.
/// Uses owned result values, with a horizontal full-estimate reference. Validates
/// complete counts and finite plotted values before graphics, but does not
/// rerun callbacks or certify edited-history provenance. No mutation, retained
/// input references or shared state. O(n) validation, owned points and storage.
/// @param sample Borrowed complete original observations.
/// @param result Borrowed same-input actual deletion result.
/// @param max_elements Complete display point cap, default10000; zero disables.
/// @return Owned native scatter/reference plot, ready for labels/title/save_svg().
/// @throws std::invalid_argument On empty/count/nonfinite displayed values.
/// @throws std::length_error On enabled complete display quota.
/// @throws std::bad_alloc On graphics allocation failure.
/// @par Example
/// @code
/// const std::vector<double> x{1,2,4};
/// const auto r=dm::stats::jackknife(x,dm::stats::jackknife_statistic(dm::stats::JackknifeStatistic::Mean));
/// std::print("{}\n",dm::stats::jackknife_influence_plot(x,r).to_svg().empty()); // false
/// @endcode
/// @see jackknife_deletion_diagram()
inline graphics::RPlot jackknife_influence_plot(
    const std::vector<double>& sample,
    const JackknifeResult&     result,
    std::size_t                max_elements = 10000) {
  if (max_elements && sample.size() > max_elements)
    throw std::length_error("jackknife complete display cap");
  if (sample.empty() || result.leave_one_out.size() != sample.size() ||
      !std::isfinite(result.estimate))
    throw std::invalid_argument("jackknife influence complete values");
  for (std::size_t i = 0; i < sample.size(); ++i)
    if (!std::isfinite(sample[i]) || !std::isfinite(result.leave_one_out[i]))
      throw std::invalid_argument("jackknife influence finite values");
  auto p = graphics::RPlot::create();
  p.points(sample, result.leave_one_out, "actual leave-one-out", {35, 95, 210}, 4)
      .abline(result.estimate, 0, {210, 55, 40}, 1.5)
      .x_label("value of deleted observation")
      .y_label("actual leave-one-out statistic");
  return p;
}
} // namespace datamunge::stats
