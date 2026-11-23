/// @file tukey_hsd.cpp
/// @brief Studentized range distribution (after R's ptukey.c) and Tukey's HSD.
#include <algorithm>
#include <cmath>
#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/tukey_hsd.hpp>
#include <format>
#include <map>
#include <numbers>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Quadrature helpers for the studentized range distribution and shared input checks.
namespace {

/// @brief Standard normal CDF at @p x.
/// @param x Argument.
/// @return Phi(x).
double phi(double x) { return 0.5 * std::erfc(-x / std::numbers::sqrt2); }

/// @brief P(range of @p cc standard normals < @p w), R's wprob() with rr = 1 (infinite df).
/// @param w Range, w >= 0.
/// @param cc Number of normals.
/// @return Probability in [0, 1].
double range_probability(double w, double cc) {
  constexpr int    nleg = 12, ihalf = 6;
  constexpr double C1 = -30, C3 = 60, bb = 8, wlar = 3;
  constexpr double xleg[ihalf] = {0.981560634246719250690549090149, 0.904117256370474856678465866119,
                                  0.769902674194304687036893833213, 0.587317954286617447296702418941,
                                  0.367831498998180193752691536644, 0.125233408511468915472441369464};
  constexpr double aleg[ihalf] = {0.047175336386511827194615961485, 0.106939325995318430960254718194,
                                  0.160078328543346226334652529543, 0.203167426723065921749064455810,
                                  0.233492536538354808760849898925, 0.249147045813402785000562436043};
  const double qsqz = w * 0.5;
  if (qsqz >= bb) return 1.0;
  // P(|Z| < w/2)^cc: every normal within w/2 of zero.
  double pr_w = 2 * phi(qsqz) - 1;
  pr_w        = pr_w >= 1 ? 1 : std::pow(pr_w, cc);
  const int wincr = w > wlar ? 2 : 3;
  double    blb = qsqz, binc = (bb - qsqz) / wincr, bub = blb + binc, einsum = 0;
  const double cc1 = cc - 1;
  for (int wi = 1; wi <= wincr; ++wi) {
    double       elsum = 0;
    const double a = 0.5 * (bub + blb), b = 0.5 * (bub - blb);
    for (int jj = 1; jj <= nleg; ++jj) {
      int    j;
      double xx;
      if (ihalf < jj) {
        j  = nleg - jj + 1;
        xx = xleg[j - 1];
      } else {
        j  = jj;
        xx = -xleg[j - 1];
      }
      const double ac = a + b * xx, qexpo = ac * ac;
      if (qexpo > C3) break;
      const double rinsum = phi(ac) - phi(ac - w);
      if (rinsum >= std::exp(C1 / cc1))
        elsum += aleg[j - 1] * std::exp(-0.5 * qexpo) * std::pow(rinsum, cc1);
    }
    einsum += elsum * (2 * b) * cc / std::sqrt(2 * std::numbers::pi);
    blb = bub;
    bub += binc;
  }
  pr_w += einsum;
  if (pr_w <= std::exp(C1)) return 0;
  return std::min(pr_w, 1.0);
}

/// @brief Rejects fewer than two means or less than one degree of freedom.
/// @param nmeans Number of means.
/// @param df Error degrees of freedom.
/// @throws std::invalid_argument Out-of-domain parameters.
void check_range_parameters(std::size_t nmeans, double df) {
  if (nmeans < 2) throw std::invalid_argument("studentized range: nmeans must be at least 2");
  if (!(df >= 1)) throw std::invalid_argument("studentized range: df must be at least 1");
}

} // namespace

double studentized_range_cdf(double q, std::size_t nmeans, double df, bool lower_tail) {
  check_range_parameters(nmeans, df);
  if (std::isnan(q)) throw std::invalid_argument("studentized range: q is NaN");
  const auto tail = [lower_tail](double p) { return lower_tail ? p : 1 - p; };
  if (q <= 0) return tail(0);
  if (!std::isfinite(q)) return tail(1);
  const double cc = static_cast<double>(nmeans);
  if (df > 25000) return tail(range_probability(q, cc));
  if (nmeans == 2) {
    const double upper = 2 * random::student_t_cdf(-q / std::numbers::sqrt2, df);
    return lower_tail ? 1 - upper : upper;
  }

  // Integrate wprob(q * sqrt(s/2)) against the chi density of s over (0, inf), in blocks of
  // width ulen with 16-point Gauss-Legendre, as R's ptukey.c does.
  constexpr int    nlegq = 16, ihalfq = 8;
  constexpr double eps1 = -30, eps2 = 1e-14;
  constexpr double xlegq[ihalfq] = {0.989400934991649932596154173450, 0.944575023073232576077988415535,
                                    0.865631202387831743880467897712, 0.755404408355003033895101194847,
                                    0.617876244402643748446671764049, 0.458016777657227386342419442984,
                                    0.281603550779258913230460501460, 0.950125098376374401853193354250e-1};
  constexpr double alegq[ihalfq] = {0.271524594117540948517805724560e-1, 0.622535239386478928628438369944e-1,
                                    0.951585116824927848099251076022e-1, 0.124628971255533872052476282192,
                                    0.149595988816576732081501730547,    0.169156519395002538189312079030,
                                    0.182603415044923588866763667969,    0.189450610455068496285396723208};
  const double f2   = df * 0.5;
  const double ulen = df <= 100 ? 1.0 : df <= 800 ? 0.5 : df <= 5000 ? 0.25 : 0.125;
  const double f2lf = f2 * std::log(df) - df * std::numbers::ln2 - std::lgamma(f2) + std::log(ulen);
  const double f21 = f2 - 1, ff4 = df * 0.25;
  double       ans = 0;
  if (df < 2) {
    // Integrate in log(u) to remove the integrable singularity at zero. For
    // 1 <= df < 2 the truncated chi-square mass is negligible at both ends.
    for (int block = -80; block < 5; ++block) {
      for (int j = 0; j < ihalfq; ++j) {
        for (const double sign : {-1.0, 1.0}) {
          const double log_u = block + 0.5 * (1 + sign * xlegq[j]);
          const double u = std::exp(log_u);
          const double density = std::exp(f2lf + f2 * log_u - u * ff4);
          ans += 0.5 * alegq[j] * density * tail(range_probability(q * std::sqrt(u * 0.5), cc));
        }
      }
    }
    return std::clamp(ans, 0.0, 1.0);
  }
  for (int i = 1; i <= 50; ++i) {
    double       otsum = 0;
    const double twa1  = (2 * i - 1) * ulen;
    for (int jj = 1; jj <= nlegq; ++jj) {
      const bool   upper = ihalfq < jj;
      const int    j     = upper ? jj - ihalfq - 1 : jj - 1;
      const double u     = upper ? twa1 + xlegq[j] * ulen : twa1 - xlegq[j] * ulen;
      const double t1    = f2lf + f21 * std::log(u) - u * ff4;
      if (t1 >= eps1) otsum += range_probability(q * std::sqrt(u * 0.5), cc) * alegq[j] * std::exp(t1);
    }
    if (i * ulen >= 1.0 && otsum <= eps2) break;
    ans += otsum;
  }
  return tail(std::clamp(ans, 0.0, 1.0));
}

double studentized_range_quantile(double p, std::size_t nmeans, double df) {
  check_range_parameters(nmeans, df);
  if (!(p > 0 && p < 1)) throw std::invalid_argument("studentized range: p must be in (0, 1)");
  double lo = 0, hi = 1;
  while (studentized_range_cdf(hi, nmeans, df) < p) {
    lo = hi;
    hi *= 2;
    if (hi > 1e6) throw std::invalid_argument("studentized range: quantile out of range");
  }
  for (int it = 0; it < 200 && hi - lo > 1e-10 * std::max(1.0, hi); ++it) {
    const double mid = 0.5 * (lo + hi);
    (studentized_range_cdf(mid, nmeans, df) < p ? lo : hi) = mid;
  }
  return 0.5 * (lo + hi);
}

TukeyHSD::TukeyHSD(const std::vector<double>& values, const std::vector<std::string>& groups, double conf_level)
    : conf_level_(conf_level) {
  detail::check_confidence(conf_level);
  if (values.size() != groups.size())
    throw std::invalid_argument("TukeyHSD: values and groups must have the same length");
  // std::map orders the levels lexicographically, like R's factor().
  std::map<std::string, std::vector<double>> by;
  for (std::size_t i = 0; i < values.size(); ++i) by[groups[i]].push_back(values[i]);
  fit(by);
}

TukeyHSD::TukeyHSD(const dstruct::DataFrame& data, const std::string& value_column,
                   const std::string& group_column, double conf_level)
    : conf_level_(conf_level) {
  detail::check_confidence(conf_level);
  fit(data.split(value_column, group_column));
}

namespace {

/// @brief Trims ASCII whitespace from both ends of @p text.
/// @param text Input.
/// @return The trimmed copy.
std::string trim_spaces(const std::string& text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

/// @brief The single grouping column named on the right of a one-way formula.
/// @param formula Formula `value ~ group`.
/// @return The trimmed group column name.
/// @throws std::invalid_argument A right-hand side that is not one plain column name.
std::string one_way_group(const Formula& formula) {
  const std::string& text  = formula.text();
  const std::string  group = trim_spaces(text.substr(text.find('~') + 1));
  if (group.find_first_of("+-*:/^()~ \t") != std::string::npos || group == "." || group == "1" || group == "0")
    throw std::invalid_argument("TukeyHSD: formula must be one-way, 'value ~ group', got \"" + text + "\"");
  return group;
}

/// @brief Throws a hint when the formula sides look swapped (string response, numeric group).
/// @param data Frame holding both columns.
/// @param value Response column name.
/// @param group Group column name.
/// @throws std::out_of_range A missing column.
/// @throws std::invalid_argument The sides look reversed.
void check_one_way_sides(const dstruct::DataFrame& data, const std::string& value, const std::string& group) {
  using Type = dstruct::DataFrame::ColumnType;
  if (data.column_type(value) == Type::String && data.column_type(group) == Type::Numeric)
    throw std::invalid_argument("TukeyHSD: response '" + value + "' is a string column and group '" + group +
                                "' is numeric; did you mean \"" + group + " ~ " + value + "\"?");
}

} // namespace

TukeyHSD::TukeyHSD(const dstruct::DataFrame& data, const Formula& formula, double conf_level)
    : conf_level_(conf_level) {
  detail::check_confidence(conf_level);
  const std::string group = one_way_group(formula);
  check_one_way_sides(data, formula.response_name(), group);
  fit(data.split(formula.response_name(), group));
}

void TukeyHSD::fit(const std::map<std::string, std::vector<double>>& by) {
  const std::size_t k = by.size();
  if (k < 2) throw std::invalid_argument("TukeyHSD: need at least two groups");

  std::vector<std::string> names;
  std::vector<double>      means, sizes;
  long double              within_ss = 0;
  std::size_t              total     = 0;
  for (const auto& [name, sample] : by) {
    detail::check_sample(sample);
    total += sample.size();
    long double mean = 0;
    for (double v : sample) mean += static_cast<long double>(v) / sample.size();
    for (double v : sample) within_ss += (v - mean) * (v - mean);
    names.push_back(name);
    means.push_back(static_cast<double>(mean));
    sizes.push_back(static_cast<double>(sample.size()));
  }
  df_ = static_cast<double>(total) - static_cast<double>(k);
  if (df_ < 2) throw std::invalid_argument("TukeyHSD: need at least two residual degrees of freedom");
  mse_ = static_cast<double>(within_ss / df_);
  if (!(mse_ > 0) || !std::isfinite(mse_))
    throw std::invalid_argument("TukeyHSD: within-group variance must be positive and finite");

  const double q_crit = studentized_range_quantile(conf_level_, k, df_);
  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = i + 1; j < k; ++j) {
      const double diff  = means[j] - means[i];
      const double se    = std::sqrt(mse_ / 2 * (1 / sizes[i] + 1 / sizes[j]));
      const double width = q_crit * se;
      comparisons_.push_back({names[j], names[i], diff, diff - width, diff + width,
                              studentized_range_cdf(std::abs(diff) / se, k, df_, false)});
    }
}

dstruct::DataFrame TukeyHSD::frame() const {
  std::vector<std::string> label, group1, group2;
  std::vector<double>      diff, lwr, upr, p_adj;
  for (const auto& c : comparisons_) {
    label.push_back(c.label());
    group1.push_back(c.group1);
    group2.push_back(c.group2);
    diff.push_back(c.diff);
    lwr.push_back(c.lower);
    upr.push_back(c.upper);
    p_adj.push_back(c.p_adjusted);
  }
  dstruct::DataFrame out;
  out.add_column("comparison", label);
  out.add_column("group1", group1);
  out.add_column("group2", group2);
  out.add_column("diff", diff);
  out.add_column("lwr", lwr);
  out.add_column("upr", upr);
  out.add_column("p_adj", p_adj);
  return out;
}

graphics::RPlot TukeyHSD::plot(graphics::RGBA significant, graphics::RGBA not_significant) const {
  const std::size_t        n = comparisons_.size();
  std::vector<std::string> labels(n);
  auto                     plt = graphics::RPlot::create();
  const double             top = static_cast<double>(n) - 0.4;
  plt.lines({0.0, 0.0}, {-0.6, top}, graphics::RGBA{71, 85, 105}, 1.5);  // Zero: no difference.
  for (std::size_t k = 0; k < n; ++k) {
    const auto&  c     = comparisons_[k];
    const double y     = static_cast<double>(n - 1 - k);  // First comparison on top, as R draws it.
    const auto   color = c.p_adjusted < 1 - conf_level_ ? significant : not_significant;
    labels[n - 1 - k]  = c.label();
    plt.lines({c.lower, c.upper}, {y, y}, color, 2.5)
        .lines({c.lower, c.lower}, {y - 0.12, y + 0.12}, color, 2.5)
        .lines({c.upper, c.upper}, {y - 0.12, y + 0.12}, color, 2.5)
        .points({c.diff}, {y}, color, 5.0);
  }
  // The automatic margins don't measure text tick labels, so leave room for the longest one.
  std::size_t longest = 0;
  for (const auto& label : labels) longest = std::max(longest, label.size());
  plt.y_tick_labels(labels)
      .margins(std::max(90.0, 7.0 * static_cast<double>(longest) + 30.0), 24, 72, 72)
      .ylim(-0.6, top)
      .legend({"significant", "not significant"}, {significant, not_significant})
      .title(std::format("{:g}% family-wise confidence level", 100 * conf_level_))
      .xlab("Differences in mean levels")
      .ylab("");
  return plt;
}

} // namespace datamunge::stats
