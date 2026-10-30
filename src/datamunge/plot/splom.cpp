/// @file splom.cpp
/// @brief Generalized pairs plot construction on top of RPlot panels tiled by RLayout.

#include <datamunge/plot/splom.hpp>
#include <datamunge/plot/figure.hpp>

#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/chi_squared_test.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <numbers>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace datamunge::graphics {
/// @brief Palette, panel-layout and density helpers private to the pairs plot.
namespace {

/// @brief Default colors for categorical levels and group_by() groups (Tableau 10).
const std::vector<RGB>& default_level_palette() {
  static const std::vector<RGB> palette{
      {78, 121, 167},  {242, 142, 43},  {225, 87, 89},  {118, 183, 178}, {89, 161, 79},
      {237, 201, 72},  {176, 122, 161}, {255, 157, 167}, {156, 117, 95}, {186, 176, 172},
  };
  return palette;
}

/// @brief Outer margins of one panel in pixels: room for short tick labels and nothing else.
constexpr double kPanelLeft = 44.0;
constexpr double kPanelRight = 12.0;
constexpr double kPanelTop = 8.0;
constexpr double kPanelBottom = 28.0;
/// @brief Tick label size inside a panel.
constexpr double kPanelTickFont = 9.0;
/// @brief Conservative average advance of one tick-label character in pixels (Computer Modern
///        lowercase averages about half an em, so 9px text runs near 4.5px per character).
constexpr double kGlyphPx = 5.0;
/// @brief Longest categorical label shown on a vertical axis before truncation.
constexpr std::size_t kMaxYLabelChars = 10;
/// @brief Text color for names and annotations, matching the default axis color.
constexpr RGB kLabelColor{55, 65, 81};
/// @brief Outline and marker color for violins and bars.
constexpr RGB kInkColor{31, 41, 55};
/// @brief Opacity of violin and stacked-bar fills.
constexpr double kFillAlpha = 0.75;
/// @brief Half the width of the widest violin, in level units (levels sit one unit apart).
constexpr double kViolinHalfWidth = 0.42;
/// @brief Points at which each violin's density is evaluated.
constexpr std::size_t kViolinGrid = 64;
/// @brief Gap between mosaic tiles as a fraction of the unit square.
constexpr double kMosaicGap = 0.02;

/// @brief Mosaic shades by Pearson residual: neutral, then moderate and strong excess/deficit.
constexpr RGB kMosaicNeutral{226, 228, 232};
constexpr RGB kMosaicMoreLight{146, 197, 222};
constexpr RGB kMosaicMoreDark{33, 102, 172};
constexpr RGB kMosaicFewerLight{244, 165, 130};
constexpr RGB kMosaicFewerDark{178, 24, 43};

/// @brief Cuts @p label to at most @p max_chars characters (at least one).
std::string fit_label(const std::string& label, std::size_t max_chars) {
  max_chars = std::max<std::size_t>(max_chars, 1);
  return label.size() <= max_chars ? label : label.substr(0, max_chars);
}

/// @brief Characters of the small tick font that fit in @p pixels.
std::size_t chars_in(double pixels) {
  return pixels <= kGlyphPx ? 1 : static_cast<std::size_t>(pixels / kGlyphPx);
}

/// @brief R's default (type 7) quantile of already-sorted, non-empty @p sorted.
double quantile7(const std::vector<double>& sorted, double p) {
  const double h = (static_cast<double>(sorted.size()) - 1.0) * p;
  const auto   lo = static_cast<std::size_t>(std::floor(h));
  const auto   hi = std::min(lo + 1, sorted.size() - 1);
  return sorted[lo] + (h - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
}

/// @brief Silverman's rule-of-thumb Gaussian bandwidth (R's `bw.nrd0`); 0 for constant data.
double silverman_bandwidth(const std::vector<double>& sorted) {
  const double n = static_cast<double>(sorted.size());
  if (sorted.size() < 2) return 0.0;
  double mean = 0.0;
  for (double v : sorted) mean += v;
  mean /= n;
  double ss = 0.0;
  for (double v : sorted) ss += (v - mean) * (v - mean);
  const double sd = std::sqrt(ss / (n - 1.0));
  const double iqr = quantile7(sorted, 0.75) - quantile7(sorted, 0.25);
  double spread = std::min(sd, iqr / 1.34);
  if (!(spread > 0.0)) spread = sd;
  return 0.9 * spread * std::pow(n, -0.2);
}

/// @brief Gaussian kernel density of @p values at @p at with bandwidth @p bw.
double kernel_density(const std::vector<double>& values, double bw, double at) {
  double sum = 0.0;
  for (double v : values) {
    const double z = (at - v) / bw;
    sum += std::exp(-0.5 * z * z);
  }
  return sum / (static_cast<double>(values.size()) * bw * std::sqrt(2.0 * std::numbers::pi));
}

/// @brief Formats a p-value the way R prints small ones: "p<0.001" or "p=0.042".
std::string format_p_value(double p) {
  if (p < 0.001) return "chi2 p<0.001";
  char buffer[32];
  std::snprintf(buffer, sizeof buffer, "chi2 p=%.3f", p);
  return buffer;
}

/// @brief Shade of a mosaic tile with Pearson residual @p residual.
RGB mosaic_shade(double residual) {
  if (residual >= 4.0) return kMosaicMoreDark;
  if (residual >= 2.0) return kMosaicMoreLight;
  if (residual <= -4.0) return kMosaicFewerDark;
  if (residual <= -2.0) return kMosaicFewerLight;
  return kMosaicNeutral;
}

/// @brief Adds the axis-aligned rectangle [x0, x1] x [y0, y1] as a filled polygon.
void add_rect(RPlot& p, double x0, double x1, double y0, double y1, RGB fill) {
  p.polygon({x0, x1, x1, x0}, {y0, y0, y1, y1}, fill, true);
}

/// @brief Points at which smooths and one-dimensional densities are evaluated.
constexpr std::size_t kCurveGrid = 64;
/// @brief Grid points per axis for bivariate densities.
constexpr std::size_t kDensityGrid2D = 48;
/// @brief Probability mass enclosed by each bivariate density contour, outermost first.
constexpr std::array<double, 4> kContourMass{0.95, 0.75, 0.50, 0.25};
/// @brief Opacity of diagonal density fills and of prediction bands.
constexpr double kDensityAlpha = 0.30;
constexpr double kBandAlpha = 0.18;
/// @brief Opacity multiplier for points drawn under a smooth.
constexpr double kSmoothPointAlpha = 0.55;
/// @brief Fewest points a group needs for a smooth, a 1-D density, or a 2-D density.
constexpr std::size_t kMinSmoothPoints = 5;
constexpr std::size_t kMinDensityPoints = 2;
constexpr std::size_t kMinDensity2DPoints = 3;

/// @brief Copy of @p color with its opacity scaled to @p alpha (times its own alpha).
RGB with_alpha(RGB color, double alpha) {
  color.alpha *= alpha;
  return color;
}

/// @brief Opaque mix of @p color toward white: @p strength 1 keeps the color, 0 gives white.
///        Used instead of transparency where overlapping segment ends would show as dots.
RGB toward_white(RGB color, double strength) {
  const auto mix = [strength](int c) {
    return static_cast<int>(std::lround(255.0 - strength * (255.0 - static_cast<double>(c))));
  };
  return RGB{mix(color.r), mix(color.g), mix(color.b)};
}

/// @brief Equivalent-kernel weights of one local fit: the fitted value at x0 is
///        sum_i linear[i] * y[i]; kernel[i] are the raw tricube weights.
struct LocalWeights {
  std::vector<double> linear; ///< local linear (degree 1) weights l_i
  std::vector<double> kernel; ///< tricube kernel weights w_i
};

/// @brief Local linear regression weights at @p x0 with a tricube kernel over the @p k nearest
///        x values (R's `loess(degree = 1, family = "gaussian")` without iterations).
LocalWeights local_linear_weights(const std::vector<double>& xs, double x0, std::size_t k) {
  const std::size_t n = xs.size();
  std::vector<double> distance(n);
  for (std::size_t i = 0; i < n; ++i) distance[i] = std::abs(xs[i] - x0);
  std::vector<double> sorted = distance;
  std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(k - 1), sorted.end());
  double h = sorted[k - 1];
  if (!(h > 0.0)) h = *std::max_element(distance.begin(), distance.end());
  if (!(h > 0.0)) h = 1.0;
  h *= 1.0 + 1e-9; // the k-th neighbour keeps a tiny positive weight
  LocalWeights out{std::vector<double>(n, 0.0), std::vector<double>(n, 0.0)};
  double s0 = 0.0, s1 = 0.0, s2 = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double u = distance[i] / h;
    if (u >= 1.0) continue;
    const double t = 1.0 - u * u * u;
    const double w = t * t * t;
    const double dx = xs[i] - x0;
    out.kernel[i] = w;
    s0 += w;
    s1 += w * dx;
    s2 += w * dx * dx;
  }
  const double det = s0 * s2 - s1 * s1;
  const bool   linear = det > 1e-10 * s0 * s2;
  for (std::size_t i = 0; i < n; ++i) {
    const double w = out.kernel[i];
    out.linear[i] = linear ? w * (s2 - (xs[i] - x0) * s1) / det : w / s0;
  }
  return out;
}

/// @brief A smoothed mean curve and its pointwise prediction band on a grid of x values.
struct SmoothBand {
  std::vector<double> x;     ///< grid over the data's x range
  std::vector<double> fit;   ///< estimated mean at each grid point
  std::vector<double> lower; ///< lower prediction limit
  std::vector<double> upper; ///< upper prediction limit
};

/// @brief Local linear smooth of @p ys on @p xs with a @p level prediction band.
///
/// The mean is a local linear fit (tricube kernel, nearest-neighbour bandwidth covering the
/// fraction @p span of the points). The band allows for unequal spread: the squared residuals,
/// divided by their expectation factor 1 - 2 l_ii + sum_j l_ij^2, are smoothed with the same
/// kernel to estimate the local variance s^2(x), and the band is
/// fit(x) +/- t * sqrt(s^2(x) * (1 + sum_i l_i(x)^2)), with t the Student-t quantile on
/// n - trace(L) residual degrees of freedom (Cleveland's equivalent number of parameters).
/// Returns nothing when there are too few points or no spread in x.
std::optional<SmoothBand> smooth_band(const std::vector<double>& xs, const std::vector<double>& ys, double span,
                                      double level) {
  const std::size_t n = xs.size();
  if (n < kMinSmoothPoints) return std::nullopt;
  const auto [min_it, max_it] = std::minmax_element(xs.begin(), xs.end());
  const double lo = *min_it;
  const double hi = *max_it;
  if (!(hi > lo)) return std::nullopt;
  const std::size_t k =
      std::min(n, std::max<std::size_t>(4, static_cast<std::size_t>(std::ceil(span * static_cast<double>(n)))));

  std::vector<double> adjusted(n);
  double              trace = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto   w = local_linear_weights(xs, xs[i], k);
    double       fitted = 0.0;
    double       sum_sq = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      fitted += w.linear[j] * ys[j];
      sum_sq += w.linear[j] * w.linear[j];
    }
    const double lii = w.linear[i];
    trace += lii;
    const double r = ys[i] - fitted;
    adjusted[i] = r * r / std::max(1.0 - 2.0 * lii + sum_sq, 0.05);
  }
  const double df = std::max(static_cast<double>(n) - trace, 1.0);
  const double t = random::student_t_quantile(0.5 + 0.5 * level, df);

  SmoothBand band;
  for (std::size_t g = 0; g < kCurveGrid; ++g) {
    const double x0 = lo + (hi - lo) * static_cast<double>(g) / static_cast<double>(kCurveGrid - 1);
    const auto   w = local_linear_weights(xs, x0, k);
    double fit = 0.0, sum_sq = 0.0, var_num = 0.0, var_den = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      fit += w.linear[j] * ys[j];
      sum_sq += w.linear[j] * w.linear[j];
      var_num += w.kernel[j] * adjusted[j];
      var_den += w.kernel[j];
    }
    const double variance = var_den > 0.0 ? var_num / var_den : 0.0;
    const double half = t * std::sqrt(variance * (1.0 + sum_sq));
    band.x.push_back(x0);
    band.fit.push_back(fit);
    band.lower.push_back(fit - half);
    band.upper.push_back(fit + half);
  }
  return band;
}

/// @brief Sample standard deviation of @p values (n - 1 denominator); 0 for fewer than two.
double sample_sd(const std::vector<double>& values) {
  if (values.size() < 2) return 0.0;
  double mean = 0.0;
  for (double v : values) mean += v;
  mean /= static_cast<double>(values.size());
  double ss = 0.0;
  for (double v : values) ss += (v - mean) * (v - mean);
  return std::sqrt(ss / static_cast<double>(values.size() - 1));
}

/// @brief Normal-reference bandwidth used by MASS's `kde2d()` (`bandwidth.nrd`, divided by 4 to
///        give the Gaussian kernel's standard deviation); 0 for constant data.
double kde2d_bandwidth(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const double iqr = quantile7(values, 0.75) - quantile7(values, 0.25);
  const double sd = sample_sd(values);
  double spread = std::min(sd, iqr / 1.34);
  if (!(spread > 0.0)) spread = sd;
  return 4.0 * 1.06 * spread * std::pow(static_cast<double>(values.size()), -0.2) / 4.0;
}

/// @brief Density thresholds whose upper level sets hold the fractions @p mass of the total
///        probability on a regular grid (highest-density-region levels).
std::vector<double> hdr_levels(std::vector<double> density, const std::array<double, 4>& mass) {
  std::sort(density.begin(), density.end(), std::greater<>());
  double total = 0.0;
  for (double d : density) total += d;
  std::vector<double> levels;
  for (double target : mass) {
    double cumulative = 0.0;
    double level = density.front();
    for (double d : density) {
      cumulative += d;
      level = d;
      if (cumulative >= target * total) break;
    }
    levels.push_back(level);
  }
  return levels;
}

/// @brief Line segments of the contour at @p level of @p z (z[ix][iy] at gx[ix], gy[iy]), by
///        marching squares; saddle cells are resolved with the cell's mean value.
void contour_segments(const std::vector<std::vector<double>>& z, const std::vector<double>& gx,
                      const std::vector<double>& gy, double level, std::vector<double>& x0,
                      std::vector<double>& y0, std::vector<double>& x1, std::vector<double>& y1) {
  const auto cross = [&](double xa, double ya, double za, double xb, double yb, double zb) {
    const double t = (level - za) / (zb - za);
    return std::pair<double, double>{xa + t * (xb - xa), ya + t * (yb - ya)};
  };
  for (std::size_t i = 0; i + 1 < gx.size(); ++i) {
    for (std::size_t j = 0; j + 1 < gy.size(); ++j) {
      // Corners counter-clockwise from bottom-left.
      const double xs[4] = {gx[i], gx[i + 1], gx[i + 1], gx[i]};
      const double ys[4] = {gy[j], gy[j], gy[j + 1], gy[j + 1]};
      const double zs[4] = {z[i][j], z[i + 1][j], z[i + 1][j + 1], z[i][j + 1]};
      int mask = 0;
      for (int c = 0; c < 4; ++c) {
        if (zs[c] >= level) mask |= 1 << c;
      }
      if (mask == 0 || mask == 15) continue;
      // Crossing points on each edge whose ends straddle the level (edge e joins corners e, e+1).
      std::pair<double, double> points[4];
      bool                      has[4] = {false, false, false, false};
      for (int e = 0; e < 4; ++e) {
        const int a = e;
        const int b = (e + 1) % 4;
        if ((zs[a] >= level) != (zs[b] >= level)) {
          points[e] = cross(xs[a], ys[a], zs[a], xs[b], ys[b], zs[b]);
          has[e] = true;
        }
      }
      const auto emit = [&](int e, int f) {
        x0.push_back(points[e].first);
        y0.push_back(points[e].second);
        x1.push_back(points[f].first);
        y1.push_back(points[f].second);
      };
      if (mask == 5 || mask == 10) {
        // Saddle: four crossings. Join them so the high corners connect through the centre
        // when the centre is above the level.
        const bool centre_high = (zs[0] + zs[1] + zs[2] + zs[3]) / 4.0 >= level;
        const bool corner0_high = (mask & 1) != 0;
        if (centre_high == corner0_high) {
          emit(0, 1);
          emit(2, 3);
        } else {
          emit(3, 0);
          emit(1, 2);
        }
        continue;
      }
      int first = -1;
      for (int e = 0; e < 4; ++e) {
        if (!has[e]) continue;
        if (first < 0) {
          first = e;
        } else {
          emit(first, e);
          break;
        }
      }
    }
  }
}

} // namespace

Splom Splom::create(const dstruct::DataFrame& data, std::vector<std::string> columns) {
  using ColumnType = dstruct::DataFrame::ColumnType;
  Splom result;
  result.rows_ = data.nrows();

  // Sorted unique non-null labels of a string column, and each row's index into them.
  const auto encode = [&](const std::string& name, std::vector<std::string>& levels, std::vector<int>& codes) {
    const auto& column = data.string_column(name);
    std::set<std::string> unique;
    for (std::size_t i = 0; i < column.size(); ++i) {
      if (!column.is_null(i)) unique.insert(column.at(i));
    }
    levels.assign(unique.begin(), unique.end());
    codes.assign(column.size(), -1);
    for (std::size_t i = 0; i < column.size(); ++i) {
      if (column.is_null(i)) continue;
      codes[i] = static_cast<int>(std::lower_bound(levels.begin(), levels.end(), column.at(i)) - levels.begin());
    }
  };

  if (columns.empty()) {
    for (const auto& name : data.columns()) {
      if (data.column_type(name) == ColumnType::Numeric) {
        columns.push_back(name);
      } else {
        std::vector<std::string> levels;
        std::vector<int>         codes;
        encode(name, levels, codes);
        if (!levels.empty() && levels.size() <= max_default_levels) columns.push_back(name);
      }
    }
  }
  for (const auto& name : columns) {
    if (!data.has_column(name)) {
      throw std::invalid_argument("Splom::create: unknown column " + name);
    }
    result.names_.push_back(name);
    if (data.column_type(name) == ColumnType::Numeric) {
      const auto& column = data.numeric_column(name);
      std::vector<double> values(column.size());
      for (std::size_t i = 0; i < column.size(); ++i) {
        values[i] = column.is_null(i) ? std::numeric_limits<double>::quiet_NaN() : column.at(i);
      }
      result.categorical_.push_back(false);
      result.values_.push_back(std::move(values));
      result.codes_.emplace_back();
      result.levels_.emplace_back();
    } else {
      std::vector<std::string> levels;
      std::vector<int>         codes;
      encode(name, levels, codes);
      result.categorical_.push_back(true);
      result.values_.emplace_back();
      result.codes_.push_back(std::move(codes));
      result.levels_.push_back(std::move(levels));
    }
  }
  if (result.names_.size() < 2) {
    throw std::invalid_argument("Splom::create: a pairs plot needs at least two columns");
  }
  for (const auto& name : data.columns()) {
    if (data.column_type(name) != ColumnType::String) continue;
    const auto& column = data.string_column(name);
    std::vector<std::string> labels(column.size());
    for (std::size_t i = 0; i < column.size(); ++i) {
      labels[i] = column.is_null(i) ? std::string() : column.at(i);
    }
    result.string_names_.push_back(name);
    result.string_cols_.push_back(std::move(labels));
  }
  result.group_index_.assign(result.rows_, -1);
  return result;
}

Splom& Splom::title(std::string value) {
  title_ = std::move(value);
  return *this;
}

Splom& Splom::size(std::size_t width, std::size_t height) {
  if (width == 0 || height == 0) {
    throw std::invalid_argument("Splom::size: width and height must be positive");
  }
  width_ = width;
  height_ = height;
  return *this;
}

Splom& Splom::color(RGB value) {
  color_ = value;
  group_by("");
  return *this;
}

Splom& Splom::group_by(const std::string& column, std::vector<RGB> palette) {
  point_colors_.clear();
  factor_codes_.clear();
  factor_palette_.clear();
  if (column.empty()) {
    group_column_.clear();
    group_names_.clear();
    palette_.clear();
    std::fill(group_index_.begin(), group_index_.end(), -1);
    return *this;
  }
  const auto it = std::find(string_names_.begin(), string_names_.end(), column);
  if (it == string_names_.end()) {
    throw std::invalid_argument("Splom::group_by: " + column + " is not a string column of the data");
  }
  const auto& labels = string_cols_[static_cast<std::size_t>(it - string_names_.begin())];
  std::set<std::string> unique;
  for (const auto& label : labels) {
    // An empty label is a null cell (see create()); nulls never form a group.
    if (!label.empty()) unique.insert(label);
  }
  group_column_ = column;
  group_names_.assign(unique.begin(), unique.end());
  for (std::size_t i = 0; i < labels.size(); ++i) {
    group_index_[i] = labels[i].empty()
        ? -1
        : static_cast<int>(std::lower_bound(group_names_.begin(), group_names_.end(), labels[i])
                           - group_names_.begin());
  }
  palette_ = palette.empty() ? default_level_palette() : std::move(palette);
  return *this;
}

Splom& Splom::point_colors(std::vector<RGB> colors) {
  group_by("");
  point_colors_ = std::move(colors);
  return *this;
}

Splom& Splom::color_by_factor(std::vector<int> factor, std::vector<RGB> palette) {
  if (factor.size() != rows_) {
    throw std::invalid_argument("Splom::color_by_factor: factor needs one code per data row");
  }
  if (palette.empty()) {
    throw std::invalid_argument("Splom::color_by_factor: palette must not be empty");
  }
  std::vector<RGB> colors(rows_, color_);
  for (std::size_t i = 0; i < rows_; ++i) {
    if (factor[i] >= 1) colors[i] = palette[static_cast<std::size_t>(factor[i] - 1) % palette.size()];
  }
  point_colors(std::move(colors));
  factor_codes_ = std::move(factor);
  factor_palette_ = std::move(palette);
  return *this;
}

bool Splom::matches_factor(std::size_t var) const {
  if (factor_codes_.empty() || !categorical_[var]) return false;
  const auto& codes = codes_[var];
  for (std::size_t i = 0; i < codes.size(); ++i) {
    if (codes[i] + 1 != std::max(factor_codes_[i], 0)) return false;
  }
  return true;
}

Splom& Splom::level_palette(std::vector<RGB> colors) {
  level_palette_ = std::move(colors);
  return *this;
}

Splom& Splom::marker_size(double value) {
  if (!(value > 0.0)) {
    throw std::invalid_argument("Splom::marker_size: size must be positive");
  }
  marker_size_ = value;
  return *this;
}

Splom& Splom::diagonal(Diagonal kind) {
  diagonal_ = kind;
  return *this;
}

Splom& Splom::bins(std::size_t value) {
  if (value == 0) {
    throw std::invalid_argument("Splom::bins: bin count must be positive");
  }
  bins_ = value;
  return *this;
}

Splom& Splom::span(double value) {
  if (!(value > 0.0 && value <= 1.0)) {
    throw std::invalid_argument("Splom::span: span must be in (0, 1]");
  }
  span_ = value;
  return *this;
}

Splom& Splom::prediction_level(double value) {
  if (!(value > 0.0 && value < 1.0)) {
    throw std::invalid_argument("Splom::prediction_level: level must be in (0, 1)");
  }
  level_ = value;
  return *this;
}

Splom& Splom::show_grid(bool enabled) {
  show_grid_ = enabled;
  return *this;
}

bool Splom::is_categorical(std::size_t index) const {
  if (index >= names_.size()) throw std::out_of_range("Splom::is_categorical: index out of range");
  return categorical_[index];
}

const std::vector<std::string>& Splom::levels(std::size_t index) const {
  if (index >= names_.size()) throw std::out_of_range("Splom::levels: index out of range");
  return levels_[index];
}

RGB Splom::point_color(std::size_t index) const {
  if (!point_colors_.empty()) return point_colors_[index % point_colors_.size()];
  if (group_names_.empty() || group_index_[index] < 0) return color_;
  return palette_[static_cast<std::size_t>(group_index_[index]) % palette_.size()];
}

std::pair<double, double> Splom::axis_range(std::size_t var) const {
  double lo = std::numeric_limits<double>::infinity();
  double hi = -lo;
  for (double v : values_[var]) {
    if (!std::isfinite(v)) continue;
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  if (!(lo <= hi)) return {0.0, 1.0};
  const double pad = hi > lo ? 0.06 * (hi - lo) : 0.5;
  return {lo - pad, hi + pad};
}

RGB Splom::group_color(std::size_t g) const {
  if (group_names_.empty() || !point_colors_.empty()) return color_;
  return palette_[g % palette_.size()];
}

std::vector<std::pair<std::vector<double>, std::vector<double>>> Splom::grouped_pairs(std::size_t xvar,
                                                                                      std::size_t yvar) const {
  const bool grouped = point_colors_.empty() && !group_names_.empty();
  std::vector<std::pair<std::vector<double>, std::vector<double>>> out(grouped ? group_names_.size() : 1);
  const auto& xs = values_[xvar];
  const auto& ys = values_[yvar];
  for (std::size_t i = 0; i < rows_; ++i) {
    if (!std::isfinite(xs[i]) || !std::isfinite(ys[i])) continue;
    std::size_t g = 0;
    if (grouped) {
      if (group_index_[i] < 0) continue;
      g = static_cast<std::size_t>(group_index_[i]);
    }
    out[g].first.push_back(xs[i]);
    out[g].second.push_back(ys[i]);
  }
  return out;
}

RGB Splom::level_color(std::size_t var, std::size_t level) const {
  RGB color;
  if (matches_factor(var)) {
    color = factor_palette_[level % factor_palette_.size()];
  } else if (!group_column_.empty() && names_[var] == group_column_) {
    color = palette_[level % palette_.size()];
  } else {
    const auto& palette = level_palette_.empty() ? default_level_palette() : level_palette_;
    color = palette[level % palette.size()];
  }
  color.alpha = kFillAlpha;
  return color;
}

void Splom::panel_geometry(std::size_t& cell_w, std::size_t& cell_h, double& left) const {
  const std::size_t n = names_.size();
  const std::size_t band = title_.empty() ? 0 : RLayout::title_band_px;
  cell_w = std::max<std::size_t>(width_ / n, 1);
  cell_h = std::max<std::size_t>((height_ > band ? height_ - band : 1) / n, 1);
  // Every panel shares one left margin so plotting areas line up down each column; widen it
  // just enough for the longest (truncated) categorical label drawn on a vertical axis.
  std::size_t longest = 0;
  for (std::size_t v = 0; v < n; ++v) {
    for (const auto& level : levels_[v]) longest = std::max(longest, std::min(level.size(), kMaxYLabelChars));
  }
  left = std::max(kPanelLeft, kGlyphPx * static_cast<double>(longest) + 10.0);
}

RPlot Splom::panel(std::size_t row, std::size_t col) const {
  const std::size_t n = names_.size();
  if (row >= n || col >= n) {
    throw std::out_of_range("Splom::panel: index out of range");
  }
  std::size_t cell_w = 0;
  std::size_t cell_h = 0;
  double      left = kPanelLeft;
  panel_geometry(cell_w, cell_h, left);

  RPlot p = RPlot::create();
  p.size(cell_w, cell_h)
      .title("")
      .x_label("")
      .y_label("")
      .margins(left, kPanelRight, kPanelTop, kPanelBottom)
      .tick_font_size(kPanelTickFont)
      .pretty_ticks(true)
      .show_grid(show_grid_);

  if (row == col) return diagonal_panel(row, std::move(p));
  const bool x_cat = categorical_[col];
  const bool y_cat = categorical_[row];
  if (!x_cat && !y_cat) {
    return row < col ? smooth_panel(row, col, std::move(p)) : density2d_panel(row, col, std::move(p));
  }
  if (x_cat && y_cat) return mosaic_panel(row, col, std::move(p));
  if (x_cat) return violin_panel(col, row, true, std::move(p));
  return violin_panel(row, col, false, std::move(p));
}

RPlot Splom::diagonal_panel(std::size_t var, RPlot p) const {
  if (diagonal_ == Diagonal::Names) {
    p.hide_axes(true).show_grid(false);
    p.text(0.5, 0.5, names_[var], kLabelColor, 14.0);
    return p;
  }

  if (!categorical_[var] && diagonal_ == Diagonal::Density) {
    // One kernel density per group (or one overall), on a shared grid over the data's range.
    const auto groups = grouped_pairs(var, var);
    double lo = std::numeric_limits<double>::infinity();
    double hi = -lo;
    for (const auto& group : groups) {
      for (double v : group.first) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
      }
    }
    if (!(lo <= hi)) {
      throw std::invalid_argument("Splom::panel: column " + names_[var] + " has no finite values");
    }
    std::tie(lo, hi) = axis_range(var);
    double peak = 0.0;
    std::vector<std::pair<std::size_t, std::vector<double>>> curves;
    for (std::size_t g = 0; g < groups.size(); ++g) {
      auto values = groups[g].first;
      if (values.size() < kMinDensityPoints) continue;
      std::sort(values.begin(), values.end());
      const double bw = silverman_bandwidth(values);
      if (!(bw > 0.0)) continue;
      std::vector<double> density(kCurveGrid);
      for (std::size_t k = 0; k < kCurveGrid; ++k) {
        const double at = lo + (hi - lo) * static_cast<double>(k) / static_cast<double>(kCurveGrid - 1);
        density[k] = kernel_density(values, bw, at);
        peak = std::max(peak, density[k]);
      }
      curves.emplace_back(g, std::move(density));
    }
    if (!curves.empty() && peak > 0.0) {
      std::vector<double> grid(kCurveGrid);
      for (std::size_t k = 0; k < kCurveGrid; ++k) {
        grid[k] = lo + (hi - lo) * static_cast<double>(k) / static_cast<double>(kCurveGrid - 1);
      }
      for (const auto& [g, density] : curves) {
        std::vector<double> px = grid;
        std::vector<double> py = density;
        px.push_back(hi);
        py.push_back(0.0);
        px.push_back(lo);
        py.push_back(0.0);
        p.polygon(std::move(px), std::move(py), with_alpha(group_color(g), kDensityAlpha), true);
      }
      for (const auto& [g, density] : curves) p.lines(grid, density, group_color(g), 1.5);
      p.x_limits(lo, hi).y_limits(0.0, 1.3 * peak);
      p.text((lo + hi) * 0.5, 1.15 * peak, names_[var], kLabelColor, 12.0);
      return p;
    }
    // Constant data has no density; fall through to the histogram.
  }

  if (!categorical_[var]) {
    std::vector<double> finite;
    finite.reserve(values_[var].size());
    for (double v : values_[var]) {
      if (std::isfinite(v)) finite.push_back(v);
    }
    if (finite.empty()) {
      throw std::invalid_argument("Splom::panel: column " + names_[var] + " has no finite values");
    }
    RPlot hist = RPlot::hist(finite, bins_, "", color_);
    hist.size(p.width(), p.height())
        .title("")
        .x_label("")
        .y_label("")
        .margins(p.margin_left(), p.margin_right(), p.margin_top(), p.margin_bottom())
        .tick_font_size(p.tick_font_px())
        .pretty_ticks(true)
        .show_grid(p.grid_visible());
    p = std::move(hist);
    double max_count = 0.0;
    double x_lo = finite.front();
    double x_hi = finite.front();
    for (const auto& series : p.series()) {
      for (double y : series.y) max_count = std::max(max_count, y);
      for (double x : series.x) {
        x_lo = std::min(x_lo, x);
        x_hi = std::max(x_hi, x);
      }
    }
    // Reserve headroom above the tallest bar so the name never overlaps it.
    p.y_limits(0.0, 1.3 * max_count);
    p.text((x_lo + x_hi) * 0.5, 1.15 * max_count, names_[var], kLabelColor, 12.0);
    return p;
  }

  // Categorical: one stacked bar of level counts, bottom to top in level order.
  const auto&         levels = levels_[var];
  std::vector<double> counts(levels.size(), 0.0);
  double              total = 0.0;
  for (int code : codes_[var]) {
    if (code < 0) continue;
    counts[static_cast<std::size_t>(code)] += 1.0;
    total += 1.0;
  }
  if (total == 0.0) {
    throw std::invalid_argument("Splom::panel: column " + names_[var] + " has no values");
  }
  constexpr double half = 0.35;
  const double     plot_w = static_cast<double>(p.width()) - p.margin_left() - p.margin_right();
  const double     plot_h = static_cast<double>(p.height()) - p.margin_top() - p.margin_bottom();
  const std::size_t label_chars = chars_in(plot_w * (2.0 * half) / 1.2 - 4.0);
  double cumulative = 0.0;
  for (std::size_t k = 0; k < levels.size(); ++k) {
    if (counts[k] == 0.0) continue;
    add_rect(p, -half, half, cumulative, cumulative + counts[k], level_color(var, k));
    cumulative += counts[k];
  }
  // Outline the whole bar, then label each segment tall enough to hold a line of text.
  p.polygon({-half, half, half, -half}, {0.0, 0.0, total, total}, kInkColor, false);
  cumulative = 0.0;
  for (std::size_t k = 0; k < levels.size(); ++k) {
    if (counts[k] == 0.0) continue;
    if (counts[k] / (1.3 * total) * plot_h >= 11.0) {
      p.text(0.0, cumulative + counts[k] * 0.5, fit_label(levels[k], label_chars), kInkColor, 9.0);
    }
    cumulative += counts[k];
  }
  p.show_grid(false)
      .x_limits(-0.6, 0.6)
      .y_limits(0.0, 1.3 * total)
      .x_ticks({0.0}, {""});
  p.text(0.0, 1.15 * total, names_[var], kLabelColor, 12.0);
  return p;
}

RPlot Splom::smooth_panel(std::size_t row, std::size_t col, RPlot p) const {
  const auto groups = grouped_pairs(col, row);
  std::size_t total = 0;
  for (const auto& group : groups) total += group.first.size();
  if (total == 0) {
    throw std::invalid_argument("Splom::panel: no complete rows for " + names_[col] + " vs " + names_[row]);
  }
  std::vector<std::optional<SmoothBand>> bands;
  for (const auto& group : groups) bands.push_back(smooth_band(group.first, group.second, span_, level_));
  const auto [x_lo, x_hi] = axis_range(col);
  const auto [y_lo, y_hi] = axis_range(row);
  // The renderer does not clip, so keep bands inside the panel: clamping the vertices clips a
  // band exactly at grid points, which are dense enough for the error to be invisible.
  for (auto& band : bands) {
    if (!band) continue;
    for (auto& v : band->lower) v = std::clamp(v, y_lo, y_hi);
    for (auto& v : band->upper) v = std::clamp(v, y_lo, y_hi);
  }

  // Prediction bands first, then the points over them, then the mean curves on top.
  for (std::size_t g = 0; g < groups.size(); ++g) {
    if (!bands[g]) continue;
    std::vector<double> bx = bands[g]->x;
    std::vector<double> by = bands[g]->upper;
    for (std::size_t k = bands[g]->x.size(); k-- > 0;) {
      bx.push_back(bands[g]->x[k]);
      by.push_back(bands[g]->lower[k]);
    }
    p.polygon(std::move(bx), std::move(by), with_alpha(group_color(g), kBandAlpha), true);
  }
  if (!point_colors_.empty()) {
    // Per-row colors: points keep their own color; the smooth is for all rows together.
    std::vector<double> x;
    std::vector<double> y;
    std::vector<RGB>    colors;
    for (std::size_t i = 0; i < rows_; ++i) {
      if (!std::isfinite(values_[col][i]) || !std::isfinite(values_[row][i])) continue;
      x.push_back(values_[col][i]);
      y.push_back(values_[row][i]);
      colors.push_back(with_alpha(point_color(i), kSmoothPointAlpha));
    }
    p.points_colored(std::move(x), std::move(y), std::move(colors), marker_size_);
  } else {
    for (std::size_t g = 0; g < groups.size(); ++g) {
      if (groups[g].first.empty()) continue;
      p.points(groups[g].first, groups[g].second, with_alpha(group_color(g), kSmoothPointAlpha), marker_size_);
    }
  }
  for (std::size_t g = 0; g < groups.size(); ++g) {
    if (!bands[g]) continue;
    // Draw only the runs of the mean curve that stay inside the panel.
    std::vector<double> run_x;
    std::vector<double> run_y;
    const auto flush = [&]() {
      if (run_x.size() >= 2) p.lines(run_x, run_y, group_color(g), 2.0);
      run_x.clear();
      run_y.clear();
    };
    for (std::size_t k = 0; k < bands[g]->x.size(); ++k) {
      const double y = bands[g]->fit[k];
      if (y < y_lo || y > y_hi) {
        flush();
        continue;
      }
      run_x.push_back(bands[g]->x[k]);
      run_y.push_back(y);
    }
    flush();
  }
  p.x_limits(x_lo, x_hi).y_limits(y_lo, y_hi);
  return p;
}

RPlot Splom::density2d_panel(std::size_t row, std::size_t col, RPlot p) const {
  const auto groups = grouped_pairs(col, row);
  std::size_t total = 0;
  for (const auto& group : groups) total += group.first.size();
  if (total == 0) {
    throw std::invalid_argument("Splom::panel: no complete rows for " + names_[col] + " vs " + names_[row]);
  }
  const auto [x_lo, x_hi] = axis_range(col);
  const auto [y_lo, y_hi] = axis_range(row);
  std::vector<double> gx(kDensityGrid2D);
  std::vector<double> gy(kDensityGrid2D);
  for (std::size_t k = 0; k < kDensityGrid2D; ++k) {
    const double t = static_cast<double>(k) / static_cast<double>(kDensityGrid2D - 1);
    gx[k] = x_lo + t * (x_hi - x_lo);
    gy[k] = y_lo + t * (y_hi - y_lo);
  }

  bool drawn = false;
  for (std::size_t g = 0; g < groups.size(); ++g) {
    const auto& xs = groups[g].first;
    const auto& ys = groups[g].second;
    if (xs.size() < kMinDensity2DPoints) continue;
    const double hx = kde2d_bandwidth(xs);
    const double hy = kde2d_bandwidth(ys);
    if (!(hx > 0.0) || !(hy > 0.0)) continue;
    // Product Gaussian kernel; the x and y factors are computed once per grid line.
    std::vector<std::vector<double>> kx(kDensityGrid2D, std::vector<double>(xs.size()));
    std::vector<std::vector<double>> ky(kDensityGrid2D, std::vector<double>(xs.size()));
    for (std::size_t k = 0; k < kDensityGrid2D; ++k) {
      for (std::size_t i = 0; i < xs.size(); ++i) {
        const double zx = (gx[k] - xs[i]) / hx;
        const double zy = (gy[k] - ys[i]) / hy;
        kx[k][i] = std::exp(-0.5 * zx * zx);
        ky[k][i] = std::exp(-0.5 * zy * zy);
      }
    }
    std::vector<std::vector<double>> z(kDensityGrid2D, std::vector<double>(kDensityGrid2D, 0.0));
    std::vector<double>              flat;
    flat.reserve(kDensityGrid2D * kDensityGrid2D);
    for (std::size_t a = 0; a < kDensityGrid2D; ++a) {
      for (std::size_t b = 0; b < kDensityGrid2D; ++b) {
        double sum = 0.0;
        for (std::size_t i = 0; i < xs.size(); ++i) sum += kx[a][i] * ky[b][i];
        z[a][b] = sum;
        flat.push_back(sum);
      }
    }
    const auto levels = hdr_levels(std::move(flat), kContourMass);
    const RGB  color = group_color(g);
    for (std::size_t l = 0; l < levels.size(); ++l) {
      std::vector<double> x0, y0, x1, y1;
      contour_segments(z, gx, gy, levels[l], x0, y0, x1, y1);
      if (x0.empty()) continue;
      // Inner (higher-density) contours are drawn more strongly.
      const double strength = 0.45 + 0.55 * static_cast<double>(l) / static_cast<double>(levels.size() - 1);
      p.segments(std::move(x0), std::move(y0), std::move(x1), std::move(y1), toward_white(color, strength), 1.3);
      drawn = true;
    }
  }
  if (!drawn) {
    // Too few or constant points for a density: show them instead so the panel is not empty.
    for (std::size_t g = 0; g < groups.size(); ++g) {
      if (!groups[g].first.empty()) p.points(groups[g].first, groups[g].second, group_color(g), marker_size_);
    }
  }
  p.x_limits(x_lo, x_hi).y_limits(y_lo, y_hi);
  return p;
}

RPlot Splom::violin_panel(std::size_t cat, std::size_t num, bool vertical, RPlot p) const {
  const auto& levels = levels_[cat];
  const auto& codes = codes_[cat];
  const auto& values = values_[num];

  std::vector<std::vector<double>> groups(levels.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (codes[i] < 0 || !std::isfinite(values[i])) continue;
    groups[static_cast<std::size_t>(codes[i])].push_back(values[i]);
  }

  // Evaluate every density first so all violins share one width scale (equal peak density,
  // equal width), which keeps their shapes comparable across levels.
  struct Violin {
    std::vector<double> at;      ///< evaluation points along the value axis
    std::vector<double> density; ///< kernel density at each point
  };
  std::vector<Violin> violins(levels.size());
  double              peak = 0.0;
  bool                any = false;
  for (std::size_t k = 0; k < levels.size(); ++k) {
    auto& group = groups[k];
    if (group.empty()) continue;
    any = true;
    std::sort(group.begin(), group.end());
    const double bw = silverman_bandwidth(group);
    if (!(bw > 0.0)) continue; // constant group: drawn as a flat bar below
    const double lo = group.front();
    const double hi = group.back();
    for (std::size_t g = 0; g < kViolinGrid; ++g) {
      const double at = lo + (hi - lo) * static_cast<double>(g) / static_cast<double>(kViolinGrid - 1);
      const double d = kernel_density(group, bw, at);
      violins[k].at.push_back(at);
      violins[k].density.push_back(d);
      peak = std::max(peak, d);
    }
  }
  if (!any) {
    throw std::invalid_argument("Splom::panel: no complete rows for " + names_[cat] + " vs " + names_[num]);
  }

  // (level offset, value) -> plot coordinates for either orientation.
  const auto add_polygon = [&](std::vector<double> across, std::vector<double> along, RGB color, bool filled) {
    if (vertical) {
      p.polygon(std::move(across), std::move(along), color, filled);
    } else {
      p.polygon(std::move(along), std::move(across), color, filled);
    }
  };
  std::vector<double> centres;
  std::vector<double> q1s;
  std::vector<double> q3s;
  std::vector<double> medians;
  for (std::size_t k = 0; k < levels.size(); ++k) {
    const auto& group = groups[k];
    if (group.empty()) continue;
    const double centre = static_cast<double>(k);
    std::vector<double> across;
    std::vector<double> along;
    if (violins[k].at.empty() || !(peak > 0.0)) {
      // A single value (or a constant group) has no spread: draw a thin flat bar.
      const double v = group.front();
      const double thickness = 0.01 * std::max(1.0, std::abs(v));
      across = {centre - kViolinHalfWidth, centre + kViolinHalfWidth, centre + kViolinHalfWidth,
                centre - kViolinHalfWidth};
      along = {v - thickness, v - thickness, v + thickness, v + thickness};
    } else {
      const auto& v = violins[k];
      for (std::size_t g = 0; g < v.at.size(); ++g) {
        across.push_back(centre + kViolinHalfWidth * v.density[g] / peak);
        along.push_back(v.at[g]);
      }
      for (std::size_t g = v.at.size(); g-- > 0;) {
        across.push_back(centre - kViolinHalfWidth * v.density[g] / peak);
        along.push_back(v.at[g]);
      }
    }
    add_polygon(across, along, level_color(cat, k), true);
    add_polygon(std::move(across), std::move(along), kInkColor, false);
    centres.push_back(centre);
    q1s.push_back(quantile7(group, 0.25));
    q3s.push_back(quantile7(group, 0.75));
    medians.push_back(quantile7(group, 0.5));
  }

  const double      plot_w = static_cast<double>(p.width()) - p.margin_left() - p.margin_right();
  const double      span = static_cast<double>(levels.size()) + 0.2;
  const std::size_t x_chars = chars_in(plot_w / span - 2.0);
  std::vector<double>      positions;
  std::vector<std::string> labels;
  for (std::size_t k = 0; k < levels.size(); ++k) {
    positions.push_back(static_cast<double>(k));
    labels.push_back(fit_label(levels[k], vertical ? x_chars : kMaxYLabelChars));
  }
  const double lo = -0.6;
  const double hi = static_cast<double>(levels.size()) - 0.4;
  // Interquartile range as a thick bar, median as a white dot, like a box plot inside each violin.
  if (vertical) {
    p.segments(centres, q1s, centres, q3s, kInkColor, 3.0);
    p.points(centres, medians, RGB{255, 255, 255}, 2.0);
    p.x_limits(lo, hi).x_ticks(std::move(positions), std::move(labels));
    const auto [v_lo, v_hi] = axis_range(num);
    p.y_limits(v_lo, v_hi);
  } else {
    p.segments(q1s, centres, q3s, centres, kInkColor, 3.0);
    p.points(medians, centres, RGB{255, 255, 255}, 2.0);
    p.y_limits(lo, hi).y_ticks(std::move(positions), std::move(labels));
    const auto [v_lo, v_hi] = axis_range(num);
    p.x_limits(v_lo, v_hi);
  }
  return p;
}

RPlot Splom::mosaic_panel(std::size_t row, std::size_t col, RPlot p) const {
  // Columns of the mosaic are the levels of the x variable; tiles stack the y levels.
  const auto&       x_levels = levels_[col];
  const auto&       y_levels = levels_[row];
  const auto&       x_codes = codes_[col];
  const auto&       y_codes = codes_[row];
  std::vector<std::vector<double>> table(x_levels.size(), std::vector<double>(y_levels.size(), 0.0));
  double total = 0.0;
  for (std::size_t i = 0; i < x_codes.size(); ++i) {
    if (x_codes[i] < 0 || y_codes[i] < 0) continue;
    table[static_cast<std::size_t>(x_codes[i])][static_cast<std::size_t>(y_codes[i])] += 1.0;
    total += 1.0;
  }
  if (total == 0.0) {
    throw std::invalid_argument("Splom::panel: no complete rows for " + names_[col] + " vs " + names_[row]);
  }
  // Levels absent from this panel's complete rows would make empty margins; drop them.
  std::vector<std::size_t> xs;
  std::vector<std::size_t> ys;
  std::vector<double>      x_margin(x_levels.size(), 0.0);
  std::vector<double>      y_margin(y_levels.size(), 0.0);
  for (std::size_t a = 0; a < x_levels.size(); ++a) {
    for (std::size_t b = 0; b < y_levels.size(); ++b) {
      x_margin[a] += table[a][b];
      y_margin[b] += table[a][b];
    }
  }
  for (std::size_t a = 0; a < x_levels.size(); ++a) {
    if (x_margin[a] > 0.0) xs.push_back(a);
  }
  for (std::size_t b = 0; b < y_levels.size(); ++b) {
    if (y_margin[b] > 0.0) ys.push_back(b);
  }

  // Chi-squared test of independence on the reduced table (Yates-corrected when 2x2, like R).
  double p_value = std::numeric_limits<double>::quiet_NaN();
  if (xs.size() >= 2 && ys.size() >= 2) {
    std::vector<double> flat;
    for (std::size_t a : xs) {
      for (std::size_t b : ys) flat.push_back(table[a][b]);
    }
    try {
      p_value = stats::chi_squared_test_independence(flat, xs.size(), ys.size()).p_value;
    } catch (const std::exception&) {
      p_value = std::numeric_limits<double>::quiet_NaN();
    }
  }

  const double x_space = 1.0 - kMosaicGap * static_cast<double>(xs.size() - 1);
  const double y_space = 1.0 - kMosaicGap * static_cast<double>(ys.size() - 1);
  const double plot_w = static_cast<double>(p.width()) - p.margin_left() - p.margin_right();
  std::vector<double>      x_positions;
  std::vector<std::string> x_labels;
  std::vector<double>      y_positions;
  std::vector<std::string> y_labels;
  double x0 = 0.0;
  for (std::size_t ai = 0; ai < xs.size(); ++ai) {
    const std::size_t a = xs[ai];
    const double      width = x_space * x_margin[a] / total;
    double            y0 = 0.0;
    for (std::size_t bi = 0; bi < ys.size(); ++bi) {
      const std::size_t b = ys[bi];
      const double      height = y_space * table[a][b] / x_margin[a];
      const double      expected = x_margin[a] * y_margin[b] / total;
      const double      residual = (table[a][b] - expected) / std::sqrt(expected);
      if (table[a][b] > 0.0) {
        add_rect(p, x0, x0 + width, y0, y0 + height, std::isnan(p_value) ? kMosaicNeutral : mosaic_shade(residual));
      }
      y0 += height + kMosaicGap;
    }
    x_positions.push_back(x0 + width * 0.5);
    x_labels.push_back(fit_label(x_levels[a], chars_in(width * plot_w - 2.0)));
    x0 += width + kMosaicGap;
  }
  // Label the vertical axis at the overall (marginal) share of each y level: R labels it from the
  // first column's tiles, but those can be thin enough for labels to collide.
  double y0 = 0.0;
  for (std::size_t b : ys) {
    const double height = y_space * y_margin[b] / total;
    y_positions.push_back(y0 + height * 0.5);
    y_labels.push_back(fit_label(y_levels[b], kMaxYLabelChars));
    y0 += height + kMosaicGap;
  }
  p.show_grid(false)
      .x_limits(0.0, 1.0)
      .y_limits(0.0, 1.14)
      .x_ticks(std::move(x_positions), std::move(x_labels))
      .y_ticks(std::move(y_positions), std::move(y_labels));
  p.text(0.5, 1.07, std::isnan(p_value) ? std::string("chi2 n/a") : format_p_value(p_value), kLabelColor, 9.0);
  return p;
}

RLayout Splom::layout() const {
  const std::size_t n = names_.size();
  RLayout           grid = RLayout::create(n, n);
  grid.size(width_, height_).title(title_);
  for (std::size_t row = 0; row < n; ++row) {
    for (std::size_t col = 0; col < n; ++col) {
      grid.add(panel(row, col));
    }
  }
  return grid;
}

std::string Splom::to_svg() const { return layout().to_svg(); }

void Splom::save_svg(const std::string& path) const {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Splom::save_svg: could not open output file");
  }
  out << to_svg();
}

void Splom::save(const std::string& path) const {
  const auto ext = std::filesystem::path(path).extension().string();
  if (ext == ".svg" || ext.empty()) {
    save_svg(path);
  } else {
    layout().save(path);
  }
}

void Splom::show(const std::string& title_hint) const {
  layout().show(title_hint.empty() ? (title_.empty() ? "datamunge splom" : title_) : title_hint);
}

void Splom::view(const std::string& title_hint) const { show(title_hint); }

Figure::Figure(const Splom& splom) : Figure(splom.layout()) {}

} // namespace datamunge::graphics

namespace datamunge {

graphics::Splom splom(const dstruct::DataFrame& data, std::vector<std::string> columns, const std::string& group_by) {
  graphics::Splom matrix = graphics::Splom::create(data, std::move(columns));
  if (!group_by.empty()) matrix.group_by(group_by);
  matrix.show();
  return matrix;
}

} // namespace datamunge
