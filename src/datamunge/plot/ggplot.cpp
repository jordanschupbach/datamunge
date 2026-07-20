#include <datamunge/plot/ggplot.hpp>

#include <datamunge/stats/lm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::plot {
namespace {

using dstruct::DataFrame;

constexpr std::array<RGB, 8> kGGDefaultPalette = {{
    {248, 118, 109},
    {124, 174, 0},
    {0, 191, 196},
    {199, 124, 255},
    {231, 107, 243},
    {0, 176, 246},
    {245, 133, 24},
    {148, 148, 148},
}};

std::string format_number(double v) {
  std::ostringstream out;
  out << v;
  return out.str();
}

std::vector<double> numeric_values(const DataFrame& df, const std::string& column, const char* fn) {
  if (column.empty()) {
    throw std::invalid_argument(std::string(fn) + ": aesthetic mapping is missing a required column");
  }
  if (df.column_type(column) != DataFrame::ColumnType::Numeric) {
    throw std::invalid_argument(std::string(fn) + ": column \"" + column + "\" must be numeric");
  }
  return df.numeric_column(column).values();
}

/// Splits `df` into groups keyed by the (string or numeric) values of `column`, sorted by key.
/// An empty `column` yields a single ungrouped entry with an empty label.
std::vector<std::pair<std::string, DataFrame>> split_by_group(const DataFrame& df, const std::string& column) {
  if (column.empty()) {
    return {{"", df}};
  }
  std::vector<std::pair<std::string, DataFrame>> groups;
  if (df.column_type(column) == DataFrame::ColumnType::String) {
    std::set<std::string> labels;
    for (const auto& v : df.string_column(column).values()) {
      labels.insert(v);
    }
    for (const auto& label : labels) {
      groups.emplace_back(label, df.filter([&](const DataFrame::Row& row) { return row.get_string(column) == label; }));
    }
  } else {
    std::set<double> keys;
    for (double v : df.numeric_column(column).values()) {
      keys.insert(v);
    }
    for (double key : keys) {
      groups.emplace_back(format_number(key),
                           df.filter([&](const DataFrame::Row& row) { return row.get_double(column) == key; }));
    }
  }
  return groups;
}

/// Sorts the rows of a two-column (x, y) pair ascending by x. Used by geoms (line/area/ribbon/
/// smooth) whose rendering assumes points are connected in x order.
void sort_by_x(std::vector<double>& x, std::vector<double>& y) {
  std::vector<std::size_t> order(x.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return x[a] < x[b]; });
  std::vector<double> sx(x.size());
  std::vector<double> sy(y.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    sx[i] = x[order[i]];
    sy[i] = y[order[i]];
  }
  x = std::move(sx);
  y = std::move(sy);
}

/// Sorts x/ymin/ymax together ascending by x, using one shared permutation so ymin and ymax
/// stay aligned with the row each originally came from (unlike two independent sort_by_x calls,
/// which would silently misalign one of the two y-arrays once x itself has been reordered).
void sort_by_x(std::vector<double>& x, std::vector<double>& ymin, std::vector<double>& ymax) {
  std::vector<std::size_t> order(x.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return x[a] < x[b]; });
  std::vector<double> sx(x.size());
  std::vector<double> smin(ymin.size());
  std::vector<double> smax(ymax.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    sx[i] = x[order[i]];
    smin[i] = ymin[order[i]];
    smax[i] = ymax[order[i]];
  }
  x = std::move(sx);
  ymin = std::move(smin);
  ymax = std::move(smax);
}

/// Gaussian KDE evaluated at `grid_size` points, bandwidth chosen via Silverman's rule of thumb.
std::pair<std::vector<double>, std::vector<double>> gaussian_kde(std::vector<double> data,
                                                                  std::size_t         grid_size = 200) {
  const auto n = data.size();
  std::sort(data.begin(), data.end());
  const double mean = std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(n);
  double       variance = 0.0;
  for (double v : data) {
    variance += (v - mean) * (v - mean);
  }
  variance /= static_cast<double>(n - 1);
  const double sd = std::sqrt(variance);
  const double q1 = data[static_cast<std::size_t>(0.25 * static_cast<double>(n - 1))];
  const double q3 = data[static_cast<std::size_t>(0.75 * static_cast<double>(n - 1))];
  const double iqr = q3 - q1;
  const double spread = std::min(sd, iqr / 1.34 > 0.0 ? iqr / 1.34 : sd);
  const double bandwidth = 0.9 * (spread > 0.0 ? spread : 1.0) * std::pow(static_cast<double>(n), -1.0 / 5.0);

  const double lo = data.front() - 3.0 * bandwidth;
  const double hi = data.back() + 3.0 * bandwidth;
  std::vector<double> grid(grid_size);
  std::vector<double> density(grid_size, 0.0);
  for (std::size_t g = 0; g < grid_size; ++g) {
    grid[g] = lo + (hi - lo) * static_cast<double>(g) / static_cast<double>(grid_size - 1);
    double sum = 0.0;
    for (double v : data) {
      const double z = (grid[g] - v) / bandwidth;
      sum += std::exp(-0.5 * z * z);
    }
    density[g] = sum / (static_cast<double>(n) * bandwidth * std::sqrt(2.0 * 3.14159265358979323846));
  }
  return {grid, density};
}

} // namespace

GGPlot GGPlot::create(const dstruct::DataFrame& data, Aes mapping) {
  GGPlot g;
  g.data_ = data;
  g.mapping_ = std::move(mapping);
  return g;
}

GGPlot& GGPlot::geom_point(RGB color, double size) {
  layers_.push_back(Layer{GeomKind::Point, color, size, 0, "", ""});
  return *this;
}

GGPlot& GGPlot::geom_line(RGB color, double width) {
  layers_.push_back(Layer{GeomKind::Line, color, width, 0, "", ""});
  return *this;
}

GGPlot& GGPlot::geom_bar(RGB color) {
  layers_.push_back(Layer{GeomKind::Bar, color, 0.0, 0, "", ""});
  return *this;
}

GGPlot& GGPlot::geom_col(RGB color) {
  layers_.push_back(Layer{GeomKind::Col, color, 0.0, 0, "", ""});
  return *this;
}

GGPlot& GGPlot::geom_histogram(std::size_t bins, RGB color) {
  layers_.push_back(Layer{GeomKind::Histogram, color, 0.0, bins, "", ""});
  return *this;
}

GGPlot& GGPlot::geom_boxplot(RGB color) {
  layers_.push_back(Layer{GeomKind::Boxplot, color, 0.0, 0, "", ""});
  return *this;
}

GGPlot& GGPlot::geom_smooth(RGB color) {
  layers_.push_back(Layer{GeomKind::Smooth, color, 0.0, 0, "", ""});
  return *this;
}

GGPlot& GGPlot::geom_area(RGB color) {
  layers_.push_back(Layer{GeomKind::Area, color, 0.0, 0, "", ""});
  return *this;
}

GGPlot& GGPlot::geom_ribbon(std::string ymin_col, std::string ymax_col, RGB color) {
  layers_.push_back(Layer{GeomKind::Ribbon, color, 0.0, 0, std::move(ymin_col), std::move(ymax_col)});
  return *this;
}

GGPlot& GGPlot::geom_density(RGB color) {
  layers_.push_back(Layer{GeomKind::Density, color, 0.0, 0, "", ""});
  return *this;
}

GGPlot& GGPlot::facet_wrap(std::string column, std::size_t ncol) {
  facet_column_ = std::move(column);
  facet_ncol_ = ncol;
  return *this;
}

GGPlot& GGPlot::theme_minimal() {
  theme_ = GGTheme::Minimal;
  return *this;
}

GGPlot& GGPlot::theme_bw() {
  theme_ = GGTheme::BW;
  return *this;
}

GGPlot& GGPlot::theme_classic() {
  theme_ = GGTheme::Classic;
  return *this;
}

GGPlot& GGPlot::scale_color_manual(std::vector<RGB> values) {
  palette_override_ = std::move(values);
  return *this;
}

GGPlot& GGPlot::labs(std::string title, std::string x, std::string y) {
  title_override_ = std::move(title);
  x_label_override_ = std::move(x);
  y_label_override_ = std::move(y);
  return *this;
}

void GGPlot::apply_theme(Plot& panel) const {
  switch (theme_) {
    case GGTheme::Gray:
      break; // Plot's own defaults (white background, light gridlines) apply unchanged.
    case GGTheme::Minimal:
      panel.background({255, 255, 255}).grid_color({229, 231, 235}).axis_color({156, 163, 175}).show_grid(true);
      break;
    case GGTheme::BW:
      panel.background({255, 255, 255}).grid_color({209, 213, 219}).axis_color({17, 24, 39}).show_grid(true);
      break;
    case GGTheme::Classic:
      panel.background({255, 255, 255}).axis_color({17, 24, 39}).show_grid(false);
      break;
  }
}

RPlot GGPlot::build_panel(const dstruct::DataFrame& panel_data, const std::string& panel_title) const {
  RPlot       panel = RPlot::create();
  const auto& palette = palette_override_.empty() ? std::vector<RGB>(kGGDefaultPalette.begin(), kGGDefaultPalette.end())
                                                    : palette_override_;
  const std::string group_column =
      !mapping_.group.empty() ? mapping_.group : (!mapping_.color.empty() ? mapping_.color : mapping_.fill);

  for (const auto& layer : layers_) {
    switch (layer.kind) {
      case GeomKind::Point: {
        auto           groups = split_by_group(panel_data, group_column);
        std::size_t    idx = 0;
        for (auto& [label, gdf] : groups) {
          auto xs = numeric_values(gdf, mapping_.x, "geom_point");
          auto ys = numeric_values(gdf, mapping_.y, "geom_point");
          const RGB c = groups.size() > 1 ? palette[idx % palette.size()] : layer.color;
          panel.points(std::move(xs), std::move(ys), label, c, layer.size);
          ++idx;
        }
        break;
      }
      case GeomKind::Line: {
        auto           groups = split_by_group(panel_data, group_column);
        std::size_t    idx = 0;
        for (auto& [label, gdf] : groups) {
          auto xs = numeric_values(gdf, mapping_.x, "geom_line");
          auto ys = numeric_values(gdf, mapping_.y, "geom_line");
          sort_by_x(xs, ys);
          const RGB c = groups.size() > 1 ? palette[idx % palette.size()] : layer.color;
          panel.line(std::move(xs), std::move(ys), label, c, layer.size);
          ++idx;
        }
        break;
      }
      case GeomKind::Bar: {
        if (mapping_.x.empty()) {
          throw std::invalid_argument("geom_bar: aes(x=...) is required");
        }
        std::map<std::string, double> counts;
        if (panel_data.column_type(mapping_.x) == DataFrame::ColumnType::String) {
          for (const auto& v : panel_data.string_column(mapping_.x).values()) {
            counts[v] += 1.0;
          }
        } else {
          for (double v : panel_data.numeric_column(mapping_.x).values()) {
            counts[format_number(v)] += 1.0;
          }
        }
        std::vector<double>      positions;
        std::vector<double>      heights;
        std::vector<std::string> labels;
        for (const auto& [label, count] : counts) {
          positions.push_back(static_cast<double>(labels.size()));
          heights.push_back(count);
          labels.push_back(label);
        }
        panel.bars(std::move(positions), std::move(heights), "", layer.color, 0.9);
        panel.x_tick_labels(std::move(labels));
        break;
      }
      case GeomKind::Col: {
        if (mapping_.x.empty() || mapping_.y.empty()) {
          throw std::invalid_argument("geom_col: aes(x=..., y=...) is required");
        }
        auto ys = numeric_values(panel_data, mapping_.y, "geom_col");
        if (panel_data.column_type(mapping_.x) == DataFrame::ColumnType::String) {
          const auto&               xs_str = panel_data.string_column(mapping_.x).values();
          std::vector<std::string>  labels(xs_str.begin(), xs_str.end());
          std::vector<double>       positions(labels.size());
          for (std::size_t i = 0; i < positions.size(); ++i) {
            positions[i] = static_cast<double>(i);
          }
          panel.bars(std::move(positions), std::move(ys), "", layer.color, 0.9);
          panel.x_tick_labels(std::move(labels));
        } else {
          auto xs = numeric_values(panel_data, mapping_.x, "geom_col");
          panel.bars(std::move(xs), std::move(ys), "", layer.color, 0.8);
        }
        break;
      }
      case GeomKind::Histogram: {
        auto data = numeric_values(panel_data, mapping_.x, "geom_histogram");
        if (data.empty()) {
          throw std::invalid_argument("geom_histogram: data must not be empty");
        }
        double lo = *std::min_element(data.begin(), data.end());
        double hi = *std::max_element(data.begin(), data.end());
        if (hi <= lo) {
          hi = lo + 1.0;
        }
        const double bin_width = (hi - lo) / static_cast<double>(layer.bins);
        std::vector<double> counts(layer.bins, 0.0);
        for (double value : data) {
          auto idx = static_cast<std::ptrdiff_t>((value - lo) / bin_width);
          idx = std::clamp<std::ptrdiff_t>(idx, 0, static_cast<std::ptrdiff_t>(layer.bins) - 1);
          counts[static_cast<std::size_t>(idx)] += 1.0;
        }
        std::vector<double> centers(layer.bins);
        for (std::size_t i = 0; i < layer.bins; ++i) {
          centers[i] = lo + (static_cast<double>(i) + 0.5) * bin_width;
        }
        panel.bars(std::move(centers), std::move(counts), "", layer.color, bin_width);
        break;
      }
      case GeomKind::Boxplot: {
        if (mapping_.y.empty()) {
          throw std::invalid_argument("geom_boxplot: aes(y=...) is required");
        }
        auto groups = split_by_group(panel_data, mapping_.x);
        std::vector<std::string> labels;
        for (std::size_t i = 0; i < groups.size(); ++i) {
          auto&                y = groups[i].second;
          std::vector<double>  values = numeric_values(y, mapping_.y, "geom_boxplot");
          std::sort(values.begin(), values.end());
          const auto quantile = [&](double p) {
            const double h = (static_cast<double>(values.size()) - 1.0) * p;
            const auto   lo = static_cast<std::size_t>(std::floor(h));
            const auto   hi = static_cast<std::size_t>(std::ceil(h));
            return values[lo] + (h - static_cast<double>(lo)) * (values[hi] - values[lo]);
          };
          const double q1 = quantile(0.25);
          const double median = quantile(0.5);
          const double q3 = quantile(0.75);
          const double iqr = q3 - q1;
          const double lower_fence = q1 - 1.5 * iqr;
          const double upper_fence = q3 + 1.5 * iqr;
          double       whisker_lo = values.front();
          double       whisker_hi = values.back();
          std::vector<double> outliers;
          for (double v : values) {
            if (v < lower_fence || v > upper_fence) {
              outliers.push_back(v);
            } else {
              whisker_lo = std::min(whisker_lo, v);
            }
          }
          for (auto it = values.rbegin(); it != values.rend(); ++it) {
            if (*it <= upper_fence) {
              whisker_hi = *it;
              break;
            }
          }
          for (double v : values) {
            if (v >= lower_fence) {
              whisker_lo = v;
              break;
            }
          }
          panel.box(static_cast<double>(i), whisker_lo, q1, median, q3, whisker_hi, outliers, layer.color, 0.6);
          labels.push_back(groups[i].first);
        }
        if (!mapping_.x.empty()) {
          panel.x_tick_labels(std::move(labels));
        }
        break;
      }
      case GeomKind::Smooth: {
        auto        groups = split_by_group(panel_data, group_column);
        std::size_t idx = 0;
        for (auto& [label, gdf] : groups) {
          auto xs = numeric_values(gdf, mapping_.x, "geom_smooth");
          auto ys = numeric_values(gdf, mapping_.y, "geom_smooth");
          if (xs.size() < 2) {
            continue;
          }
          DataFrame fit_data;
          fit_data.add_column(mapping_.x, xs);
          fit_data.add_column(mapping_.y, ys);
          const stats::LM lm(fit_data, mapping_.y + " ~ " + mapping_.x);
          const double     x_min = *std::min_element(xs.begin(), xs.end());
          const double     x_max = *std::max_element(xs.begin(), xs.end());
          DataFrame newdata;
          newdata.add_column(mapping_.x, std::vector<double>{x_min, x_max});
          const auto fitted = lm.predict(newdata);
          const RGB  c = groups.size() > 1 ? palette[idx % palette.size()] : layer.color;
          panel.line({x_min, x_max}, fitted, label, c, 2.0);
          ++idx;
        }
        break;
      }
      case GeomKind::Area: {
        auto xs = numeric_values(panel_data, mapping_.x, "geom_area");
        auto ys = numeric_values(panel_data, mapping_.y, "geom_area");
        sort_by_x(xs, ys);
        std::vector<double> px;
        std::vector<double> py;
        px.push_back(xs.front());
        py.push_back(0.0);
        for (std::size_t i = 0; i < xs.size(); ++i) {
          px.push_back(xs[i]);
          py.push_back(ys[i]);
        }
        px.push_back(xs.back());
        py.push_back(0.0);
        panel.polygon(std::move(px), std::move(py), layer.color, true);
        break;
      }
      case GeomKind::Ribbon: {
        auto xs = numeric_values(panel_data, mapping_.x, "geom_ribbon");
        auto ymin = numeric_values(panel_data, layer.ymin_col, "geom_ribbon");
        auto ymax = numeric_values(panel_data, layer.ymax_col, "geom_ribbon");
        sort_by_x(xs, ymin, ymax);
        std::vector<double> px;
        std::vector<double> py;
        for (std::size_t i = 0; i < xs.size(); ++i) {
          px.push_back(xs[i]);
          py.push_back(ymax[i]);
        }
        for (std::size_t i = xs.size(); i-- > 0;) {
          px.push_back(xs[i]);
          py.push_back(ymin[i]);
        }
        panel.polygon(std::move(px), std::move(py), layer.color, true);
        break;
      }
      case GeomKind::Density: {
        auto data = numeric_values(panel_data, mapping_.x, "geom_density");
        if (data.size() < 2) {
          throw std::invalid_argument("geom_density: need at least two values");
        }
        auto [grid, density] = gaussian_kde(data);
        std::vector<double> px;
        std::vector<double> py;
        px.push_back(grid.front());
        py.push_back(0.0);
        for (std::size_t i = 0; i < grid.size(); ++i) {
          px.push_back(grid[i]);
          py.push_back(density[i]);
        }
        px.push_back(grid.back());
        py.push_back(0.0);
        panel.polygon(px, py, layer.color, true);
        panel.line(std::move(grid), std::move(density), "", layer.color, 1.5);
        break;
      }
    }
  }

  panel.title(!title_override_.empty() ? title_override_ : panel_title);
  panel.x_label(!x_label_override_.empty() ? x_label_override_ : mapping_.x);
  panel.y_label(!y_label_override_.empty() ? y_label_override_ : mapping_.y);
  apply_theme(panel);
  return panel;
}

void GGPlot::save_svg(const std::string& path) const {
  if (facet_column_.empty()) {
    build_panel(data_, "").save_svg(path);
    return;
  }

  auto facets = split_by_group(data_, facet_column_);
  if (facets.empty()) {
    throw std::runtime_error("GGPlot::save: facet_wrap column has no values");
  }
  const std::size_t cols =
      facet_ncol_ > 0 ? facet_ncol_ : static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(facets.size()))));
  const std::size_t rows = (facets.size() + cols - 1) / cols;

  auto layout = RLayout::create(rows, cols);
  for (const auto& [label, panel_data] : facets) {
    RPlot panel = build_panel(panel_data, label);
    layout.add(panel);
  }
  layout.save_svg(path);
}

void GGPlot::save(const std::string& path) const {
  save_svg(path);
}

} // namespace datamunge::plot
