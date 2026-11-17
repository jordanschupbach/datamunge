#pragma once
/// @file gram_schmidt_analysis.hpp
/// @brief Native orthogonality diagnostics, controlled rows and centered dataframe-column inputs.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/gram_schmidt.hpp>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/plot/plot.hpp>
#include <map>
#include <stdexcept>
#include <string>

namespace datamunge::algorithms {
/// @brief Build three nearly parallel independent rows in four dimensions.
/// Rows are (1,e,0,0), (1,0,e,0), (1,0,0,e). Small e isolates loss of
/// orthogonality; absolute-threshold skipping is a separate effect.
/// @param e Finite positive perturbation.
/// @return Owned 3-by-4 rectangular rows.
/// @throws std::invalid_argument If e is nonpositive or nonfinite.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto v=dm::algorithms::gram_schmidt_near_parallel(.01);
/// std::print("{} {}\n",v.size(),v[0].size()); // 3 4
/// @endcode
/// @see gram_schmidt_run()
inline std::vector<std::vector<double>> gram_schmidt_near_parallel(double e) {
  if (!std::isfinite(e) || e <= 0)
    throw std::invalid_argument("positive finite perturbation required");
  return {{1, e, 0, 0}, {1, 0, e, 0}, {1, 0, 0, e}};
}
/// @brief Measure a supplied basis's Gram defect and direct orthogonal-projection residual.
/// Computes max|q_i dot q_j - identity_ij| and ||V - sum_i(v dot q_i)q_i||F/||V||F
/// with ordinary double dots and hypot norm accumulation. Rounded diagnostics
/// can underreport a true stored-basis defect, especially below machine precision;
/// they are not interval bounds or certified accuracy. For a nonorthogonal
/// basis this is NOT a least-squares or best-span residual. Empty basis has
/// vacuous zero Gram defect; a nonzero input then has projection residual one.
/// Zero input norm defines relative residual zero. All rows must have a common
/// width; inputs are unmodified, no references survive. O(r²d+krd) work.
/// @param vectors Original finite rectangular row vectors.
/// @param basis Supplied finite rectangular rows, same width when both are nonempty.
/// @return One owned row: inputs, dimension, accepted, max_gram_error and
/// relative_projection_residual.
/// @throws std::invalid_argument For malformed or nonfinite rows.
/// @throws std::overflow_error If diagnostic arithmetic becomes nonfinite.
/// @throws std::bad_alloc On output/temporary allocation failure.
/// @throws std::length_error On container exhaustion.
/// @par Example
/// @code
/// const auto v=std::vector<std::vector<double>>{{1,0},{0,1}};
/// const auto d=dm::algorithms::gram_schmidt_quality(v,v);
/// std::print("{}\n",d.get_column<double>("max_gram_error").values()[0]); // 0
/// @endcode
/// @see gram_schmidt_run()
inline dstruct::DataFrame gram_schmidt_quality(
    const std::vector<std::vector<double>>& vectors,
    const std::vector<std::vector<double>>& basis) {
  const std::size_t d =
      !vectors.empty() ? vectors[0].size() : (!basis.empty() ? basis[0].size() : 0);
  for (const auto* m : {&vectors, &basis})
    for (const auto& row : *m) {
      if (row.size() != d)
        throw std::invalid_argument("quality needs equal row widths");
      for (double x : row)
        if (!std::isfinite(x))
          throw std::invalid_argument("quality needs finite rows");
    }
  double error = 0;
  for (std::size_t i = 0; i < basis.size(); ++i)
    for (std::size_t j = 0; j < basis.size(); ++j) {
      double dot = 0;
      for (std::size_t k = 0; k < d; ++k)
        dot += basis[i][k] * basis[j][k];
      if (!std::isfinite(dot))
        throw std::overflow_error("quality Gram overflow");
      error = std::max(error, std::abs(dot - (i == j ? 1. : 0.)));
    }
  double input_norm = 0, residual_norm = 0;
  for (const auto& row : vectors) {
    std::vector<double> projection(d, 0);
    for (const auto& q : basis) {
      double dot = 0;
      for (std::size_t k = 0; k < d; ++k)
        dot += row[k] * q[k];
      for (std::size_t k = 0; k < d; ++k)
        projection[k] += dot * q[k];
    }
    for (std::size_t k = 0; k < d; ++k) {
      input_norm    = std::hypot(input_norm, row[k]);
      residual_norm = std::hypot(residual_norm, row[k] - projection[k]);
    }
  }
  if (!std::isfinite(input_norm) || !std::isfinite(residual_norm))
    throw std::overflow_error("quality projection overflow");
  dstruct::DataFrame t;
  t.add_column("inputs", std::vector<double>{double(vectors.size())});
  t.add_column("dimension", std::vector<double>{double(d)});
  t.add_column("accepted", std::vector<double>{double(basis.size())});
  t.add_column("max_gram_error", std::vector<double>{error});
  t.add_column(
      "relative_projection_residual",
      std::vector<double>{input_norm ? residual_norm / input_norm : 0});
  return t;
}
/// @brief Copy selected numeric dataframe columns as optionally centered input vectors.
/// Each selected variable becomes a row; original dataframe rows become vector
/// coordinates. Column order is preserved, duplicates allowed. Centering subtracts
/// the ordinary sequential arithmetic mean, without scaling variances. Finite
/// complete numeric columns are required; no imputation or labels enter the basis.
/// No dataframe storage is retained or mutated. Empty selection returns empty.
/// @param data Borrowed dataframe containing requested double columns.
/// @param columns Ordered numeric column names.
/// @param center Subtract each column mean, default true; empty columns remain empty.
/// @return Owned rectangular vectors, one per requested variable.
/// @throws std::invalid_argument For null or nonfinite values.
/// @throws std::out_of_range For a missing column.
/// @throws std::invalid_argument For an incompatible column type.
/// @throws std::overflow_error If centering arithmetic becomes nonfinite.
/// @throws std::bad_alloc On allocation failure.
/// @throws std::length_error On container exhaustion.
/// @par Example
/// @code
/// dm::dstruct::DataFrame d; d.add_column("x",std::vector<double>{1,3});
/// const auto v=dm::algorithms::gram_schmidt_columns(d,{"x"});
/// std::print("{} {}\n",v[0][0],v[0][1]); // -1 1
/// @endcode
/// @see gram_schmidt_run(), gram_schmidt_scores_plot()
inline std::vector<std::vector<double>> gram_schmidt_columns(
    const dstruct::DataFrame&       data,
    const std::vector<std::string>& columns,
    bool                            center = true) {
  std::vector<std::vector<double>> result;
  for (const auto& name : columns) {
    const auto&         column = data.get_column<double>(name);
    std::vector<double> row;
    row.reserve(column.size());
    for (std::size_t i = 0; i < column.size(); ++i)
      row.push_back(column.at(i));
    double sum = 0;
    for (double x : row) {
      if (!std::isfinite(x))
        throw std::invalid_argument("centered columns need finite values");
      if (center)
        sum += x;
    }
    if (center && !row.empty()) {
      const double mean = sum / row.size();
      if (!std::isfinite(mean))
        throw std::overflow_error("column mean overflow");
      for (double& x : row) {
        x -= mean;
        if (!std::isfinite(x))
          throw std::overflow_error("centered value overflow");
      }
    }
    result.push_back(std::move(row));
  }
  return result;
}
/// @brief Plot the first two accepted basis rows as per-observation coordinates, grouped by labels.
/// Requires at least two basis rows and a label per coordinate. This is an
/// order-dependent orthogonal feature representation, not PCA, a fitted classifier
/// or held-out prediction. All points are shown; labels affect colors only.
/// Copies coordinates and grouping into an owned native plot.
/// @param run Actual result whose first two rows have matching finite widths.
/// @param labels Borrowed display labels aligned with vector coordinates.
/// @return Owned editable scatter plot.
/// @throws std::invalid_argument For missing rows, unequal widths, label mismatch or nonfinite
/// values.
/// @throws std::bad_alloc On allocation failure.
/// @throws std::length_error On container exhaustion.
/// @par Example
/// @code
/// const auto r=dm::algorithms::gram_schmidt_run({{1,0},{0,1}});
/// dm::algorithms::gram_schmidt_scores_plot(r,{"a","b"}).save_svg("build/gram_scores.svg");
/// @endcode
/// @see gram_schmidt_columns()
inline graphics::RPlot gram_schmidt_scores_plot(
    const GramSchmidtRun&           run,
    const std::vector<std::string>& labels) {
  if (run.basis.size() < 2 || run.basis[0].size() != run.basis[1].size() ||
      run.basis[0].size() != labels.size() || labels.empty())
    throw std::invalid_argument("score plot needs two rows and matching labels");
  std::map<std::string, std::pair<std::vector<double>, std::vector<double>>> groups;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    double x = run.basis[0][i], y = run.basis[1][i];
    if (!std::isfinite(x) || !std::isfinite(y))
      throw std::invalid_argument("finite scores required");
    groups[labels[i]].first.push_back(x);
    groups[labels[i]].second.push_back(y);
  }
  auto                p        = graphics::RPlot::create();
  const graphics::RGB colors[] = {{37, 99, 235}, {220, 38, 38}, {5, 150, 105}, {202, 138, 4}};
  std::size_t         i        = 0;
  for (const auto& [name, xy] : groups)
    p.points(xy.first, xy.second, name, colors[i++ % 4], 4);
  p.title("Ordered Gram-Schmidt coordinates: first two directions")
      .x_label("q1 coordinate per observation")
      .y_label("q2 coordinate per observation");
  return p;
}
} // namespace datamunge::algorithms
