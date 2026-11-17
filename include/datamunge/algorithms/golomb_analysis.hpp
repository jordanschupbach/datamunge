#pragma once
/// @file golomb_analysis.hpp
/// @brief Native Golomb length profiles and explicitly parameterized geometric-source expectations.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/golomb_coding.hpp>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/plot/plot.hpp>
#include <format>

namespace datamunge::algorithms {
/// @brief Expected Golomb bits for P(N=n)=(1-a)a^n on n=0,1,... .
/// Uses b+ a^c/(1-a^M), where b=ceil(log2 M), c=2^b-M.
/// Stable expm1 evaluates the denominator for a near one. Returns the complete
/// infinite-source expectation, not a finite-window renormalization or measured
/// codec timing. a=0 is a point mass at zero. No allocation or input mutation.
/// @param a Finite tail ratio in [0,1); success probability is 1-a.
/// @param M Positive signed-int Golomb parameter.
/// @return Expected logical bits per source symbol, possibly infinity on numeric overflow.
/// @throws std::invalid_argument For invalid a or M.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::golomb_geometric_expected(.5,1)); // 2
/// @endcode
/// @par References
/// - S. W. Golomb (1966), Run-Length Encodings, doi:10.1109/TIT.1966.1053907.
///   The expectation here follows the stated zero-based source and truncated-binary convention.
/// @see golomb_geometric_table()
inline double golomb_geometric_expected(double a, int M) {
  golomb_detail::parameter(M);
  if (!std::isfinite(a) || a < 0 || a >= 1)
    throw std::invalid_argument("geometric tail ratio must be in [0,1)");
  if (a == 0)
    return double(golomb_code_length(0, M));
  const double loga = std::log(a);
  return golomb_detail::width(M) +
         std::exp(double(golomb_detail::cutoff(M)) * loga) / (-std::expm1(double(M) * loga));
}
/// @brief Compare full and conditional finite-window Golomb expectations.
/// Each row uses the same zero-based geometric source. The finite mean conditions
/// on N<window; omitted_probability is a^window. Scalar length calls do not emit
/// codewords. Empty parameters return an empty frame with all numeric columns.
/// O(parameters*window) work, O(parameters) output. No source samples are generated.
/// @param a Finite tail ratio in [0,1).
/// @param parameters Ordered positive parameters, duplicates permitted.
/// @param window Positive number of retained integer values, default 200.
/// @return Owned columns M, expected_bits, conditional_bits and omitted_probability.
/// @throws std::invalid_argument For invalid a, any M or window zero.
/// @throws std::bad_alloc On output allocation failure.
/// @throws std::length_error On container capacity limits.
/// @par Example
/// @code
/// const auto d=dm::algorithms::golomb_geometric_table(.5,{1,2});
/// std::print("{}\n",d.nrows()); // 2
/// @endcode
/// @see golomb_geometric_expected()
inline dstruct::DataFrame
golomb_geometric_table(double a, const std::vector<int>& parameters, std::size_t window = 200) {
  (void)golomb_geometric_expected(a, 1);
  if (!window)
    throw std::invalid_argument("geometric window must be positive");
  for (int M : parameters)
    golomb_detail::parameter(M);
  std::vector<double> m, full, finite, tail;
  for (int M : parameters) {
    double weight = 1, sum = 0, mass = 0;
    for (std::size_t n = 0; n < window; ++n) {
      sum += weight * double(golomb_code_length(n, M));
      mass += weight;
      weight *= a;
    }
    m.push_back(M);
    full.push_back(golomb_geometric_expected(a, M));
    finite.push_back(sum / mass);
    tail.push_back(std::pow(a, double(window)));
  }
  dstruct::DataFrame d;
  d.add_column("M", m);
  d.add_column("expected_bits", full);
  d.add_column("conditional_bits", finite);
  d.add_column("omitted_probability", tail);
  return d;
}
/// @brief Build a native exact-length table, without emitting potentially huge unary strings.
/// Counts are converted to double for presentation and are exact only through 2^53.
/// O(values*parameters) work/output. No mutation, random data or retained references.
/// @param values Borrowed nonnegative integers in requested display order.
/// @param parameters Ordered positive parameters, duplicates permitted.
/// @return Owned long-form columns value, M, bits, unary_bits and remainder_bits.
/// @throws std::invalid_argument For invalid M, even with empty values.
/// @throws std::length_error On unrepresentable bit counts/container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto d=dm::algorithms::golomb_length_table({0,8},{3});
/// std::print("{}\n",d.nrows()); // 2
/// @endcode
/// @see golomb_code_length(), golomb_length_plot()
inline dstruct::DataFrame golomb_length_table(
    const std::vector<std::uint64_t>& values,
    const std::vector<int>&           parameters) {
  for (int M : parameters)
    golomb_detail::parameter(M);
  std::vector<double> v, m, l, u, r;
  for (auto n : values)
    for (int M : parameters) {
      auto size = golomb_code_length(n, M);
      v.push_back(double(n));
      m.push_back(M);
      l.push_back(double(size));
      u.push_back(double(n / M) + 1);
      r.push_back(double(golomb_detail::payload(n, M)));
    }
  dstruct::DataFrame d;
  d.add_column("value", v);
  d.add_column("M", m);
  d.add_column("bits", l);
  d.add_column("unary_bits", u);
  d.add_column("remainder_bits", r);
  return d;
}
/// @brief Plot native logical word lengths on a shared numerical value axis.
/// Lines connect the requested values in their supplied order; use a sorted grid
/// for a curve. Output remains editable through the native plotting API.
/// @param values Nonempty display grid, values converted to double.
/// @param parameters Nonempty ordered positive parameters.
/// @return Owned native line plot with one curve per M and legend headroom.
/// @throws std::invalid_argument For empty inputs or invalid M.
/// @throws std::length_error On unrepresentable logical counts/container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// dm::algorithms::golomb_length_plot({0,2,4},{2,4}).save_svg("build/golomb.svg");
/// @endcode
/// @see golomb_length_table()
inline graphics::RPlot golomb_length_plot(
    const std::vector<std::uint64_t>& values,
    const std::vector<int>&           parameters) {
  if (values.empty() || parameters.empty())
    throw std::invalid_argument("Golomb plot needs values and parameters");
  for (int M : parameters)
    golomb_detail::parameter(M);
  auto                p        = graphics::RPlot::create();
  const graphics::RGB colors[] = {{37, 99, 235}, {220, 38, 38}, {5, 150, 105}, {202, 138, 4}};
  std::vector<double> x;
  for (auto n : values)
    x.push_back(double(n));
  double maximum = 0;
  for (std::size_t j = 0; j < parameters.size(); ++j) {
    std::vector<double> y;
    for (auto n : values) {
      auto l = double(golomb_code_length(n, parameters[j]));
      y.push_back(l);
      maximum = std::max(maximum, l);
    }
    p.line(x, y, std::format("M={}", parameters[j]), colors[j % 4], 2);
  }
  p.title("Golomb codeword length: unary versus remainder")
      .x_label("nonnegative source value")
      .y_label("logical codeword bits")
      .y_limits(0, maximum * 1.3 + 1);
  return p;
}
} // namespace datamunge::algorithms
