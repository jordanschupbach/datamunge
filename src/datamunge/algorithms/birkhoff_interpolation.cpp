/// @file birkhoff_interpolation.cpp
/// @brief Scaled derivative interpolation and checked polynomial evaluation.
#include <datamunge/algorithms/birkhoff_interpolation.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private finite-arithmetic checks for interpolation and evaluation.
namespace {
/// @brief Reject nonfinite arithmetic instead of propagating NaN coefficients.
/// @param value Computed result.
/// @return The unchanged finite value.
/// @throws std::overflow_error If value is not finite.
double finite_result(double value) {
  if (!std::isfinite(value)) throw std::overflow_error("birkhoff: nonfinite arithmetic result");
  return value;
}

/// @brief Validate finite polynomial coefficients and query.
/// @param coeffs Borrowed ascending coefficients.
/// @param x Query coordinate.
/// @throws std::invalid_argument For nonfinite inputs.
void validate_evaluation(const std::vector<double>& coeffs,double x) {
  if (!std::isfinite(x)) throw std::invalid_argument("birkhoff: query must be finite");
  for (double c:coeffs)
    if (!std::isfinite(c)) throw std::invalid_argument("birkhoff: coefficients must be finite");
}
} // namespace

namespace detail {
double birkhoff_entry(int k,int order,double node) {
  if (k<order) return 0;
  double falling=1;
  for (int t=0;t<order;++t) falling=finite_result(falling*(k-t));
  return finite_result(falling*std::pow(node,k-order));
}
} // namespace detail

std::vector<std::vector<double>> birkhoff_matrix(const std::vector<BirkhoffConstraint>& constraints) {
  const auto n=constraints.size();
  if (n>static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("birkhoff: too many constraints");
  for (const auto& c:constraints)
    if (c.order<0 || !std::isfinite(c.node) || !std::isfinite(c.value))
      throw std::invalid_argument("birkhoff: finite conditions with nonnegative orders required");
  std::vector<std::vector<double>> matrix(n,std::vector<double>(n));
  for (std::size_t i=0;i<n;++i)
    for (std::size_t k=0;k<n;++k)
      matrix[i][k]=detail::birkhoff_entry(static_cast<int>(k),constraints[i].order,constraints[i].node);
  return matrix;
}

std::vector<double> birkhoff_polynomial(const std::vector<BirkhoffConstraint>& constraints) {
  auto matrix=birkhoff_matrix(constraints);
  const auto n=matrix.size();
  if (!n) return {};
  for (std::size_t i=0;i<n;++i) {
    double scale=0;
    for (double value:matrix[i]) scale=std::max(scale,std::abs(value));
    if (scale==0) return {};
    for (double& value:matrix[i]) value/=scale;
    matrix[i].push_back(finite_result(constraints[i].value/scale));
  }
  std::vector<double> column_scale(n);
  for (std::size_t k=0;k<n;++k) {
    for (std::size_t i=0;i<n;++i) column_scale[k]=std::max(column_scale[k],std::abs(matrix[i][k]));
    if (column_scale[k]==0) return {};
    for (std::size_t i=0;i<n;++i) matrix[i][k]/=column_scale[k];
  }
  const double tolerance=64*std::numeric_limits<double>::epsilon()*n;
  for (std::size_t col=0;col<n;++col) {
    std::size_t pivot=col;
    for (std::size_t row=col+1;row<n;++row)
      if (std::abs(matrix[row][col])>std::abs(matrix[pivot][col])) pivot=row;
    if (std::abs(matrix[pivot][col])<=tolerance) return {};
    std::swap(matrix[col],matrix[pivot]);
    const double diagonal=matrix[col][col];
    for (std::size_t k=col+1;k<=n;++k) matrix[col][k]=finite_result(matrix[col][k]/diagonal);
    matrix[col][col]=1;
    for (std::size_t row=0;row<n;++row) {
      if (row==col) continue;
      const double multiplier=matrix[row][col];
      for (std::size_t k=col+1;k<=n;++k)
        matrix[row][k]=finite_result(std::fma(-multiplier,matrix[col][k],matrix[row][k]));
      matrix[row][col]=0;
    }
  }
  std::vector<double> coefficients(n);
  for (std::size_t i=0;i<n;++i) coefficients[i]=finite_result(matrix[i][n]/column_scale[i]);
  return coefficients;
}

double birkhoff_eval(const std::vector<double>& coeffs,double x) {
  validate_evaluation(coeffs,x);
  double value=0;
  for (std::size_t i=coeffs.size();i-- >0;) value=finite_result(std::fma(value,x,coeffs[i]));
  return value;
}

double birkhoff_eval_derivative(const std::vector<double>& coeffs,double x,int order) {
  if (order<0) throw std::invalid_argument("birkhoff: derivative order must be nonnegative");
  validate_evaluation(coeffs,x);
  if (static_cast<std::size_t>(order)>=coeffs.size()) return 0;
  double value=0;
  for (std::size_t k=coeffs.size();k-- >static_cast<std::size_t>(order);) {
    double coefficient=coeffs[k];
    for (int t=0;t<order && coefficient!=0;++t) coefficient=finite_result(coefficient*(k-t));
    value=finite_result(std::fma(value,x,coefficient));
  }
  return value;
}
} // namespace datamunge::algorithms
