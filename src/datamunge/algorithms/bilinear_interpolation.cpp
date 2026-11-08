/// @file bilinear_interpolation.cpp
/// @brief Validated rectangular-grid lookup and stable bilinear blending.
#include <datamunge/algorithms/bilinear_interpolation.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private validation and interval lookup for bilinear interpolation.
namespace {
/// @brief Require at least two finite strictly increasing coordinates.
/// @param axis Borrowed axis; no mutation.
/// @throws std::invalid_argument For invalid size, finiteness or ordering.
void validate_axis(const std::vector<double>& axis) {
  if (axis.size() < 2) throw std::invalid_argument("bilinear_interpolate: axis needs two points");
  for (std::size_t i=0;i<axis.size();++i)
    if (!std::isfinite(axis[i]) || (i && !(axis[i] > axis[i-1])))
      throw std::invalid_argument("bilinear_interpolate: axes must be finite and increasing");
}

/// @brief Locate a cell, retaining the nearest end cell outside the axis.
/// @param axis Validated increasing axis, size at least two.
/// @param query Finite coordinate.
/// @return Lower cell index in [0,axis.size()-2].
std::size_t cell(const std::vector<double>& axis, double query) {
  if (query <= axis.front()) return 0;
  if (query >= axis[axis.size()-2]) return axis.size()-2;
  return static_cast<std::size_t>(std::upper_bound(axis.begin(),axis.end(),query)-axis.begin()-1);
}

/// @brief Form a dimensionless cell coordinate with wider intermediate arithmetic.
/// @param lo Lower coordinate.
/// @param hi Strictly larger upper coordinate.
/// @param query Finite query.
/// @return Relative position, including extrapolation values outside [0,1].
/// @throws std::overflow_error If a finite double position cannot be represented.
double fraction(double lo, double hi, double query) {
  const double t=static_cast<double>((static_cast<long double>(query)-lo)/
                                    (static_cast<long double>(hi)-lo));
  if (!std::isfinite(t)) throw std::overflow_error("bilinear_interpolate: coordinate overflow");
  return t;
}
} // namespace

double bilinear_interpolate(const std::vector<double>& xs, const std::vector<double>& ys,
                            const std::vector<double>& z, double x, double y) {
  validate_axis(xs);
  validate_axis(ys);
  if (xs.size() > z.max_size()/ys.size() || z.size() != xs.size()*ys.size())
    throw std::invalid_argument("bilinear_interpolate: wrong grid size");
  if (!std::isfinite(x) || !std::isfinite(y))
    throw std::invalid_argument("bilinear_interpolate: query must be finite");
  for (double value:z)
    if (!std::isfinite(value)) throw std::invalid_argument("bilinear_interpolate: nonfinite sample");
  const auto i=cell(xs,x), j=cell(ys,y), ny=ys.size();
  const double tx=fraction(xs[i],xs[i+1],x), ty=fraction(ys[j],ys[j+1],y);
  const double a=std::lerp(z[i*ny+j],z[(i+1)*ny+j],tx);
  const double b=std::lerp(z[i*ny+j+1],z[(i+1)*ny+j+1],tx);
  const double result=std::lerp(a,b,ty);
  if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(result))
    throw std::overflow_error("bilinear_interpolate: value overflow");
  return result;
}
} // namespace datamunge::algorithms
