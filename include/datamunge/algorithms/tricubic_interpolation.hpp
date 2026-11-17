#pragma once
/// @file tricubic_interpolation.hpp
/// @brief Checked tricubic interpolation operations and producer diagnostics.
#include <cmath>
#include <cstddef>
#include <datamunge/algorithms/bicubic_interpolation.hpp>
#include <stdexcept>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual tricubic neighborhood and final separable accumulation.
/// @par Example
/// @code
/// dm::algorithms::TricubicWork r; //empty until a query
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
struct TricubicWork {
  double                   value{0}; ///< Actual separable result.
  std::vector<std::size_t> indices;  ///<64 actual clamped flattened sample indices.
  std::vector<double>      weights;  ///<64 actual product weights in l,m,n order.
};
/// @brief Interpolate using Keys cubic convolution with a=-1/2.
/// Relative spacing tolerance1e-10; boundary samples repeat at edges. Interior trilinear reproduction; boundary clamping need not reproduce linear functions. Full input validation O(volume) per call;64-sample query after validation; no shared state.
/// @param xs At least2 finite increasing uniform nodes.
/// @param ys At least2 finite increasing uniform nodes.
/// @param zs At least2 finite increasing uniform nodes.
/// @param v Finite row-major nx*ny*nz volume; at most1000000 cells.
/// @param x Finite query within closed x axis.
/// @param y Finite query within closed y axis.
/// @param z Finite query within closed z axis.
/// @return Actual64 clamped sample indices, weights and result.
/// @throws std::invalid_argument Malformed axes, shape, nonfinite input or out-of-domain query.
/// @throws std::length_error Volume exceeds1000000 samples.
/// @throws std::overflow_error Nonfinite result.
/// @par Example
/// @code
/// std::vector<double> axis{0,1},v(8,3); auto r=dm::algorithms::tricubic_work(axis,axis,axis,v,.5,.5,.5); //3
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
inline TricubicWork tricubic_work(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    const std::vector<double>& zs,
    const std::vector<double>& v,
    double                     x,
    double                     y,
    double                     z) {
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
    throw std::invalid_argument("tricubic: nonfinite query");
  std::size_t cells = 1;
  for (const auto* axis : {&xs, &ys, &zs}) {
    if (axis->size() < 2 || axis->size() > 1000000)
      throw std::invalid_argument("tricubic: axis size");
    if (cells > 1000000 / axis->size())
      throw std::length_error("tricubic: volume limit");
    cells *= axis->size();
    double h = (*axis)[1] - (*axis)[0];
    if (!std::isfinite(h) || h <= 0)
      throw std::invalid_argument("tricubic: spacing");
    for (std::size_t i = 0; i < axis->size(); ++i)
      if (!std::isfinite((*axis)[i]) ||
          (i && (!((*axis)[i] > (*axis)[i - 1]) ||
                 std::abs(((*axis)[i] - (*axis)[i - 1]) - h) > 1e-10 * h)))
        throw std::invalid_argument("tricubic: nonuniform axis");
  }
  if (v.size() != cells)
    throw std::invalid_argument("tricubic: volume shape");
  for (double q : v)
    if (!std::isfinite(q))
      throw std::invalid_argument("tricubic: nonfinite volume");
  if (x < xs.front() || x > xs.back() || y < ys.front() || y > ys.back() || z < zs.front() ||
      z > zs.back())
    throw std::invalid_argument("tricubic: query outside volume");
  TricubicWork      work;
  const std::size_t nx = xs.size(), ny = ys.size(), nz = zs.size();
  const double      hx = xs[1] - xs[0], hy = ys[1] - ys[0], hz = zs[1] - zs[0];

  const double fx = (x - xs[0]) / hx, fy = (y - ys[0]) / hy, fz = (z - zs[0]) / hz;
  const long   ix = static_cast<long>(std::floor(fx)), iy = static_cast<long>(std::floor(fy)),
             iz   = static_cast<long>(std::floor(fz));
  const double tx = fx - ix, ty = fy - iy, tz = fz - iz;

  double result = 0.0;
  for (int l = -1; l <= 2; ++l) {
    const double      wx = detail::bicubic_kernel(tx - l);
    const std::size_t gi = detail::bicubic_clamp(ix + l, nx);
    double            sy = 0.0;
    for (int m = -1; m <= 2; ++m) {
      const double      wy = detail::bicubic_kernel(ty - m);
      const std::size_t gj = detail::bicubic_clamp(iy + m, ny);
      double            sz = 0.0;
      for (int n = -1; n <= 2; ++n) {
        const double      wz    = detail::bicubic_kernel(tz - n);
        const std::size_t gk    = detail::bicubic_clamp(iz + n, nz);
        auto              index = (gi * ny + gj) * nz + gk;
        work.indices.push_back(index);
        work.weights.push_back(wx * wy * wz);
        sz += wz * v[index];
      }
      sy += wy * sz;
    }
    result += wx * sy;
  }
  if (!std::isfinite(result))
    throw std::overflow_error("tricubic: result overflow");
  work.value = result;
  return work;
}

/// @brief Return the same checked tricubic producer value.
/// @param xs Uniform x axis.
/// @param ys Uniform y axis.
/// @param zs Uniform z axis.
/// @param v Finite complete volume.
/// @param x In-domain x.
/// @param y In-domain y.
/// @param z In-domain z.
/// @return Interpolated scalar.
/// @throws std::invalid_argument Invalid query or grid.
/// @throws std::length_error Volume limit.
/// @throws std::overflow_error Nonfinite result.
/// @par Example
/// @code
/// std::vector<double> axis{0,1},v(8,3); auto y=dm::algorithms::tricubic_interpolate(axis,axis,axis,v,.5,.5,.5);
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
inline double tricubic_interpolate(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    const std::vector<double>& zs,
    const std::vector<double>& v,
    double                     x,
    double                     y,
    double                     z) {
  return tricubic_work(xs, ys, zs, v, x, y, z).value;
}
/// @brief Original smooth-volume line experiment and both interpolants.
/// @par Example
/// @code
/// auto r=dm::algorithms::tricubic_example();
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
struct TricubicExample {
  std::vector<double> x;      ///< Original line coordinates -2.5..2.5 at.05 spacing.
  std::vector<double> truth;  ///< Actual analytic field on the line.
  std::vector<double> cubic;  ///< Actual checked tricubic outputs.
  std::vector<double> linear; ///< Actual trilinear8-corner outputs.
};
/// @brief Run the original9-cubed smooth-volume experiment.
/// Grid spacing.7, origin-2.8; field exp(-.08*(x²+y²+z²))*cos(.6x+.4y-.3z); line y.35,z-.5. Boundary clamping is retained; no global cubic-accuracy claim.
/// @return Owned original line, analytic truth, native tricubic and trilinear comparison.
/// @throws std::invalid_argument Internal fixture violates checked grid contract.
/// @throws std::length_error Internal fixture exceeds budget.
/// @throws std::overflow_error Nonfinite result.
/// @par Example
/// @code
/// auto r=dm::algorithms::tricubic_example(); //101 line points
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
TricubicExample tricubic_example();

} // namespace datamunge::algorithms
