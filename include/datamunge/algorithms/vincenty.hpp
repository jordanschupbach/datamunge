#pragma once
/// @file vincenty.hpp
/// @brief Checked native vincenty and actual producing diagnostics.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual Vincenty inverse outcome, including unresolved status.
/// @par Example
/// @code
/// dm::algorithms::VincentyResult r{};
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
struct VincentyResult {
  double distance_m{0};    ///< WGS84 geodesic metres if converged, NaN otherwise.
  int    iterations{0};    ///< Actual fixed-point updates,0 for detected coincident points.
  bool   converged{false}; ///< Lambda update tolerance reached, not a certified error bound.
};
/// @brief Actual auxiliary-longitude fixed-point update.
/// @par Example
/// @code
/// dm::algorithms::VincentyStep r{};
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
struct VincentyStep {
  double before{0}; ///< Actual lambda input, radians.
  double after{0};  ///< Actual lambda output, radians.
  double delta{0};  ///< Actual absolute update, radians.
};
/// @brief Actual Vincenty result with bounded optional history.
/// @par Example
/// @code
/// auto r=dm::algorithms::vincenty_work(0,0,0,1);
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
struct VincentyWork {
  VincentyResult            result; ///< Actual final status and valid distance.
  std::vector<VincentyStep> steps;  ///< First max_records actual updates.
};
/// @brief Solve the WGS84 inverse problem with actual lambda updates.
/// Longitude difference reduced modulo360. Coincident geographic poles are detected. Near-antipodal fixed points may fail to converge; no spherical fallback is substituted. O(max_iter) work and O(max_records) history; calls independent.
/// @param lat1_deg Finite source latitude in[-90,90].
/// @param lon1_deg Finite source longitude in[-180,180].
/// @param lat2_deg Finite target latitude in[-90,90].
/// @param lon2_deg Finite target longitude in[-180,180].
/// @param max_iter Positive update cap<=1000000, default200.
/// @param tol Finite positive lambda-update tolerance radians, default1e-12.
/// @param max_records Actual update records retained, default0; <=1000000.
/// @return Valid metres only when converged; NaN on iteration exhaustion.
/// @throws std::invalid_argument Invalid coordinates, cap or tolerance.
/// @throws std::length_error Too many history records.
/// @par Example
/// @code
/// auto r=dm::algorithms::vincenty_work(0,0,0,1,200,1e-12,10);
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline VincentyWork vincenty_work(
    double      lat1_deg,
    double      lon1_deg,
    double      lat2_deg,
    double      lon2_deg,
    int         max_iter    = 200,
    double      tol         = 1e-12,
    std::size_t max_records = 0) {
  for (double q : {lat1_deg, lon1_deg, lat2_deg, lon2_deg, tol})
    if (!std::isfinite(q))
      throw std::invalid_argument("Vincenty finite inputs");
  if (std::abs(lat1_deg) > 90 || std::abs(lat2_deg) > 90 || std::abs(lon1_deg) > 180 ||
      std::abs(lon2_deg) > 180 || max_iter <= 0 || max_iter > 1000000 || tol <= 0)
    throw std::invalid_argument("Vincenty input range");
  if (max_records > 1000000)
    throw std::length_error("Vincenty history cap");
  VincentyWork work;
  if (lat1_deg == lat2_deg &&
      (std::abs(lat1_deg) == 90 || std::remainder(lon2_deg - lon1_deg, 360.) == 0)) {
    work.result = {0, 0, true};
    return work;
  }
  constexpr double a   = 6378137.0;           // WGS-84 semi-major axis (m)
  constexpr double f   = 1.0 / 298.257223563; // flattening
  const double     b   = (1.0 - f) * a;       // semi-minor axis
  constexpr double deg = 3.14159265358979323846 / 180.0;

  double U1 = std::atan((1 - f) * std::tan(lat1_deg * deg));
  double U2 = std::atan((1 - f) * std::tan(lat2_deg * deg));
  double L  = std::remainder(lon2_deg - lon1_deg, 360.0) * deg;

  double sinU1 = std::sin(U1), cosU1 = std::cos(U1);
  double sinU2 = std::sin(U2), cosU2 = std::cos(U2);

  double lambda   = L, lambda_prev;
  double sinSigma = 0, cosSigma = 0, sigma = 0, cos2Alpha = 0, cos2SigmaM = 0;
  int    it        = 0;
  bool   converged = false;

  for (; it < max_iter; ++it) {
    double sinLambda = std::sin(lambda), cosLambda = std::cos(lambda);
    sinSigma = std::sqrt(
        std::pow(cosU2 * sinLambda, 2) + std::pow(cosU1 * sinU2 - sinU1 * cosU2 * cosLambda, 2));
    if (sinSigma == 0) {
      work.result = {0.0, it, true};
      return work;
    } // coincident points
    cosSigma        = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda;
    sigma           = std::atan2(sinSigma, cosSigma);
    double sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma;
    cos2Alpha       = 1.0 - sinAlpha * sinAlpha;
    cos2SigmaM      = (cos2Alpha != 0.0) ? cosSigma - 2 * sinU1 * sinU2 / cos2Alpha : 0.0;
    double C        = f / 16 * cos2Alpha * (4 + f * (4 - 3 * cos2Alpha));
    lambda_prev     = lambda;
    lambda          = L + (1 - C) * f * sinAlpha *
                     (sigma + C * sinSigma *
                                  (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM)));
    if (work.steps.size() < max_records)
      work.steps.push_back({lambda_prev, lambda, std::fabs(lambda - lambda_prev)});
    if (std::fabs(lambda - lambda_prev) < tol) {
      converged = true;
      ++it;
      break;
    }
  }

  double u2         = cos2Alpha * (a * a - b * b) / (b * b);
  double A          = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)));
  double B          = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)));
  double deltaSigma = B * sinSigma *
                      (cos2SigmaM + B / 4 *
                                        (cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM) -
                                         B / 6 * cos2SigmaM * (-3 + 4 * sinSigma * sinSigma) *
                                             (-3 + 4 * cos2SigmaM * cos2SigmaM)));
  double s    = b * A * (sigma - deltaSigma);
  work.result = {converged ? s : std::numeric_limits<double>::quiet_NaN(), it, converged};
  return work;
}

/// @brief Return the same checked Vincenty inverse outcome.
/// @param lat1_deg Finite source latitude in[-90,90].
/// @param lon1_deg Finite source longitude in[-180,180].
/// @param lat2_deg Finite target latitude in[-90,90].
/// @param lon2_deg Finite target longitude in[-180,180].
/// @param max_iter Positive update cap<=1000000, default200.
/// @param tol Finite positive lambda-update tolerance radians, default1e-12.
/// @return Actual distance/status; NaN distance if unresolved.
/// @throws std::invalid_argument Invalid coordinates or iteration controls.
/// @par Example
/// @code
/// auto r=dm::algorithms::vincenty_inverse(0,0,0,1);
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline VincentyResult vincenty_inverse(
    double lat1_deg,
    double lon1_deg,
    double lat2_deg,
    double lon2_deg,
    int    max_iter = 200,
    double tol      = 1e-12) {
  return vincenty_work(lat1_deg, lon1_deg, lat2_deg, lon2_deg, max_iter, tol).result;
}
/// @brief Compute a spherical great-circle comparison distance.
/// Clamps the computed haversine to[0,1] against rounding at antipodes; O(1), calls independent.
/// @param lat1_deg Finite source latitude in[-90,90].
/// @param lon1_deg Finite source longitude in[-180,180].
/// @param lat2_deg Finite target latitude in[-90,90].
/// @param lon2_deg Finite target longitude in[-180,180].
/// @param radius_m Finite positive sphere radius metres, default6371000.
/// @return Spherical distance metres, not ellipsoid ground truth.
/// @throws std::invalid_argument Invalid coordinates or radius.
/// @par Example
/// @code
/// double d=dm::algorithms::haversine_distance(0,0,0,1);
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline double haversine_distance(
    double lat1_deg,
    double lon1_deg,
    double lat2_deg,
    double lon2_deg,
    double radius_m = 6371000.0) {
  for (double q : {lat1_deg, lon1_deg, lat2_deg, lon2_deg, radius_m})
    if (!std::isfinite(q))
      throw std::invalid_argument("haversine finite input");
  if (std::abs(lat1_deg) > 90 || std::abs(lat2_deg) > 90 || std::abs(lon1_deg) > 180 ||
      std::abs(lon2_deg) > 180 || radius_m <= 0)
    throw std::invalid_argument("haversine range");
  constexpr double deg  = 3.14159265358979323846 / 180.0;
  double           dlat = (lat2_deg - lat1_deg) * deg;
  double           dlon = (lon2_deg - lon1_deg) * deg;
  double           h    = std::sin(dlat / 2) * std::sin(dlat / 2) + std::cos(lat1_deg * deg) *
                                                           std::cos(lat2_deg * deg) *
                                                           std::sin(dlon / 2) * std::sin(dlon / 2);
  return 2 * radius_m * std::asin(std::sqrt(std::clamp(h, 0., 1.)));
}

} // namespace datamunge::algorithms
