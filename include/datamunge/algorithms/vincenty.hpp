#pragma once

/// \file vincenty.hpp
/// \brief Vincenty's inverse formula: geodesic distance on an ellipsoid.
///
/// Vincenty's formulae (Thaddeus Vincenty, 1975) compute the distance between two
/// latitude/longitude points on an oblate *ellipsoid* (by default WGS-84, the shape GPS
/// uses) to sub-millimetre accuracy, far better than the spherical *haversine* formula which
/// ignores the ~0.3% equatorial bulge. The inverse method solves for the geodesic iteratively
/// on an auxiliary sphere, correcting for flattening at each step. This module provides the
/// Vincenty inverse distance and, for comparison, the haversine great-circle distance.

#include <cmath>

namespace datamunge::algorithms {

/// Result of a Vincenty inverse solution.
struct VincentyResult {
    double distance_m;   ///< Geodesic distance in metres.
    int    iterations;   ///< Fixed-point iterations used.
    bool   converged;    ///< False for (near-)antipodal points that fail to converge.
};

/// \brief Geodesic distance between two lat/lon points (degrees) on the WGS-84 ellipsoid.
inline VincentyResult vincenty_inverse(double lat1_deg, double lon1_deg,
                                       double lat2_deg, double lon2_deg,
                                       int max_iter = 200, double tol = 1e-12) {
    constexpr double a = 6378137.0;              // WGS-84 semi-major axis (m)
    constexpr double f = 1.0 / 298.257223563;    // flattening
    const double     b = (1.0 - f) * a;          // semi-minor axis
    constexpr double deg = 3.14159265358979323846 / 180.0;

    double U1 = std::atan((1 - f) * std::tan(lat1_deg * deg));
    double U2 = std::atan((1 - f) * std::tan(lat2_deg * deg));
    double L  = (lon2_deg - lon1_deg) * deg;

    double sinU1 = std::sin(U1), cosU1 = std::cos(U1);
    double sinU2 = std::sin(U2), cosU2 = std::cos(U2);

    double lambda = L, lambda_prev;
    double sinSigma = 0, cosSigma = 0, sigma = 0, cos2Alpha = 0, cos2SigmaM = 0;
    int    it = 0;
    bool   converged = false;

    for (; it < max_iter; ++it) {
        double sinLambda = std::sin(lambda), cosLambda = std::cos(lambda);
        sinSigma = std::sqrt(std::pow(cosU2 * sinLambda, 2) +
                             std::pow(cosU1 * sinU2 - sinU1 * cosU2 * cosLambda, 2));
        if (sinSigma == 0) return {0.0, it, true};  // coincident points
        cosSigma      = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda;
        sigma         = std::atan2(sinSigma, cosSigma);
        double sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma;
        cos2Alpha     = 1.0 - sinAlpha * sinAlpha;
        cos2SigmaM    = (cos2Alpha != 0.0) ? cosSigma - 2 * sinU1 * sinU2 / cos2Alpha : 0.0;
        double C      = f / 16 * cos2Alpha * (4 + f * (4 - 3 * cos2Alpha));
        lambda_prev   = lambda;
        lambda = L + (1 - C) * f * sinAlpha *
                     (sigma + C * sinSigma *
                          (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM)));
        if (std::fabs(lambda - lambda_prev) < tol) { converged = true; ++it; break; }
    }

    double u2 = cos2Alpha * (a * a - b * b) / (b * b);
    double A  = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)));
    double B  = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)));
    double deltaSigma =
        B * sinSigma *
        (cos2SigmaM + B / 4 * (cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM) -
                               B / 6 * cos2SigmaM * (-3 + 4 * sinSigma * sinSigma) *
                                   (-3 + 4 * cos2SigmaM * cos2SigmaM)));
    double s = b * A * (sigma - deltaSigma);
    return {s, it, converged};
}

/// \brief Great-circle distance (metres) via the spherical haversine formula (mean radius).
inline double haversine_distance(double lat1_deg, double lon1_deg,
                                 double lat2_deg, double lon2_deg,
                                 double radius_m = 6371000.0) {
    constexpr double deg = 3.14159265358979323846 / 180.0;
    double dlat = (lat2_deg - lat1_deg) * deg;
    double dlon = (lon2_deg - lon1_deg) * deg;
    double h = std::sin(dlat / 2) * std::sin(dlat / 2) +
               std::cos(lat1_deg * deg) * std::cos(lat2_deg * deg) *
                   std::sin(dlon / 2) * std::sin(dlon / 2);
    return 2 * radius_m * std::asin(std::sqrt(h));
}

}  // namespace datamunge::algorithms
