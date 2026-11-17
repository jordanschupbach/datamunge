#pragma once
#include <datamunge/algorithms/kahan.hpp>
#include <datamunge/algorithms/newton_division.hpp>
#include <datamunge/algorithms/nth_root.hpp>

/// @file elementary.hpp
/// @brief Elementary numerical routines: compensated summation, Newton division and roots,
///        and square-root-free magnitude estimates (alpha max plus beta min).

#include <vector>

namespace datamunge::algorithms {





/// @brief The *alpha max plus beta min algorithm*: a fast, division- and square-root-free
///        approximation of the Euclidean magnitude @c sqrt(x^2+y^2), as
///        @c alpha·max(|x|,|y|) + beta·min(|x|,|y|). With the peak-error-minimizing coefficients
///        @c alpha≈0.96043, @c beta≈0.39782 the worst-case relative error is about @c 3.96%. It is a
///        classic DSP/embedded trick for estimating vector magnitude (or complex modulus) where a true
///        @c hypot is too expensive.
///
/// @param x,y the two components.
/// @return the approximate magnitude @c ~sqrt(x^2+y^2).
double alpha_max_beta_min(double x, double y);

/// @brief Coefficients of an alpha-max-plus-beta-min estimate @c alpha·hi + beta·lo.
struct AlphaBetaCoefficients {
    double alpha{1.0};  ///< weight on the larger absolute component.
    double beta{0.5};   ///< weight on the smaller absolute component.
};

/// @brief The range of @c estimate/true over every direction for a coefficient pair.
struct AlphaBetaErrorRange {
    double min_ratio{1.0};   ///< smallest @c estimate/|v| over all directions.
    double max_ratio{1.0};   ///< largest @c estimate/|v| over all directions.
    double peak_error{0.0};  ///< @c max(|min_ratio-1|, |max_ratio-1|), the worst relative error.
};

/// @brief The general estimate @c alpha·max(|x|,|y|) + beta·min(|x|,|y|).
/// @param x,y the two components.
/// @param coefficients the pair to use.
/// @return the estimated magnitude.
double alpha_max_beta_min(double x, double y, const AlphaBetaCoefficients& coefficients);

/// @brief The pair minimizing the worst relative error, in closed form:
///        \f$\alpha=2\cos(\pi/8)/(1+\cos(\pi/8))\f$, \f$\beta=2\sin(\pi/8)/(1+\cos(\pi/8))\f$,
///        with peak error \f$\tan^2(\pi/16)\approx3.96\%\f$.
/// @return the minimax coefficients (the ones alpha_max_beta_min(x, y) uses).
AlphaBetaCoefficients alpha_max_beta_min_minimax();

/// @brief The pair with the smallest worst error among those that never underestimate:
///        \f$\alpha=1\f$, \f$\beta=\tan(\pi/8)\f$, peak error \f$1/\cos(\pi/8)-1\approx8.24\%\f$.
/// @return the coefficients.
AlphaBetaCoefficients alpha_max_beta_min_no_underestimate();

/// @brief The pair minimizing the mean squared relative error over directions uniform in angle,
///        \f$\int_0^{\pi/4}(\alpha\cos\theta+\beta\sin\theta-1)^2\,d\theta\f$, from its 2x2 normal equations.
/// @return the least-squares coefficients.
AlphaBetaCoefficients alpha_max_beta_min_least_squares();

/// @brief Exact error range of a coefficient pair over all directions.
///
/// With @c hi = r·cos(theta), @c lo = r·sin(theta), theta in [0, pi/4], the ratio is
/// @c alpha·cos(theta) + beta·sin(theta); its extremes lie at the ends of the interval or at
/// @c theta = atan(beta/alpha), so no sampling is needed.
/// @param coefficients the pair to analyse.
/// @return the minimum and maximum ratio and the peak relative error.
AlphaBetaErrorRange alpha_max_beta_min_error(const AlphaBetaCoefficients& coefficients);

/// @brief The minimax k-line generalisation: the octant [0, pi/4] is split into @p lines equal
///        sectors, each with its own minimax pair, and the estimate is the largest of the
///        @p lines linear forms (a regular polygon around the circle).
/// @param lines number of linear forms per octant (at least 1); 1 gives alpha_max_beta_min_minimax().
/// @return the @p lines coefficient pairs, in order of increasing angle.
/// @throws std::invalid_argument if @p lines is below 1.
std::vector<AlphaBetaCoefficients> alpha_max_beta_min_polygon(int lines);

/// @brief The polygon estimate @c max_j(alpha_j·hi + beta_j·lo); its peak relative error is
///        \f$\tan^2(\pi/(16k))\f$ for @c k lines from alpha_max_beta_min_polygon().
/// @param x,y the two components.
/// @param lines the pairs, e.g. from alpha_max_beta_min_polygon().
/// @return the estimated magnitude.
/// @throws std::invalid_argument if @p lines is empty.
double alpha_max_beta_min(double x, double y, const std::vector<AlphaBetaCoefficients>& lines);

} // namespace datamunge::algorithms
