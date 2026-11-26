#pragma once

/// @file geostatistics.hpp
/// @brief Empirical variograms, parametric variogram models and kriging of spatial data.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <limits>
#include <vector>

namespace datamunge::stats {

/// @brief Options for empirical_variogram().
/// @par Example
/// @code
/// const dm::stats::VariogramOptions o{.cutoff = 1000, .width = 100};
/// std::println("{}", o.cressie); // false
/// @endcode
struct VariogramOptions {
    double cutoff{std::numeric_limits<double>::quiet_NaN()}; ///< Largest lag; NaN uses 1/3 of the bounding-box diagonal.
    double width{std::numeric_limits<double>::quiet_NaN()};  ///< Lag width; NaN uses cutoff / 15.
    std::vector<double> boundaries; ///< Explicit increasing lag boundaries (overrides cutoff and width).
    std::vector<double> directions; ///< Directions in degrees clockwise from north (+y); empty is isotropic.
    double angular_tolerance{std::numeric_limits<double>::quiet_NaN()}; ///< Half-width in degrees; NaN uses 90 / #directions.
    bool cressie{false}; ///< Use the robust Cressie-Hawkins estimator instead of the method of moments.
};

/// @brief One lag bin of an empirical variogram.
/// @par Example
/// @code
/// dm::stats::VariogramBin b{.distance = 100, .gamma = 0.3, .pairs = 57};
/// std::println("{}", b.pairs); // 57
/// @endcode
struct VariogramBin {
    double distance{};  ///< Mean distance of the pairs in the bin.
    double gamma{};     ///< Semivariance estimate.
    std::size_t pairs{}; ///< Number of point pairs.
    double direction{std::numeric_limits<double>::quiet_NaN()}; ///< Direction in degrees (NaN when isotropic).
    double lower{};     ///< Lower lag boundary (exclusive).
    double upper{};     ///< Upper lag boundary (inclusive).
};

/// @brief Empirical (sample) semivariogram.
///
/// For each lag bin \f$(b_k,b_{k+1}]\f$ with pairs \f$N(h)\f$, the method-of-moments estimator is
/// \f$\hat\gamma(h)=\frac1{2|N(h)|}\sum_{N(h)}(z_i-z_j)^2\f$. The Cressie-Hawkins estimator,
/// \f$\bar{|z_i-z_j|^{1/2}}^4/(2(0.457+0.494/|N(h)|))\f$, resists outliers. Pairs at distance 0 or
/// beyond the last boundary are ignored, and empty bins are omitted. Directional variograms keep a
/// pair when the azimuth of its separation vector, in degrees clockwise from north and taken modulo
/// 180, is within the tolerance of the direction. The defaults follow `gstat::variogram`, which
/// also reports the mean pair distance per bin. Pass residuals to estimate a variogram after removing
/// a trend.
/// @param coordinates n-by-d locations (d = 2 for directional variograms).
/// @param values Observed values.
/// @param options Lags, directions and estimator.
/// @return Nonempty bins, ordered by direction and then lag.
/// @throws std::invalid_argument For mismatched sizes, fewer than two points, nonfinite inputs,
/// nonincreasing boundaries, a nonpositive cutoff or width, or directions without 2-D coordinates.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> xy(3, 2, std::vector<double>{0, 0, 1, 0, 3, 0});
/// const auto v = dm::stats::empirical_variogram(xy, {1, 2, 4}, {.boundaries = {0, 1, 2, 3}});
/// std::println("{} {} {}", v[0].gamma, v[1].gamma, v[2].gamma); // 0.5 2 4.5
/// @endcode
std::vector<VariogramBin> empirical_variogram(const linalg::DenseMatrix<double>& coordinates,
                                              const std::vector<double>& values, const VariogramOptions& options = {});

/// @brief Shape of a parametric variogram model (parameterised as in gstat).
enum class VariogramModelKind {
    spherical,   ///< \f$1.5h/a-0.5(h/a)^3\f$ for \f$h<a\f$, 1 beyond; reaches the sill at the range a.
    exponential, ///< \f$1-e^{-h/a}\f$; practical range about 3a.
    gaussian,    ///< \f$1-e^{-(h/a)^2}\f$; practical range about 1.73a.
    matern,      ///< \f$1-\frac{(h/a)^\kappa K_\kappa(h/a)}{2^{\kappa-1}\Gamma(\kappa)}\f$; kappa 0.5 is exponential.
};

/// @brief A nugget plus one structured variogram component.
/// @par Example
/// @code
/// const dm::stats::VariogramModel m{.kind = dm::stats::VariogramModelKind::spherical,
///                                    .nugget = 0.1, .partial_sill = 1, .range = 5};
/// std::println("{}", m.nugget + m.partial_sill); // 1.1
/// @endcode
struct VariogramModel {
    VariogramModelKind kind{VariogramModelKind::spherical}; ///< Shape of the structured component.
    double nugget{};       ///< Discontinuity at the origin (measurement error and micro-scale variation).
    double partial_sill{1}; ///< Variance of the structured component; sill = nugget + partial_sill.
    double range{1};       ///< Range (spherical) or scale (other shapes) parameter a.
    double kappa{0.5};     ///< Matern smoothness (ignored by the other shapes).
};

/// @brief Value of a variogram model at lag h.
///
/// gamma(0) = 0 and gamma(h) = nugget + partial_sill * shape(h) for h > 0, as `gstat::variogramLine`.
/// @param model The model.
/// @param h Lag distance (nonnegative).
/// @return The semivariance.
/// @throws std::invalid_argument If h is negative or the model parameters are invalid
/// (negative sills, nonpositive range or kappa).
/// @par Example
/// @code
/// const dm::stats::VariogramModel m{.kind = dm::stats::VariogramModelKind::exponential,
///                                    .nugget = 0.2, .partial_sill = 1, .range = 300};
/// std::println("{:.4f}", dm::stats::variogram_model_value(m, 300)); // 0.8321
/// @endcode
double variogram_model_value(const VariogramModel& model, double h);

/// @brief Covariance implied by a bounded variogram model, C(h) = sill - gamma(h).
///
/// C(0) = nugget + partial_sill; for h > 0 the nugget does not contribute.
/// @param model The model.
/// @param h Lag distance (nonnegative).
/// @return The covariance.
/// @throws std::invalid_argument As for variogram_model_value().
/// @par Example
/// @code
/// const dm::stats::VariogramModel m{.kind = dm::stats::VariogramModelKind::spherical,
///                                    .nugget = 0.1, .partial_sill = 1, .range = 5};
/// std::println("{} {}", dm::stats::variogram_model_covariance(m, 0),
///              dm::stats::variogram_model_covariance(m, 6)); // 1.1 0
/// @endcode
double variogram_model_covariance(const VariogramModel& model, double h);

/// @brief Weights of the least-squares variogram fit (gstat `fit.method` codes in brackets).
enum class VariogramWeighting {
    pairs,                       ///< N_j (1).
    cressie,                     ///< N_j / gamma(h_j)^2 from the current model, iterated to a fixed point (2).
    uniform,                     ///< Ordinary least squares (6).
    pairs_over_distance_squared, ///< N_j / h_j^2, the gstat default (7).
};

/// @brief Options for fit_variogram().
/// @par Example
/// @code
/// const dm::stats::VariogramFitOptions o{.fit_nugget = false};
/// std::println("{}", o.fit_range); // true
/// @endcode
struct VariogramFitOptions {
    VariogramWeighting weighting{VariogramWeighting::pairs_over_distance_squared}; ///< Bin weights.
    bool fit_nugget{true}; ///< Estimate the nugget (false keeps the initial nugget).
    bool fit_range{true};  ///< Estimate the range (false keeps the initial range).
    std::size_t max_iterations{100}; ///< Reweighting iterations for the Cressie weights.
};

/// @brief Result of fit_variogram().
/// @par Example
/// @code
/// dm::stats::VariogramFit f;
/// std::println("{}", f.converged); // true
/// @endcode
struct VariogramFit {
    VariogramModel model;    ///< Fitted model.
    double weighted_sse{};   ///< Weighted residual sum of squares at the fit.
    std::size_t iterations{}; ///< Reweighting iterations (1 unless Cressie weights are used).
    bool converged{true};    ///< Whether the Cressie reweighting converged.
};

/// @brief Weighted least-squares fit of a variogram model to an empirical variogram.
///
/// Minimises \f$\sum_jw_j(\hat\gamma_j-\gamma(h_j;\theta))^2\f$ over a nonnegative nugget and partial
/// sill and a positive range, with the shape and kappa fixed. For each range the sills are the
/// exact nonnegative weighted least-squares solution, and the range is profiled by a grid of 400
/// log-spaced values between min(h)/10 and 10 max(h) followed by golden-section refinement. This
/// finds the global optimum that `gstat::fit.variogram` approaches iteratively from its starting
/// values. Cressie weights depend on the model and are recomputed from the previous fit, starting from
/// `initial`, until the parameters stop changing. Fit to the variogram of detrended data, and inspect
/// fits that reach a bound (nugget or partial sill 0, or a range at the grid limit).
/// @param bins Empirical variogram (all directions are pooled; distance > 0 required).
/// @param initial Shape, kappa, and the values kept fixed (or used to start Cressie weights).
/// @param options Weighting and which parameters are fitted.
/// @return The fitted model and its weighted SSE.
/// @throws std::invalid_argument For fewer than two bins, nonpositive distances, nonfinite values or
/// invalid initial parameters.
/// @par Example
/// @code
/// using K = dm::stats::VariogramModelKind;
/// const dm::stats::VariogramModel truth{.kind = K::exponential, .nugget = 0.1, .partial_sill = 1, .range = 3};
/// std::vector<dm::stats::VariogramBin> bins;
/// for (double h = 0.5; h < 10; h += 1)
///     bins.push_back({.distance = h, .gamma = dm::stats::variogram_model_value(truth, h), .pairs = 50});
/// const auto f = dm::stats::fit_variogram(bins, {.kind = K::exponential});
/// std::println("{:.3f} {:.3f} {:.3f}", f.model.nugget, f.model.partial_sill, f.model.range); // 0.100 1.000 3.000
/// @endcode
VariogramFit fit_variogram(const std::vector<VariogramBin>& bins, const VariogramModel& initial,
                           const VariogramFitOptions& options = {});

/// @brief Neighbourhood options for kriging.
/// @par Example
/// @code
/// const dm::stats::KrigingOptions o{.max_neighbours = 20};
/// std::println("{}", o.max_neighbours); // 20
/// @endcode
struct KrigingOptions {
    std::size_t max_neighbours{0}; ///< Use only the nearest observations per target (0 uses all, a global neighbourhood).
    double max_distance{std::numeric_limits<double>::infinity()}; ///< Ignore observations farther than this.
};

/// @brief Kriging predictions and variances at target locations.
/// @par Example
/// @code
/// dm::stats::KrigingResult r{.prediction = {1.0}, .variance = {0.2}};
/// std::println("{}", r.prediction.size()); // 1
/// @endcode
struct KrigingResult {
    std::vector<double> prediction; ///< Best linear unbiased predictions (NaN where no observation is in range).
    std::vector<double> variance;   ///< Kriging (prediction error) variances.
    std::vector<double> trend_coefficients; ///< GLS trend estimate from all observations (empty for simple kriging).
};

/// @brief Simple kriging with a known constant mean.
///
/// \f$\hat Z(s_0)=\mu+c_0^\top C^{-1}(z-\mu1)\f$ with variance \f$C(0)-c_0^\top C^{-1}c_0\f$, where C and
/// c_0 come from variogram_model_covariance(). Predictions interpolate the data exactly (a target at an
/// observed location returns the observation with variance 0), as `gstat::krige` with `beta`.
/// @param coordinates n-by-d observation locations.
/// @param values Observations.
/// @param targets m-by-d prediction locations.
/// @param model Bounded variogram model.
/// @param mean Known mean.
/// @param options Neighbourhood.
/// @return Predictions and variances.
/// @throws std::invalid_argument For mismatched sizes or dimensions, nonfinite inputs, or an invalid model.
/// @throws std::runtime_error If a kriging system is singular (for example duplicated locations without a nugget).
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> xy(2, 1, std::vector<double>{0, 10});
/// const dm::linalg::DenseMatrix<double> t(1, 1, std::vector<double>{100});
/// const dm::stats::VariogramModel m{.kind = dm::stats::VariogramModelKind::exponential, .range = 1};
/// std::println("{:.3f}", dm::stats::simple_kriging(xy, {3, 5}, t, m, 1).prediction[0]); // 1.000
/// @endcode
KrigingResult simple_kriging(const linalg::DenseMatrix<double>& coordinates, const std::vector<double>& values,
                             const linalg::DenseMatrix<double>& targets, const VariogramModel& model, double mean,
                             const KrigingOptions& options = {});

/// @brief Ordinary kriging: unknown constant mean, weights constrained to sum to one.
///
/// Equivalent to universal_kriging() with an intercept-only trend; matches `gstat::krige(z ~ 1, ...)`,
/// including `nmax` neighbourhoods (nearest observations, ties by index).
/// @param coordinates n-by-d observation locations.
/// @param values Observations.
/// @param targets m-by-d prediction locations.
/// @param model Bounded variogram model.
/// @param options Neighbourhood.
/// @return Predictions, variances and the GLS mean.
/// @throws std::invalid_argument As for simple_kriging().
/// @throws std::runtime_error As for simple_kriging().
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> xy(3, 1, std::vector<double>{0, 1, 2});
/// const dm::linalg::DenseMatrix<double> t(1, 1, std::vector<double>{1});
/// const dm::stats::VariogramModel m{.kind = dm::stats::VariogramModelKind::exponential, .range = 1};
/// std::println("{:.6f}", dm::stats::ordinary_kriging(xy, {4, 7, 5}, t, m).prediction[0]); // 7.000000
/// @endcode
KrigingResult ordinary_kriging(const linalg::DenseMatrix<double>& coordinates, const std::vector<double>& values,
                               const linalg::DenseMatrix<double>& targets, const VariogramModel& model,
                               const KrigingOptions& options = {});

/// @brief Universal kriging (kriging with an external drift) for a trend linear in known covariates.
///
/// With trend design X (n-by-p) at the observations and x_0 at a target, the predictor is
/// \f$x_0^\top\hat\beta+c_0^\top C^{-1}(z-X\hat\beta)\f$ with GLS
/// \f$\hat\beta=(X^\top C^{-1}X)^{-1}X^\top C^{-1}z\f$ and variance
/// \f$C(0)-c_0^\top C^{-1}c_0+u^\top(X^\top C^{-1}X)^{-1}u\f$, \f$u=x_0-X^\top C^{-1}c_0\f$. The model
/// is the variogram of the residuals from the trend. Include a column of ones for an intercept.
/// Matches `gstat::krige(z ~ covariates, ...)`. With a local neighbourhood, the trend is re-estimated
/// within each neighbourhood.
/// @param coordinates n-by-d observation locations.
/// @param values Observations.
/// @param trend n-by-p trend design at the observations.
/// @param targets m-by-d prediction locations.
/// @param target_trend m-by-p trend design at the targets.
/// @param model Bounded residual variogram model.
/// @param options Neighbourhood.
/// @return Predictions, variances and the global GLS trend coefficients.
/// @throws std::invalid_argument As for simple_kriging(), or if the trend designs do not match.
/// @throws std::runtime_error If a kriging system or X'C^{-1}X is singular.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> xy(4, 1, std::vector<double>{0, 1, 2, 3});
/// const dm::linalg::DenseMatrix<double> x(4, 2, std::vector<double>{1, 0, 1, 1, 1, 2, 1, 3});
/// const dm::linalg::DenseMatrix<double> t(1, 1, std::vector<double>{10});
/// const dm::linalg::DenseMatrix<double> t_x(1, 2, std::vector<double>{1, 10});
/// const dm::stats::VariogramModel m{.kind = dm::stats::VariogramModelKind::exponential, .range = 0.5};
/// const auto r = dm::stats::universal_kriging(xy, {1, 3, 5, 7}, x, t, t_x, m);
/// std::println("{:.3f}", r.prediction[0]); // 21.000
/// @endcode
KrigingResult universal_kriging(const linalg::DenseMatrix<double>& coordinates, const std::vector<double>& values,
                                const linalg::DenseMatrix<double>& trend, const linalg::DenseMatrix<double>& targets,
                                const linalg::DenseMatrix<double>& target_trend, const VariogramModel& model,
                                const KrigingOptions& options = {});

/// @brief Leave-one-out cross-validation of ordinary or universal kriging.
/// @par Example
/// @code
/// dm::stats::KrigingCrossValidation cv{.residual = {0.1, -0.2}};
/// std::println("{}", cv.residual.size()); // 2
/// @endcode
struct KrigingCrossValidation {
    std::vector<double> prediction; ///< Prediction of each observation from the others.
    std::vector<double> variance;   ///< Its kriging variance.
    std::vector<double> residual;   ///< Observed minus predicted.
    std::vector<double> zscore;     ///< Residual divided by the kriging standard deviation.
    double mean_error{};            ///< Mean residual (near 0 for an unbiased predictor).
    double rmse{};                  ///< Root mean squared residual.
    double mean_squared_zscore{};   ///< Mean squared z-score (near 1 when the variances are calibrated).
};

/// @brief Leave-one-out cross-validation of kriging, as `gstat::krige.cv` with `nfold = n`.
///
/// Each observation is predicted from all the others with the same model and neighbourhood, so the
/// trend (or mean) is re-estimated every time. A mean squared z-score far from 1 indicates a misfitting
/// sill or nugget; a large RMSE relative to the data SD indicates weak spatial dependence.
/// @param coordinates n-by-d observation locations.
/// @param values Observations.
/// @param model Bounded variogram model.
/// @param trend n-by-p trend design; an empty matrix (0 rows) means ordinary kriging.
/// @param options Neighbourhood.
/// @return Per-observation predictions, variances, residuals and z-scores with summaries.
/// @throws std::invalid_argument As for universal_kriging(), or for fewer than three observations.
/// @throws std::runtime_error As for universal_kriging().
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> xy(4, 1, std::vector<double>{0, 1, 2, 3});
/// const dm::stats::VariogramModel m{.kind = dm::stats::VariogramModelKind::exponential, .range = 2};
/// const auto cv = dm::stats::kriging_cross_validation(xy, {1, 2, 2, 3}, m);
/// std::println("{}", cv.residual.size()); // 4
/// @endcode
KrigingCrossValidation kriging_cross_validation(const linalg::DenseMatrix<double>& coordinates,
                                                const std::vector<double>& values, const VariogramModel& model,
                                                const linalg::DenseMatrix<double>& trend = {},
                                                const KrigingOptions& options = {});

} // namespace datamunge::stats
