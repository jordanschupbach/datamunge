#pragma once

/// @file gam.hpp
/// @brief Generalized additive regression and random-intercept additive mixed models.

#include <datamunge/fda/bspline.hpp>
#include <datamunge/stats/glm.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Specifies a centered cubic P-spline or a centered linear term.
///
/// @par Example
/// @code
/// const dm::stats::GAMSmooth dose{.basis_size = 9, .lambda = 2};
/// @endcode
/// @see GAMOptions, fit_gam()
struct GAMSmooth {
    std::size_t basis_size{8}; ///< Cubic basis size (at least four), or zero for a linear term.
    double lambda{1}; ///< Nonnegative finite multiplier of summed squared second differences, ignored for linear terms.
};

/// @brief Controls additive term structure and penalized IRLS.
/// @par Example
/// @code
/// const dm::stats::GAMOptions options{.family = dm::stats::GLMFamily::Poisson,
///                                  .smooths = {{.basis_size = 6, .lambda = 2}}};
/// @endcode
/// @see fit_gam(), fit_gamm()
struct GAMOptions {
    GLMFamily family{GLMFamily::Gaussian}; ///< Gaussian, binary Binomial, Poisson or Gamma, with the usual GLM link.
    std::vector<GAMSmooth> smooths; ///< One spec per predictor column, or empty to use defaults for every column.
    std::size_t max_iter{60}; ///< Positive upper bound on penalized IRLS updates.
    double tol{1e-8}; ///< Positive finite relative tolerance for the full working coefficient update.
};

/// @brief Gaussian random-intercept prior and additive fit controls.
/// @par Example
/// @code
/// dm::stats::GAMMOptions options;
/// options.random_intercept_sd = 0.7;
/// @endcode
/// @see fit_gamm()
struct GAMMOptions {
    GAMOptions gam; ///< Fixed-part family, smooth terms and iteration controls.
    double random_intercept_sd{1}; ///< Positive finite group-effect SD on the link scale.
    double dispersion{1}; ///< Positive finite Gaussian/Gamma dispersion for the group penalty. Use one for Poisson/Binomial.
};

/// @brief Fitted additive terms, predictions and optional group effects.
///
/// Models own the training bases, their centers and the coefficient estimates.
/// Fixed terms sum to zero over the training rows. Spline evaluation clamps to the
/// training boundary, and linear terms extrapolate linearly. Group effects can
/// be included for known groups. Predictions without effects are conditional
/// means at group effect zero.
/// @par Example
/// @code
/// const auto model = dm::stats::fit_gam({1, 2, 3, 4}, {{0, 1, 2, 3}},
///                                     {.smooths = {{.basis_size = 0}}});
/// std::print("{:.0f}", model.predict({{2}}).front()); // 3
/// @endcode
/// @see fit_gam(), fit_gamm()
class AdditiveModel {
public:
    /// @brief Predicts response-scale means from columns in training predictor order.
    /// @param predictors One finite vector per fitted additive term, with equal lengths.
    /// @param groups Optional group label per row. Empty uses zero group effects.
    /// Unknown nonempty labels also use zero. Only mixed models accept labels.
    /// @return Owned means in query order.
    /// @throws std::invalid_argument Invalid columns, nonfinite values or invalid group labels.
    /// @throws std::runtime_error A prediction leaves the family domain or overflows.
    /// @par Example
    /// See the AdditiveModel class example.
    /// @see smooth(), fitted_values()
    [[nodiscard]] std::vector<double> predict(const std::vector<std::vector<double>>& predictors,
                                              const std::vector<std::string>& groups = {}) const;

    /// @brief Evaluates the centered contribution of one additive term on the link scale.
    /// @param term Zero-based predictor index.
    /// @param x Finite new coordinates.
    /// @return Owned term contribution at each coordinate.
    /// @throws std::invalid_argument If an index or coordinate is invalid.
    /// @throws std::runtime_error If evaluation overflows.
    /// @par Example
    /// @code
    /// const auto model = dm::stats::fit_gam({1, 2, 3, 4}, {{0, 1, 2, 3}},
    ///                                     {.smooths = {{.basis_size = 0}}});
    /// std::print("{:.1f}", model.smooth(0, {0}).front()); // -1.5
    /// @endcode
    /// @see plot_smooth(), predict()
    [[nodiscard]] std::vector<double> smooth(std::size_t term, const std::vector<double>& x) const;

    /// @brief Plots one fitted additive component over its training range.
    /// @param term Zero-based predictor index.
    /// @param points Number of equally spaced plot points, at least two, default 120.
    /// @return Editable link-scale plot.
    /// @throws std::invalid_argument Invalid term or point count.
    /// @throws std::runtime_error If evaluation overflows.
    /// @par Example
    /// @code
    /// auto fit = dm::stats::fit_gam({1, 2, 3, 4}, {{0, 1, 2, 3}},
    ///                              {.smooths = {{.basis_size = 0}}});
    /// auto plot = fit.plot_smooth(0);
    /// @endcode
    /// @see smooth()
    [[nodiscard]] graphics::RPlot plot_smooth(std::size_t term, std::size_t points = 120) const;

    /// @brief Estimates the intercept before adding any centered term or group effect.
    /// @return Link-scale intercept.
    [[nodiscard]] double intercept() const { return coefficients_.front(); }

    /// @brief Borrows training means, including any fitted group intercepts.
    /// @return Means in training order, valid while the model lives.
    [[nodiscard]] const std::vector<double>& fitted_values() const noexcept { return fitted_; }

    /// @brief Reports whether penalized IRLS met the requested coefficient-change tolerance.
    /// @return True if the stopping criterion was met. This does not test for separation.
    [[nodiscard]] bool converged() const noexcept { return converged_; }

    /// @brief Counts accepted penalized IRLS updates.
    /// @return Number of updates.
    [[nodiscard]] std::size_t iterations() const noexcept { return iterations_; }

    /// @brief Returns the unpenalized conditional deviance at the fitted means.
    /// @return Sum of unit-family deviances.
    [[nodiscard]] double deviance() const noexcept { return deviance_; }

    /// @brief Borrows group labels in the same order as random_intercepts().
    /// @return Training group labels in sorted order or empty for a GAM.
    [[nodiscard]] const std::vector<std::string>& group_levels() const noexcept { return group_levels_; }

    /// @brief Returns fitted group effects in group_levels() order.
    /// @return Link-scale group modes or an empty vector for a GAM.
    /// @par Example
    /// See fit_gamm().
    [[nodiscard]] std::vector<double> random_intercepts() const;

private:
    /// @brief Owns one training basis and the column means removed from its evaluations.
    struct Term {
        fda::BSplineBasis basis; ///< Cubic basis when spline is true.
        bool spline; ///< True for a cubic spline, false for a linear term.
        std::vector<double> centers; ///< Mean of each included design column.
        double lower; ///< Minimum training coordinate.
        double upper; ///< Maximum training coordinate.
        std::size_t offset; ///< Index of the first coefficient for this term.
    };

    /// @brief Fits a validated additive design with an optional diagonal group penalty.
    /// @param y Training response.
    /// @param predictors Predictor columns.
    /// @param groups Empty or training group labels.
    /// @param group_penalty Group penalty precision, used when groups are supplied.
    /// @param options Family, terms and solver controls.
    AdditiveModel(const std::vector<double>& y, const std::vector<std::vector<double>>& predictors,
                  const std::vector<std::string>& groups, double group_penalty, const GAMOptions& options);

    /// @brief Evaluates the design columns for a single additive term.
    /// @param term Stored training term.
    /// @param x Finite coordinate.
    /// @return Owned centered design row.
    [[nodiscard]] static std::vector<double> term_row(const Term& term, double x);

    GLMFamily family_; ///< Fixed family and link.
    std::vector<Term> terms_; ///< Additive terms in input order.
    std::vector<double> coefficients_; ///< Intercept, term coefficients, then optional group effects.
    std::vector<double> fitted_; ///< Fitted conditional means.
    std::vector<std::string> group_levels_; ///< Sorted observed group labels.
    bool converged_{false}; ///< Whether the configured stopping tolerance was met.
    std::size_t iterations_{0}; ///< Accepted update count.
    double deviance_{0}; ///< Unpenalized fitted deviance.

    friend AdditiveModel fit_gam(const std::vector<double>&, const std::vector<std::vector<double>>&, const GAMOptions&);
    friend AdditiveModel fit_gamm(const std::vector<double>&, const std::vector<std::vector<double>>&,
                                   const std::vector<std::string>&, const GAMMOptions&);
};

/// @brief Fits a generalized additive model with fixed spline penalties.
/// @param y Nonempty finite response in the selected GLM family domain.
/// @param predictors Nonempty predictor columns, each finite, varying, and the same length as y.
/// @param options Family, additive terms and IRLS controls. Defaults to cubic splines and Gaussian response.
/// @return Owned fitted model. Check converged() before use.
/// @throws std::invalid_argument For invalid data, shapes or options.
/// @throws std::runtime_error For nonidentifiable fits or invalid numerical steps.
/// @par Example
/// See the AdditiveModel class example.
/// @par References
/// - Eilers and Marx (1996), Flexible smoothing with B-splines and penalties.
///   https://doi.org/10.1214/ss/1038425655
/// @see fit_gamm(), GLM
AdditiveModel fit_gam(const std::vector<double>& y, const std::vector<std::vector<double>>& predictors,
                      const GAMOptions& options = {});

/// @brief Fits a generalized additive mixed model with independent Gaussian group intercepts.
///
/// Fits conditional modes jointly with the spline terms. The group ridge penalty
/// is dispersion divided by random-intercept variance. Both variance and smoothness
/// controls are supplied by the caller.
/// @param y Nonempty finite response in the selected GLM family domain.
/// @param predictors Nonempty predictor columns, each finite, varying, and the same length as y.
/// @param groups One nonempty label per response, with at least two distinct groups.
/// @param options Family, smoothness, dispersion and group SD controls.
/// @return Owned fitted model with group modes. Check converged() before use.
/// @throws std::invalid_argument For invalid data, shapes or options.
/// @throws std::runtime_error For nonidentifiable fits or invalid numerical steps.
/// @par Example
/// @code
/// dm::stats::GAMMOptions options;
/// options.gam.smooths = {{.basis_size = 0}};
/// auto fit = dm::stats::fit_gamm({1, 2, 3, 4}, {{0, 1, 2, 3}}, {"A", "A", "B", "B"}, options);
/// std::print("{}", fit.group_levels().size()); // 2
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/random.effects.html
/// @see fit_gam(), GLMM
AdditiveModel fit_gamm(const std::vector<double>& y, const std::vector<std::vector<double>>& predictors,
                       const std::vector<std::string>& groups, const GAMMOptions& options = {});

} // namespace datamunge::stats
