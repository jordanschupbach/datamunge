#pragma once

/// @file item_response.hpp
/// @brief Item-response models fitted by marginal maximum likelihood: Rasch and two-parameter logistic for
/// binary items, Samejima's graded response model for ordered categories; person scores and information.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <limits>
#include <vector>

namespace datamunge::stats {

/// @brief Which binary IRT model to fit.
/// @par Example
/// @code
/// const auto m = dm::stats::IrtModel::rasch;
/// std::println("{}", m == dm::stats::IrtModel::rasch); // true
/// @endcode
enum class IrtModel {
    rasch,         ///< One discrimination shared by all items (fixed or estimated).
    two_parameter, ///< Item-specific discriminations (2PL).
};

/// @brief Options for fit_irt().
/// @par Example
/// @code
/// dm::stats::IrtOptions o{.model = dm::stats::IrtModel::rasch, .rasch_discrimination = 1};
/// std::println("{}", o.rasch_discrimination); // 1
/// @endcode
struct IrtOptions {
    IrtModel model{IrtModel::two_parameter}; ///< Model.
    double rasch_discrimination{std::numeric_limits<double>::quiet_NaN()}; ///< Rasch: fixed common slope, or NaN to estimate it.
    std::size_t quadrature_points{21};   ///< Gauss–Hermite points for the N(0, 1) latent distribution.
    std::size_t max_iterations{500};     ///< Newton iterations.
    double tolerance{1e-9};              ///< Convergence tolerance on the largest gradient component per person.
};

/// @brief Parameters of one item, \f$P(X=1\mid\theta)=\operatorname{logit}^{-1}(a(\theta-b))\f$.
/// @par Example
/// @code
/// dm::stats::IrtItem i{.difficulty = 0, .discrimination = 1};
/// std::println("{}", dm::stats::irt_probability(i, 0)); // 0.5
/// @endcode
struct IrtItem {
    double difficulty{};       ///< b: the ability with a 50% chance of a correct response.
    double discrimination{1};  ///< a: slope of the item characteristic curve at b.
    double difficulty_se{};    ///< Standard error of b (delta method from the observed information).
    double discrimination_se{}; ///< Standard error of a (NaN when fixed).
};

/// @brief Result of fit_irt().
/// @par Example
/// @code
/// dm::stats::IrtFit f;
/// std::println("{}", f.items.size()); // 0
/// @endcode
struct IrtFit {
    IrtModel model{};              ///< Fitted model.
    std::size_t n{};               ///< Respondents.
    std::vector<IrtItem> items;    ///< Item parameters.
    double log_likelihood{};       ///< Maximized marginal log-likelihood.
    std::size_t parameters{};      ///< Number of estimated parameters.
    double aic{};                  ///< -2 logL + 2 k.
    double bic{};                  ///< -2 logL + k log n.
    std::size_t iterations{};      ///< Newton iterations.
    bool converged{};              ///< Whether the gradient criterion was met.
    std::size_t quadrature_points{}; ///< Quadrature used.
};

/// @brief Fits a Rasch or 2PL model by marginal maximum likelihood.
///
/// The latent trait is N(0, 1) and integrated out by Gauss–Hermite quadrature; identical response patterns are
/// grouped. The likelihood is maximized in the slope–intercept form \f$c_j+a_j\theta\f$ by damped Newton steps
/// on the analytic gradient (Fisher's identity) with a finite-difference Hessian; difficulties are
/// \f$b_j=-c_j/a_j\f$ and their standard errors use the delta method. This is the estimator of `ltm::rasch`
/// and `ltm::ltm(X ~ z1)` with the same number of quadrature points.
/// @param responses n-by-p matrix of 0/1 responses (p >= 2; every item must have both responses).
/// @param options Model and numerical options.
/// @return The fit.
/// @throws std::invalid_argument For non-binary data, an item answered all correct or all wrong, a nonpositive
/// fixed discrimination or no quadrature points.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(6, 2, {1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0});
/// const auto f = dm::stats::fit_irt(x, {.model = dm::stats::IrtModel::rasch, .rasch_discrimination = 1});
/// std::println("{:.3f}", f.items[0].difficulty); // -0.000
/// @endcode
IrtFit fit_irt(const linalg::DenseMatrix<double>& responses, const IrtOptions& options = {});

/// @brief Probability of a correct response.
/// @param item Item parameters.
/// @param theta Ability.
/// @return \f$\operatorname{logit}^{-1}(a(\theta-b))\f$.
/// @par Example
/// @code
/// std::println("{:.3f}", dm::stats::irt_probability({.difficulty = 1, .discrimination = 2}, 1.5)); // 0.731
/// @endcode
double irt_probability(const IrtItem& item, double theta);

/// @brief Test information \f$I(\theta)=\sum_ja_j^2P_j(1-P_j)\f$; its inverse square root is the SE of the ML
/// ability estimate.
/// @param fit A fit.
/// @param theta Ability.
/// @return Information.
/// @par Example
/// @code
/// dm::stats::IrtFit f;
/// f.items = {{.difficulty = 0, .discrimination = 2}};
/// std::println("{}", dm::stats::irt_test_information(f, 0)); // 1
/// @endcode
double irt_test_information(const IrtFit& fit, double theta);

/// @brief Ability estimates of one respondent.
/// @par Example
/// @code
/// dm::stats::IrtScore s{.eap = 0.5};
/// std::println("{}", s.eap); // 0.5
/// @endcode
struct IrtScore {
    double map{};    ///< Posterior mode (empirical Bayes, `ltm` "EB").
    double map_se{}; ///< Inverse square root of the posterior curvature at the mode.
    double eap{};    ///< Posterior mean (quadrature).
    double eap_sd{}; ///< Posterior standard deviation.
};

/// @brief Posterior ability scores under a N(0, 1) prior.
/// @param fit A fit.
/// @param responses Rows of 0/1 responses (same items as the fit).
/// @param quadrature_points Gauss–Hermite points for EAP (default as the fit's quadrature).
/// @return One score per row.
/// @throws std::invalid_argument On an item-count mismatch or non-binary data.
/// @par Example
/// @code
/// dm::stats::IrtFit f;
/// f.items = {{.difficulty = 0, .discrimination = 1}};
/// f.quadrature_points = 21;
/// const auto s = dm::stats::irt_scores(f, dm::linalg::DenseMatrix<double>(1, 1, {1.0}));
/// std::println("{:.3f}", s[0].map); // 0.401
/// @endcode
std::vector<IrtScore> irt_scores(const IrtFit& fit, const linalg::DenseMatrix<double>& responses,
                                 std::size_t quadrature_points = 0);

/// @brief Parameters of one graded (ordered-category) item, \f$P(X\ge k\mid\theta)=\operatorname{logit}^{-1}(a(\theta-b_k))\f$.
/// @par Example
/// @code
/// dm::stats::GradedItem i{.thresholds = {-1, 1}, .discrimination = 1};
/// std::println("{:.3f}", dm::stats::graded_category_probabilities(i, 0)[1]); // 0.462
/// @endcode
struct GradedItem {
    std::vector<double> thresholds;   ///< Increasing b_1..b_{K-1}: the ability at which P(X >= k) = 1/2.
    double discrimination{1};         ///< Common slope a of the item's cumulative curves.
    std::vector<double> threshold_se; ///< Standard errors of the thresholds (delta method).
    double discrimination_se{};       ///< Standard error of a.
};

/// @brief Options for fit_graded_response().
/// @par Example
/// @code
/// dm::stats::GradedOptions o{.common_discrimination = true};
/// std::println("{}", o.common_discrimination); // true
/// @endcode
struct GradedOptions {
    bool common_discrimination{false}; ///< Constrain all items to one discrimination.
    std::size_t quadrature_points{21}; ///< Gauss–Hermite points for the N(0, 1) latent distribution.
    std::size_t max_iterations{500};   ///< Newton iterations.
    double tolerance{1e-9};            ///< Convergence tolerance on the largest gradient component per person.
};

/// @brief Result of fit_graded_response().
/// @par Example
/// @code
/// dm::stats::GradedFit f;
/// std::println("{}", f.items.size()); // 0
/// @endcode
struct GradedFit {
    std::size_t n{};                 ///< Respondents.
    std::vector<GradedItem> items;   ///< Item parameters.
    double log_likelihood{};         ///< Maximized marginal log-likelihood.
    std::size_t parameters{};        ///< Number of estimated parameters.
    double aic{};                    ///< -2 logL + 2 k.
    double bic{};                    ///< -2 logL + k log n.
    std::size_t iterations{};        ///< Newton iterations.
    bool converged{};                ///< Whether the gradient criterion was met.
    std::size_t quadrature_points{}; ///< Quadrature used.
};

/// @brief Fits Samejima's graded response model by marginal maximum likelihood.
///
/// Item j has categories 0..K_j-1 and cumulative curves \f$P(X_j\ge k\mid\theta)=\operatorname{logit}^{-1}
/// (a_j(\theta-b_{jk}))\f$, so category probabilities are differences of adjacent curves. The latent trait is
/// N(0, 1), integrated by Gauss–Hermite quadrature over grouped response patterns; the likelihood is maximized
/// in slope–intercept form by damped Newton steps on the analytic gradient with a finite-difference Hessian.
/// This is the estimator of `ltm::grm` (optionally `constrained = TRUE`). With two categories it is the 2PL.
/// @param responses n-by-p matrix of integer categories starting at 0; every category from 0 to an item's
/// maximum must occur.
/// @param options Options.
/// @return The fit.
/// @throws std::invalid_argument For non-integer or negative codes, an item with fewer than two categories or
/// an unobserved intermediate category.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(6, 2, {0, 0, 1, 1, 2, 2, 0, 1, 1, 2, 2, 1});
/// std::println("{}", dm::stats::fit_graded_response(x).items[0].thresholds.size()); // 2
/// @endcode
GradedFit fit_graded_response(const linalg::DenseMatrix<double>& responses, const GradedOptions& options = {});

/// @brief Category response curves of a graded item at one ability.
/// @param item Item.
/// @param theta Ability.
/// @return K probabilities summing to 1.
/// @par Example
/// @code
/// const auto p = dm::stats::graded_category_probabilities({.thresholds = {0}, .discrimination = 1}, 0);
/// std::println("{} {}", p[0], p[1]); // 0.5 0.5
/// @endcode
std::vector<double> graded_category_probabilities(const GradedItem& item, double theta);

/// @brief Fisher information of a graded item, \f$\sum_k (P_k')^2/P_k\f$.
/// @param item Item.
/// @param theta Ability.
/// @return Information.
/// @par Example
/// @code
/// std::println("{}", dm::stats::graded_item_information({.thresholds = {0}, .discrimination = 2}, 0)); // 1
/// @endcode
double graded_item_information(const GradedItem& item, double theta);

/// @brief Test information: the sum of item informations.
/// @param fit A fit.
/// @param theta Ability.
/// @return Information.
/// @par Example
/// @code
/// dm::stats::GradedFit f;
/// f.items = {{.thresholds = {0}, .discrimination = 2}, {.thresholds = {0}, .discrimination = 2}};
/// std::println("{}", dm::stats::graded_test_information(f, 0)); // 2
/// @endcode
double graded_test_information(const GradedFit& fit, double theta);

/// @brief Posterior ability scores (MAP and EAP) under a N(0, 1) prior for graded responses.
/// @param fit A fit.
/// @param responses Rows of category codes.
/// @param quadrature_points Gauss–Hermite points for EAP (0: the fit's).
/// @return One score per row.
/// @throws std::invalid_argument On an item-count mismatch or a code outside an item's categories.
/// @par Example
/// @code
/// dm::stats::GradedFit f;
/// f.items = {{.thresholds = {-1, 1}, .discrimination = 1}};
/// f.quadrature_points = 21;
/// const auto s = dm::stats::graded_scores(f, dm::linalg::DenseMatrix<double>(1, 1, {1.0}));
/// std::println("{:.3f}", std::abs(s[0].eap)); // 0.000
/// @endcode
std::vector<IrtScore> graded_scores(const GradedFit& fit, const linalg::DenseMatrix<double>& responses,
                                    std::size_t quadrature_points = 0);

} // namespace datamunge::stats
