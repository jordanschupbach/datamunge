#pragma once

/// @file survey.hpp
/// @brief Design-based estimation for probability samples: Horvitz–Thompson and Hájek estimators with with-replacement or exact (Sen–Yates–Grundy or Horvitz–Thompson) variances, Taylor-linearized totals, means, ratios and domain estimates for stratified multistage designs, and replicate-weight (BRR, Fay, jackknife) variances.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <optional>
#include <vector>

namespace datamunge::stats {

/// @brief A design-based point estimate with its standard error and normal-theory interval.
/// @par Example
/// @code
/// const dm::stats::SurveyEstimate e{.estimate = 10, .standard_error = 2, .variance = 4};
/// std::println("{}", e.estimate); // 10
/// @endcode
struct SurveyEstimate {
    double estimate{};        ///< Point estimate.
    double standard_error{};  ///< Estimated standard error (0 when the variance estimate is negative).
    double lower{};           ///< Lower interval end.
    double upper{};           ///< Upper interval end.
    double variance{};        ///< Variance estimate as computed, which an exact estimator can make negative.
};

/// @brief Options for horvitz_thompson().
/// @par Example
/// @code
/// const dm::stats::HorvitzThompsonOptions o{.population_size = 6194};
/// std::println("{}", *o.population_size); // 6194
/// @endcode
struct HorvitzThompsonOptions {
    std::optional<double> population_size{}; ///< Known population size N, enabling the Horvitz–Thompson mean.
    double confidence{0.95};                 ///< Coverage of the normal intervals.
};

/// @brief Horvitz–Thompson total and the two design-weighted means.
/// @par Example
/// @code
/// const dm::stats::HorvitzThompsonResult r;
/// std::println("{}", r.ht_mean.has_value()); // false
/// @endcode
struct HorvitzThompsonResult {
    SurveyEstimate total;                  ///< \f$\hat t_y=\sum_i y_i/\pi_i\f$.
    SurveyEstimate hajek_mean;             ///< \f$\hat t_y/\hat N\f$, a ratio estimator with linearized variance.
    std::optional<SurveyEstimate> ht_mean; ///< \f$\hat t_y/N\f$ when N is known.
    double estimated_population_size{};    ///< \f$\hat N=\sum_i 1/\pi_i\f$.
    std::size_t sample_size{};             ///< Number of sampled units.
    bool negative_variance{false};         ///< True if an exact variance estimate was negative (its standard error is then reported as 0).
};

/// @brief Horvitz–Thompson and Hájek estimation from a sample with unequal inclusion probabilities.
///
/// Each sampled unit stands for \f$1/\pi_i\f$ population units. The total estimator
/// \f$\hat t_y=\sum y_i/\pi_i\f$ is design-unbiased for any design with positive inclusion
/// probabilities. The Hájek mean \f$\sum (y_i/\pi_i)/\sum(1/\pi_i)\f$ divides by the estimated
/// population size instead of the true one. It is slightly biased, but much more stable when the
/// sample size or the \f$y\f$ level varies with the weights. Its variance is linearized with
/// residuals \f$(y_i-\hat{\bar y})/(\pi_i\hat N)\f$.
///
/// Standard errors use the with-replacement approximation
/// \f$\frac{n}{n-1}\sum_i(z_i-\bar z)^2\f$ on the weighted values \f$z_i\f$. This is
/// `survey::svydesign(ids = ~1, probs = ~pi)` without a finite-population correction. It is
/// conservative for without-replacement designs with small sampling fractions. Exact
/// fixed-size-design variances need joint inclusion probabilities.
/// @param y Study variable for each sampled unit.
/// @param inclusion_probabilities First-order inclusion probability of each sampled unit, in (0, 1].
/// @param options Known population size and interval coverage.
/// @return The total, Hájek mean, Horvitz–Thompson mean (with N) and \f$\hat N\f$.
/// @throws std::invalid_argument For mismatched sizes, fewer than two units, non-finite values,
/// probabilities outside (0, 1] or a nonpositive population size or invalid confidence.
/// @par Example
/// @code
/// const auto r = dm::stats::horvitz_thompson({3, 5, 10}, {0.5, 0.25, 0.1}, {.population_size = 20});
/// std::println("{} {:.3f} {}", r.total.estimate, r.hajek_mean.estimate, r.ht_mean->estimate); // 126 7.875 6.3
/// @endcode
/// @par References
/// - D. G. Horvitz and D. J. Thompson (1952). "A generalization of sampling without replacement
///   from a finite universe." JASA 47(260), 663-685. doi:10.1080/01621459.1952.10483446
/// - C.-E. Särndal, B. Swensson and J. Wretman (1992). Model Assisted Survey Sampling. Springer.
///   doi:10.1007/978-1-4612-4378-6
[[nodiscard]] HorvitzThompsonResult horvitz_thompson(const std::vector<double>& y,
                                                     const std::vector<double>& inclusion_probabilities,
                                                     const HorvitzThompsonOptions& options = {});

/// @brief Exact design-variance estimator for horvitz_thompson_joint().
/// @par Example
/// @code
/// const auto m = dm::stats::SurveyVariance::sen_yates_grundy;
/// std::println("{}", m == dm::stats::SurveyVariance::horvitz_thompson); // false
/// @endcode
enum class SurveyVariance {
    sen_yates_grundy, ///< \f$\tfrac12\sum_{i\ne j}\frac{\pi_i\pi_j-\pi_{ij}}{\pi_{ij}}(z_i-z_j)^2\f$; unbiased for fixed-size designs.
    horvitz_thompson  ///< \f$\sum_i(1-\pi_i)z_i^2+\sum_{i\ne j}\frac{\pi_{ij}-\pi_i\pi_j}{\pi_{ij}}z_iz_j\f$; unbiased for any design.
};

/// @brief Options for horvitz_thompson_joint().
/// @par Example
/// @code
/// const dm::stats::JointInclusionOptions o{.method = dm::stats::SurveyVariance::horvitz_thompson};
/// std::println("{}", o.confidence); // 0.95
/// @endcode
struct JointInclusionOptions {
    SurveyVariance method{SurveyVariance::sen_yates_grundy}; ///< Variance estimator.
    std::optional<double> population_size{};                 ///< Known population size N, enabling the Horvitz–Thompson mean.
    double confidence{0.95};                                 ///< Coverage of the normal intervals.
};

/// @brief Horvitz–Thompson and Hájek estimation with exact variances from joint inclusion probabilities.
///
/// Uses the same point estimates as horvitz_thompson(), with variances estimated from the
/// second-order inclusion probabilities \f$\pi_{ij}\f$ instead of the with-replacement
/// approximation, as `survey::svydesign(pps = ppsmat(joint), variance = "YG" or "HT")`. With
/// \f$z_i=u_i/\pi_i\f$, the Sen–Yates–Grundy form uses only pairwise differences. It is exactly
/// unbiased, and it is nonnegative when \f$\pi_i\pi_j\ge\pi_{ij}\f$, for fixed-size designs. The
/// Horvitz–Thompson form is unbiased for any design, but it can be negative and is usually less
/// stable. The Hájek mean applies the chosen form to linearized residuals
/// \f$u_i=(y_i-\hat{\bar y})/\hat N\f$. Every \f$\pi_{ij}\f$ must be positive, or the design admits no
/// unbiased variance estimator.
/// @param y Study variable for each sampled unit.
/// @param joint_inclusion Symmetric matrix of joint inclusion probabilities of the sampled units; the
/// diagonal holds the first-order probabilities.
/// @param options Variance form, known population size and interval coverage.
/// @return Estimates as in horvitz_thompson(), with a flag for negative variance estimates.
/// @throws std::invalid_argument For mismatched sizes, fewer than two units, non-finite values, an
/// asymmetric matrix, probabilities outside (0, 1], joint probabilities outside (0, min(pi_i, pi_j)],
/// or invalid population size or confidence.
/// @par Example
/// Two of four units drawn by simple random sampling have \f$\pi_i=1/2\f$ and \f$\pi_{ij}=1/6\f$.
/// @code
/// const dm::linalg::DenseMatrix<double> j(2, 2, {0.5, 1.0 / 6, 1.0 / 6, 0.5});
/// const auto r = dm::stats::horvitz_thompson_joint({3, 7}, j, {.population_size = 4});
/// std::println("{} {:.4f}", r.total.estimate, r.ht_mean->standard_error); // 20 1.4142
/// @endcode
/// @par References
/// - A. R. Sen (1953). "On the estimate of the variance in sampling with varying probabilities."
///   Journal of the Indian Society of Agricultural Statistics 5, 119-127.
/// - F. Yates and P. M. Grundy (1953). "Selection without replacement from within strata with
///   probability proportional to size." JRSS B 15(2), 253-261. doi:10.1111/j.2517-6161.1953.tb00140.x
/// @see horvitz_thompson()
[[nodiscard]] HorvitzThompsonResult horvitz_thompson_joint(const std::vector<double>& y,
                                                           const linalg::DenseMatrix<double>& joint_inclusion,
                                                           const JointInclusionOptions& options = {});

/// @brief Treatment of strata with a single sampled primary sampling unit, as `options(survey.lonely.psu)`.
/// @par Example
/// @code
/// const auto l = dm::stats::LonelyPsu::certainty;
/// std::println("{}", l == dm::stats::LonelyPsu::fail); // false
/// @endcode
enum class LonelyPsu {
    fail,     ///< Throw: the stratum's variance cannot be estimated.
    certainty ///< Contribute zero, as for a PSU selected with certainty.
};

/// @brief A stratified multistage sample described by its first stage, for Taylor linearization.
///
/// Rows are sampled units. Variances use the ultimate-cluster approximation: weighted PSU totals
/// vary independently within strata, which is exact for with-replacement first-stage sampling.
/// A finite-population correction is applied at the first stage when population PSU counts are
/// given.
/// @par Example
/// @code
/// const dm::stats::SurveyDesign d{.weights = {10, 10, 20, 20}, .strata = {1, 1, 2, 2}};
/// std::println("{}", d.clusters.empty()); // true
/// @endcode
struct SurveyDesign {
    std::vector<double> weights;              ///< Sampling weight (inverse inclusion probability) per unit, positive.
    std::vector<int> strata{};                ///< Stratum label per unit (empty: one stratum).
    std::vector<int> clusters{};              ///< Primary sampling unit label per unit, read within its stratum as `nest = TRUE` (empty: each unit is its own PSU).
    std::vector<double> population_psus{};    ///< Number of PSUs in the population of each unit's stratum (empty: no correction).
    LonelyPsu lonely_psu{LonelyPsu::fail};    ///< Policy for strata with one sampled PSU.
    double confidence{0.95};                  ///< Coverage of the normal intervals.
};

/// @brief Design-weighted total of a variable, optionally within a domain, with a linearized standard error.
///
/// \f$\hat t=\sum_i w_id_iy_i\f$, where \f$d_i\f$ indicates domain membership. The variance is
/// \f$\sum_h(1-f_h)\frac{n_h}{n_h-1}\sum_c(z_{hc}-\bar z_h)^2\f$ over the PSU totals \f$z_{hc}\f$ of
/// \f$w_id_iy_i\f$, as `survey::svytotal`. Units outside the domain stay in the design with value 0,
/// so the domain's random sample size is reflected in the variance.
/// @param design Weights, strata, clusters and correction.
/// @param y Variable per unit.
/// @param domain Optional domain membership per unit (empty: everyone).
/// @return The estimate with standard error and normal interval.
/// @throws std::invalid_argument For an invalid design or mismatched sizes, an empty domain, or a lonely
/// PSU under LonelyPsu::fail.
/// @par Example
/// @code
/// const dm::stats::SurveyDesign d{.weights = {10, 10, 20, 20}, .strata = {1, 1, 2, 2}};
/// std::println("{:.2f}", dm::stats::survey_total(d, {1, 3, 2, 6}).estimate); // 200.00
/// @endcode
/// @par References
/// - K. M. Wolter (2007). Introduction to Variance Estimation, 2nd ed. Springer, chapter 6.
///   doi:10.1007/978-0-387-35099-8
/// - T. Lumley (2004). "Analysis of complex survey samples." Journal of Statistical Software 9(8).
///   doi:10.18637/jss.v009.i08
/// @see survey_mean(), survey_ratio()
[[nodiscard]] SurveyEstimate survey_total(const SurveyDesign& design, const std::vector<double>& y,
                                          const std::vector<bool>& domain = {});

/// @brief Design-weighted (Hájek) mean, optionally within a domain, with a linearized standard error.
///
/// The mean \f$\sum w_id_iy_i/\sum w_id_i\f$ is a ratio. Its linearized values are
/// \f$w_id_i(y_i-\hat\mu)/\hat N_d\f$, which go into the same stratified PSU-total variance as
/// survey_total(). This is `survey::svymean`, and `svyby` for domains.
/// @param design Weights, strata, clusters and correction.
/// @param y Variable per unit.
/// @param domain Optional domain membership per unit (empty: everyone).
/// @return The estimate with standard error and normal interval.
/// @throws std::invalid_argument As survey_total().
/// @par Example
/// @code
/// const dm::stats::SurveyDesign d{.weights = {10, 10, 20, 20}, .strata = {1, 1, 2, 2}};
/// std::println("{:.4f}", dm::stats::survey_mean(d, {1, 3, 2, 6}).estimate); // 3.3333
/// @endcode
/// @see survey_total(), survey_ratio()
[[nodiscard]] SurveyEstimate survey_mean(const SurveyDesign& design, const std::vector<double>& y,
                                         const std::vector<bool>& domain = {});

/// @brief Ratio of two design-weighted totals, optionally within a domain, with a linearized standard error.
///
/// \f$\hat R=\hat t_y/\hat t_x\f$, with linearized values \f$w_id_i(y_i-\hat Rx_i)/\hat t_x\f$, as
/// `survey::svyratio`.
/// @param design Weights, strata, clusters and correction.
/// @param numerator Numerator variable y per unit.
/// @param denominator Denominator variable x per unit (its weighted total must be nonzero).
/// @param domain Optional domain membership per unit (empty: everyone).
/// @return The estimate with standard error and normal interval.
/// @throws std::invalid_argument As survey_total(), or for a zero denominator total.
/// @par Example
/// @code
/// const dm::stats::SurveyDesign d{.weights = {10, 10, 20, 20}, .strata = {1, 1, 2, 2}};
/// std::println("{:.4f}", dm::stats::survey_ratio(d, {1, 3, 2, 6}, {2, 2, 4, 8}).estimate); // 0.7143
/// @endcode
/// @see survey_total(), survey_mean()
[[nodiscard]] SurveyEstimate survey_ratio(const SurveyDesign& design, const std::vector<double>& numerator,
                                          const std::vector<double>& denominator,
                                          const std::vector<bool>& domain = {});

/// @brief Linearized covariance matrix of the totals of several per-unit score variables.
///
/// For scores \(z_{ij}\) (already multiplied by the weights, so that the statistic is
/// \(\sum_iz_{i\cdot}\)), sums them to PSU totals and returns the ultimate-cluster covariance
/// \(\sum_h(1-f_h)\frac{n_h}{n_h-1}\sum_c(z_{hc}-\bar z_h)(z_{hc}-\bar z_h)^\top\), the
/// multivariate form of the variance used by survey_total() and survey_mean(). Supplying
/// weighted influence functions gives the covariance of any smooth estimator, such as a vector of
/// cell proportions or regression coefficients (the "meat" of a design-based sandwich).
/// @param design Sampling design.
/// @param scores Weighted scores, one row per unit and one column per statistic.
/// @return Symmetric covariance matrix (columns by columns).
/// @throws std::invalid_argument For an invalid design, mismatched rows, non-finite scores, or
/// a single-PSU stratum under LonelyPsu::fail.
/// @par Example
/// @code
/// const dm::stats::SurveyDesign d{.weights = {2, 2, 2, 2}};
/// const dm::linalg::DenseMatrix<double> z(4, 1, {2, 4, 6, 8});
/// std::println("{:.4f}", dm::stats::survey_score_covariance(d, z)(0, 0)); // 26.6667
/// @endcode
/// @see survey_degrees_of_freedom()
[[nodiscard]] linalg::DenseMatrix<double> survey_score_covariance(const SurveyDesign& design,
                                                                  const linalg::DenseMatrix<double>& scores);

/// @brief Design degrees of freedom: number of PSUs minus number of strata, as `survey::degf`.
///
/// With a domain, only PSUs and strata that contain domain units are counted, as `degf()` of a
/// subsetted `survey` design.
/// @param design Sampling design.
/// @param domain Units in the domain (empty: all).
/// @return Degrees of freedom for t and F reference distributions.
/// @throws std::invalid_argument For an invalid design or a domain of the wrong length.
/// @par Example
/// @code
/// const dm::stats::SurveyDesign d{.weights = {1, 1, 1, 1}, .strata = {0, 0, 1, 1}};
/// std::println("{}", dm::stats::survey_degrees_of_freedom(d)); // 2
/// @endcode
/// @see survey_score_covariance()
[[nodiscard]] double survey_degrees_of_freedom(const SurveyDesign& design, const std::vector<bool>& domain = {});

/// @brief Replicate-weight construction method, as `survey::as.svrepdesign(type =)`.
/// @par Example
/// @code
/// const auto m = dm::stats::ReplicateMethod::fay;
/// std::println("{}", m == dm::stats::ReplicateMethod::brr); // false
/// @endcode
enum class ReplicateMethod {
    brr, ///< Balanced repeated replication: exactly two PSUs per stratum, half-samples from a Hadamard matrix.
    fay, ///< Fay's BRR: the dropped half keeps weight factor rho and the kept half 2 - rho.
    jk1, ///< Delete-one-PSU jackknife for an unstratified design.
    jkn  ///< Stratified delete-one-PSU jackknife.
};

/// @brief Options for make_replicate_design().
/// @par Example
/// @code
/// const dm::stats::ReplicateOptions o{.method = dm::stats::ReplicateMethod::fay, .fay_rho = 0.3};
/// std::println("{}", o.fay_rho); // 0.3
/// @endcode
struct ReplicateOptions {
    ReplicateMethod method{ReplicateMethod::jkn}; ///< Construction method.
    double fay_rho{0.5};                          ///< Fay perturbation factor in [0, 1) (Fay only).
    bool mse{false};                              ///< Centre on the full-sample estimate instead of the replicate mean.
};

/// @brief Full-sample and replicate weights with the variance scaling, usable with published replicate weights.
///
/// The variance of an estimator \f$\hat\theta\f$ is \f$c\sum_r a_r(\hat\theta_r-\tilde\theta)^2\f$,
/// where \f$\hat\theta_r\f$ is recomputed with replicate weights \f$r\f$, \f$c\f$ is `scale`, \f$a_r\f$
/// is `rscales`, and \f$\tilde\theta\f$ is the replicate mean (or \f$\hat\theta\f$ when `mse`). This is
/// `survey::svrepdesign`.
/// @par Example
/// @code
/// const dm::stats::ReplicateDesign d{.weights = {2, 2},
///                                    .replicate_weights = dm::linalg::DenseMatrix<double>(2, 2, {4, 0, 0, 4})};
/// std::println("{}", d.scale); // 1
/// @endcode
struct ReplicateDesign {
    std::vector<double> weights;                   ///< Full-sample weight per unit.
    linalg::DenseMatrix<double> replicate_weights; ///< Units by replicates.
    double scale{1};                               ///< Overall multiplier c.
    std::vector<double> rscales{};                 ///< Per-replicate multipliers (empty: all 1).
    bool mse{false};                               ///< Centre on the full-sample estimate.
    double confidence{0.95};                       ///< Coverage of the normal intervals.
};

/// @brief Builds BRR, Fay or jackknife replicate weights from a stratified cluster design.
///
/// BRR and Fay need exactly two PSUs per stratum. Replicates are the rows of a normalized Hadamard
/// matrix of the smallest available order (Sylvester doublings of 1, 2 or a Paley order) above the
/// number of strata. Stratum \f$h\f$ uses column \f$h+1\f$, and the first PSU in the stratum is kept
/// when the entry is +1. The scale is \f$1/R\f$ for BRR and \f$1/\{R(1-\rho)^2\}\f$ for Fay. JK1 deletes one
/// PSU at a time and reweights the rest by \f$n/(n-1)\f$, with scale \f$(n-1)/n\f$. JKn does so within each
/// stratum, with per-replicate scale \f$(n_h-1)/n_h\f$. The jackknives include the first-stage
/// finite-population correction \f$1-f_h\f$ when the design has population PSU counts. These
/// conventions are those of `survey::as.svrepdesign`.
/// @param design Weights, strata, clusters and correction.
/// @param options Method, Fay factor and centring.
/// @return The replicate design, sharing the design's confidence.
/// @throws std::invalid_argument For an invalid design, a BRR or Fay stratum without exactly two PSUs, JK1
/// with several strata, a jackknife stratum with one PSU, or a Fay factor outside [0, 1).
/// @par Example
/// @code
/// const dm::stats::SurveyDesign d{.weights = {1, 1, 1, 1}, .strata = {1, 1, 2, 2}};
/// const auto r = dm::stats::make_replicate_design(d, {.method = dm::stats::ReplicateMethod::brr});
/// std::println("{} {}", r.replicate_weights.cols(), r.scale); // 4 0.25
/// @endcode
/// @par References
/// - K. M. Wolter (2007). Introduction to Variance Estimation, 2nd ed. Springer, chapters 3-4.
///   doi:10.1007/978-0-387-35099-8
/// - D. Judkins (1990). "Fay's method for variance estimation." Journal of Official Statistics 6(3), 223-239.
/// @see replicate_mean()
[[nodiscard]] ReplicateDesign make_replicate_design(const SurveyDesign& design, const ReplicateOptions& options = {});

/// @brief Replicate-weight total, optionally within a domain.
/// @param design Replicate design.
/// @param y Variable per unit.
/// @param domain Optional domain membership per unit (empty: everyone).
/// @return The estimate with replicate standard error and normal interval.
/// @throws std::invalid_argument For an inconsistent design, mismatched sizes or an empty domain.
/// @par Example
/// @code
/// const dm::stats::SurveyDesign d{.weights = {1, 1, 1, 1}, .strata = {1, 1, 2, 2}};
/// const auto r = dm::stats::make_replicate_design(d, {.method = dm::stats::ReplicateMethod::brr});
/// const auto e = dm::stats::replicate_total(r, {1, 3, 2, 6});
/// std::println("{} {:.4f}", e.estimate, e.standard_error); // 12 4.4721
/// @endcode
/// @see replicate_mean(), replicate_ratio(), survey_total()
[[nodiscard]] SurveyEstimate replicate_total(const ReplicateDesign& design, const std::vector<double>& y,
                                             const std::vector<bool>& domain = {});

/// @brief Replicate-weight (Hájek) mean, optionally within a domain.
/// @param design Replicate design.
/// @param y Variable per unit.
/// @param domain Optional domain membership per unit (empty: everyone).
/// @return The estimate with replicate standard error and normal interval.
/// @throws std::invalid_argument As replicate_total(), or for a replicate with zero domain weight.
/// @par Example
/// @code
/// const dm::stats::SurveyDesign d{.weights = {1, 1, 1, 1}, .strata = {1, 1, 2, 2}};
/// const auto r = dm::stats::make_replicate_design(d, {.method = dm::stats::ReplicateMethod::jkn});
/// std::println("{}", dm::stats::replicate_mean(r, {1, 3, 2, 6}).estimate); // 3
/// @endcode
/// @see replicate_total(), survey_mean()
[[nodiscard]] SurveyEstimate replicate_mean(const ReplicateDesign& design, const std::vector<double>& y,
                                            const std::vector<bool>& domain = {});

/// @brief Replicate-weight ratio of totals, optionally within a domain.
/// @param design Replicate design.
/// @param numerator Numerator variable per unit.
/// @param denominator Denominator variable per unit.
/// @param domain Optional domain membership per unit (empty: everyone).
/// @return The estimate with replicate standard error and normal interval.
/// @throws std::invalid_argument As replicate_total(), or for a zero denominator total in any replicate.
/// @par Example
/// @code
/// const dm::stats::SurveyDesign d{.weights = {1, 1, 1, 1}, .strata = {1, 1, 2, 2}};
/// const auto r = dm::stats::make_replicate_design(d);
/// std::println("{:.4f}", dm::stats::replicate_ratio(r, {1, 3, 2, 6}, {2, 2, 4, 8}).estimate); // 0.7500
/// @endcode
/// @see replicate_total(), survey_ratio()
[[nodiscard]] SurveyEstimate replicate_ratio(const ReplicateDesign& design, const std::vector<double>& numerator,
                                             const std::vector<double>& denominator,
                                             const std::vector<bool>& domain = {});

} // namespace datamunge::stats
