#pragma once

/// @file knockoffs.hpp
/// @brief Model-X Gaussian knockoffs and the knockoff false-discovery-rate filter.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace datamunge::stats {

/// @brief Rule for choosing the knockoff covariance gap \f$D=\operatorname{diag}(s)\f$.
///
/// Both rules work on the correlation scale \f$G\f$ of the covariance and return
/// \f$s\f$ multiplied by the variances. They then shrink \f$s\f$ by the smallest
/// factor \f$1-10^{-8}\cdot10^k\f$ for which \f$2G-\operatorname{diag}(s)\f$ has a
/// smallest eigenvalue above \f$10^{-8}\f$, as the R knockoff package does.
/// @par Example
/// @code
/// const auto scale = dm::stats::KnockoffScale::equicorrelated;
/// std::println("{}", scale == dm::stats::KnockoffScale::sdp); // false
/// @endcode
enum class KnockoffScale {
    equicorrelated, ///< Common \f$s_j=\min(1,2\lambda_{\min}(G))\f$; one eigenvalue problem.

    /// Maximizes \f$\sum_j s_j\f$ subject to \f$0\le s_j\le1\f$ and
    /// \f$2G-\operatorname{diag}(s)\succeq0\f$, solved by a log-barrier Newton method.
    sdp,
};

/// @brief Construction controls for Gaussian model-X knockoffs.
/// @par Example
/// @code
/// const dm::stats::GaussianKnockoffOptions options{.scale=dm::stats::KnockoffScale::sdp,.seed=7};
/// std::println("{}", options.seed); // 7
/// @endcode
struct GaussianKnockoffOptions {
    KnockoffScale scale{KnockoffScale::sdp}; ///< Rule for \f$s\f$ when `diagonal` is empty.

    /// Optional caller-chosen \f$s\f$ on the covariance scale, one nonnegative value per
    /// predictor; it must keep \f$2\Sigma-\operatorname{diag}(s)\f$ positive semidefinite.
    std::vector<double> diagonal{};
    std::uint64_t seed{1}; ///< Seed for the Gaussian innovations of the knockoff draw.
};

/// @brief One knockoff matrix and the joint covariance quantities used to draw it.
/// @par Example
/// @code
/// const dm::stats::GaussianKnockoffs draw;
/// std::println("{}", draw.diagonal.size()); // 0
/// @endcode
struct GaussianKnockoffs {
    linalg::DenseMatrix<double> knockoffs; ///< Rows of \f$\tilde X\f$, matching the rows of \f$X\f$.
    std::vector<double> diagonal;          ///< \f$s\f$ on the covariance scale.

    /// Conditional covariance \f$2D-D\Sigma^{-1}D\f$ of each knockoff row given its original row.
    linalg::DenseMatrix<double> conditional_covariance;
};

/// @brief Antisymmetric feature statistics \f$W_j\f$ computed from predictors, knockoffs and response.
///
/// Large positive values should favor the original predictor over its knockoff, and
/// swapping \f$X_j\f$ with \f$\tilde X_j\f$ must flip the sign of \f$W_j\f$ only.
using KnockoffStatistic = std::function<std::vector<double>(
    const linalg::DenseMatrix<double>& predictors, const linalg::DenseMatrix<double>& knockoffs,
    const std::vector<double>& response)>;

/// @brief Knockoff filter controls.
/// @par Example
/// @code
/// const dm::stats::KnockoffFilterOptions options{.fdr=.2};
/// std::println("{}", options.plus); // true
/// @endcode
struct KnockoffFilterOptions {
    double fdr{.1};                    ///< Target false discovery rate in (0, 1).
    bool plus{true};                   ///< Use knockoff+ (offset 1, exact FDR control) instead of offset 0.
    GaussianKnockoffOptions knockoffs{}; ///< Knockoff construction controls.
};

/// @brief Knockoff draw, feature statistics, data-dependent threshold and selected predictors.
/// @par Example
/// @code
/// const dm::stats::KnockoffFilterResult result;
/// std::println("{}", result.selected.size()); // 0
/// @endcode
struct KnockoffFilterResult {
    GaussianKnockoffs draw;              ///< Knockoffs passed to the statistic.
    std::vector<double> statistics;      ///< \f$W_j\f$ in predictor order.
    double threshold{};                  ///< Selection threshold; infinity when nothing can be selected.
    std::vector<std::size_t> selected;   ///< Zero-based predictors with \f$W_j\ge T\f$, ascending.
};

/// @brief Chooses the diagonal \f$s\f$ of the knockoff covariance gap for a Gaussian design.
///
/// Mirrors `create.solve_equi` and `create.solve_sdp` of the R knockoff package.
/// The SDP rule may differ from that package's DSDP output by the solver tolerance,
/// and it returns the maximizer reached by an interior-point path when the optimum
/// is not unique.
/// @param covariance Symmetric positive definite \f$p\times p\f$ covariance \f$\Sigma\f$.
/// @param scale Equicorrelated or SDP rule.
/// @return \f$s\f$ on the covariance scale, each in \f$[0,\Sigma_{jj}]\f$.
/// @throws std::invalid_argument If the matrix is empty, nonsquare, asymmetric, nonfinite or not positive definite.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> sigma(2,2,{1,.5,.5,1});
/// const auto s=dm::stats::knockoff_diagonal(sigma,dm::stats::KnockoffScale::equicorrelated);
/// std::println("{:.6f}", s[0]); // 1.000000
/// @endcode
/// @par References
/// - E. Candès, Y. Fan, L. Janson and J. Lv (2018). "Panning for gold: model-X knockoffs for high
///   dimensional controlled variable selection". JRSS B 80(3), 551-577. doi:10.1111/rssb.12265
/// - R. F. Barber and E. J. Candès (2015). "Controlling the false discovery rate via knockoffs".
///   Annals of Statistics 43(5), 2055-2085. doi:10.1214/15-AOS1337
/// - https://cran.r-project.org/package=knockoff
/// @see gaussian_knockoffs()
[[nodiscard]] std::vector<double> knockoff_diagonal(
    const linalg::DenseMatrix<double>& covariance, KnockoffScale scale = KnockoffScale::sdp);

/// @brief Draws Gaussian model-X knockoffs \f$\tilde X\f$ for rows \f$X\sim N(\mu,\Sigma)\f$.
///
/// Each knockoff row is \f$\tilde x=x-D\Sigma^{-1}(x-\mu)+C^\top z\f$ with standard normal
/// \f$z\f$ and \f$C^\top C=2D-D\Sigma^{-1}D\f$, so that \f$(X,\tilde X)\f$ has covariance
/// \f$\begin{pmatrix}\Sigma&\Sigma-D\\\Sigma-D&\Sigma\end{pmatrix}\f$ and swapping any
/// \f$X_j\f$ with \f$\tilde X_j\f$ leaves the joint law unchanged. The square root is
/// symmetric-eigen based, so predictors with \f$s_j=0\f$ get exact copies. Knockoffs are
/// valid only when the supplied mean and covariance are the true design law;
/// estimated moments give approximate knockoffs.
/// @param predictors \f$n\times p\f$ finite observations in rows.
/// @param mean Mean vector of length \f$p\f$.
/// @param covariance Positive definite \f$p\times p\f$ covariance.
/// @param options Diagonal rule or caller-chosen diagonal, and seed.
/// @return Knockoff matrix with its diagonal and conditional covariance.
/// @throws std::invalid_argument For mismatched shapes, nonfinite values, a non-positive-definite
/// covariance, or a supplied diagonal that is negative or makes the conditional covariance indefinite.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(3,2,{0,1,1,0,2,2});
/// const dm::linalg::DenseMatrix<double> sigma(2,2,{1,0,0,1});
/// const auto draw=dm::stats::gaussian_knockoffs(x,{0,0},sigma);
/// std::println("{}x{}", draw.knockoffs.rows(), draw.knockoffs.cols()); // 3x2
/// @endcode
/// @par References
/// E. Candès, Y. Fan, L. Janson and J. Lv (2018), doi:10.1111/rssb.12265
/// @see knockoff_diagonal(), knockoff_filter()
[[nodiscard]] GaussianKnockoffs gaussian_knockoffs(
    const linalg::DenseMatrix<double>& predictors, const std::vector<double>& mean,
    const linalg::DenseMatrix<double>& covariance, const GaussianKnockoffOptions& options = {});

/// @brief Knockoff or knockoff+ threshold \f$T\f$ for antisymmetric statistics \f$W\f$.
///
/// Returns the smallest \f$t\in\{0\}\cup\{|W_j|\}\f$ with
/// \f$(o+\#\{j:W_j\le-t\})/\max(1,\#\{j:W_j\ge t\})\le q\f$, where \f$o=1\f$ for knockoff+
/// and \f$o=0\f$ otherwise, as `knockoff.threshold` in R. Selecting \f$W_j\ge T\f$ controls
/// the FDR at \f$q\f$ for knockoff+ and a modified FDR for offset 0.
/// @param statistics Finite feature statistics.
/// @param fdr Target level \f$q\in(0,1)\f$.
/// @param plus True for knockoff+ (offset 1).
/// @return The threshold, or positive infinity if no candidate satisfies the bound.
/// @throws std::invalid_argument If a statistic is nonfinite or the level is outside (0, 1).
/// @par Example
/// @code
/// const auto t=dm::stats::knockoff_threshold({3,2,1,-.5},.5);
/// std::println("{}", t); // 1
/// @endcode
/// @par References
/// R. F. Barber and E. J. Candès (2015), doi:10.1214/15-AOS1337
/// @see knockoff_filter()
[[nodiscard]] double knockoff_threshold(const std::vector<double>& statistics, double fdr, bool plus = true);

/// @brief Draws Gaussian knockoffs, computes \f$W\f$ and selects predictors with \f$W_j\ge T\f$.
///
/// The statistic callback receives the original predictors, the knockoffs and the
/// response unchanged. Selection is random through the knockoff draw; fix the seed
/// to reproduce it.
/// @param predictors \f$n\times p\f$ finite observations in rows.
/// @param response Finite response values in the same row order.
/// @param mean Mean of the predictor law.
/// @param covariance Positive definite covariance of the predictor law.
/// @param statistic Callback returning \f$p\f$ finite antisymmetric statistics.
/// @param options Target level, knockoff+ choice and knockoff construction.
/// @return Draw, statistics, threshold and selected predictors.
/// @throws std::invalid_argument For invalid data, an empty callback or malformed statistics,
/// and for the errors of gaussian_knockoffs() and knockoff_threshold().
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(3,2,{0,1,1,0,2,2});
/// const dm::linalg::DenseMatrix<double> sigma(2,2,{1,0,0,1});
/// const auto fit=dm::stats::knockoff_filter(x,{1,2,3},{0,0},sigma,
///     [](const auto&,const auto&,const auto&) { return std::vector<double>{-1,-1}; });
/// std::println("{}", fit.selected.size()); // 0
/// @endcode
/// @par References
/// E. Candès, Y. Fan, L. Janson and J. Lv (2018), doi:10.1111/rssb.12265
/// @see gaussian_knockoffs(), knockoff_threshold()
[[nodiscard]] KnockoffFilterResult knockoff_filter(
    const linalg::DenseMatrix<double>& predictors, const std::vector<double>& response,
    const std::vector<double>& mean, const linalg::DenseMatrix<double>& covariance,
    const KnockoffStatistic& statistic, const KnockoffFilterOptions& options = {});

} // namespace datamunge::stats
