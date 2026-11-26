#pragma once

/// @file d_optimal.hpp
/// @brief D-optimal exact designs selected from a candidate set by Fedorov exchange, with efficiency criteria.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::stats {

/// @brief Information-based summaries of an exact design for a linear model.
///
/// With \f$n\f$ runs, \f$p\f$ parameters and information matrix \f$M=X^\top X\f$, these follow
/// `AlgDesign::optFederov`: \f$D=|M|^{1/p}/n\f$, \f$A=\operatorname{tr}\{(M/n)^{-1}\}/p\f$ and
/// \f$G_e=p/\max_{x}n f(x)^\top M^{-1}f(x)\f$, the maximum taken over the candidate set.
/// @par Example
/// @code
/// const dm::stats::DesignCriteria c;
/// std::println("{}", c.estimable); // false
/// @endcode
struct DesignCriteria {
    std::size_t runs{};         ///< Number of runs n.
    std::size_t parameters{};   ///< Number of model columns p.
    std::size_t rank{};         ///< Numerical rank of M (eigenvalues above \f$10^{-10}\f$ times the largest).
    bool estimable{};           ///< Whether every parameter is estimable (rank equals p).
    double log_det{};           ///< \f$\log|M|\f$; \f$-\infty\f$ when not estimable.
    double d_criterion{};       ///< \f$|M|^{1/p}/n\f$; 0 when not estimable.
    double a_criterion{};       ///< Average variance factor \f$\operatorname{tr}\{(M/n)^{-1}\}/p\f$; \f$\infty\f$ when not estimable.

    /// G-efficiency \f$p/\max_x n\,d(x)\f$ in (0, 1], where \f$d(x)=f(x)^\top M^{-1}f(x)\f$ is the
    /// standardized prediction variance; 1 for a G-optimal design and 0 when not estimable.
    double g_efficiency{};
};

/// @brief Evaluates design criteria for the rows of a model matrix.
/// @param design Model matrix of the design, n by p (rows are \f$f(x_i)^\top\f$, intercept included if wanted).
/// @param candidates Model matrix of the region over which G-efficiency is maximized; p columns.
/// @return D, A and G criteria, rank and estimability.
/// @throws std::invalid_argument For no runs or columns, mismatched column counts, an empty candidate set,
/// or nonfinite entries.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(3, 3, {1, -1, 1, 1, 0, 0, 1, 1, 1}); // 1, x, x^2 at -1, 0, 1
/// const auto c = dm::stats::design_criteria(x, x);
/// std::println("{:.4f} {:.3f}", c.d_criterion, c.g_efficiency); // 0.5291 1.000
/// @endcode
/// @see d_optimal_design()
[[nodiscard]] DesignCriteria design_criteria(const linalg::DenseMatrix<double>& design,
                                             const linalg::DenseMatrix<double>& candidates);

/// @brief Search controls for d_optimal_design().
/// @par Example
/// @code
/// const dm::stats::DOptimalOptions options{.starts = 50};
/// std::println("{} {}", options.seed, options.allow_repeats); // 1 false
/// @endcode
struct DOptimalOptions {
    std::size_t starts{20};           ///< Independent random starts; the best final design is kept.
    std::uint64_t seed{1};            ///< Seed for the starting designs.
    std::size_t max_exchanges{10000}; ///< Exchange limit per start.
    bool allow_repeats{false};        ///< Whether a candidate may be run more than once (AlgDesign forbids it).
};

/// @brief An exact D-optimal design found by exchange.
/// @par Example
/// @code
/// const dm::stats::DOptimalDesign d;
/// std::println("{}", d.rows.size()); // 0
/// @endcode
struct DOptimalDesign {
    std::vector<std::size_t> rows;    ///< Selected candidate rows, in increasing order (repeats adjacent).
    DesignCriteria criteria;          ///< Criteria of the selected design, G over the candidates.
    std::size_t exchanges{};          ///< Exchanges made in the winning start.
};

/// @brief Selects n runs from a candidate set to maximize \f$|X^\top X|\f$ by Fedorov exchange.
///
/// Each start draws a random design (a tiny ridge \f$10^{-9}\max|f|^2\,I\f$ is added to
/// \f$M\f$ during the search so that a singular start can be repaired). It then repeatedly makes
/// the single swap of a design row \f$i\f$ for a candidate \f$j\f$ that most increases the
/// determinant, \f$|M'|=|M|\,(1+d_j-d_i-d_id_j+d_{ij}^2)\f$ with
/// \f$d_{ij}=f_i^\top M^{-1}f_j\f$, until no swap gains more than a relative \f$10^{-10}\f$.
/// Exchange reaches a local optimum, so several starts are used. The model is supplied as
/// a candidate model matrix, so any linear model (polynomial, mixture, blocked) works.
/// @param candidates Candidate model matrix, N by p, of rank p.
/// @param runs Number of runs n, at least p; at most N unless repeats are allowed.
/// @param options Starts, seed, exchange limit and repeat policy.
/// @return Selected rows, their criteria and the exchange count.
/// @throws std::invalid_argument For an empty or nonfinite candidate matrix, a candidate set of rank
/// below p, fewer runs than parameters, more runs than candidates without repeats, or zero starts.
/// @throws std::runtime_error When no start reaches a nonsingular design.
/// @par Example
/// @code
/// std::vector<double> f; // 1, x, x^2 on 11 points in [-1, 1]
/// for (int i = 0; i <= 10; ++i) {
///     const double x = -1 + .2 * i;
///     f.insert(f.end(), {1, x, x * x});
/// }
/// const auto d = dm::stats::d_optimal_design(dm::linalg::DenseMatrix<double>(11, 3, f), 3);
/// std::println("{} {} {}", d.rows[0], d.rows[1], d.rows[2]); // 0 5 10
/// @endcode
/// @par References
/// - V. V. Fedorov (1972). Theory of Optimal Experiments. Academic Press. ISBN 978-0-12-250750-9
/// - A. C. Atkinson, A. N. Donev and R. D. Tobias (2007). Optimum Experimental Designs, with SAS.
///   Oxford University Press, ch. 12. ISBN 978-0-19-929660-6
/// - R. B. Wheeler, AlgDesign: https://cran.r-project.org/package=AlgDesign
/// @see design_criteria(), fit_quadratic_surface()
[[nodiscard]] DOptimalDesign d_optimal_design(const linalg::DenseMatrix<double>& candidates, std::size_t runs,
                                              const DOptimalOptions& options = {});

} // namespace datamunge::stats
