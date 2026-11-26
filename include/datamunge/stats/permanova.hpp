#pragma once

/// @file permanova.hpp
/// @brief Distance-based multivariate analysis: Bray-Curtis dissimilarities, permutational MANOVA (PERMANOVA,
/// Anderson 2001; McArdle and Anderson 2001) partitioning a distance matrix by model terms, and the test of
/// multivariate homogeneity of group dispersions (PERMDISP, Anderson 2006) that separates location from spread.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Bray-Curtis dissimilarities \f$\sum_k|x_{ik}-x_{jk}|/\sum_k(x_{ik}+x_{jk})\f$ between rows (as
/// `vegan::vegdist(method = "bray")`).
/// @param x Nonnegative abundances, one row per sample.
/// @return n x n symmetric matrix (0 between two all-zero rows).
/// @throws std::invalid_argument On negative or nonfinite entries.
/// @par Example
/// @code
/// const auto d = dm::stats::bray_curtis_distances(dm::linalg::DenseMatrix<double>(2, 2, {1, 3, 3, 1}));
/// std::println("{}", d(0, 1)); // 0.5
/// @endcode
linalg::DenseMatrix<double> bray_curtis_distances(const linalg::DenseMatrix<double>& x);

/// @brief One model term: a name and its design columns (no intercept; they are centered internally).
/// @par Example
/// @code
/// dm::stats::PermanovaTerm t{"dose", dm::linalg::DenseMatrix<double>(3, 1, {1, 2, 3})};
/// std::println("{}", t.name); // dose
/// @endcode
struct PermanovaTerm {
    std::string name;                    ///< Label in the output table.
    linalg::DenseMatrix<double> columns; ///< n x k design columns.
};

/// @brief Treatment-coded factor: one indicator per level except the first in sorted order (as R's default).
/// @param name Term name.
/// @param labels Level of each observation.
/// @return The term.
/// @throws std::invalid_argument If labels is empty.
/// @par Example
/// @code
/// std::println("{}", dm::stats::factor_term("g", {"a", "b", "c", "a"}).columns.cols()); // 2
/// @endcode
PermanovaTerm factor_term(const std::string& name, const std::vector<std::string>& labels);

/// @brief Numeric covariate term.
/// @param name Term name.
/// @param values One value per observation.
/// @return The term.
/// @par Example
/// @code
/// std::println("{}", dm::stats::numeric_term("x", {1, 2, 3}).columns.rows()); // 3
/// @endcode
PermanovaTerm numeric_term(const std::string& name, const std::vector<double>& values);

/// @brief Interaction term: all products of a column of `a` with a column of `b`.
/// @param name Term name.
/// @param a First term.
/// @param b Second term with the same rows.
/// @return The term.
/// @throws std::invalid_argument On a row mismatch.
/// @par Example
/// @code
/// const auto a = dm::stats::factor_term("a", {"x", "y", "x", "y"});
/// const auto b = dm::stats::factor_term("b", {"u", "u", "v", "v"});
/// std::println("{}", dm::stats::interaction_term("a:b", a, b).columns(3, 0)); // 1
/// @endcode
PermanovaTerm interaction_term(const std::string& name, const PermanovaTerm& a, const PermanovaTerm& b);

/// @brief How PERMANOVA partitions the model sum of squares.
/// @par Example
/// @code
/// const auto by = dm::stats::PermanovaBy::margin;
/// std::println("{}", by == dm::stats::PermanovaBy::margin); // true
/// @endcode
enum class PermanovaBy {
    overall, ///< One test of all terms together (`adonis2` default, `by = NULL`).
    terms,   ///< Sequential (type I) sums of squares in the given order (`by = "terms"`).
    margin,  ///< Each term adjusted for all others (type III-like, `by = "margin"`).
};

/// @brief Options for permanova().
/// @par Example
/// @code
/// dm::stats::PermanovaOptions o{.permutations = 99};
/// std::println("{}", o.permutations); // 99
/// @endcode
struct PermanovaOptions {
    PermanovaBy by{PermanovaBy::terms}; ///< Partition.
    std::size_t permutations{999};      ///< Random permutations (ignored when permutation_matrix is given).
    std::uint64_t seed{1};              ///< Permutation seed.
    std::vector<std::size_t> strata{};  ///< Optional block of each observation: permute only within blocks.
    std::vector<std::vector<std::size_t>> permutation_matrix{}; ///< Explicit 0-based permutations, one per row.
};

/// @brief One row of a PERMANOVA table.
/// @par Example
/// @code
/// dm::stats::PermanovaRow r{.term = "Residual"};
/// std::println("{}", r.term); // Residual
/// @endcode
struct PermanovaRow {
    std::string term;        ///< Term, "Model", "Residual" or "Total".
    std::size_t df{};        ///< Degrees of freedom.
    double sum_of_squares{}; ///< Sum of squared distances attributed to the row.
    double r_squared{};      ///< Share of the total sum of squares.
    double f{};              ///< Pseudo-F (NaN for Residual and Total).
    double p_value{};        ///< Permutation p-value \f$(\#\{F^*\ge F\}+1)/(B+1)\f$ (NaN for Residual and Total).
};

/// @brief A PERMANOVA table.
/// @par Example
/// @code
/// dm::stats::Permanova p;
/// std::println("{}", p.rows.size()); // 0
/// @endcode
struct Permanova {
    std::vector<PermanovaRow> rows; ///< Terms (or "Model"), then "Residual" and "Total".
    std::size_t permutations{};     ///< Number of permutations used.
};

/// @brief Permutational MANOVA of a distance matrix (as `vegan::adonis2`).
///
/// The Gower matrix \f$G=-\frac12 JD^{(2)}J\f$ is projected onto the centered design. The pseudo-F of a term is
/// \f$(SS_k/df_k)/(SS_{res}/(n-r-1))\f$, and every permutation recomputes it on \f$G[\pi,\pi]\f$. For the absence
/// of any effect this is exact. For sequential and marginal terms it permutes raw data, as vegan does without
/// conditioning variables. Aliased columns (relative tolerance 1e-7) are dropped.
/// @param distances n x n symmetric dissimilarities.
/// @param terms Model terms with n rows each.
/// @param options Options.
/// @return The table.
/// @throws std::invalid_argument On a non-square, asymmetric or negative matrix, mismatched terms, a model
/// without residual degrees of freedom, or invalid strata or permutations.
/// @par Example
/// @code
/// const auto d = dm::stats::bray_curtis_distances(
///     dm::linalg::DenseMatrix<double>(4, 2, {5, 1, 6, 1, 1, 5, 1, 6}));
/// const auto p = dm::stats::permanova(d, {dm::stats::factor_term("g", {"a", "a", "b", "b"})});
/// std::println("{:.3f}", p.rows[0].r_squared); // 0.988
/// @endcode
Permanova permanova(const linalg::DenseMatrix<double>& distances, const std::vector<PermanovaTerm>& terms,
                    const PermanovaOptions& options = {});

/// @brief Group center used by dispersion_test().
/// @par Example
/// @code
/// const auto c = dm::stats::DispersionCenter::centroid;
/// std::println("{}", c == dm::stats::DispersionCenter::centroid); // true
/// @endcode
enum class DispersionCenter {
    median,   ///< Spatial median (`betadisper` default), computed separately on real and imaginary axes.
    centroid, ///< Group centroid.
};

/// @brief Options for dispersion_test().
/// @par Example
/// @code
/// dm::stats::DispersionOptions o{.bias_adjust = true};
/// std::println("{}", o.bias_adjust); // true
/// @endcode
struct DispersionOptions {
    DispersionCenter center{DispersionCenter::median}; ///< Group center.
    bool bias_adjust{false};       ///< Multiply distances by \f$\sqrt{n_g/(n_g-1)}\f$ (Stier et al. 2013).
    std::size_t permutations{999}; ///< Random permutations of ANOVA residuals.
    std::uint64_t seed{1};         ///< Permutation seed.
    std::vector<std::vector<std::size_t>> permutation_matrix{}; ///< Explicit 0-based permutations.
};

/// @brief Result of dispersion_test().
/// @par Example
/// @code
/// dm::stats::Dispersion d;
/// std::println("{}", d.groups.size()); // 0
/// @endcode
struct Dispersion {
    std::vector<std::string> groups;          ///< Group levels in sorted order.
    std::vector<double> distances;            ///< Distance of each observation to its group center.
    std::vector<double> group_mean_distances; ///< Mean distance per group (average dispersion).
    std::vector<double> eigenvalues;          ///< PCoA eigenvalues kept (negative ones give imaginary axes).
    linalg::DenseMatrix<double> coordinates;  ///< PCoA coordinates scaled by \f$\sqrt{|\lambda|}\f$.
    std::size_t df_groups{};                  ///< g - 1.
    std::size_t df_residual{};                ///< n - g.
    double sum_sq_groups{};                   ///< Between-group sum of squares of the distances.
    double sum_sq_residual{};                 ///< Within-group sum of squares.
    double f{};                               ///< One-way ANOVA F on the distances.
    double parametric_p_value{};              ///< F-distribution p-value.
    double permutation_p_value{};             ///< Permutation p-value (NaN with 0 permutations).
    std::size_t permutations{};               ///< Permutations used.
};

/// @brief Test homogeneity of multivariate dispersions (as `vegan::betadisper` + `permutest`).
///
/// Principal coordinates of the distances, including the imaginary axes of a non-Euclidean dissimilarity, give
/// each observation's distance to its group center. The distances are then compared by a one-way ANOVA, whose
/// residuals are permuted.
/// @param distances n x n symmetric dissimilarities.
/// @param groups Group label of each observation (at least 2 groups).
/// @param options Options.
/// @return The test.
/// @throws std::invalid_argument On invalid distances, mismatched or single-group labels, or invalid permutations.
/// @par Example
/// @code
/// const auto d = dm::stats::bray_curtis_distances(
///     dm::linalg::DenseMatrix<double>(4, 2, {5, 1, 6, 1, 1, 5, 1, 9}));
/// std::println("{}", dm::stats::dispersion_test(d, {"a", "a", "b", "b"}).df_groups); // 1
/// @endcode
Dispersion dispersion_test(const linalg::DenseMatrix<double>& distances, const std::vector<std::string>& groups,
                           const DispersionOptions& options = {});

} // namespace datamunge::stats
