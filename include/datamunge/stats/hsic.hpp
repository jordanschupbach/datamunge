#pragma once

/// @file hsic.hpp
/// @brief Hilbert-Schmidt independence criterion (Gretton et al. 2005, 2008) and its d-variable extension dHSIC
/// (Pfister et al. 2018): kernel statistics with gamma-approximation and permutation tests of joint independence.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace datamunge::stats {

/// @brief Kernel applied to each variable.
/// @par Example
/// @code
/// const auto k = dm::stats::HsicKernel::discrete;
/// std::println("{}", k == dm::stats::HsicKernel::discrete); // true
/// @endcode
enum class HsicKernel {
    gaussian,       ///< \f$k(x,x')=\exp(-\lVert x-x'\rVert^2/(2\sigma^2))\f$, \f$\sigma\f$ by the median heuristic
                    ///< (\f$\sigma^2\f$ = upper median of squared pairwise distances / 2).
    gaussian_fixed, ///< Gaussian kernel with the bandwidth given in HsicOptions::bandwidth.
    discrete,       ///< \f$k(x,x')=1\{x=x'\}\f$ for categorical variables (coded as numbers).
};

/// @brief Options for hsic().
/// @par Example
/// @code
/// dm::stats::HsicOptions o{.permutations = 999};
/// std::println("{}", o.permutations); // 999
/// @endcode
struct HsicOptions {
    HsicKernel kernel{HsicKernel::gaussian}; ///< Kernel for every variable.
    double bandwidth{1.0};                   ///< Bandwidth for HsicKernel::gaussian_fixed.
    double bandwidth_scale{1.0};             ///< Multiplier on each median-heuristic bandwidth (HsicKernel::gaussian).
    std::size_t permutations{0};             ///< Permutation replicates (0: gamma approximation only).
    std::uint64_t seed{1};                   ///< Permutation seed.
};

/// @brief dHSIC statistic and tests.
/// @par Example
/// @code
/// dm::stats::Hsic h{.n = 10};
/// std::println("{}", h.n); // 10
/// @endcode
struct Hsic {
    std::size_t n{};                ///< Sample size.
    double statistic{};             ///< V-statistic dHSIC (`dHSIC::dhsic`); HSIC for two variables.
    double scaled_statistic{};      ///< n * statistic, the test statistic.
    std::vector<double> bandwidths; ///< Gaussian bandwidths used (NaN for the discrete kernel).
    double gamma_shape{};           ///< Shape of the moment-matched gamma null for n * statistic.
    double gamma_scale{};           ///< Scale of the gamma null.
    double gamma_p_value{};         ///< Gamma-approximation p-value (`dhsic.test(method = "gamma")`).
    double permutation_p_value{std::numeric_limits<double>::quiet_NaN()}; ///< (1 + #{replicates >= observed}) / (B + 1).
    std::size_t permutations{};     ///< Replicates used.
};

/// @brief Test joint (mutual) independence of d >= 2 variables with dHSIC.
///
/// Each variable is an n x p_j matrix with observations in rows. The permutation test permutes variables 2..d
/// independently. The gamma approximation matches the null mean and variance (Pfister et al. 2018) and is
/// conservative-to-accurate for moderate n; it requires n >= 2d.
/// @param variables The variables.
/// @param options Options.
/// @return Statistic and tests.
/// @throws std::invalid_argument With fewer than 2 variables, unequal row counts, n < 2d, nonfinite values or a
/// nonpositive fixed bandwidth.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(6, 1, {1, 2, 3, 4, 5, 6}), y(6, 1, {1, 4, 9, 16, 25, 36});
/// std::println("{:.4f}", dm::stats::hsic({x, y}).statistic); // 0.1122
/// @endcode
Hsic hsic(const std::vector<linalg::DenseMatrix<double>>& variables, const HsicOptions& options = {});

/// @brief HSIC independence test for two univariate samples.
/// @param x First sample.
/// @param y Second sample.
/// @param options Options.
/// @return Statistic and tests.
/// @throws std::invalid_argument As for the matrix overload.
/// @par Example
/// @code
/// const std::vector<double> x{-2, -1, 0, 1, 2, 3}, y{4, 1, 0, 1, 4, 9};
/// std::println("{:.4f}", dm::stats::hsic(x, y).statistic); // 0.0610
/// @endcode
Hsic hsic(const std::vector<double>& x, const std::vector<double>& y, const HsicOptions& options = {});

} // namespace datamunge::stats
