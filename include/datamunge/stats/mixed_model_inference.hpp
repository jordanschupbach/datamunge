#pragma once

/// @file mixed_model_inference.hpp
/// @brief Small-sample fixed-effect tests for Gaussian variance-component models.

#include <datamunge/stats/variance_components.hpp>

namespace datamunge::stats {

/// @brief Denominator degrees-of-freedom approximation for a fixed-effect contrast.
/// @par Example
/// @code
/// const auto method = dm::stats::MixedModelDF::KenwardRoger;
/// std::println("{}", method == dm::stats::MixedModelDF::KenwardRoger); // true
/// @endcode
enum class MixedModelDF {
    Satterthwaite, ///< Delta-method moment matching with observed REML information.
    KenwardRoger   ///< Expected REML information, adjusted covariance, and a scaled F statistic.
};

/// @brief F test and fixed-effect covariance used to test the supplied contrasts.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(6,1,1), z(6,3,{1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1});
/// const std::vector<double> y{0,1,3,4,6,7};
/// const auto fit = dm::stats::fit_variance_components(x,y,{z});
/// const auto test = dm::stats::mixed_model_contrast_test(fit,x,y,{z},dm::linalg::DenseMatrix<double>(1,1,1));
/// std::println("{:.0f}", test.denominator_df); // 2
/// @endcode
struct MixedModelContrastTest {
    linalg::DenseMatrix<double> covariance; ///< Full fixed-effect covariance used by the selected method.
    double statistic;                       ///< F statistic, including any Kenward–Roger scaling.
    std::size_t numerator_df;               ///< Number of independent contrast rows.
    double denominator_df;                  ///< Estimated denominator degrees of freedom.
    double scale;                           ///< Multiplier of the covariance-based Wald F statistic.
    double p_value;                         ///< Upper-tail probability.
};

/// @brief Tests a linear hypothesis with Satterthwaite or Kenward–Roger inference.
///
/// Supply the original training inputs for a converged REML fit. Satterthwaite
/// uses the eigen-contrast moment match for hypotheses with multiple rows.
/// Kenward–Roger uses the covariance adjustment for a linear sum of variance
/// components and scales the F statistic by its first two moments. For a
/// boundary fit, zero components are held fixed and inference is conditional
/// on the active covariance model.
/// @param fit Converged REML variance-component fit.
/// @param design Original fixed-effect design.
/// @param response Original Gaussian response.
/// @param random_designs Original random-effect designs in fit order.
/// @param contrast Full-row-rank matrix with one column per fixed effect.
/// @param method Degrees-of-freedom approximation, default Satterthwaite.
/// @param null_value Hypothesized contrast values, empty for zero.
/// @return Scaled F statistic, its degrees of freedom and p-value, and the covariance used.
/// @throws std::invalid_argument Invalid input, rank-deficient hypothesis or inadmissible fit.
/// @throws std::length_error Dense working matrices exceed storage capacity.
/// @throws std::runtime_error Singular covariance or information, or undefined moment approximation.
/// @throws std::overflow_error A finite result exceeds the reporting range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(6,1,1), z(6,3,{1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1});
/// const std::vector<double> y{0,1,3,4,6,7};
/// const auto fit = dm::stats::fit_variance_components(x,y,{z});
/// const auto test = dm::stats::mixed_model_contrast_test(fit,x,y,{z},dm::linalg::DenseMatrix<double>(1,1,1),
///     dm::stats::MixedModelDF::KenwardRoger);
/// std::println("{:.0f}", test.denominator_df); // 2
/// @endcode
/// @par References
/// - Satterthwaite (1946), https://doi.org/10.2307/3002019
/// - Kenward and Roger (1997), https://doi.org/10.2307/2533558
/// - Halekoh and Højsgaard (2014), https://doi.org/10.18637/jss.v059.i09
[[nodiscard]] MixedModelContrastTest mixed_model_contrast_test(const VarianceComponentsFit& fit,
    const linalg::DenseMatrix<double>& design, const std::vector<double>& response,
    const std::vector<linalg::DenseMatrix<double>>& random_designs, const linalg::DenseMatrix<double>& contrast,
    MixedModelDF method = MixedModelDF::Satterthwaite, const std::vector<double>& null_value = {});

} // namespace datamunge::stats
