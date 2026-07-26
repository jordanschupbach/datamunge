#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct KernelRidgeRegressionOptions {
    /// @brief Kernel: "rbf" (k(x,x')=exp(-gamma||x-x'||^2)), "linear" (k=x.x'), or "polynomial"
    ///        (k=(gamma x.x' + coef0)^degree).
    std::string kernel{"rbf"};
    /// @brief Ridge (Tikhonov) regularization strength lambda >= 0. A negative value (the default)
    ///        means "choose automatically" by minimizing the closed-form leave-one-out CV error
    ///        over an internal grid.
    double lambda{-1.0};
    /// @brief rbf/polynomial kernel scale gamma > 0. A negative value (the default) means "choose
    ///        automatically" jointly with lambda; ignored for the linear kernel.
    double gamma{-1.0};
    /// @brief Polynomial kernel exponent and offset (ignored for other kernels).
    double degree{3.0};
    double coef0{1.0};
    std::size_t n_lambda_grid{40};
    std::size_t n_gamma_grid{12};
    bool        standardize{true};
};

/// @brief Kernel Ridge Regression: ridge regression performed in a (possibly infinite-dimensional)
///        kernel feature space. It minimizes the squared error plus an RKHS-norm penalty,
///        which by the representer theorem has the closed-form dual solution
///        alpha = (K + lambda I)^{-1} y, so a prediction is f(x) = sum_i alpha_i k(x, x_i). Unlike
///        Nadaraya-Watson kernel regression (a local weighted average), KRR is a global penalized
///        least-squares fit; unlike Gaussian process regression it is a point estimator with no
///        posterior variance, but its predictive mean is identical to a GP's. With a linear kernel
///        it reduces exactly to ordinary ridge regression. Both lambda and the RBF bandwidth can be
///        chosen automatically by an exact, refit-free leave-one-out cross-validation (via the ridge
///        hat-matrix identity), which is also what fitted_values()/r_squared()/rmse() report. Rows
///        with a null value in any model column are dropped before fitting.
class KernelRidgeRegression {
public:
    KernelRidgeRegression(const dstruct::DataFrame& data, const std::string& formula,
                          KernelRidgeRegressionOptions options = {});

    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    [[nodiscard]] std::size_t                     observations() const { return observations_; }
    [[nodiscard]] const std::string&              kernel() const { return options_.kernel; }

    /// @brief The regularization and bandwidth actually used (caller-supplied or CV-selected).
    [[nodiscard]] double lambda() const { return lambda_used_; }
    [[nodiscard]] double gamma() const { return gamma_used_; }
    [[nodiscard]] bool   lambda_was_selected() const { return lambda_was_selected_; }
    [[nodiscard]] bool   gamma_was_selected() const { return gamma_was_selected_; }

    /// @brief Dual coefficients alpha = (K + lambda I)^{-1} (y - ybar), one per training row.
    [[nodiscard]] const std::vector<double>& dual_coefficients() const { return alpha_; }

    /// @brief Effective degrees of freedom, tr[K (K + lambda I)^{-1}] = sum_i d_i/(d_i + lambda)
    ///        over the kernel eigenvalues d_i: a continuous "model complexity" that falls from ~n
    ///        (interpolation) toward ~0 (a constant) as lambda grows.
    [[nodiscard]] double effective_degrees_of_freedom() const { return effective_dof_; }

    /// @brief The swept lambda grid and its leave-one-out mean squared error (at the selected
    ///        gamma). Non-empty only when lambda was auto-selected.
    [[nodiscard]] const std::vector<double>& lambda_grid() const { return lambda_grid_; }
    [[nodiscard]] const std::vector<double>& cv_mean_squared_error() const { return cv_mse_; }

    /// @brief Leave-one-out predictions/metrics (see class docs) -- not resubstitution.
    [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_; }
    [[nodiscard]] double                     r_squared() const;
    [[nodiscard]] double                     rmse() const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;

    /// @brief Scatter of `data` plus the fitted KRR curve; single-predictor models only (throws otherwise).
    [[nodiscard]] plot::RPlot plot_fit(const dstruct::DataFrame& data, std::size_t grid_resolution = 200) const;
    [[nodiscard]] plot::RPlot plot_predicted_vs_actual() const;
    [[nodiscard]] plot::RPlot plot_residuals_vs_fitted() const;
    /// @brief LOO-CV error across the lambda grid with the selection marked; throws unless lambda was auto-selected.
    [[nodiscard]] plot::RPlot plot_cv_curve() const;

private:
    void   fit(const dstruct::DataFrame& data);
    double kernel_scalar(const std::vector<double>& a, const std::vector<double>& b, double gamma) const;
    // k(query, training row j) for every j, in standardized predictor units.
    std::vector<double> kernel_vector(const std::vector<double>& query_std) const;

    Formula                        formula_;
    DesignInfo                      design_;
    KernelRidgeRegressionOptions    options_;

    std::vector<std::string> predictor_names_;
    std::vector<double>      feature_mean_;
    std::vector<double>      feature_scale_;
    std::vector<double>      feature_min_;
    std::vector<double>      feature_max_;

    linalg::DenseMatrix<double> training_X_std_;
    std::vector<double>          training_y_;
    double                        y_mean_{0.0};

    std::vector<double> alpha_;
    double              lambda_used_{0.0};
    double              gamma_used_{1.0};
    double              effective_dof_{0.0};
    bool                lambda_was_selected_{false};
    bool                gamma_was_selected_{false};

    std::vector<double> lambda_grid_;
    std::vector<double> cv_mse_;

    std::vector<double> fitted_; // leave-one-out, original scale
    std::size_t           observations_{0};
};

} // namespace datamunge::stats
