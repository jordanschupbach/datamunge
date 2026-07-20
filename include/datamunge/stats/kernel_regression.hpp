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

enum class KernelRegressionKernel { Gaussian, Epanechnikov, Uniform, Triangular };

struct KernelRegressionOptions {
    KernelRegressionKernel kernel{KernelRegressionKernel::Gaussian};
    // Bandwidth in standardized-predictor units. A negative value (the
    // default) means "choose automatically" by leave-one-out
    // cross-validation over an internally generated grid -- the classical
    // bandwidth selector for kernel regression.
    double      bandwidth{-1.0};
    std::size_t n_bandwidth{50}; // grid size when auto-selecting
    bool        standardize{true};
};

// Nadaraya-Watson kernel regression: a nonparametric fit where each
// prediction is a kernel-weighted average of the training responses,
// weighted by distance in (standardized) predictor space --
//
//   f(x) = sum_i K((x - x_i) / h) * y_i / sum_i K((x - x_i) / h)
//
// using the same formula + DataFrame constructor style as the other stats
// classes:
//
//   KernelRegression model(iris, "Petal.Length ~ Petal.Width");
//   model.print_summary();
//   auto preds = model.predict(newdata);
//
// As with KNNRegressor, a training point's own kernel weight dominates as
// the bandwidth shrinks (in the limit it interpolates the training data
// exactly), so fitted_values()/r_squared()/rmse() report leave-one-out
// performance rather than a trivial resubstitution fit; the same
// leave-one-out criterion is what selects the bandwidth automatically.
class KernelRegression {
 public:
    KernelRegression(const dstruct::DataFrame& data, const std::string& formula,
                     KernelRegressionOptions options = {});

    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    [[nodiscard]] std::size_t                     observations() const { return observations_; }
    [[nodiscard]] KernelRegressionKernel           kernel() const { return options_.kernel; }

    // The bandwidth actually used -- either the caller-supplied value or the cross-validation-selected one.
    [[nodiscard]] double bandwidth() const { return bandwidth_used_; }
    [[nodiscard]] bool   bandwidth_was_selected() const { return bandwidth_was_selected_; }
    // Nonempty only when the bandwidth was selected automatically: the swept grid and the leave-one-out mean
    // squared error at each of them (same length and order).
    [[nodiscard]] const std::vector<double>& bandwidth_grid() const { return bandwidth_grid_; }
    [[nodiscard]] const std::vector<double>& cv_mean_squared_error() const { return cv_mse_; }

    // Leave-one-out predictions/metrics (see class docs) -- not resubstitution.
    [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_; }
    [[nodiscard]] double                     r_squared() const;
    [[nodiscard]] double                     rmse() const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    // Uses every training row (no leave-one-out exclusion) since newdata is
    // assumed genuinely new. Returns NaN for rows outside a compact-support
    // kernel's window (Epanechnikov/Uniform/Triangular) of every training
    // point, in addition to rows with missing predictors.
    [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;

    // Scatter of `data` plus the fitted kernel-regression curve across the
    // predictor's range; only valid for a single-predictor model (throws otherwise).
    [[nodiscard]] plot::RPlot plot_fit(const dstruct::DataFrame& data, std::size_t grid_resolution = 200) const;
    [[nodiscard]] plot::RPlot plot_predicted_vs_actual() const;
    [[nodiscard]] plot::RPlot plot_residuals_vs_fitted() const;
    // Leave-one-out CV curve across the bandwidth grid with the selected bandwidth marked; throws unless the
    // bandwidth was auto-selected.
    [[nodiscard]] plot::RPlot plot_cv_curve() const;

 private:
    void   fit(const dstruct::DataFrame& data);
    double predict_one(const std::vector<double>& query_std, double bandwidth, std::size_t exclude_row = SIZE_MAX) const;

    Formula                  formula_;
    DesignInfo                design_;
    KernelRegressionOptions   options_;

    std::vector<std::string> predictor_names_;
    std::vector<double>      feature_mean_;
    std::vector<double>      feature_scale_;
    std::vector<double>      feature_min_;
    std::vector<double>      feature_max_;

    linalg::DenseMatrix<double> training_X_std_;
    std::vector<double>          training_y_;

    double                bandwidth_used_{0.0};
    bool                  bandwidth_was_selected_{false};
    std::vector<double>  bandwidth_grid_;
    std::vector<double>  cv_mse_;

    std::vector<double> fitted_; // leave-one-out predictions
    std::size_t           observations_{0};
};

} // namespace datamunge::stats
