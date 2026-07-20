#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace datamunge::stats {

struct GaussianProcessRegressionOptions {
    // RBF (squared-exponential) kernel length scale, in standardized-predictor units:
    //   k(x, x') = signal_variance * exp(-||x - x'||^2 / (2 * length_scale^2))
    // A negative value (the default) means "choose automatically" by maximizing the exact log
    // marginal likelihood over an internal grid.
    double      length_scale{-1.0};
    // noise_variance / signal_variance. A negative value (the default) means "choose
    // automatically" the same way; 0 is a legal fixed value (noiseless/interpolating GP).
    // signal_variance itself is never a free option -- it is always profiled out analytically
    // (it has a closed-form maximum-likelihood value given length_scale and noise_ratio).
    double      noise_ratio{-1.0};
    std::size_t n_length_scale_grid{20};
    std::size_t n_noise_grid{15};
    bool        standardize{true}; // standardize predictors (the response is always standardized internally)
};

struct GaussianProcessRegressionPrediction {
    std::vector<double> fit;
    std::vector<double> se_fit; // posterior std of f(x), not of a new noisy observation
    std::vector<double> lower;
    std::vector<double> upper;
};

// Exact Gaussian process regression with an RBF kernel, fit via Cholesky
// decomposition, using the same formula + DataFrame constructor style as
// the other stats classes:
//
//   GaussianProcessRegression model(iris, "Petal.Length ~ Petal.Width");
//   model.print_summary();
//   auto preds = model.predict(newdata);
//
// Unlike every other regressor in this library, a GP's prediction comes
// with a principled posterior variance at every point -- see
// predict_detail() for the Gaussian confidence interval this gives for
// free. A noiseless (or near-noiseless) GP interpolates its training data
// exactly, so -- as with KNNRegressor and KernelRegression --
// fitted_values()/r_squared()/rmse() report leave-one-out performance
// (computed here via the closed-form "virtual LOO" identity for GPs,
// Rasmussen & Williams eq. 5.12, not by refitting n times) rather than a
// trivial resubstitution fit.
class GaussianProcessRegression {
 public:
    GaussianProcessRegression(const dstruct::DataFrame& data, const std::string& formula,
                              GaussianProcessRegressionOptions options = {});

    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    [[nodiscard]] std::size_t                     observations() const { return observations_; }

    [[nodiscard]] double length_scale() const { return length_scale_used_; }
    [[nodiscard]] double signal_variance() const { return signal_variance_used_; }
    [[nodiscard]] double noise_variance() const { return noise_ratio_used_ * signal_variance_used_; }
    [[nodiscard]] double log_marginal_likelihood() const { return log_marginal_likelihood_used_; }
    [[nodiscard]] bool   length_scale_was_selected() const { return length_scale_grid_.size() > 1; }
    [[nodiscard]] bool   noise_ratio_was_selected() const { return noise_ratio_grid_.size() > 1; }
    // The swept length-scale grid, and (for each length scale) the best log marginal likelihood
    // achieved after optimizing the noise ratio -- a profile likelihood curve. Only meaningful
    // when length_scale was auto-selected.
    [[nodiscard]] const std::vector<double>& length_scale_grid() const { return length_scale_grid_; }
    [[nodiscard]] const std::vector<double>& length_scale_profile_log_likelihood() const {
        return length_scale_profile_ll_;
    }

    // Leave-one-out predictions/metrics (see class docs) -- not resubstitution.
    [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_; }
    [[nodiscard]] double                     r_squared() const;
    [[nodiscard]] double                     rmse() const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;
    [[nodiscard]] GaussianProcessRegressionPrediction predict_detail(const dstruct::DataFrame& newdata,
                                                                     double level = 0.95) const;

    // Scatter of `data` plus the GP posterior mean and a Gaussian confidence band; only valid
    // for a single-predictor model (throws otherwise).
    [[nodiscard]] plot::RPlot plot_fit(const dstruct::DataFrame& data, std::size_t grid_resolution = 200,
                                             double level = 0.95) const;
    [[nodiscard]] plot::RPlot plot_predicted_vs_actual() const;
    [[nodiscard]] plot::RPlot plot_residuals_vs_fitted() const;
    // Profile log marginal likelihood across the length-scale grid; throws unless length_scale
    // was auto-selected.
    [[nodiscard]] plot::RPlot plot_length_scale_profile() const;

 private:
    void fit(const dstruct::DataFrame& data);
    // k(query, training row j) for every j, in standardized units (signal_variance already applied).
    std::vector<double> kernel_vector(const std::vector<double>& query_std) const;

    Formula                          formula_;
    DesignInfo                        design_;
    GaussianProcessRegressionOptions  options_;

    std::vector<std::string> predictor_names_;
    std::vector<double>      feature_mean_;
    std::vector<double>      feature_scale_;

    linalg::DenseMatrix<double> training_X_std_;
    std::vector<double>          training_y_;     // original scale
    double                        y_mean_{0.0};
    double                        y_scale_{1.0};
    std::vector<double>          training_y_std_;

    std::vector<double> length_scale_grid_;
    std::vector<double> length_scale_profile_ll_;
    std::vector<double> noise_ratio_grid_;

    double length_scale_used_{1.0};
    double noise_ratio_used_{0.0};
    double signal_variance_used_{1.0};
    double log_marginal_likelihood_used_{0.0};

    std::optional<linalg::CholeskyDecomposition<double>> chol_; // of the final K (standardized-y units)
    std::vector<double>                                    alpha_; // K^-1 y_std

    std::vector<double> fitted_; // leave-one-out, original scale
    std::size_t           observations_{0};
};

} // namespace datamunge::stats
