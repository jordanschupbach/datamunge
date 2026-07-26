#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace datamunge::stats {

struct QuantileRegressionOptions {
    double                     quantile{0.5};
    double                     rho{1.0};
    std::size_t                max_iterations{10000};
    double                     absolute_tolerance{1e-6};
    double                     relative_tolerance{1e-5};
    bool                       adaptive_rho{false};
    std::optional<std::string> weights_column;
};

// Linear quantile regression fit by ADMM. The model minimizes
//
//   sum_i w_i rho_tau(y_i - x_i^T beta),
//
// where rho_tau(r) = r * (tau - I[r < 0]) is the check (pinball) loss.
// Formula parsing, null-row omission, categorical encoding, and prediction
// follow the same conventions as stats::LM.
class QuantileRegression {
 public:
    QuantileRegression(const dstruct::DataFrame& data, const std::string& formula,
                       QuantileRegressionOptions options = {});

    [[nodiscard]] const std::string& formula_text() const { return formula_.text(); }
    [[nodiscard]] double             quantile() const { return options_.quantile; }
    [[nodiscard]] bool               has_intercept() const { return design_.has_intercept; }
    [[nodiscard]] std::size_t        observations() const { return observations_; }

    [[nodiscard]] const std::vector<double>&      coefficients() const { return coefficients_; }
    [[nodiscard]] const std::vector<std::string>& coefficient_names() const {
        return design_.coefficient_names;
    }
    [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_; }
    [[nodiscard]] const std::vector<double>& residuals() const { return residuals_; }

    [[nodiscard]] double      objective() const { return objective_; }
    [[nodiscard]] double      pseudo_r_squared() const { return pseudo_r_squared_; }
    [[nodiscard]] std::size_t iterations() const { return iterations_; }
    [[nodiscard]] bool        converged() const { return converged_; }
    [[nodiscard]] double      primal_residual_norm() const { return primal_residual_norm_; }
    [[nodiscard]] double      dual_residual_norm() const { return dual_residual_norm_; }

    [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

 private:
    void fit(const dstruct::DataFrame& data);
    void validate_categorical_levels(const dstruct::DataFrame& newdata) const;

    Formula                   formula_;
    DesignInfo                design_;
    QuantileRegressionOptions options_;

    std::vector<double> coefficients_;
    std::vector<double> fitted_;
    std::vector<double> residuals_;
    std::vector<double> weights_;

    double      objective_{0.0};
    double      pseudo_r_squared_{0.0};
    std::size_t observations_{0};
    std::size_t iterations_{0};
    bool        converged_{false};
    double      primal_residual_norm_{0.0};
    double      dual_residual_norm_{0.0};
};

} // namespace datamunge::stats
