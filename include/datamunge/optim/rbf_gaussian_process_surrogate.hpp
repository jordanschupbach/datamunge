#pragma once
#include <datamunge/optim/bayesian_optimization.hpp>
namespace datamunge::optim {
/// @brief Vector-native RBF Gaussian-process surrogate with expected-improvement acquisition.
class RBFGaussianProcessSurrogate : public BayesianSurrogate {
public:
    explicit RBFGaussianProcessSurrogate(double length_scale = 1.0, double noise = 1e-6);
    void fit(const std::vector<std::vector<double>>& points,const std::vector<double>& values) override;
    double acquisition(const std::vector<double>& point,double incumbent) override;
private:
    double length_scale_, noise_;
    std::vector<std::vector<double>> points_;
    std::vector<double> values_;
    std::vector<double> alpha_, cholesky_;
};
}
