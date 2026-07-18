#include <datamunge/optim/adam.hpp>

#include <cmath>

namespace datamunge::optim {

Adam::Adam(AdamOptions options) : options_(options) {}

double Adam::optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const {
    const std::size_t n = coordinates.size();
    std::vector<double> m(n, 0.0);
    std::vector<double> v(n, 0.0);
    double value = function.evaluate(coordinates);

    for (std::size_t iter = 1; iter <= options_.max_iterations; ++iter) {
        const auto grad = function.gradient(coordinates);
        double grad_norm_sq = 0.0;
        const double bias_correction1 = 1.0 - std::pow(options_.beta1, static_cast<double>(iter));
        const double bias_correction2 = 1.0 - std::pow(options_.beta2, static_cast<double>(iter));
        for (std::size_t j = 0; j < n; ++j) {
            m[j] = options_.beta1 * m[j] + (1.0 - options_.beta1) * grad[j];
            v[j] = options_.beta2 * v[j] + (1.0 - options_.beta2) * grad[j] * grad[j];
            const double m_hat = m[j] / bias_correction1;
            const double v_hat = v[j] / bias_correction2;
            coordinates[j] -= options_.step_size * m_hat / (std::sqrt(v_hat) + options_.epsilon);
            grad_norm_sq += grad[j] * grad[j];
        }
        const double new_value = function.evaluate(coordinates);
        const bool converged =
            std::sqrt(grad_norm_sq) < options_.tolerance || std::abs(new_value - value) < options_.tolerance;
        value = new_value;
        if (converged) break;
    }
    return value;
}

} // namespace datamunge::optim
