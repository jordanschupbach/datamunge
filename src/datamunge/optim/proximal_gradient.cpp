#include <datamunge/optim/proximal_gradient.hpp>
#include <cmath>
#include <stdexcept>
namespace datamunge::optim {
ProximalGradient::ProximalGradient(ProximalGradientOptions options) : options_(options) {}
double ProximalGradient::optimize(ProximalFunction& function, std::vector<double>& coordinates) const {
    if (options_.step_size <= 0.0) throw std::invalid_argument("ProximalGradient: step_size must be positive");
    for (std::size_t i = 0; i < options_.max_iterations; ++i) { auto gradient = function.gradient(coordinates); std::vector<double> trial(coordinates.size()); for (std::size_t j=0;j<trial.size();++j) trial[j]=coordinates[j]-options_.step_size*gradient[j]; auto next=function.proximal(trial, options_.step_size); if(next.size()!=coordinates.size()) throw std::invalid_argument("ProximalGradient: proximal returned wrong dimension"); double change=0; for(std::size_t j=0;j<next.size();++j){double d=next[j]-coordinates[j];change+=d*d;} coordinates=std::move(next); if(std::sqrt(change)<=options_.tolerance) break; }
    return function.evaluate(coordinates);
}
} // namespace datamunge::optim
