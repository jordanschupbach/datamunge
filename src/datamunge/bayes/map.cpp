#include <datamunge/bayes/map.hpp>

#include <datamunge/optim/lbfgs.hpp>

namespace datamunge::bayes {

namespace {

class NegatedFunction : public optim::DifferentiableFunction {
public:
    explicit NegatedFunction(optim::DifferentiableFunction& f) : f_(f) {}

    double evaluate(const std::vector<double>& x) override { return -f_.evaluate(x); }
    std::vector<double> gradient(const std::vector<double>& x) override {
        auto g = f_.gradient(x);
        for (double& v : g) v = -v;
        return g;
    }

private:
    optim::DifferentiableFunction& f_;
};

} // namespace

MAP::MAP(MAPOptions options) : options_(options) {}

double MAP::optimize(optim::DifferentiableFunction& log_posterior, std::vector<double>& coordinates) const {
    NegatedFunction negated(log_posterior);
    optim::LBFGSOptions lbfgs_options;
    lbfgs_options.max_iterations = options_.max_iterations;
    lbfgs_options.tolerance = options_.tolerance;
    lbfgs_options.history_size = options_.history_size;
    const optim::LBFGS lbfgs(lbfgs_options);
    const double negated_value = lbfgs.optimize(negated, coordinates);
    return -negated_value;
}

} // namespace datamunge::bayes
