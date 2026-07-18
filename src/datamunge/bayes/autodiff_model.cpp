#include <datamunge/bayes/autodiff_model.hpp>

namespace datamunge::bayes {

AutodiffModel::AutodiffModel(LogDensityFn log_density_fn) : log_density_fn_(std::move(log_density_fn)) {}

double AutodiffModel::evaluate(const std::vector<double>& params) {
    autodiff::Tape tape;
    std::vector<autodiff::Var> leaves;
    leaves.reserve(params.size());
    for (const double p : params) leaves.emplace_back(tape, p);
    return log_density_fn_(tape, leaves).value();
}

std::vector<double> AutodiffModel::gradient(const std::vector<double>& params) {
    autodiff::Tape tape;
    std::vector<autodiff::Var> leaves;
    leaves.reserve(params.size());
    for (const double p : params) leaves.emplace_back(tape, p);
    const autodiff::Var y = log_density_fn_(tape, leaves);
    const auto adjoint = tape.backward(y.index());
    std::vector<double> grad(params.size());
    for (std::size_t i = 0; i < params.size(); ++i) grad[i] = adjoint[leaves[i].index()];
    return grad;
}

} // namespace datamunge::bayes
