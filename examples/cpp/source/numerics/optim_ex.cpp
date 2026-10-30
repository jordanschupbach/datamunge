#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/optim/optim.hpp>
#include <datamunge/stats/stats.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace datamunge::optim;

namespace {

// ---- Function types ----

// DifferentiableFunction: a plain 3-D quadratic bowl with a known minimum.
class QuadraticBowl : public DifferentiableFunction {
public:
    explicit QuadraticBowl(std::vector<double> target) : target_(std::move(target)) {}

    double evaluate(const std::vector<double>& coordinates) override {
        double total = 0.0;
        for (std::size_t i = 0; i < coordinates.size(); ++i) {
            const double d = coordinates[i] - target_[i];
            total += d * d;
        }
        return total;
    }
    std::vector<double> gradient(const std::vector<double>& coordinates) override {
        std::vector<double> grad(coordinates.size());
        for (std::size_t i = 0; i < coordinates.size(); ++i) grad[i] = 2.0 * (coordinates[i] - target_[i]);
        return grad;
    }

private:
    std::vector<double> target_;
};

// DifferentiableFunction: the classic Rosenbrock "banana" function -- a much harder landscape.
class RosenbrockFn : public DifferentiableFunction {
public:
    double evaluate(const std::vector<double>& x) override {
        const double a = 1.0 - x[0];
        const double b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    }
    std::vector<double> gradient(const std::vector<double>& x) override {
        return {-2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]), 200.0 * (x[1] - x[0] * x[0])};
    }
};

// DifferentiableSeparableFunction: ordinary least squares as a sum of per-example losses --
// the textbook case for stochastic gradient descent.
class LinearRegressionLoss : public DifferentiableSeparableFunction {
public:
    LinearRegressionLoss(std::vector<std::vector<double>> X, std::vector<double> y)
        : X_(std::move(X)), y_(std::move(y)) {}

    [[nodiscard]] std::size_t num_functions() const override { return y_.size(); }

    double evaluate_term(const std::vector<double>& w, const std::size_t i) override {
        const double err = predict(w, i) - y_[i];
        return err * err;
    }
    std::vector<double> gradient_term(const std::vector<double>& w, const std::size_t i) override {
        const double err = predict(w, i) - y_[i];
        std::vector<double> grad(w.size());
        for (std::size_t j = 0; j < w.size(); ++j) grad[j] = 2.0 * err * X_[i][j];
        return grad;
    }

private:
    double predict(const std::vector<double>& w, const std::size_t i) const {
        double pred = 0.0;
        for (std::size_t j = 0; j < w.size(); ++j) pred += w[j] * X_[i][j];
        return pred;
    }
    std::vector<std::vector<double>> X_;
    std::vector<double> y_;
};

// ArbitraryFunction: a bumpy, multimodal landscape -- no gradient available, so only a
// derivative-free method (SimulatedAnnealing) can be used here.
class BumpyFunction : public ArbitraryFunction {
public:
    double evaluate(const std::vector<double>& x) override {
        const double bowl = (x[0] - 3.0) * (x[0] - 3.0) + (x[1] + 1.0) * (x[1] + 1.0);
        const double ripples = 5.0 * std::sin(x[0]) * std::cos(x[1]);
        return bowl + ripples;
    }
};

// ArbitraryFunction: the classic Rastrigin function -- highly multimodal (many local minima
// arranged in a regular grid), global minimum f=0 at the origin. A standard torture test for
// population-based methods, since local/gradient-based methods get stuck in the first basin
// they land in.
class RastriginFunction : public ArbitraryFunction {
public:
    double evaluate(const std::vector<double>& x) override {
        double total = 10.0 * static_cast<double>(x.size());
        for (const double xi : x) total += xi * xi - 10.0 * std::cos(2.0 * M_PI * xi);
        return total;
    }
};

void print_vec(const std::vector<double>& v) {
    std::cout << "[";
    for (std::size_t i = 0; i < v.size(); ++i) std::cout << (i ? ", " : "") << v[i];
    std::cout << "]";
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "=================== DifferentiableFunction: three optimizers, one bowl ===================\n";
    const std::vector<double> target{4.0, -2.0, 1.0};
    {
        QuadraticBowl f(target);
        std::vector<double> x{0.0, 0.0, 0.0};
        const double value = GradientDescent(GradientDescentOptions{0.1, 0.0, 1000, 1e-10}).optimize(f, x);
        std::cout << "GradientDescent: f=" << value << " x=";
        print_vec(x);
        std::cout << "\n";
    }
    {
        QuadraticBowl f(target);
        std::vector<double> x{0.0, 0.0, 0.0};
        const double value = Adam().optimize(f, x);
        std::cout << "Adam:             f=" << value << " x=";
        print_vec(x);
        std::cout << "\n";
    }
    {
        QuadraticBowl f(target);
        std::vector<double> x{0.0, 0.0, 0.0};
        const double value = LBFGS().optimize(f, x);
        std::cout << "LBFGS:            f=" << value << " x=";
        print_vec(x);
        std::cout << " (converges in far fewer iterations)\n";
    }

    std::cout << "\n=================== LBFGS on the Rosenbrock function ===================\n";
    {
        RosenbrockFn f;
        std::vector<double> x{-1.2, 1.0};
        const double value = LBFGS().optimize(f, x);
        std::cout << "f=" << value << " x=";
        print_vec(x);
        std::cout << " (true minimum: f=0 at [1, 1])\n";
    }

    std::cout << "\n=================== DifferentiableSeparableFunction: SGD vs. closed-form LM ===================\n";
    {
        const auto iris = datamunge::datasets::iris();
        std::vector<double> sepal_length(iris.nrows()), sepal_width(iris.nrows()), petal_length(iris.nrows());
        for (std::size_t i = 0; i < iris.nrows(); ++i) {
            sepal_length[i] = iris.double_at("Sepal.Length", i);
            sepal_width[i] = iris.double_at("Sepal.Width", i);
            petal_length[i] = iris.double_at("Petal.Length", i);
        }
        // SGD with a single constant step size converges far faster (and far more reliably)
        // on standardized features -- unnormalized predictors of very different scales give
        // the loss an ill-conditioned Hessian, which plain constant-step SGD handles poorly.
        // This is standard practice for gradient-based optimizers, not a workaround.
        auto mean_of = [](const std::vector<double>& v) {
            double s = 0.0;
            for (const double x : v) s += x;
            return s / static_cast<double>(v.size());
        };
        auto stddev_of = [](const std::vector<double>& v, const double mean) {
            double s = 0.0;
            for (const double x : v) s += (x - mean) * (x - mean);
            return std::sqrt(s / static_cast<double>(v.size()));
        };
        const double mean1 = mean_of(sepal_length), std1 = stddev_of(sepal_length, mean1);
        const double mean2 = mean_of(sepal_width), std2 = stddev_of(sepal_width, mean2);

        std::vector<std::vector<double>> X;
        X.reserve(iris.nrows());
        for (std::size_t i = 0; i < iris.nrows(); ++i)
            X.push_back({1.0, (sepal_length[i] - mean1) / std1, (sepal_width[i] - mean2) / std2});
        LinearRegressionLoss loss(X, petal_length);

        std::vector<double> w_std{0.0, 0.0, 0.0};
        SGDOptions sgd_options;
        sgd_options.step_size = 0.01;
        sgd_options.max_epochs = 300;
        sgd_options.batch_size = 8;
        SGD(sgd_options).optimize(loss, w_std);

        // Convert the standardized-space weights back to the original feature scale.
        const std::vector<double> w{w_std[0] - w_std[1] * mean1 / std1 - w_std[2] * mean2 / std2, w_std[1] / std1,
                                     w_std[2] / std2};
        std::cout << "SGD weights (intercept, Sepal.Length, Sepal.Width): ";
        print_vec(w);
        std::cout << "\n";

        datamunge::stats::LM lm(iris, "Petal.Length ~ Sepal.Length + Sepal.Width");
        std::cout << "LM  weights (intercept, Sepal.Length, Sepal.Width): ";
        print_vec(lm.coefficients());
        std::cout << " (closed-form OLS, for comparison)\n";
    }

    std::cout << "\n=================== ArbitraryFunction: derivative-free SimulatedAnnealing ===================\n";
    {
        BumpyFunction f;
        std::vector<double> x{0.0, 0.0};
        SimulatedAnnealingOptions sa_options;
        sa_options.initial_temperature = 10.0;
        sa_options.cooling_rate = 0.999;
        sa_options.max_iterations = 20000;
        sa_options.step_std_dev = 0.5;
        const double value = SimulatedAnnealing(sa_options).optimize(f, x);
        std::cout << "f=" << value << " x=";
        print_vec(x);
        std::cout << " (found without ever computing a gradient)\n";
    }

    std::cout << "\n=================== Population-based methods on the Rastrigin function ===================\n";
    const std::vector<double> lower{-5.12, -5.12}, upper{5.12, 5.12};
    {
        RastriginFunction f;
        std::vector<double> x{3.0, -4.0};
        PSOOptions options;
        options.topology = "global";
        options.inertia_strategy = "constant";
        const double value = PSO(options).optimize(f, x, lower, upper);
        std::cout << "PSO (global topology, constant inertia):    f=" << value << " x=";
        print_vec(x);
        std::cout << "\n";
    }
    {
        RastriginFunction f;
        std::vector<double> x{3.0, -4.0};
        PSOOptions options;
        options.topology = "ring";
        options.inertia_strategy = "linear_decay";
        const double value = PSO(options).optimize(f, x, lower, upper);
        std::cout << "PSO (ring topology, linear-decay inertia):  f=" << value << " x=";
        print_vec(x);
        std::cout << "\n";
    }
    {
        RastriginFunction f;
        std::vector<double> x{3.0, -4.0};
        DEOptions options;
        options.mutation_strategy = "rand1";
        options.crossover_strategy = "binomial";
        const double value = DifferentialEvolution(options).optimize(f, x, lower, upper);
        std::cout << "DE (rand1/binomial):                        f=" << value << " x=";
        print_vec(x);
        std::cout << "\n";
    }
    {
        RastriginFunction f;
        std::vector<double> x{3.0, -4.0};
        DEOptions options;
        options.mutation_strategy = "best1";
        options.crossover_strategy = "exponential";
        const double value = DifferentialEvolution(options).optimize(f, x, lower, upper);
        std::cout << "DE (best1/exponential):                     f=" << value << " x=";
        print_vec(x);
        std::cout << "\n";
    }
    {
        RastriginFunction f;
        std::vector<double> x{3.0, -4.0};
        GAOptions options;
        options.selection_strategy = "tournament";
        options.crossover_strategy = "blend";
        const double value = GeneticAlgorithm(options).optimize(f, x, lower, upper);
        std::cout << "GA (tournament/blend, elitism on):           f=" << value << " x=";
        print_vec(x);
        std::cout << "\n";
    }
    {
        RastriginFunction f;
        std::vector<double> x{3.0, -4.0};
        GAOptions options;
        options.selection_strategy = "rank";
        options.crossover_strategy = "uniform";
        options.elitism = false;
        const double value = GeneticAlgorithm(options).optimize(f, x, lower, upper);
        std::cout << "GA (rank/uniform, elitism off):              f=" << value << " x=";
        print_vec(x);
        std::cout << " (true minimum: f=0 at [0, 0])\n";
    }

    return 0;
}
