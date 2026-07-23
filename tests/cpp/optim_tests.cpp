#include <gtest/gtest.h>

#include <datamunge/optim/optim.hpp>

#include <cmath>
#include <random>
#include <vector>

using datamunge::optim::Adam;
using datamunge::optim::AdamOptions;
using datamunge::optim::AdaGrad;
using datamunge::optim::AdaGradOptions;
using datamunge::optim::AdaDelta;
using datamunge::optim::AdaDeltaOptions;
using datamunge::optim::AMSGrad;
using datamunge::optim::AMSGradOptions;
using datamunge::optim::ArbitraryFunction;
using datamunge::optim::ConjugateGradient;
using datamunge::optim::ConjugateGradientOptions;
using datamunge::optim::CoordinateDescent;
using datamunge::optim::CoordinateDescentOptions;
using datamunge::optim::DifferentiableFunction;
using datamunge::optim::DifferentiableSeparableFunction;
using datamunge::optim::GradientDescent;
using datamunge::optim::GradientDescentOptions;
using datamunge::optim::LBFGS;
using datamunge::optim::LBFGSOptions;
using datamunge::optim::LevenbergMarquardt;
using datamunge::optim::ResidualFunction;
using datamunge::optim::TrustRegionNewton;
using datamunge::optim::HessianFunction;
using datamunge::optim::EqualityConstrainedFunction;
using datamunge::optim::AugmentedLagrangian;
using datamunge::optim::AugmentedLagrangianOptions;
using datamunge::optim::SQP;
using datamunge::optim::SQPOptions;
using datamunge::optim::InequalityConstrainedFunction;
using datamunge::optim::InteriorPoint;
using datamunge::optim::InteriorPointOptions;
using datamunge::optim::NelderMead;
using datamunge::optim::NelderMeadOptions;
using datamunge::optim::Nadam;
using datamunge::optim::NadamOptions;
using datamunge::optim::NesterovAcceleratedGradient;
using datamunge::optim::NesterovAcceleratedGradientOptions;
using datamunge::optim::RandomizedBlockCoordinateDescent;
using datamunge::optim::RandomizedBlockCoordinateDescentOptions;
using datamunge::optim::ProximalFunction;
using datamunge::optim::ProximalGradient;
using datamunge::optim::RMSProp;
using datamunge::optim::RMSPropOptions;
using datamunge::optim::SeparableFunction;
using datamunge::optim::SGD;
using datamunge::optim::SGDOptions;
using datamunge::optim::SAGA;
using datamunge::optim::SAGAOptions;
using datamunge::optim::SVRG;
using datamunge::optim::SVRGOptions;
using datamunge::optim::SimulatedAnnealing;
using datamunge::optim::SimulatedAnnealingOptions;

namespace {

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

class RosenbrockFn : public DifferentiableFunction {
public:
    double evaluate(const std::vector<double>& x) override {
        const double a = 1.0 - x[0];
        const double b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    }
    std::vector<double> gradient(const std::vector<double>& x) override {
        const double dfdx0 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        const double dfdx1 = 200.0 * (x[1] - x[0] * x[0]);
        return {dfdx0, dfdx1};
    }
};

class ThreeTermFunction : public SeparableFunction {
public:
    [[nodiscard]] std::size_t num_functions() const override { return 3; }
    double evaluate_term(const std::vector<double>& coordinates, const std::size_t i) override {
        return coordinates[0] * static_cast<double>(i + 1); // terms: x, 2x, 3x
    }
};

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

} // namespace

TEST(SeparableFunction, DefaultEvaluateSumsTerms) {
    ThreeTermFunction f;
    const std::vector<double> x{2.0};
    EXPECT_DOUBLE_EQ(f.evaluate(x), 2.0 * (1.0 + 2.0 + 3.0));
}

TEST(DifferentiableSeparableFunction, DefaultEvaluateAndGradientSumTerms) {
    const std::vector<std::vector<double>> X{{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
    const std::vector<double> y{2.0, -3.0, -1.0}; // exactly consistent with w = [2, -3]
    LinearRegressionLoss loss(X, y);
    const std::vector<double> w{2.0, -3.0};

    EXPECT_NEAR(loss.evaluate(w), 0.0, 1e-12);
    const auto grad = loss.gradient(w);
    for (const double g : grad) EXPECT_NEAR(g, 0.0, 1e-12); // at the exact minimizer, gradient vanishes

    double manual_sum = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) manual_sum += loss.evaluate_term(w, i);
    EXPECT_NEAR(loss.evaluate(w), manual_sum, 1e-12);
}

TEST(GradientDescent, ConvergesToQuadraticMinimum) {
    QuadraticBowl f({3.0, -2.0});
    std::vector<double> x{0.0, 0.0};
    GradientDescentOptions options;
    options.step_size = 0.1;
    options.max_iterations = 1000;
    const GradientDescent optimizer(options);
    const double value = optimizer.optimize(f, x);
    EXPECT_NEAR(value, 0.0, 1e-6);
    EXPECT_NEAR(x[0], 3.0, 1e-3);
    EXPECT_NEAR(x[1], -2.0, 1e-3);
}

TEST(GradientDescent, MomentumAlsoConverges) {
    QuadraticBowl f({1.0, 1.0});
    std::vector<double> x{5.0, -5.0};
    GradientDescentOptions options;
    options.step_size = 0.05;
    options.momentum = 0.9;
    options.max_iterations = 2000;
    const GradientDescent optimizer(options);
    const double value = optimizer.optimize(f, x);
    EXPECT_NEAR(value, 0.0, 1e-4);
}

TEST(CoordinateDescent, CyclesToQuadraticMinimum) {
    QuadraticBowl f({3.0, -2.0, 1.0});
    std::vector<double> x{0.0, 0.0, 0.0};
    CoordinateDescentOptions options;
    options.step_size = 1.0;
    options.max_iterations = 20;
    const double value = CoordinateDescent(options).optimize(f, x);
    EXPECT_NEAR(value, 0.0, 1e-12);
    EXPECT_NEAR(x[0], 3.0, 1e-8);
    EXPECT_NEAR(x[1], -2.0, 1e-8);
    EXPECT_NEAR(x[2], 1.0, 1e-8);
}

TEST(ProximalGradient, SoftThresholdsL1Quadratic) {
    class L1Quadratic : public ProximalFunction {
    public:
        double evaluate(const std::vector<double>& x) override { const double d=x[0]-3.0; return 0.5*d*d+std::abs(x[0]); }
        std::vector<double> gradient(const std::vector<double>& x) override { return {x[0]-3.0}; }
        std::vector<double> proximal(const std::vector<double>& x, double step) override { return {std::copysign(std::max(0.0,std::abs(x[0])-step),x[0])}; }
    } f;
    std::vector<double> x{0.0};
    datamunge::optim::ProximalGradientOptions options; options.step_size = 1.0;
    EXPECT_NEAR(ProximalGradient(options).optimize(f, x), 2.5, 1e-12);
    EXPECT_NEAR(x[0], 2.0, 1e-12);
}

TEST(RandomizedBlockCoordinateDescent, VisitsEveryCoordinateAndConverges) {
    QuadraticBowl f({3.0, -2.0, 1.0, -4.0, 5.0});
    std::vector<double> x(5, 0.0);
    RandomizedBlockCoordinateDescentOptions options;
    options.block_size = 2;
    options.step_size = 1.0;
    options.max_iterations = 20;
    options.seed = 7;
    const double value = RandomizedBlockCoordinateDescent(options).optimize(f, x);
    EXPECT_NEAR(value, 0.0, 1e-12);
    EXPECT_NEAR(x[0], 3.0, 1e-8);
    EXPECT_NEAR(x[1], -2.0, 1e-8);
    EXPECT_NEAR(x[2], 1.0, 1e-8);
    EXPECT_NEAR(x[3], -4.0, 1e-8);
    EXPECT_NEAR(x[4], 5.0, 1e-8);
}

TEST(RandomizedBlockCoordinateDescent, RejectsZeroBlockSize) {
    QuadraticBowl f({1.0});
    std::vector<double> x{0.0};
    RandomizedBlockCoordinateDescentOptions options;
    options.block_size = 0;
    EXPECT_THROW(RandomizedBlockCoordinateDescent(options).optimize(f, x), std::invalid_argument);
}

TEST(Adam, ConvergesToQuadraticMinimum) {
    QuadraticBowl f({-4.0, 7.0});
    std::vector<double> x{0.0, 0.0};
    AdamOptions options;
    options.step_size = 0.1;
    options.max_iterations = 5000;
    const Adam optimizer(options);
    const double value = optimizer.optimize(f, x);
    EXPECT_NEAR(value, 0.0, 1e-6);
    EXPECT_NEAR(x[0], -4.0, 1e-2);
    EXPECT_NEAR(x[1], 7.0, 1e-2);
}

TEST(AdaGrad, ConvergesToQuadraticMinimum) {
    QuadraticBowl f({-4.0, 7.0});
    std::vector<double> x{0.0, 0.0};
    AdaGradOptions options;
    options.step_size = 1.0;
    options.max_iterations = 10000;
    const double value = AdaGrad(options).optimize(f, x);
    EXPECT_NEAR(value, 0.0, 1e-8);
    EXPECT_NEAR(x[0], -4.0, 1e-4);
    EXPECT_NEAR(x[1], 7.0, 1e-4);
}

TEST(AdaGrad, RejectsNonPositiveEpsilon) {
    QuadraticBowl f({0.0});
    std::vector<double> x{1.0};
    AdaGradOptions options;
    options.epsilon = 0.0;
    EXPECT_THROW(AdaGrad(options).optimize(f, x), std::invalid_argument);
}

TEST(AMSGrad, ConvergesToQuadraticMinimum) {
    QuadraticBowl f({-4.0, 7.0});
    std::vector<double> x{0.0, 0.0};
    AMSGradOptions options;
    options.step_size = 0.1;
    options.max_iterations = 5000;
    const double value = AMSGrad(options).optimize(f, x);
    EXPECT_NEAR(value, 0.0, 1e-6);
    EXPECT_NEAR(x[0], -4.0, 1e-2);
    EXPECT_NEAR(x[1], 7.0, 1e-2);
}

TEST(AMSGrad, RejectsInvalidBeta) {
    QuadraticBowl f({0.0});
    std::vector<double> x{1.0};
    AMSGradOptions options;
    options.beta2 = 1.0;
    EXPECT_THROW(AMSGrad(options).optimize(f, x), std::invalid_argument);
}

TEST(Nadam, ConvergesToQuadraticMinimum) {
    QuadraticBowl f({-4.0, 7.0});
    std::vector<double> x{0.0, 0.0};
    NadamOptions options;
    options.step_size = 0.1;
    options.max_iterations = 5000;
    const double value = Nadam(options).optimize(f, x);
    EXPECT_NEAR(value, 0.0, 1e-6);
    EXPECT_NEAR(x[0], -4.0, 1e-2);
    EXPECT_NEAR(x[1], 7.0, 1e-2);
}

TEST(Nadam, RejectsInvalidBeta) {
    QuadraticBowl f({0.0});
    std::vector<double> x{1.0};
    NadamOptions options;
    options.beta1 = 1.0;
    EXPECT_THROW(Nadam(options).optimize(f, x), std::invalid_argument);
}

TEST(AdaDelta, ReducesQuadraticObjective) {
    QuadraticBowl f({-1.0, 1.0});
    std::vector<double> x{0.0, 0.0};
    AdaDeltaOptions options;
    options.max_iterations = 10000;
    const double value = AdaDelta(options).optimize(f, x);
    EXPECT_LT(value, 1e-4);
    EXPECT_NEAR(x[0], -1.0, 0.01);
    EXPECT_NEAR(x[1], 1.0, 0.01);
}

TEST(AdaDelta, RejectsInvalidDecayRate) {
    QuadraticBowl f({0.0});
    std::vector<double> x{1.0};
    AdaDeltaOptions options;
    options.decay_rate = 1.0;
    EXPECT_THROW(AdaDelta(options).optimize(f, x), std::invalid_argument);
}

TEST(RMSProp, GetsCloseToQuadraticMinimum) {
    QuadraticBowl f({-4.0, 7.0});
    std::vector<double> x{0.0, 0.0};
    RMSPropOptions options;
    options.step_size = 0.01;
    options.max_iterations = 10000;
    const double value = RMSProp(options).optimize(f, x);
    EXPECT_LT(value, 1e-5);
    EXPECT_NEAR(x[0], -4.0, 0.01);
    EXPECT_NEAR(x[1], 7.0, 0.01);
}

TEST(RMSProp, RejectsInvalidDecayRate) {
    QuadraticBowl f({0.0});
    std::vector<double> x{1.0};
    RMSPropOptions options;
    options.decay_rate = 1.0;
    EXPECT_THROW(RMSProp(options).optimize(f, x), std::invalid_argument);
}

TEST(NesterovAcceleratedGradient, ConvergesToQuadraticMinimum) {
    QuadraticBowl f({-4.0, 7.0});
    std::vector<double> x{0.0, 0.0};
    NesterovAcceleratedGradientOptions options;
    options.step_size = 0.05;
    options.momentum = 0.8;
    options.max_iterations = 1000;
    const double value = NesterovAcceleratedGradient(options).optimize(f, x);
    EXPECT_NEAR(value, 0.0, 1e-10);
    EXPECT_NEAR(x[0], -4.0, 1e-5);
    EXPECT_NEAR(x[1], 7.0, 1e-5);
}

TEST(LBFGS, ConvergesToQuadraticMinimumInFewIterations) {
    QuadraticBowl f({10.0, -10.0, 5.0});
    std::vector<double> x{0.0, 0.0, 0.0};
    const LBFGS optimizer;
    const double value = optimizer.optimize(f, x);
    EXPECT_NEAR(value, 0.0, 1e-10);
    EXPECT_NEAR(x[0], 10.0, 1e-5);
    EXPECT_NEAR(x[1], -10.0, 1e-5);
    EXPECT_NEAR(x[2], 5.0, 1e-5);
}

TEST(LBFGS, ConvergesOnRosenbrock) {
    RosenbrockFn f;
    std::vector<double> x{-1.2, 1.0}; // classic Rosenbrock starting point
    LBFGSOptions options;
    options.max_iterations = 1000;
    const LBFGS optimizer(options);
    const double value = optimizer.optimize(f, x);
    EXPECT_NEAR(value, 0.0, 1e-8);
    EXPECT_NEAR(x[0], 1.0, 1e-3);
    EXPECT_NEAR(x[1], 1.0, 1e-3);
}

TEST(ConjugateGradient, ConvergesOnRosenbrock) {
    RosenbrockFn f;
    std::vector<double> x{-1.2, 1.0};
    ConjugateGradientOptions options;
    options.max_iterations = 5000;
    options.tolerance = 1e-7;
    const double value = ConjugateGradient(options).optimize(f, x);
    EXPECT_NEAR(value, 0.0, 1e-8);
    EXPECT_NEAR(x[0], 1.0, 1e-3);
    EXPECT_NEAR(x[1], 1.0, 1e-3);
}

TEST(NelderMead, ConvergesOnRosenbrockWithoutGradients) {
    RosenbrockFn f;
    ArbitraryFunction& generic = f;
    std::vector<double> x{-1.2, 1.0};
    NelderMeadOptions options;
    options.initial_simplex_scale = 0.5;
    options.max_iterations = 5000;
    options.tolerance = 1e-8;
    const double value = NelderMead(options).optimize(generic, x);
    EXPECT_NEAR(value, 0.0, 1e-8);
    EXPECT_NEAR(x[0], 1.0, 1e-3);
    EXPECT_NEAR(x[1], 1.0, 1e-3);
}

TEST(SGD, RecoversTrueLinearRegressionWeights) {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> unif(-5.0, 5.0);
    const std::vector<double> true_w{2.0, -3.0};

    std::vector<std::vector<double>> X;
    std::vector<double> y;
    for (int i = 0; i < 200; ++i) {
        const double x0 = unif(rng);
        const double x1 = unif(rng);
        X.push_back({x0, x1});
        y.push_back(true_w[0] * x0 + true_w[1] * x1); // noise-free
    }
    LinearRegressionLoss loss(X, y);

    std::vector<double> w{0.0, 0.0};
    SGDOptions options;
    options.step_size = 0.01;
    options.max_epochs = 200;
    options.batch_size = 4;
    const SGD optimizer(options);
    optimizer.optimize(loss, w);

    EXPECT_NEAR(w[0], true_w[0], 0.1);
    EXPECT_NEAR(w[1], true_w[1], 0.1);
}

TEST(SGD, RejectsFunctionWithZeroTerms) {
    class EmptyLoss : public DifferentiableSeparableFunction {
    public:
        [[nodiscard]] std::size_t num_functions() const override { return 0; }
        double evaluate_term(const std::vector<double>&, std::size_t) override { return 0.0; }
        std::vector<double> gradient_term(const std::vector<double>& w, std::size_t) override {
            return std::vector<double>(w.size(), 0.0);
        }
    };
    EmptyLoss loss;
    std::vector<double> w{0.0};
    const SGD optimizer;
    EXPECT_THROW(optimizer.optimize(loss, w), std::invalid_argument);
}

TEST(SVRG, RecoversTrueLinearRegressionWeights) {
    const std::vector<std::vector<double>> X{{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {2.0, -1.0}};
    const std::vector<double> y{2.0, -3.0, -1.0, 7.0};
    LinearRegressionLoss loss(X, y);
    std::vector<double> w{0.0, 0.0};
    SVRGOptions options;
    options.step_size = 0.02;
    options.max_epochs = 300;
    options.seed = 7;
    SVRG(options).optimize(loss, w);
    EXPECT_NEAR(w[0], 2.0, 0.02);
    EXPECT_NEAR(w[1], -3.0, 0.02);
}

TEST(SVRG, RejectsZeroTermFunction) {
    class EmptyLoss : public DifferentiableSeparableFunction {
    public:
        [[nodiscard]] std::size_t num_functions() const override { return 0; }
        double evaluate_term(const std::vector<double>&, std::size_t) override { return 0.0; }
        std::vector<double> gradient_term(const std::vector<double>& w, std::size_t) override { return std::vector<double>(w.size()); }
    } loss;
    std::vector<double> w{0.0};
    EXPECT_THROW(SVRG().optimize(loss, w), std::invalid_argument);
}

TEST(SAGA, RecoversTrueLinearRegressionWeights) {
    const std::vector<std::vector<double>> X{{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {2.0, -1.0}};
    const std::vector<double> y{2.0, -3.0, -1.0, 7.0};
    LinearRegressionLoss loss(X, y);
    std::vector<double> w{0.0, 0.0};
    SAGAOptions options;
    options.step_size = 0.02;
    options.max_iterations = 3000;
    options.seed = 7;
    SAGA(options).optimize(loss, w);
    EXPECT_NEAR(w[0], 2.0, 0.02);
    EXPECT_NEAR(w[1], -3.0, 0.02);
}

TEST(SAGA, RejectsZeroTermFunction) {
    class EmptyLoss : public DifferentiableSeparableFunction {
    public:
        [[nodiscard]] std::size_t num_functions() const override { return 0; }
        double evaluate_term(const std::vector<double>&, std::size_t) override { return 0.0; }
        std::vector<double> gradient_term(const std::vector<double>& w, std::size_t) override { return std::vector<double>(w.size()); }
    } loss;
    std::vector<double> w{0.0};
    EXPECT_THROW(SAGA().optimize(loss, w), std::invalid_argument);
}

TEST(SimulatedAnnealing, GetsCloseToQuadraticMinimumWithoutGradients) {
    QuadraticBowl f({2.0, 2.0});
    ArbitraryFunction& generic = f; // exercised purely through the ArbitraryFunction interface
    std::vector<double> x{0.0, 0.0};
    SimulatedAnnealingOptions options;
    options.initial_temperature = 5.0;
    options.cooling_rate = 0.999;
    options.max_iterations = 20000;
    options.step_std_dev = 0.5;
    const SimulatedAnnealing optimizer(options);
    const double value = optimizer.optimize(generic, x);
    EXPECT_LT(value, 0.05); // stochastic method: a loose but meaningful tolerance
}

TEST(LevenbergMarquardt, FitsLinearResiduals) {
    class LinearResiduals : public ResidualFunction {
    public:
        std::vector<double> residuals(const std::vector<double>& x) override { return {x[0] - 2.0, 2.0 * x[0] - 4.0}; }
        std::vector<std::vector<double>> jacobian(const std::vector<double>&) override { return {{1.0}, {2.0}}; }
    } f;
    std::vector<double> x{0.0};
    EXPECT_NEAR(LevenbergMarquardt().optimize(f, x), 0.0, 1e-10);
}

TEST(TrustRegionNewton, MinimizesQuadraticWithHessianModel) {
    class Quadratic : public HessianFunction { public:
        double evaluate(const std::vector<double>& x) override { double d=x[0]-2.0; return d*d; }
        std::vector<double> gradient(const std::vector<double>& x) override { return {2.0*(x[0]-2.0)}; }
        std::vector<std::vector<double>> hessian(const std::vector<double>&) override { return {{2.0}}; }
    } f;
    std::vector<double> x{0.0};
    EXPECT_NEAR(TrustRegionNewton().optimize(f, x), 0.0, 1e-10);
}

TEST(AugmentedLagrangian, EnforcesLinearEqualityConstraint) {
    class ConstrainedQuadratic : public EqualityConstrainedFunction { public:
        double evaluate(const std::vector<double>& x) override { double a=x[0]-2,b=x[1]-2; return a*a+b*b; }
        std::vector<double> gradient(const std::vector<double>& x) override { return {2*(x[0]-2),2*(x[1]-2)}; }
        std::vector<double> constraints(const std::vector<double>& x) override { return {x[0]+x[1]-1}; }
        std::vector<std::vector<double>> constraint_jacobian(const std::vector<double>&) override { return {{1,1}}; }
    } f;
    std::vector<double> x{0,0}; AugmentedLagrangianOptions o; o.step_size=.001; o.inner_iterations=500; o.max_outer_iterations=5;
    AugmentedLagrangian(o).optimize(f,x); EXPECT_NEAR(x[0]+x[1],1.0,1e-3);
}

TEST(SQP, SolvesProjectedEqualityConstrainedQuadratic) {
    class ConstrainedQuadratic : public EqualityConstrainedFunction { public:
        double evaluate(const std::vector<double>& x) override { double a=x[0]-2,b=x[1]-2; return a*a+b*b; }
        std::vector<double> gradient(const std::vector<double>& x) override { return {2*(x[0]-2),2*(x[1]-2)}; }
        std::vector<double> constraints(const std::vector<double>& x) override { return {x[0]+x[1]-1}; }
        std::vector<std::vector<double>> constraint_jacobian(const std::vector<double>&) override { return {{1,1}}; }
    } f;
    std::vector<double> x{0,0}; SQPOptions o; o.step_size=.1; o.max_iterations=200;
    SQP(o).optimize(f,x); EXPECT_NEAR(x[0]+x[1],1.0,1e-5); EXPECT_NEAR(x[0],.5,1e-3);
}

TEST(InteriorPoint, MaintainsStrictFeasibility) {
    class BoundedQuadratic : public InequalityConstrainedFunction { public:
        double evaluate(const std::vector<double>& x) override { return x[0]*x[0]; }
        std::vector<double> gradient(const std::vector<double>& x) override { return {2*x[0]}; }
        std::vector<double> inequalities(const std::vector<double>& x) override { return {x[0]-1}; }
        std::vector<std::vector<double>> inequality_jacobian(const std::vector<double>&) override { return {{1}}; }
    } f;
    std::vector<double> x{0.5}; InteriorPointOptions o; o.step_size=.001; o.inner_iterations=100;
    InteriorPoint(o).optimize(f,x); EXPECT_LT(x[0],1.0); EXPECT_NEAR(x[0],0.0,.1);
}
