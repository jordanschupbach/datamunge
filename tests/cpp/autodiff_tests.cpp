#include <gtest/gtest.h>

#include <datamunge/autodiff/autodiff.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

using datamunge::autodiff::Dual;
using datamunge::autodiff::HyperDual;
using datamunge::autodiff::Tape;
using datamunge::autodiff::Var;

namespace {

// A single functor template works unmodified across double, Dual, HyperDual, and Var --
// the entire point of operator-overloading-style automatic differentiation.
struct Rosenbrock {
    template <typename T>
    T operator()(const std::vector<T>& x) const {
        const auto a = 1.0 - x[0];
        const auto b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    }
};

struct Cubic {
    template <typename T>
    T operator()(T x) const {
        return x * x * x - 2.0 * x;
    }
};

struct VectorFn {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x) const {
        return {x[0] * x[0], x[0] * x[1], x[1] * x[1] * x[1]};
    }
};

struct MixedPartialFn {
    template <typename T>
    T operator()(const std::vector<T>& x) const {
        return x[0] * x[0] * x[1];
    }
};

} // namespace

TEST(Dual, BasicArithmeticMatchesHandDerivatives) {
    const Dual x(3.0, 1.0);
    const Dual y = x * x; // d/dx x^2 = 2x
    EXPECT_DOUBLE_EQ(y.value(), 9.0);
    EXPECT_DOUBLE_EQ(y.derivative(), 6.0);

    const Dual sum = x + Dual(1.0, 0.0);
    EXPECT_DOUBLE_EQ(sum.value(), 4.0);
    EXPECT_DOUBLE_EQ(sum.derivative(), 1.0);

    const Dual scaled = x * 2.0 - 1.0;
    EXPECT_DOUBLE_EQ(scaled.value(), 5.0);
    EXPECT_DOUBLE_EQ(scaled.derivative(), 2.0);
}

TEST(Dual, ProductRuleOnSinTimesExp) {
    const Dual x(1.0, 1.0);
    const Dual y = datamunge::autodiff::sin(x) * datamunge::autodiff::exp(x);
    const double expected_value = std::sin(1.0) * std::exp(1.0);
    const double expected_derivative = std::cos(1.0) * std::exp(1.0) + std::sin(1.0) * std::exp(1.0);
    EXPECT_NEAR(y.value(), expected_value, 1e-12);
    EXPECT_NEAR(y.derivative(), expected_derivative, 1e-12);
}

TEST(Dual, MathFunctionsMatchKnownDerivatives) {
    using datamunge::autodiff::abs;
    using datamunge::autodiff::pow;
    using datamunge::autodiff::tan;
    using datamunge::autodiff::tanh;

    const Dual cube = pow(Dual(2.0, 1.0), 3.0);
    EXPECT_DOUBLE_EQ(cube.value(), 8.0);
    EXPECT_DOUBLE_EQ(cube.derivative(), 12.0); // 3*2^2

    const Dual t = tan(Dual(0.0, 1.0));
    EXPECT_DOUBLE_EQ(t.value(), 0.0);
    EXPECT_DOUBLE_EQ(t.derivative(), 1.0);

    const Dual th = tanh(Dual(0.0, 1.0));
    EXPECT_DOUBLE_EQ(th.value(), 0.0);
    EXPECT_DOUBLE_EQ(th.derivative(), 1.0);

    const Dual a = abs(Dual(-3.0, 1.0));
    EXPECT_DOUBLE_EQ(a.value(), 3.0);
    EXPECT_DOUBLE_EQ(a.derivative(), -1.0);
}

TEST(HyperDual, SecondDerivativeOfCubeMatchesHandComputation) {
    const HyperDual x(2.0, 1.0, 1.0, 0.0); // both epsilon directions seeded on the same variable
    const HyperDual y = x * x * x;
    EXPECT_DOUBLE_EQ(y.value(), 8.0);
    EXPECT_DOUBLE_EQ(y.eps1(), 12.0);      // f'(2) = 3*2^2
    EXPECT_DOUBLE_EQ(y.eps2(), 12.0);
    EXPECT_DOUBLE_EQ(y.eps1eps2(), 12.0);  // f''(2) = 6*2
}

TEST(HyperDual, MixedPartialOfXSquaredTimesY) {
    // f(x, y) = x^2 * y; d^2f/dxdy = 2x
    const HyperDual x(2.0, 1.0, 0.0, 0.0);
    const HyperDual y(3.0, 0.0, 1.0, 0.0);
    const HyperDual f = x * x * y;
    EXPECT_DOUBLE_EQ(f.value(), 12.0);
    EXPECT_DOUBLE_EQ(f.eps1(), 12.0); // df/dx = 2xy = 12
    EXPECT_DOUBLE_EQ(f.eps2(), 4.0);  // df/dy = x^2 = 4
    EXPECT_DOUBLE_EQ(f.eps1eps2(), 4.0); // d2f/dxdy = 2x = 4
}

TEST(Tape, ReverseModeGradientMatchesHandDerivative) {
    Tape tape;
    const Var x1(tape, 2.0);
    const Var x2(tape, 3.0);
    const Var y = x1 * x2 + datamunge::autodiff::sin(x1);

    EXPECT_NEAR(y.value(), 6.0 + std::sin(2.0), 1e-12);

    const auto adjoint = tape.backward(y.index());
    EXPECT_NEAR(adjoint[x1.index()], 3.0 + std::cos(2.0), 1e-12); // dy/dx1 = x2 + cos(x1)
    EXPECT_NEAR(adjoint[x2.index()], 2.0, 1e-12);                 // dy/dx2 = x1
}

TEST(Tape, MixedScalarOperationsWork) {
    Tape tape;
    const Var x(tape, 4.0);
    const Var y = 2.0 * x - 1.0;
    EXPECT_DOUBLE_EQ(y.value(), 7.0);
    const auto adjoint = tape.backward(y.index());
    EXPECT_DOUBLE_EQ(adjoint[x.index()], 2.0);

    const Var z = 10.0 / x;
    EXPECT_DOUBLE_EQ(z.value(), 2.5);
    const auto adjoint_z = tape.backward(z.index());
    EXPECT_DOUBLE_EQ(adjoint_z[x.index()], -10.0 / (4.0 * 4.0));
}

TEST(Tape, OperationsAcrossDifferentTapesThrow) {
    Tape tape_a;
    Tape tape_b;
    const Var a(tape_a, 1.0);
    const Var b(tape_b, 2.0);
    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(Tape, OutOfRangeAccessThrows) {
    Tape tape;
    const Var x(tape, 1.0);
    (void)x;
    EXPECT_THROW(tape.value_at(100), std::out_of_range);
    EXPECT_THROW(tape.backward(100), std::out_of_range);
}

TEST(AutodiffDrivers, ScalarDerivativeMatchesHandComputation) {
    const double d = datamunge::autodiff::derivative(Cubic{}, 2.0);
    EXPECT_DOUBLE_EQ(d, 10.0); // f(x)=x^3-2x, f'(x)=3x^2-2, f'(2)=10
}

TEST(AutodiffDrivers, ForwardAndReverseGradientsAgreeAndMatchHandDerivative) {
    const std::vector<double> x{0.0, 0.0};
    const auto grad_fwd = datamunge::autodiff::gradient_forward(Rosenbrock{}, x);
    const auto [value, grad_rev] = datamunge::autodiff::gradient_reverse(Rosenbrock{}, x);

    EXPECT_NEAR(value, 1.0, 1e-12);
    ASSERT_EQ(grad_fwd.size(), 2u);
    ASSERT_EQ(grad_rev.size(), 2u);
    EXPECT_NEAR(grad_fwd[0], -2.0, 1e-9);
    EXPECT_NEAR(grad_fwd[1], 0.0, 1e-9);
    EXPECT_NEAR(grad_rev[0], -2.0, 1e-9);
    EXPECT_NEAR(grad_rev[1], 0.0, 1e-9);

    const auto grad_default = datamunge::autodiff::gradient(Rosenbrock{}, x);
    EXPECT_NEAR(grad_default[0], grad_rev[0], 1e-12);
    EXPECT_NEAR(grad_default[1], grad_rev[1], 1e-12);
}

TEST(AutodiffDrivers, JacobianForwardMatchesHandDerivation) {
    const std::vector<double> x{2.0, 3.0};
    const auto jac = datamunge::autodiff::jacobian_forward(VectorFn{}, x);
    ASSERT_EQ(jac.rows(), 3u);
    ASSERT_EQ(jac.cols(), 2u);
    // outputs: [x0^2, x0*x1, x1^3]; d/dx0 = [2x0, x1, 0]; d/dx1 = [0, x0, 3x1^2]
    EXPECT_DOUBLE_EQ(jac(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(jac(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(jac(2, 0), 0.0);
    EXPECT_DOUBLE_EQ(jac(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(jac(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(jac(2, 1), 27.0);
}

TEST(AutodiffDrivers, HessianOfMixedPartialFunctionMatchesHandDerivation) {
    const std::vector<double> x{2.0, 3.0};
    const auto H = datamunge::autodiff::hessian(MixedPartialFn{}, x);
    ASSERT_EQ(H.rows(), 2u);
    ASSERT_EQ(H.cols(), 2u);
    EXPECT_DOUBLE_EQ(H(0, 0), 6.0); // d2f/dx2 = 2y
    EXPECT_DOUBLE_EQ(H(1, 1), 0.0); // d2f/dy2 = 0
    EXPECT_DOUBLE_EQ(H(0, 1), 4.0); // d2f/dxdy = 2x
    EXPECT_DOUBLE_EQ(H(1, 0), 4.0);
}

TEST(AutodiffDrivers, HessianOfRosenbrockAtOriginMatchesHandDerivation) {
    const std::vector<double> x{0.0, 0.0};
    const auto H = datamunge::autodiff::hessian(Rosenbrock{}, x);
    EXPECT_NEAR(H(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(H(1, 1), 200.0, 1e-9);
    EXPECT_NEAR(H(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(H(1, 0), 0.0, 1e-9);
}
