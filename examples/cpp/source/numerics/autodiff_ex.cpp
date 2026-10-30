#include <datamunge/autodiff/autodiff.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using datamunge::autodiff::Dual;
using datamunge::autodiff::Tape;
using datamunge::autodiff::Var;

namespace {

// A single function template usable, unmodified, with double, Dual, HyperDual, or Var --
// this genericity is the whole point of operator-overloading-style automatic differentiation.
struct Rosenbrock {
    template <typename T>
    T operator()(const std::vector<T>& x) const {
        const auto a = 1.0 - x[0];
        const auto b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    }
};

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "=================== Forward mode: scalar derivative ===================\n";
    auto cube_minus_2x = [](Dual x) { return x * x * x - 2.0 * x; };
    const double x0 = 2.0;
    const double d = datamunge::autodiff::derivative(cube_minus_2x, x0);
    std::cout << "f(x) = x^3 - 2x, f'(2) = " << d << " (exact: 10)\n\n";

    std::cout << "=================== Reverse mode: build a graph by hand ===================\n";
    Tape tape;
    const Var a(tape, 2.0);
    const Var b(tape, 3.0);
    const Var y = a * b + datamunge::autodiff::sin(a);
    std::cout << "y = a*b + sin(a) at a=2, b=3 -> y = " << y.value() << "\n";
    const auto adjoint = tape.backward(y.index());
    std::cout << "dy/da = " << adjoint[a.index()] << " (exact: b + cos(a))\n";
    std::cout << "dy/db = " << adjoint[b.index()] << " (exact: a)\n\n";

    std::cout << "=================== Forward vs reverse mode agree on the Rosenbrock function ===================\n";
    const std::vector<double> p{0.0, 0.0};
    const auto grad_fwd = datamunge::autodiff::gradient_forward(Rosenbrock{}, p);
    const auto [value, grad_rev] = datamunge::autodiff::gradient_reverse(Rosenbrock{}, p);
    std::cout << "f(0,0) = " << value << "\n";
    std::cout << "gradient (forward mode): [" << grad_fwd[0] << ", " << grad_fwd[1] << "]\n";
    std::cout << "gradient (reverse mode): [" << grad_rev[0] << ", " << grad_rev[1] << "]\n\n";

    std::cout << "=================== Jacobian of a vector-valued function ===================\n";
    auto vector_fn = [](const std::vector<Dual>& v) {
        return std::vector<Dual>{v[0] * v[0], v[0] * v[1], v[1] * v[1] * v[1]};
    };
    const auto jac = datamunge::autodiff::jacobian_forward(vector_fn, {2.0, 3.0});
    std::cout << "f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:\n";
    for (std::size_t r = 0; r < jac.rows(); ++r) {
        std::cout << "  [";
        for (std::size_t c = 0; c < jac.cols(); ++c) std::cout << (c ? ", " : "") << jac(r, c);
        std::cout << "]\n";
    }
    std::cout << "\n";

    std::cout << "=================== Hessian via second-order forward mode ===================\n";
    const auto H = datamunge::autodiff::hessian(Rosenbrock{}, std::vector<double>{1.0, 1.0});
    std::cout << "Hessian of the Rosenbrock function at its minimum (1,1):\n";
    for (std::size_t r = 0; r < H.rows(); ++r) {
        std::cout << "  [";
        for (std::size_t c = 0; c < H.cols(); ++c) std::cout << (c ? ", " : "") << H(r, c);
        std::cout << "]\n";
    }
    std::cout << "\n";

    std::cout << "=================== Gradient descent driven by reverse-mode gradients ===================\n";
    std::vector<double> point{-1.2, 1.0}; // the classic Rosenbrock starting point
    constexpr double learning_rate = 0.001;
    constexpr int n_steps = 2000;
    for (int step = 0; step < n_steps; ++step) {
        const auto [loss, grad] = datamunge::autodiff::gradient_reverse(Rosenbrock{}, point);
        for (std::size_t i = 0; i < point.size(); ++i) point[i] -= learning_rate * grad[i];
        if (step == 0 || step == n_steps - 1)
            std::cout << "step " << step << ": loss = " << loss << ", x = [" << point[0] << ", " << point[1]
                      << "]\n";
    }
    std::cout << "(true minimum is at [1, 1] with loss 0)\n";

    return 0;
}
