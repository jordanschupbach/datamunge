#include <datamunge/ode/ode.hpp>
#include <datamunge/plot/plot.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace datamunge::ode;
using namespace datamunge::plot;

namespace {

// A live, user-supplied RHS (dy/dt = -k y), demonstrating the C++-native path every language
// binding also has access to via ODESolver::solve() -- as opposed to solve_builtin(), the only
// path available to bindings without SWIG director support (see rhs.hpp's doc comment).
class ExponentialDecay : public RHS {
public:
    explicit ExponentialDecay(double k) : k_(k) {}
    std::vector<double> evaluate(double, const std::vector<double>& y) override { return {-k_ * y[0]}; }

private:
    double k_;
};

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "=================== Method comparison: dy/dt = -y, y(0) = 1 ===================\n";
    ExponentialDecay decay(1.0);
    const double t_end = 2.0;
    const double exact = std::exp(-t_end);
    std::cout << "exact y(" << t_end << ") = " << exact << "\n\n";

    const std::vector<std::pair<std::string, ODEOptions>> configs = {
        {"Euler (h=0.01)", ODEOptions{StepMethod::Euler, 4, 0.01}},
        {"Midpoint/RK2 (h=0.01)", ODEOptions{StepMethod::Midpoint, 4, 0.01}},
        {"RK4 (h=0.01)", ODEOptions{StepMethod::RK4, 4, 0.01}},
        {"RK45 adaptive (tol=1e-8)", ODEOptions{StepMethod::RK45, 4, 0.1, 1e-8, 1e-8, 0.0}},
        {"Adams-Bashforth order 4 (h=0.01)", ODEOptions{StepMethod::AdamsBashforth, 4, 0.01}},
        {"Adams-Bashforth-Moulton order 4 (h=0.01)", ODEOptions{StepMethod::AdamsMoulton, 4, 0.01}},
    };
    std::cout << std::left << std::setw(42) << "method" << std::setw(14) << "y(2)" << std::setw(12) << "abs.error"
              << std::setw(10) << "steps" << "evals\n";
    for (const auto& [name, options] : configs) {
        ODESolver solver(options);
        const auto sol = solver.solve(decay, {1.0}, 0.0, t_end);
        std::cout << std::left << std::setw(42) << name << std::setw(14) << sol.y.back()[0] << std::setw(12)
                   << std::abs(sol.y.back()[0] - exact) << std::setw(10) << sol.steps_taken << sol.function_evaluations
                   << "\n";
    }

    std::cout << "\n=================== Harmonic oscillator (energy conservation) ===================\n";
    {
        ODESolver solver({StepMethod::RK4, 4, 0.01});
        const auto sol = solver.solve_builtin("harmonic_oscillator", {1.0}, {1.0, 0.0}, 0.0, 20.0);
        const double x = sol.y.back()[0], v = sol.y.back()[1];
        std::cout << "x(20) = " << x << " (cos(20) = " << std::cos(20.0) << ")\n";
        std::cout << "energy x^2+v^2 = " << (x * x + v * v) << " (should stay near 1.0)\n";

        std::vector<double> x_series, v_series;
        for (const auto& state : sol.y) {
            x_series.push_back(state[0]);
            v_series.push_back(state[1]);
        }
        auto plot = RPlot::plot(sol.t, x_series, "l", "x(t)", {37, 99, 235});
        plot.lines(sol.t, v_series, "v(t)", {220, 38, 38});
        plot.legend({"x(t)", "v(t)"}, {{37, 99, 235}, {220, 38, 38}});
        plot.title("Harmonic Oscillator").x_label("t").y_label("state").background({250, 250, 252});
        plot.save("ode_harmonic_oscillator.svg");
        std::cout << "wrote ode_harmonic_oscillator.svg\n";
    }

    std::cout << "\n=================== Lorenz attractor (phase plane) ===================\n";
    {
        ODESolver solver({StepMethod::RK4, 4, 0.005});
        const auto sol = solver.solve_builtin("lorenz", {10.0, 28.0, 8.0 / 3.0}, {1.0, 1.0, 1.0}, 0.0, 25.0);
        std::cout << "steps_taken = " << sol.steps_taken << ", final state = (" << sol.y.back()[0] << ", "
                   << sol.y.back()[1] << ", " << sol.y.back()[2] << ")\n";

        std::vector<double> xs, zs;
        for (const auto& state : sol.y) {
            xs.push_back(state[0]);
            zs.push_back(state[2]);
        }
        auto plot = RPlot::plot(xs, zs, "l", "trajectory", {124, 58, 237});
        plot.title("Lorenz Attractor (x-z phase plane)").x_label("x").y_label("z").background({250, 250, 252});
        plot.save("ode_lorenz_phase_plane.svg");
        std::cout << "wrote ode_lorenz_phase_plane.svg\n";
    }

    return 0;
}
