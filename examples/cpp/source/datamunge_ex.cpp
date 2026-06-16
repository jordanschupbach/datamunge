#include <datamunge/fda/pspline.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/random/random.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using datamunge::fda::PSplineOptions;
using datamunge::fda::fit_pspline;
using datamunge::plot::ScatterPlot;
using datamunge::random::SplitMix64;

int main() {
    constexpr std::size_t n = 500;
    constexpr double xmin = 0.0;
    constexpr double xmax = 1.0;
    constexpr double noise_sigma = 0.25;

    SplitMix64 rng(0xB51A1EULL);

    std::vector<double> x(n);
    std::vector<double> y(n);
    std::vector<double> y_true(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = xmin + (xmax - xmin) * static_cast<double>(i) / static_cast<double>(n - 1);
        y_true[i] = std::sin(20.0 / (x[i] + 0.25));
        y[i] = y_true[i] + noise_sigma * rng.normal();
    }

    PSplineOptions opts;
    opts.lambda = 0.5;
    opts.penalty_order = 2;
    const auto fit = fit_pspline(x, y, 3, 300, opts);

    std::vector<double> grid_x(200);
    for (std::size_t i = 0; i < grid_x.size(); ++i) {
        grid_x[i] = xmin + (xmax - xmin) * static_cast<double>(i)
                              / static_cast<double>(grid_x.size() - 1);
    }

    const auto grid_y = fit.predict(grid_x);
    std::vector<double> grid_true(grid_x.size());
    for (std::size_t i = 0; i < grid_x.size(); ++i) {
        grid_true[i] = std::sin(20.0 / (grid_x[i] + 0.25));
    }

    auto plot = ScatterPlot::create();
    plot.points(x, y, "samples", {37, 99, 235}, 6.0)
        .line(grid_x, grid_true, "true signal", {22, 163, 74}, 2.0)
        .line(grid_x, grid_y, "cubic B-spline fit", {220, 38, 38}, 3.0)
        .title("Nonlinear Regression With Cubic B-Splines")
        .x_label("x")
        .y_label("y")
        .background({250, 250, 252});

    plot.save_svg("build/debug/examples/datamunge_ex_pspline.svg");

    std::cout << std::fixed << std::setprecision(4)
              << "n = " << n << "\n"
              << "noise sigma = " << noise_sigma << "\n"
              << "basis functions = " << fit.basis.basis_size() << "\n"
              << "lambda = " << fit.lambda << "\n"
              << "effective dof = " << fit.effective_degrees_of_freedom << "\n"
              << "R^2 = " << fit.r_squared << "\n"
              << "adj R^2 = " << fit.adjusted_r_squared << "\n"
              << "svg = build/debug/examples/datamunge_ex_pspline.svg\n";
    return 0;
}
