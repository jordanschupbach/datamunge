#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/stats.hpp>

#include <iostream>

using datamunge::stats::KernelRegression;
using datamunge::stats::KernelRegressionOptions;

namespace {
constexpr const char* kFormula = "Petal.Length ~ Petal.Width";
}

int main() {
    const auto iris = datamunge::datasets::iris();
    std::cout << "iris: " << iris.nrows() << " rows x " << iris.ncols() << " cols\n";
    std::cout << "formula: " << kFormula << "\n\n";

    KernelRegression model(iris, kFormula); // bandwidth auto-selected via leave-one-out CV
    model.print_summary(std::cout);

    model.plot_fit(iris).save("kernel_regression_iris_fit.svg");
    model.plot_cv_curve().save("kernel_regression_iris_cv.svg");
    std::cout << "\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg\n";

    // A too-small and a too-large bandwidth, for comparison -- the classic
    // kernel regression bias-variance story: small h chases noise (wiggly,
    // low-bias/high-variance), large h oversmooths (flat, high-bias/low-variance).
    KernelRegressionOptions small_h;
    small_h.bandwidth = 0.05;
    KernelRegression small(iris, kFormula, small_h);
    std::cout << "\nbandwidth=0.05 (too small): LOO R-squared=" << small.r_squared()
               << "  LOO RMSE=" << small.rmse() << "\n";
    small.plot_fit(iris).save("kernel_regression_iris_fit_small_bandwidth.svg");

    KernelRegressionOptions large_h;
    large_h.bandwidth = 5.0;
    KernelRegression large(iris, kFormula, large_h);
    std::cout << "bandwidth=5.0 (too large):  LOO R-squared=" << large.r_squared()
               << "  LOO RMSE=" << large.rmse() << "\n";
    large.plot_fit(iris).save("kernel_regression_iris_fit_large_bandwidth.svg");

    std::cout << "bandwidth=" << model.bandwidth() << " (CV-selected): LOO R-squared=" << model.r_squared()
               << "  LOO RMSE=" << model.rmse() << "\n";
    std::cout << "\nSaved kernel_regression_iris_fit_small_bandwidth.svg and "
                 "kernel_regression_iris_fit_large_bandwidth.svg\n";

    return 0;
}
