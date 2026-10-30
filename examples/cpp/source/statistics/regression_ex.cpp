#include <datamunge/linalg/linalg.hpp>
#include <datamunge/random/random.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace datamunge::linalg;

namespace {

void print_vec(const char* label, const std::vector<double>& v) {
    std::cout << label << " [";
    for (std::size_t i = 0; i < v.size(); ++i)
        std::cout << (i ? ", " : "") << std::fixed << std::setprecision(5) << v[i];
    std::cout << "]\n";
}

void print_summary(const char* title,
                   const LinearRegressionResult<double>& fit,
                   const std::vector<double>&            truth) {
    std::cout << "\n---- " << title << " ----\n";
    print_vec("true beta      =", truth);
    print_vec("estimated beta =", fit.coefficients);
    print_vec("std. errors    =", fit.standard_errors);
    print_vec("var(beta) diag =", fit.parameter_variances);
    std::cout << "sigma^2 = " << fit.sigma2 << "\n";
    std::cout << "sigma   = " << fit.sigma << "\n";
    std::cout << "R^2     = " << fit.r_squared << "\n";
    std::cout << "adj R^2 = " << fit.adjusted_r_squared << "\n";
    std::cout << "RSS     = " << fit.residual_sum_of_squares << "\n";
}

} // namespace

int main() {
    constexpr std::size_t n = 200;
    constexpr std::size_t p = 5;
    const std::vector<double> beta_true = {1.25, -0.75, 2.00, 0.50, -1.50};
    const double noise_sigma = 0.25;

    datamunge::random::SplitMix64 rng(0xDADA5EEDULL);

    DenseMatrix<double> X_dense(n, p, 0.0);
    SparseCOO<double>   X_sparse(n, p, n * 3);
    std::vector<double> y(n, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        const double x1 = rng.normal();
        const double x2 = (i % 4 == 0) ? 0.0 : rng.uniform01() - 0.5;
        const double x3 = (i % 5 == 0) ? 0.0 : rng.normal();
        const double x4 = (i % 3 == 0) ? 0.0 : (rng.uniform01() < 0.5 ? -1.0 : 1.0);

        const double row[p] = {1.0, x1, x2, x3, x4};
        for (std::size_t j = 0; j < p; ++j) {
            X_dense(i, j) = row[j];
            if (row[j] != 0.0) X_sparse.set(i, j, row[j]);
        }

        y[i] = beta_true[0]
             + beta_true[1] * x1
             + beta_true[2] * x2
             + beta_true[3] * x3
             + beta_true[4] * x4
             + noise_sigma * rng.normal();
    }

    X_sparse.compress();

    auto dense_fit = linear_regression(X_dense, y);

    SparseLinearRegressionOptions sparse_opts;
    sparse_opts.solver.tol      = 1e-10;
    sparse_opts.solver.max_iter = 4 * p;
    sparse_opts.use_jacobi      = true;
    auto sparse_fit = sparse_linear_regression(X_sparse, y, sparse_opts);

    print_summary("Dense QR linear regression", dense_fit, beta_true);
    print_summary("Sparse normal-equation regression", sparse_fit, beta_true);

    print_vec("\ndense residual head =", std::vector<double>(
        dense_fit.residuals.begin(), dense_fit.residuals.begin() + 5));
    print_vec("sparse residual head =", std::vector<double>(
        sparse_fit.residuals.begin(), sparse_fit.residuals.begin() + 5));

    return 0;
}
