/// @file graphical_lasso.cpp
/// @brief Sparse precision via exact component screening and ADMM.

#include <datamunge/stats/graphical_lasso.hpp>

#include "empirical_covariance.hpp"
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/eigen.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Standardized covariance and sparse log-determinant optimization.
namespace {

/// @brief Wide-precision matrix for standardized quantities.
using Matrix = linalg::DenseMatrix<long double>;
/// @brief Original column indices in an independent component.
using Indices = std::vector<std::size_t>;
/// @brief Triangular factor used to test positive definiteness and invert a precision.
using Cholesky = linalg::CholeskyDecomposition<long double>;

/// @brief Validates public solver controls.
void validate(double penalty, const GraphicalLassoOptions& options) {
    if (!(penalty >= 0 && std::isfinite(penalty)) || options.max_iterations == 0 ||
        !(options.tolerance > 0 && options.tolerance < 1))
        throw std::invalid_argument("graphical lasso: invalid penalty or solver controls");
}

/// @brief Converts a working entry to the reporting precision.
double output(long double value, bool require_nonzero = false) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("graphical lasso: fitted result exceeds the finite reporting range");
    const auto result = static_cast<double>(value);
    if (require_nonzero && value != 0 && result == 0)
        throw std::overflow_error("graphical lasso: a fitted entry underflows double precision");
    return result;
}

/// @brief Reads a finite symmetric covariance with positive diagonal.
Matrix read_covariance(const linalg::DenseMatrix<double>& covariance) {
    const auto p = covariance.rows();
    if (p == 0 || p != covariance.cols() ||
        !std::ranges::all_of(covariance.data(),[](double value) { return std::isfinite(value); }))
        throw std::invalid_argument("graphical lasso: expected a finite square covariance");
    if (p > std::vector<long double>{}.max_size()/p)
        throw std::length_error("graphical lasso: dense workspace exceeds storage capacity");
    Matrix result(p,p);
    for (std::size_t j = 0; j < p; ++j) {
        if (!(covariance(j,j) > 0))
            throw std::domain_error("graphical lasso: all variables must have positive variance");
        result(j,j) = covariance(j,j);
    }
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t k = 0; k < j; ++k) {
            const long double a = covariance(j,k), b = covariance(k,j);
            const auto scale = std::max({std::abs(a),std::abs(b),
                std::sqrt(result(j,j))*std::sqrt(result(k,k))});
            if (std::abs(a-b) > 64*std::numeric_limits<double>::epsilon()*scale)
                throw std::invalid_argument("graphical lasso: covariance must be symmetric");
            result(j,k) = result(k,j) = std::midpoint(a,b);
        }
    return result;
}

/// @brief Finds the graph components that must be independent in the solution.
std::vector<Indices> components(const Matrix& correlation, const Matrix& penalty) {
    const auto p = correlation.rows();
    std::vector<Indices> result;
    std::vector<bool> visited(p);
    for (std::size_t start = 0; start < p; ++start) {
        if (visited[start])
            continue;
        auto& block = result.emplace_back(1,start);
        visited[start] = true;
        for (std::size_t k = 0; k < block.size(); ++k) {
            const auto node = block[k];
            for (std::size_t j = 0; j < p; ++j)
                if (!visited[j] && std::abs(correlation(node,j)) > penalty(node,j)) {
                    block.push_back(j);
                    visited[j] = true;
                }
        }
        std::ranges::sort(block);
    }
    return result;
}

/// @brief State and primal diagnostics for a feasible solution of one block.
struct BlockFit {
    Matrix precision;         ///< Sparse standardized precision.
    Matrix covariance;        ///< Inverse standardized precision.
    long double kkt{};        ///< Maximum residual of the subgradient optimality conditions.
    long double objective{};  ///< Objective in standardized coordinates.
    std::size_t iterations{}; ///< Iterations performed.
};

/// @brief Evaluates a feasible precision and its KKT conditions.
BlockFit evaluate(Matrix precision, const Cholesky& factor, const Matrix& sample, const Matrix& penalty,
    std::size_t iterations) {
    const auto p = sample.rows();
    BlockFit result{std::move(precision),factor.solve(Matrix::identity(p)),0,0,iterations};
    for (std::size_t j = 0; j < p; ++j) {
        result.kkt = std::max(result.kkt,std::abs(result.covariance(j,j)-sample(j,j)));
        result.objective += sample(j,j)*result.precision(j,j)-2*std::log(factor.L(j,j));
        for (std::size_t k = 0; k < j; ++k) {
            const auto covariance = std::midpoint(result.covariance(j,k),result.covariance(k,j));
            result.covariance(j,k) = result.covariance(k,j) = covariance;
            const auto residual = covariance-sample(j,k);
            const auto entry = result.precision(j,k);
            const auto violation = entry == 0 ? std::max(0.0L,std::abs(residual)-penalty(j,k)) :
                std::abs(residual-std::copysign(penalty(j,k),entry));
            result.kkt = std::max(result.kkt,violation);
            result.objective += 2*(sample(j,k)*entry+penalty(j,k)*std::abs(entry));
        }
    }
    return result;
}

/// @brief Applies the positive-definite log-determinant proximal step by symmetric eigendecomposition.
Matrix update_precision(const Matrix& z, const Matrix& dual, const Matrix& sample, long double rho) {
    const auto p = z.rows();
    Matrix shifted(p,p);
    for (std::size_t i = 0; i < shifted.data().size(); ++i)
        shifted.data()[i] = rho*(z.data()[i]-dual.data()[i])-sample.data()[i];
    const auto decomposition = linalg::symmetric_eigen(shifted);
    Matrix result(p,p);
    for (std::size_t r = 0; r < p; ++r) {
        const auto value = decomposition.eigenvalues[r];
        const auto root = std::hypot(value,2*std::sqrt(rho));
        const auto eigenvalue = value >= 0 ? (root+value)/(2*rho) : 2/(root-value);
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t k = 0; k <= j; ++k)
                result(j,k) += eigenvalue*decomposition.eigenvectors(j,r)*decomposition.eigenvectors(k,r);
    }
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t k = 0; k < j; ++k)
            result(k,j) = result(j,k);
    return result;
}

/// @brief Solves one component and returns a positive-definite sparse iterate on exhaustion.
BlockFit solve_block(const Matrix& sample, const Matrix& penalty, bool unpenalized, const GraphicalLassoOptions& options) {
    const auto p = sample.rows();
    if (p == 1)
        return {Matrix(1,1,1),Matrix(1,1,1),0,1,0};
    if (unpenalized) {
        const auto covariance_factor = linalg::cholesky(sample);
        if (!covariance_factor.ok)
            throw std::domain_error("graphical lasso: zero penalty requires a positive-definite covariance");
        auto precision = covariance_factor.solve(Matrix::identity(p));
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t k = 0; k < j; ++k)
                precision(j,k) = precision(k,j) = std::midpoint(precision(j,k),precision(k,j));
        const auto factor = linalg::cholesky(precision);
        if (!factor.ok)
            throw std::domain_error("graphical lasso: covariance inverse is numerically singular");
        return evaluate(std::move(precision),factor,sample,penalty,0);
    }

    auto z = Matrix::identity(p);
    Matrix dual(p,p);
    long double rho = 1;
    for (std::size_t iteration = 0; iteration < options.max_iterations; ++iteration) {
        const auto count = iteration+1;
        const auto x = update_precision(z,dual,sample,rho);
        long double residual = 0, dual_residual = 0, norm = 0;
        for (std::size_t j = 0; j < p; ++j) {
            for (std::size_t k = 0; k <= j; ++k) {
                const auto previous = z(j,k), input = x(j,k)+dual(j,k);
                const auto threshold = penalty(j,k)/rho;
                const auto value = std::copysign(std::max(0.0L,std::abs(input)-threshold),input);
                const auto difference = x(j,k)-value;
                z(j,k) = z(k,j) = value;
                dual(j,k) = dual(k,j) = dual(j,k)+difference;
                residual = std::max(residual,std::abs(difference));
                dual_residual = std::max(dual_residual,rho*std::abs(value-previous));
                norm = std::max(norm,std::abs(value));
            }
        }
        if (count == options.max_iterations ||
            std::max(residual,dual_residual) <= options.tolerance*std::max(1.0L,norm)) {
            auto factor = linalg::cholesky(z);
            if (count == options.max_iterations) {
                // Zero off-diagonals stay zero while an indefinite last iterate moves into the cone.
                for (int attempt = 0; !factor.ok && attempt < 64; ++attempt) {
                    for (std::size_t j = 0; j < p; ++j)
                        for (std::size_t k = 0; k < j; ++k) {
                            z(j,k) *= .5L;
                            z(k,j) = z(j,k);
                        }
                    factor = linalg::cholesky(z);
                }
                if (!factor.ok) {
                    z = Matrix::identity(p);
                    factor = {z,true};
                }
                return evaluate(std::move(z),factor,sample,penalty,count);
            }
            if (factor.ok) {
                auto trial = evaluate(z,factor,sample,penalty,count);
                if (trial.kkt <= options.tolerance)
                    return trial;
            }
        }
        if (count % 20 == 0) {
            const auto previous_rho = rho;
            if (residual > 10*dual_residual && rho < 1e6L)
                rho *= 2;
            else if (dual_residual > 10*residual && rho > 1e-6L)
                rho *= .5L;
            if (rho != previous_rho)
                for (auto& value : dual.data())
                    value *= previous_rho/rho;
        }
    }
    throw std::runtime_error("graphical lasso: optimizer exited without an iterate");
}

/// @brief Standardizes features, solves independent blocks, and restores original units.
GraphicalLassoResult fit_covariance(Matrix sample, long double penalty, long double scale,
    bool validate_psd, const GraphicalLassoOptions& options) {
    const auto p = sample.rows();
    Matrix weights(p,p);
    std::vector<long double> deviations(p);
    for (std::size_t j = 0; j < p; ++j) {
        if (!(sample(j,j) > 0))
            throw std::domain_error("graphical lasso: all variables must have positive variance");
        deviations[j] = std::sqrt(sample(j,j));
    }
    for (std::size_t j = 0; j < p; ++j) {
        sample(j,j) = 1;
        for (std::size_t k = 0; k < j; ++k) {
            const auto product = deviations[j]*deviations[k];
            sample(j,k) = sample(k,j) = sample(j,k)/product;
            weights(j,k) = weights(k,j) = penalty/product;
        }
    }
    if (validate_psd && penalty > 0) {
        const auto eigen = linalg::symmetric_eigen(sample);
        const auto cutoff = 64*p*std::numeric_limits<double>::epsilon()*std::max(1.0L,eigen.eigenvalues.front());
        if (eigen.eigenvalues.back() < -cutoff)
            throw std::domain_error("graphical lasso: empirical covariance is indefinite");
    }
    GraphicalLassoResult result{linalg::DenseMatrix<double>(p,p),linalg::DenseMatrix<double>(p,p),{},0,0,0,false};
    long double objective = 0;
    for (auto& value : deviations) {
        value *= scale;
        objective += 2*std::log(value);
    }
    for (const auto& indices : components(sample,weights)) {
        Matrix block(indices.size(),indices.size()), block_weights(indices.size(),indices.size());
        for (std::size_t j = 0; j < indices.size(); ++j)
            for (std::size_t k = 0; k < indices.size(); ++k) {
                block(j,k) = sample(indices[j],indices[k]);
                block_weights(j,k) = weights(indices[j],indices[k]);
            }
        const auto fit = solve_block(block,block_weights,penalty == 0,options);
        objective += fit.objective;
        result.iterations = std::max(result.iterations,fit.iterations);
        result.max_kkt_violation = std::max(result.max_kkt_violation,output(fit.kkt));
        for (std::size_t j = 0; j < indices.size(); ++j)
            for (std::size_t k = 0; k <= j; ++k) {
                const auto a = indices[j], b = indices[k];
                const auto covariance = (fit.covariance(j,k)*deviations[a])*deviations[b];
                const auto precision = (fit.precision(j,k)/deviations[a])/deviations[b];
                result.precision(a,b) = result.precision(b,a) = output(precision,true);
                result.covariance(a,b) = result.covariance(b,a) = output(covariance,a == b);
                if (k != j && precision != 0) {
                    const auto partial = (-fit.precision(j,k)/std::sqrt(fit.precision(j,j)))/
                        std::sqrt(fit.precision(k,k));
                    result.edges.push_back({b,a,output(partial)});
                }
            }
    }
    std::ranges::sort(result.edges,{},[](const auto& edge) { return std::pair{edge.first,edge.second}; });
    result.objective = output(objective);
    result.converged = result.max_kkt_violation <= options.tolerance;
    return result;
}

} // namespace

GraphicalLassoResult graphical_lasso(const linalg::DenseMatrix<double>& empirical_covariance, double penalty,
    const GraphicalLassoOptions& options) {
    validate(penalty,options);
    return fit_covariance(read_covariance(empirical_covariance),penalty,1,true,options);
}

GraphicalLassoResult fit_graphical_lasso(const linalg::DenseMatrix<double>& observations, double penalty,
    const GraphicalLassoOptions& options) {
    validate(penalty,options);
    auto sample = detail::empirical_covariance(observations,false);
    return fit_covariance(std::move(sample.covariance),(penalty/sample.scale)/sample.scale,sample.scale,false,options);
}

} // namespace datamunge::stats
