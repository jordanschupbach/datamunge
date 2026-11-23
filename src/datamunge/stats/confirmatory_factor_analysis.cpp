/// @file confirmatory_factor_analysis.cpp
/// @brief Gaussian factor covariance discrepancy and analytic constrained scores.

#include <datamunge/stats/confirmatory_factor_analysis.hpp>

#include "likelihood_bfgs.hpp"
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/svd.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::stats {
/// @brief Scaled factor model matrices and optimizer workspaces.
namespace {

/// @brief Working matrix for likelihood derivatives.
using Matrix = linalg::DenseMatrix<long double>;
/// @brief Working coordinates and scales.
using Vector = std::vector<long double>;
/// @brief Named matrices in the order used for parameterization.
enum Block {
    Loadings, ///< Indicator loadings.
    Latent,   ///< Latent factor covariance.
    Residual  ///< Indicator residual covariance.
};
/// @brief Loading, factor and residual matrices.
using Blocks = std::array<Matrix,3>;
/// @brief Positive-definite matrix factor.
using Factor = linalg::CholeskyDecomposition<long double>;

/// @brief Address of one freely estimated entry.
struct Parameter {
    Block block;     ///< Model matrix.
    std::size_t row; ///< Row of the free entry.
    std::size_t col; ///< Column, at or below the diagonal for covariances.
    bool log_scale;  ///< Diagonal variance uses a log coordinate.
};

/// @brief Objective and analytic score at a feasible model.
struct Evaluation {
    long double value{-std::numeric_limits<long double>::infinity()}; ///< Negative half discrepancy.
    Vector score; ///< Gradient in working coordinates.
    Blocks model; ///< Candidate model matrices.
    Matrix implied; ///< Scaled model covariance.
};

/// @brief Checks that a dense working matrix fits its storage.
void check_storage(std::size_t rows, std::size_t cols) {
    if (cols && rows > Vector{}.max_size()/cols)
        throw std::length_error("confirmatory factor analysis: dense workspace exceeds storage capacity");
}

/// @brief Reads a finite symmetric covariance and mirrors its averaged entries.
Matrix covariance(const linalg::DenseMatrix<double>& values, std::size_t size) {
    if (values.rows() != size || values.cols() != size ||
        !std::ranges::all_of(values.data(),[](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("confirmatory factor analysis: invalid covariance shape or values");
    Matrix result(size,size);
    for (std::size_t i = 0; i < size; ++i) {
        result(i,i) = values(i,i);
        for (std::size_t j = 0; j < i; ++j) {
            const long double a = values(i,j), b = values(j,i);
            if (std::abs(a-b) > 64*std::numeric_limits<double>::epsilon()*std::max(std::abs(a),std::abs(b)))
                throw std::invalid_argument("confirmatory factor analysis: covariance must be symmetric");
            result(i,j) = result(j,i) = std::midpoint(a,b);
        }
    }
    return result;
}

/// @brief Computes log determinant or minus infinity for an unusable factor.
long double log_determinant(const Factor& factor) {
    if (!factor.ok)
        return -std::numeric_limits<long double>::infinity();
    long double value = 0;
    for (std::size_t i = 0; i < factor.L.rows(); ++i) {
        if (!(factor.L(i,i)>0) || !std::isfinite(factor.L(i,i)))
            return -std::numeric_limits<long double>::infinity();
        value += 2*std::log(factor.L(i,i));
    }
    return value;
}

/// @brief Averages opposite off-diagonal entries to remove arithmetic asymmetry.
void symmetrize(Matrix& matrix) {
    for (std::size_t i = 0; i < matrix.rows(); ++i)
        for (std::size_t j = 0; j < i; ++j)
            matrix(i,j) = matrix(j,i) = std::midpoint(matrix(i,j),matrix(j,i));
}

/// @brief Records free coordinates and checks the dimensions and symmetry of their mask.
void append_parameters(std::vector<Parameter>& parameters, const CFAParameters& input, Block block) {
    const auto& mask = input.free_entries;
    if (mask.rows() == 0 && mask.cols() == 0)
        return;
    if (mask.rows() != input.values.rows() || mask.cols() != input.values.cols() ||
        !std::ranges::all_of(mask.data(),[](auto x) { return x <= 1; }))
        throw std::invalid_argument("confirmatory factor analysis: invalid free-entry mask");
    for (std::size_t i = 0; i < mask.rows(); ++i)
        for (std::size_t j = 0; j < (block == Loadings ? mask.cols() : i+1); ++j) {
            if (block != Loadings && mask(i,j) != mask(j,i))
                throw std::invalid_argument("confirmatory factor analysis: covariance mask must be symmetric");
            if (mask(i,j))
                parameters.push_back({block,i,j,block != Loadings && i == j});
        }
}

/// @brief Builds covariance components from the current free parameters.
Blocks unpack(Blocks model, const std::vector<Parameter>& free, const Vector& coordinates) {
    for (std::size_t k = 0; k < free.size(); ++k) {
        const auto& [block,i,j,log_scale] = free[k];
        auto& matrix = model[block];
        matrix(i,j) = log_scale ? std::exp(coordinates[k]) : coordinates[k];
        if (block != Loadings)
            matrix(j,i) = matrix(i,j);
    }
    return model;
}

/// @brief Tests local identification using derivatives of the distinct implied covariance entries.
bool identified(const Blocks& model, const std::vector<Parameter>& free) {
    if (free.empty())
        return true;
    const auto& loading = model[Loadings];
    const auto p = loading.rows(), entries = p*(p+1)/2;
    check_storage(entries,free.size());
    Matrix jacobian(entries,free.size());
    const auto weighted = loading*model[Latent];
    for (std::size_t k = 0; k < free.size(); ++k) {
        const auto& [block,i,j,log_scale] = free[k];
        // Log coordinates differ by a positive scalar and have the same rank.
        long double scale = 0;
        std::size_t row = 0;
        for (std::size_t a = 0; a < p; ++a)
            for (std::size_t b = 0; b <= a; ++b) {
                long double value = 0;
                if (block == Loadings)
                    value = (a == i ? weighted(b,j) : 0)+(b == i ? weighted(a,j) : 0);
                else if (block == Latent)
                    value = loading(a,i)*loading(b,j)+(i == j ? 0 : loading(a,j)*loading(b,i));
                else
                    value = (a == i && b == j);
                jacobian(row++,k) = value;
                scale = std::max(scale,std::abs(value));
            }
        if (!(scale > 0))
            return false;
        for (std::size_t i = 0; i < entries; ++i)
            jacobian(i,k) /= scale;
    }
    const auto factor = linalg::svd(jacobian);
    if (!factor.converged)
        throw std::runtime_error("confirmatory factor analysis: identification factorization failed");
    return factor.rank(entries*std::numeric_limits<double>::epsilon()) == free.size();
}

/// @brief Evaluates Gaussian covariance discrepancy and gradients at one parameter vector.
Evaluation evaluate(Blocks model, const Matrix& sample, long double sample_log_det,
    const std::vector<Parameter>& free) {
    if (!std::isfinite(log_determinant(linalg::cholesky(model[Latent]))) ||
        !std::isfinite(log_determinant(linalg::cholesky(model[Residual]))))
        return {};
    const auto& loading = model[Loadings];
    const auto transpose = loading.transpose();
    const auto weighted = loading*model[Latent];
    auto implied = weighted*transpose + model[Residual];
    symmetrize(implied);
    const auto factor = linalg::cholesky(implied);
    const auto determinant = log_determinant(factor);
    if (!std::isfinite(determinant))
        return {};
    const auto p = loading.rows();
    const auto inverse = factor.solve(Matrix::identity(p));
    const auto inverse_sample = inverse*sample;
    long double trace = 0;
    for (std::size_t i = 0; i < p; ++i)
        trace += inverse_sample(i,i);
    const auto value = -.5L*(determinant+trace-sample_log_det-p);
    auto derivative = (inverse_sample*inverse-inverse)*.5L;
    symmetrize(derivative);
    const auto loading_score = (derivative*weighted)*2.0L;
    const auto latent_score = (transpose*derivative)*loading;
    Vector score;
    score.reserve(free.size());
    for (const auto& [block,i,j,log_scale] : free) {
        if (block == Loadings)
            score.push_back(loading_score(i,j));
        else {
            const auto& gradient = block == Latent ? latent_score : derivative;
            score.push_back(log_scale ? gradient(i,i)*model[block](i,i) : gradient(i,j)+gradient(j,i));
        }
    }
    if (!std::isfinite(value) || !std::ranges::all_of(score,[](auto x) { return std::isfinite(x); }))
        return {};
    return {value,std::move(score),std::move(model),std::move(implied)};
}

/// @brief Checks conversion of a fitted parameter in original units.
double output(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("confirmatory factor analysis: result exceeds the finite reporting range");
    return static_cast<double>(value);
}

/// @brief Converts a matrix from working to original covariance coordinates.
linalg::DenseMatrix<double> restore(const Matrix& values, const Vector& rows, const Vector& cols) {
    linalg::DenseMatrix<double> result(values.rows(),values.cols());
    for (std::size_t i = 0; i < values.rows(); ++i)
        for (std::size_t j = 0; j < values.cols(); ++j)
            result(i,j) = output((values(i,j)*rows[i])*cols[j]);
    return result;
}

} // namespace

CFAFit fit_confirmatory_factor_analysis(const linalg::DenseMatrix<double>& empirical_covariance,
    const CFASpecification& specification, const CFAOptions& options) {
    const auto p = empirical_covariance.rows(), k = specification.loadings.values.cols();
    if (p == 0 || k == 0 || k > p || options.max_iterations == 0 || !(options.tolerance > 0 && options.tolerance < 1) ||
        specification.loadings.values.rows() != p ||
        !std::ranges::all_of(specification.loadings.values.data(),[](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("confirmatory factor analysis: invalid model dimensions, loadings or controls");
    check_storage(p,p);
    auto sample = covariance(empirical_covariance,p);
    Vector scales(p);
    for (std::size_t i = 0; i < p; ++i) {
        if (!(sample(i,i)>0))
            throw std::domain_error("confirmatory factor analysis: nonpositive empirical variance");
        scales[i] = std::sqrt(sample(i,i));
    }

    Blocks initial{Matrix(p,k),covariance(specification.factor_covariance.values,k),
        covariance(specification.residual_covariance.values,p)};
    for (std::size_t i = 0; i < p; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            const auto scale = scales[i]*scales[j];
            sample(i,j) /= scale;
            initial[Residual](i,j) /= scale;
        }
        for (std::size_t j = 0; j < k; ++j)
            initial[Loadings](i,j) = specification.loadings.values(i,j)/scales[i];
    }
    const auto sample_log_det = log_determinant(linalg::cholesky(sample));
    if (!std::isfinite(sample_log_det))
        throw std::domain_error("confirmatory factor analysis: empirical covariance is singular");
    if (!std::isfinite(log_determinant(linalg::cholesky(initial[Latent]))) ||
        !std::isfinite(log_determinant(linalg::cholesky(initial[Residual]))))
        throw std::invalid_argument("confirmatory factor analysis: starting covariances must be positive definite");

    std::vector<Parameter> free;
    append_parameters(free,specification.loadings,Loadings);
    append_parameters(free,specification.factor_covariance,Latent);
    append_parameters(free,specification.residual_covariance,Residual);
    const auto entries = p*(p+1)/2;
    if (free.size() > entries)
        throw std::domain_error("confirmatory factor analysis: more parameters than distinct covariance entries");
    check_storage(free.size(),free.size());
    Vector parameters;
    for (const auto& [block,i,j,log_scale] : free)
        parameters.push_back(log_scale ? std::log(initial[block](i,j)) : initial[block](i,j));
    const auto objective = [&](const Vector& coordinates) {
        return evaluate(unpack(initial,free,coordinates),sample,sample_log_det,free);
    };
    Evaluation best;
    auto status = CFAStatus::NumericalFailure;
    if (free.empty()) {
        best = objective(parameters);
        status = CFAStatus::Converged;
    } else {
        bool has_identified_start = false;
        for (int start = 0; start < 3; ++start) {
            auto coordinates = parameters;
            if (start)
                for (std::size_t a = 0; a < free.size(); ++a)
                    if (free[a].block == Loadings)
                        coordinates[a] += .35L*std::sin((a+1.0L)*(start+1));
            if (!identified(unpack(initial,free,coordinates),free))
                continue;
            has_identified_start = true;
            auto fit = detail::maximize_likelihood<CFAStatus>(std::move(coordinates),options.max_iterations,
                options.tolerance,objective);
            const auto rounding = 64*std::numeric_limits<long double>::epsilon();
            if (fit.state.value > best.value+rounding ||
                (fit.status == CFAStatus::Converged && status != CFAStatus::Converged &&
                 fit.state.value >= best.value-rounding)) {
                best = std::move(fit.state);
                status = fit.status;
            }
        }
        if (!has_identified_start)
            throw std::domain_error("confirmatory factor analysis: model covariance Jacobian is rank deficient");
    }
    if (!std::isfinite(best.value))
        throw std::runtime_error("confirmatory factor analysis: no feasible model covariance found");
    if (!identified(best.model,free))
        throw std::domain_error("confirmatory factor analysis: fitted model is locally unidentified");
    const Vector ones(k,1);
    return {restore(best.model[Loadings],scales,ones),restore(best.model[Latent],ones,ones),
        restore(best.model[Residual],scales,scales),restore(best.implied,scales,scales),
        restore(sample-best.implied,scales,scales),output(std::max(0.0L,-2*best.value)),entries-free.size(),status};
}

} // namespace datamunge::stats
