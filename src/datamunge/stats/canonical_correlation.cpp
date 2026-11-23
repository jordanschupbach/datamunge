/// @file canonical_correlation.cpp
/// @brief Orthogonal-score canonical correlation and sequential inference.

#include <datamunge/stats/canonical_correlation.hpp>

#include <datamunge/linalg/svd.hpp>
#include <datamunge/stats/detail/test_common.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Stable within-view whitening and canonical score coordinates.
namespace {

/// @brief Working matrix for standardized data and projections.
using Matrix = linalg::DenseMatrix<long double>;
/// @brief Working vector for singular values or data scales.
using Vector = std::vector<long double>;
/// @brief Public input or result matrix.
using PublicMatrix = linalg::DenseMatrix<double>;

/// @brief Thin SVD of one centered and standardized view, reduced to estimable directions.
struct View {
    Matrix observations;        ///< Orthonormal observation-space basis.
    Matrix variables;           ///< Orthonormal variable-space basis.
    Vector singular_values;     ///< Nonzero singular values in the selected subspace.
    Vector scales;              ///< Norm of each centered original feature.
    std::vector<double> means;  ///< Original feature means.
};

/// @brief Rotated scores, coefficients, and loadings for one view.
struct CanonicalView {
    PublicMatrix weights;  ///< Weights for centered variables in original units.
    PublicMatrix scores;   ///< Unit sample variance scores.
    PublicMatrix loadings; ///< Correlations of variables with the scores.
};

/// @brief Checks that a dense matrix fits its working storage type.
void check_storage(std::size_t rows, std::size_t cols) {
    if (cols && rows > Vector{}.max_size() / cols)
        throw std::length_error("canonical correlation: dense workspace exceeds storage capacity");
}

/// @brief Checks finite output on the original feature scale.
double output(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("canonical correlation: result exceeds the finite reporting range");
    return static_cast<double>(value);
}

/// @brief Centers each feature stably, normalizes its norm, and retains the estimable subspace.
View decompose(const PublicMatrix& data, long double tolerance) {
    const auto n = data.rows(), p = data.cols();
    check_storage(n,p);
    Matrix centered(n,p);
    View result{{},{},{},Vector(p),std::vector<double>(p)};
    for (std::size_t j = 0; j < p; ++j) {
        const auto origin = static_cast<long double>(data(0,j));
        long double offset = 0;
        for (std::size_t i = 0; i < n; ++i) {
            centered(i,j) = data(i,j)-origin;
            offset += centered(i,j);
        }
        offset /= n;
        result.means[j] = output(origin+offset);
        for (std::size_t i = 0; i < n; ++i) {
            centered(i,j) -= offset;
            result.scales[j] = std::hypot(result.scales[j],centered(i,j));
        }
        if (result.scales[j] > 0)
            for (std::size_t i = 0; i < n; ++i)
                centered(i,j) /= result.scales[j];
    }
    const auto factor = linalg::svd(centered);
    if (!factor.converged)
        throw std::runtime_error("canonical correlation: within-view factorization failed");
    const auto rank = std::min(n-1,factor.rank(tolerance));
    if (rank == 0)
        throw std::domain_error("canonical correlation: a view has no centered variation");
    result.observations = Matrix(n,rank);
    result.variables = Matrix(p,rank);
    result.singular_values.assign(factor.s.begin(),factor.s.begin()+rank);
    for (std::size_t k = 0; k < rank; ++k) {
        for (std::size_t i = 0; i < n; ++i)
            result.observations(i,k) = factor.U(i,k);
        for (std::size_t j = 0; j < p; ++j)
            result.variables(j,k) = factor.V(j,k);
    }
    return result;
}

/// @brief Completes the paired canonical coordinates when some correlations are zero.
Matrix complete_rotation(const Matrix& selected, std::size_t active) {
    Matrix basis(selected.rows(),selected.cols());
    for (std::size_t i = 0; i < basis.rows(); ++i)
        for (std::size_t k = 0; k < active; ++k)
            basis(i,k) = selected(i,k);
    for (std::size_t k = active; k < basis.cols(); ++k) {
        std::size_t best = 0;
        long double best_remainder = -1;
        for (std::size_t i = 0; i < basis.rows(); ++i) {
            long double remainder = 1;
            for (std::size_t j = 0; j < k; ++j)
                remainder -= basis(i,j)*basis(i,j);
            if (remainder > best_remainder) {
                best = i;
                best_remainder = remainder;
            }
        }
        basis(best,k) = 1;
        // Reorthogonalize to protect the unit-variance scores at a zero root.
        for (int pass = 0; pass < 2; ++pass) {
            for (std::size_t j = 0; j < k; ++j) {
                long double projection = 0;
                for (std::size_t i = 0; i < basis.rows(); ++i)
                    projection += basis(i,j)*basis(i,k);
                for (std::size_t i = 0; i < basis.rows(); ++i)
                    basis(i,k) -= projection*basis(i,j);
            }
        }
        long double norm = 0;
        for (std::size_t i = 0; i < basis.rows(); ++i)
            norm = std::hypot(norm,basis(i,k));
        if (!(norm > 0))
            throw std::runtime_error("canonical correlation: failed to complete zero-correlation directions");
        for (std::size_t i = 0; i < basis.rows(); ++i)
            basis(i,k) /= norm;
    }
    return basis;
}

/// @brief Builds first- or second-view coefficients, scores, and structure correlations.
CanonicalView rotate(const View& view, const Matrix& rotation) {
    const auto n = view.observations.rows(), p = view.variables.rows(), pairs = rotation.cols();
    CanonicalView result{PublicMatrix(p,pairs),PublicMatrix(n,pairs),PublicMatrix(p,pairs)};
    const auto variance = std::sqrt(static_cast<long double>(n-1));
    const auto scores = view.observations * rotation;
    for (std::size_t k = 0; k < pairs; ++k) {
        for (std::size_t i = 0; i < n; ++i)
            result.scores(i,k) = output(variance*scores(i,k));
        for (std::size_t j = 0; j < p; ++j) {
            if (view.scales[j] == 0)
                continue;
            long double weight = 0, loading = 0;
            for (std::size_t r = 0; r < rotation.rows(); ++r) {
                const auto coefficient = view.variables(j,r)*rotation(r,k);
                weight += coefficient/view.singular_values[r];
                loading += coefficient*view.singular_values[r];
            }
            result.weights(j,k) = output(variance*weight/view.scales[j]);
            result.loadings(j,k) = static_cast<double>(std::clamp(loading,-1.0L,1.0L));
        }
    }
    return result;
}

/// @brief Applies the same canonical sign change to every output in one view.
void flip(CanonicalView& view, std::size_t column) {
    for (auto* matrix : {&view.weights,&view.scores,&view.loadings})
        for (std::size_t i = 0; i < matrix->rows(); ++i)
            (*matrix)(i,column) = -(*matrix)(i,column);
}

/// @brief Computes sequential Wilks tests under the full-rank Gaussian reference model.
std::vector<CanonicalCorrelationTest> tests(std::size_t n, std::size_t p, std::size_t q, const Vector& correlations) {
    std::vector<CanonicalCorrelationTest> result(correlations.size());
    const auto multiplier = n-1 - (static_cast<long double>(p)+q+1)/2;
    long double log_lambda = 0;
    for (std::size_t k = correlations.size(); k-- > 0;) {
        log_lambda += std::log1p(-correlations[k]*correlations[k]);
        const auto df = (p-k)*(q-k);
        const auto statistic = static_cast<double>(-multiplier*log_lambda);
        result[k] = {k+1,static_cast<double>(std::exp(log_lambda)),statistic,df,
            detail::chi_squared_test_survival(statistic,df)};
    }
    return result;
}

} // namespace

CanonicalCorrelationFit canonical_correlation(const linalg::DenseMatrix<double>& x,
    const linalg::DenseMatrix<double>& y, double relative_tolerance) {
    const auto n = x.rows(), p = x.cols(), q = y.cols();
    const auto finite = [](double value) { return std::isfinite(value); };
    if (n < 2 || n != y.rows() || p == 0 || q == 0 ||
        !(relative_tolerance >= 0 && relative_tolerance < 1) ||
        !std::ranges::all_of(x.data(),finite) || !std::ranges::all_of(y.data(),finite))
        throw std::invalid_argument("canonical correlation: incompatible views, tolerance or nonfinite observations");
    const auto floor = std::max({n,p,q})*static_cast<long double>(std::numeric_limits<double>::epsilon());
    const auto tolerance = std::max(floor,static_cast<long double>(relative_tolerance));
    auto first = decompose(x,tolerance);
    auto second = decompose(y,tolerance);
    const auto rx = first.observations.cols(), ry = second.observations.cols();
    check_storage(rx,ry);
    const auto cross = linalg::svd(first.observations.transpose()*second.observations);
    if (!cross.converged)
        throw std::runtime_error("canonical correlation: cross-view factorization failed");
    const auto active = static_cast<std::size_t>(std::ranges::count_if(cross.s,[&](auto value) { return value > floor; }));
    Vector correlations(cross.s.size());
    for (std::size_t k = 0; k < active; ++k)
        correlations[k] = std::min(1.0L,cross.s[k]);
    auto first_rotated = rotate(first,complete_rotation(cross.U,active));
    auto second_rotated = rotate(second,complete_rotation(cross.V,active));
    for (std::size_t k = 0; k < correlations.size(); ++k) {
        std::size_t index = 0;
        for (std::size_t j = 1; j < p; ++j)
            if (std::abs(first_rotated.loadings(j,k)) > std::abs(first_rotated.loadings(index,k)))
                index = j;
        if (first_rotated.loadings(index,k) < 0) {
            flip(first_rotated,k);
            flip(second_rotated,k);
        }
    }
    CanonicalCorrelationFit result{};
    result.correlations.assign(correlations.begin(),correlations.end());
    result.x_means = std::move(first.means);
    result.y_means = std::move(second.means);
    result.x_rank = rx;
    result.y_rank = ry;
    result.x_weights = std::move(first_rotated.weights);
    result.y_weights = std::move(second_rotated.weights);
    result.x_scores = std::move(first_rotated.scores);
    result.y_scores = std::move(second_rotated.scores);
    result.x_loadings = std::move(first_rotated.loadings);
    result.y_loadings = std::move(second_rotated.loadings);
    const auto perfect = 1 - 32*std::numeric_limits<double>::epsilon();
    if (rx == p && ry == q && q <= n-1-p && correlations.front() < perfect)
        result.sequential_tests = tests(n,p,q,correlations);
    return result;
}

} // namespace datamunge::stats
