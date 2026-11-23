/// @file simpls.cpp
/// @brief SIMPLS covariance deflation and prefix validation.

#include <datamunge/stats/simpls.hpp>

#include <datamunge/linalg/svd.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Workspaces, preprocessing and component-path helpers.
namespace {

/// @brief Extended-precision matrix for the iterative fit.
using Matrix = linalg::DenseMatrix<long double>;
/// @brief Extended-precision scores, loadings and prediction accumulators.
using Vector = std::vector<long double>;
/// @brief Public input or result matrix.
using PublicMatrix = linalg::DenseMatrix<double>;
/// @brief Training or held-out row indices.
using Rows = std::vector<std::size_t>;

/// @brief Centered training data and parameters for restoring original units.
struct Centered {
    Matrix values;              ///< Training rows in centered and scaled coordinates.
    Vector scales;              ///< Scaling divisors in column order.
    std::vector<double> means;  ///< Means in original data units.
};

/// @brief Checks finite public data.
bool finite(const std::vector<double>& values) {
    return std::ranges::all_of(values, [](double value) { return std::isfinite(value); });
}

/// @brief Checks capacity before allocating a working or result matrix.
void check_storage(std::size_t rows, std::size_t cols) {
    if (cols && rows > Vector{}.max_size() / cols)
        throw std::length_error("SIMPLS: dense workspace exceeds storage capacity");
}

/// @brief Converts a value to its public precision with a finite-range check.
double output(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("SIMPLS: result exceeds the finite reporting range");
    return static_cast<double>(value);
}

/// @brief Validates training responses, predictors, and numerical controls.
void validate(const PublicMatrix& predictors, const PublicMatrix& response, const SIMPLSOptions& options) {
    if (predictors.rows() < 2 || predictors.rows() != response.rows() ||
        predictors.cols() == 0 || response.cols() == 0 || !finite(predictors.data()) || !finite(response.data()) ||
        !(options.tolerance > 0 && options.tolerance < 1))
        throw std::invalid_argument("SIMPLS: incompatible data or numerical controls");
}

/// @brief Computes the Euclidean norm in a scaled working space.
long double norm(const Vector& values) {
    return std::sqrt(std::inner_product(values.begin(),values.end(),values.begin(),0.0L));
}

/// @brief Centers and scales using a fold's training rows only.
Centered center(const PublicMatrix& data, const Rows& rows, bool standardize) {
    const auto n = rows.size(), p = data.cols();
    check_storage(n,p);
    Centered result{Matrix(n,p),Vector(p),std::vector<double>(p)};
    long double common = 0;
    for (std::size_t j = 0; j < p; ++j) {
        const auto reference = static_cast<long double>(data(rows[0],j));
        long double offset = 0;
        for (std::size_t i = 0; i < n; ++i) {
            result.values(i,j) = data(rows[i],j)-reference;
            offset += result.values(i,j);
        }
        offset /= n;
        result.means[j] = output(reference+offset);
        long double scale = 0;
        for (std::size_t i = 0; i < n; ++i) {
            result.values(i,j) -= offset;
            scale = std::hypot(scale,result.values(i,j));
        }
        result.scales[j] = scale;
        common = std::max(common,scale);
    }
    for (std::size_t j = 0; j < p; ++j) {
        auto divisor = standardize ? result.scales[j] / std::sqrt(static_cast<long double>(n-1)) : common;
        if (divisor == 0)
            divisor = 1;
        result.scales[j] = divisor;
        for (std::size_t i = 0; i < n; ++i)
            result.values(i,j) /= divisor;
    }
    return result;
}

/// @brief Projects a vector off the existing orthonormal basis with reorthogonalization.
void orthogonalize(Vector& target, const Matrix& basis, std::size_t count, Vector* weights = nullptr,
    const Matrix* old_weights = nullptr) {
    for (int pass = 0; pass < 2; ++pass)
        for (std::size_t k = 0; k < count; ++k) {
            long double overlap = 0;
            for (std::size_t i = 0; i < target.size(); ++i)
                overlap += basis(i,k)*target[i];
            for (std::size_t i = 0; i < target.size(); ++i)
                target[i] -= overlap*basis(i,k);
            if (weights)
                for (std::size_t j = 0; j < weights->size(); ++j)
                    (*weights)[j] -= overlap*(*old_weights)(j,k);
        }
}

/// @brief Copies the fitted columns and applies coordinate-specific output scales.
PublicMatrix restore(const Matrix& work, std::size_t count, const Vector& scales, bool divide = false) {
    PublicMatrix result(work.rows(),count);
    for (std::size_t j = 0; j < work.rows(); ++j) {
        const auto multiplier = scales.empty() ? 1 : (divide ? 1/scales[j] : scales[j]);
        for (std::size_t k = 0; k < count; ++k)
            result(j,k) = output(work(j,k)*multiplier);
    }
    return result;
}

/// @brief Fits a single component path on the supplied training rows.
SIMPLSFit fit_path(const PublicMatrix& predictors, const PublicMatrix& response, const Rows& rows,
    std::size_t max_components, const SIMPLSOptions& options) {
    const auto n = rows.size(), p = predictors.cols(), q = response.cols();
    const auto limit = std::min({max_components,p,n-1});
    auto x = center(predictors,rows,options.scale_predictors);
    auto y = center(response,rows,options.scale_responses);
    if (limit == 0)
        return {std::move(x.means),std::move(y.means),PublicMatrix(p,0),PublicMatrix(n,0),PublicMatrix(p,0),PublicMatrix(q,0)};
    check_storage(p,q);
    check_storage(n,limit);
    check_storage(p,limit);
    check_storage(q,limit);
    const auto xt = x.values.transpose(), yt = y.values.transpose();
    auto covariance = xt*y.values;
    const auto tolerance = std::max(static_cast<long double>(options.tolerance),
        std::max({n,p,q})*std::numeric_limits<long double>::epsilon());
    const auto covariance_cutoff = tolerance*norm(covariance.data());
    const auto score_cutoff = tolerance*norm(x.values.data());
    Matrix weights(p,limit), scores(n,limit), x_loadings(p,limit), y_loadings(q,limit), deflation(p,limit);
    std::size_t fitted = 0;
    for (; fitted < limit; ++fitted) {
        Vector r;
        if (q == 1) {
            r = covariance.col(0);
            const auto length = norm(r);
            if (length <= covariance_cutoff)
                break;
            for (auto& value : r)
                value /= length;
        } else {
            const auto factor = linalg::svd(covariance);
            if (!factor.converged)
                throw std::runtime_error("SIMPLS: covariance factorization failed");
            if (factor.s.front() <= covariance_cutoff)
                break;
            r = factor.U.col(0);
        }
        auto score = x.values*r;
        orthogonalize(score,scores,fitted,&r,&weights);
        const auto length = norm(score);
        if (length <= score_cutoff)
            break;
        const auto biggest = std::ranges::max_element(r, {}, [](auto value) { return std::abs(value); });
        const auto signed_length = std::copysign(length,*biggest);
        for (auto& value : r)
            value /= signed_length;
        for (auto& value : score)
            value /= signed_length;
        const auto x_loading = xt*score, y_loading = yt*score;
        auto direction = x_loading;
        orthogonalize(direction,deflation,fitted);
        const auto direction_norm = norm(direction);
        if (direction_norm <= score_cutoff)
            break;
        for (auto& value : direction)
            value /= direction_norm;
        for (std::size_t j = 0; j < q; ++j) {
            long double overlap = 0;
            for (std::size_t a = 0; a < p; ++a)
                overlap += direction[a]*covariance(a,j);
            for (std::size_t a = 0; a < p; ++a)
                covariance(a,j) -= direction[a]*overlap;
            y_loadings(j,fitted) = y_loading[j];
        }
        for (std::size_t i = 0; i < n; ++i)
            scores(i,fitted) = score[i];
        for (std::size_t a = 0; a < p; ++a) {
            weights(a,fitted) = r[a];
            x_loadings(a,fitted) = x_loading[a];
            deflation(a,fitted) = direction[a];
        }
    }
    return {std::move(x.means),std::move(y.means),restore(weights,fitted,x.scales,true),restore(scores,fitted,{}),
        restore(x_loadings,fitted,x.scales),restore(y_loadings,fitted,y.scales)};
}

/// @brief Checks the component prefix for predictions and coefficient extraction.
std::size_t component_count(const SIMPLSFit& fit, std::optional<std::size_t> requested) {
    const auto count = requested.value_or(fit.components());
    if (count > fit.components() || fit.score_weights.rows() != fit.predictor_means.size() ||
        fit.y_loadings.rows() != fit.response_means.size() || fit.y_loadings.cols() != fit.components() ||
        !finite(fit.score_weights.data()) || !finite(fit.y_loadings.data()) ||
        !finite(fit.predictor_means) || !finite(fit.response_means))
        throw std::invalid_argument("SIMPLS: invalid component count or fitted model");
    return count;
}

/// @brief Applies one score contribution to a row's response predictions.
void add_component(const SIMPLSFit& fit, const PublicMatrix& predictors, std::size_t row, std::size_t k, Vector& prediction) {
    long double score = 0;
    for (std::size_t j = 0; j < predictors.cols(); ++j)
        score += (static_cast<long double>(predictors(row,j)) - fit.predictor_means[j])*fit.score_weights(j,k);
    for (std::size_t j = 0; j < prediction.size(); ++j)
        prediction[j] += score*fit.y_loadings(j,k);
}

/// @brief Copies the prefix of a finished path's matrix when cross-validation selects fewer components.
void trim(PublicMatrix& values, std::size_t count) {
    if (count == values.cols())
        return;
    PublicMatrix selected(values.rows(),count);
    for (std::size_t i = 0; i < selected.rows(); ++i)
        for (std::size_t j = 0; j < count; ++j)
            selected(i,j) = values(i,j);
    values = std::move(selected);
}

} // namespace

linalg::DenseMatrix<double> SIMPLSFit::coefficients(std::optional<std::size_t> count) const {
    const auto k = component_count(*this,count);
    const auto p = predictor_means.size(), q = response_means.size();
    check_storage(p,q);
    PublicMatrix result(p,q);
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < q; ++j) {
            long double total = 0;
            for (std::size_t r = 0; r < k; ++r)
                total += static_cast<long double>(score_weights(i,r))*y_loadings(j,r);
            result(i,j) = output(total);
        }
    return result;
}

linalg::DenseMatrix<double> SIMPLSFit::predict(const linalg::DenseMatrix<double>& predictors,
    std::optional<std::size_t> count) const {
    const auto k = component_count(*this,count);
    if (predictors.cols() != predictor_means.size() || !finite(predictors.data()))
        throw std::invalid_argument("SIMPLS: incompatible predictors");
    check_storage(predictors.rows(),response_means.size());
    PublicMatrix result(predictors.rows(),response_means.size());
    for (std::size_t i = 0; i < predictors.rows(); ++i) {
        Vector prediction(response_means.begin(),response_means.end());
        for (std::size_t r = 0; r < k; ++r)
            add_component(*this,predictors,i,r,prediction);
        for (std::size_t j = 0; j < prediction.size(); ++j)
            result(i,j) = output(prediction[j]);
    }
    return result;
}

SIMPLSFit fit_simpls(const linalg::DenseMatrix<double>& predictors, const linalg::DenseMatrix<double>& response,
    std::size_t max_components, const SIMPLSOptions& options) {
    validate(predictors,response,options);
    Rows training(predictors.rows());
    std::iota(training.begin(),training.end(),0uz);
    return fit_path(predictors,response,training,max_components,options);
}

SIMPLSCrossValidation cross_validate_simpls(const linalg::DenseMatrix<double>& predictors,
    const linalg::DenseMatrix<double>& response, const std::vector<std::int64_t>& fold,
    std::size_t max_components, const SIMPLSOptions& options) {
    validate(predictors,response,options);
    const auto n = predictors.rows(), q = response.cols();
    if (fold.size() != n)
        throw std::invalid_argument("SIMPLS: fold assignments must match the data rows");
    std::map<std::int64_t,Rows> held_out;
    for (std::size_t i = 0; i < n; ++i)
        held_out[fold[i]].push_back(i);
    if (held_out.size() < 2 || std::ranges::any_of(held_out,[&](const auto& group) { return n-group.second.size() < 2; }))
        throw std::invalid_argument("SIMPLS: each of at least two folds must leave two training rows");

    Rows training(n);
    std::iota(training.begin(),training.end(),0uz);
    auto fit = fit_path(predictors,response,training,max_components,options);
    auto common = fit.components();
    Vector squared_loss(common+1);
    for (const auto& [label,rows] : held_out) {
        training.clear();
        for (std::size_t i = 0; i < n; ++i)
            if (fold[i] != label)
                training.push_back(i);
        const auto model = fit_path(predictors,response,training,common,options);
        common = std::min(common,model.components());
        for (const auto row : rows) {
            Vector prediction(model.response_means.begin(),model.response_means.end());
            for (std::size_t k = 0;; ++k) {
                for (std::size_t j = 0; j < q; ++j) {
                    const auto error = response(row,j)-prediction[j];
                    squared_loss[k] += error*error;
                }
                if (k == common)
                    break;
                add_component(model,predictors,row,k,prediction);
            }
        }
    }
    squared_loss.resize(common+1);
    const auto selected = static_cast<std::size_t>(
        std::ranges::min_element(squared_loss)-squared_loss.begin());
    for (auto* matrix : {&fit.score_weights,&fit.scores,&fit.x_loadings,&fit.y_loadings})
        trim(*matrix,selected);
    std::vector<double> mse(common+1);
    const auto observations = static_cast<long double>(n)*q;
    for (std::size_t k = 0; k <= common; ++k)
        mse[k] = output(squared_loss[k]/observations);
    return {selected,std::move(mse),std::move(fit)};
}

} // namespace datamunge::stats
