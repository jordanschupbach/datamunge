/// @file minimum_covariance_determinant.cpp
/// @brief Exact and concentration searches with Gaussian-consistent MCD reweighting.

#include <datamunge/stats/minimum_covariance_determinant.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Working scatter factors and subset search helpers.
namespace {

/// @brief Scaled observation or correlation matrix.
using Matrix = linalg::DenseMatrix<long double>;
/// @brief Extended-precision location, scale, or distance vector.
using Vector = std::vector<long double>;
/// @brief Subset of original observation rows.
using Rows = std::vector<std::size_t>;

/// @brief Moment estimate and its factored covariance in working coordinates.
struct Scatter {
    Vector origin;          ///< First selected row, used for stable centering.
    Vector offset;          ///< Mean shift from the centering row.
    Vector deviation;       ///< Marginal standard deviations.
    Matrix lower;           ///< Lower triangular Cholesky factor of the correlation.
    long double log_det{};  ///< Log determinant before Gaussian correction.
};

/// @brief Candidate support, scatter and convergence outcome.
struct Candidate {
    Rows rows;         ///< Sorted observation indices.
    Scatter scatter;   ///< Moments fitted to those indices.
    bool converged{};  ///< The concentration steps stabilized.
};

/// @brief Factors centered contrasts directly when forming cross-products loses a pivot.
Matrix factor_contrasts(const Matrix& x, Rows rows, const Vector& deviation) {
    Matrix upper(x.cols(),x.cols());
    Vector radius(x.rows());
    for (const auto row : rows)
        for (std::size_t j = 0; j < x.cols(); ++j)
            radius[row] = std::max(radius[row],std::abs(x(row,j)));
    std::ranges::stable_sort(rows,{},[&](auto row) { return radius[row]; });
    auto mean = x.row(rows.front());
    Vector contrast(x.cols());
    for (std::size_t i = 1; i < rows.size(); ++i) {
        const auto factor = std::sqrt((static_cast<long double>(i)/(i+1))/rows.size());
        for (std::size_t j = 0; j < x.cols(); ++j) {
            const auto difference = x(rows[i],j)-mean[j];
            contrast[j] = (difference/deviation[j])*factor;
            mean[j] += difference/(i+1);
        }
        // Insert an orthonormal centered contrast into the triangular factor.
        for (std::size_t j = 0; j < x.cols(); ++j) {
            if (contrast[j] == 0)
                continue;
            const auto length = std::hypot(upper(j,j),contrast[j]);
            const auto cosine = upper(j,j)/length, sine = contrast[j]/length;
            upper(j,j) = length;
            for (std::size_t k = j+1; k < x.cols(); ++k) {
                const auto previous = upper(j,k);
                upper(j,k) = cosine*previous+sine*contrast[k];
                contrast[k] = cosine*contrast[k]-sine*previous;
            }
        }
    }
    return upper.transpose();
}

/// @brief Forms the maximum-likelihood covariance and factors its correlation matrix.
Scatter fit_subset(const Matrix& x, const Rows& rows) {
    const auto p = x.cols();
    Scatter result{Vector(p),Vector(p),Vector(p),{},0};
    result.origin = x.row(rows.front());
    for (const auto row : rows)
        for (std::size_t j = 0; j < p; ++j)
            result.offset[j] += x(row,j)-result.origin[j];
    for (auto& offset : result.offset)
        offset /= rows.size();

    Matrix scatter(p,p);
    Vector difference(p);
    for (const auto row : rows) {
        for (std::size_t j = 0; j < p; ++j)
            difference[j] = (x(row,j)-result.origin[j])-result.offset[j];
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t k = 0; k <= j; ++k)
                scatter(j,k) += difference[j]*difference[k];
    }
    for (std::size_t j = 0; j < p; ++j) {
        if (!(scatter(j,j) > 0))
            return {};
        result.deviation[j] = std::sqrt(scatter(j,j)/rows.size());
    }
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t k = 0; k < j; ++k)
            scatter(j,k) = scatter(k,j) = ((scatter(j,k)/rows.size())/result.deviation[j])/result.deviation[k];
        scatter(j,j) = 1;
    }
    auto factor = linalg::cholesky(scatter);
    const auto cutoff = 8*std::sqrt(std::numeric_limits<double>::epsilon());
    bool resolved = factor.ok;
    for (std::size_t j = 0; j < p && resolved; ++j)
        resolved = factor.L(j,j) > cutoff;
    result.lower = resolved ? std::move(factor.L) : factor_contrasts(x,rows,result.deviation);
    const auto precision = 64*std::numeric_limits<long double>::epsilon()*std::max(p,rows.size());
    for (std::size_t j = 0; j < p; ++j) {
        if (!(result.lower(j,j) > precision))
            return {};
        result.log_det += 2*(std::log(result.deviation[j])+std::log(result.lower(j,j)));
    }
    return result;
}

/// @brief Squared distances to a full-rank covariance via triangular whitening.
Vector distances(const Matrix& x, const Scatter& scatter) {
    const auto n = x.rows(), p = x.cols();
    Vector whitened(p), result(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            auto value = ((x(i,j)-scatter.origin[j])-scatter.offset[j])/scatter.deviation[j];
            for (std::size_t k = 0; k < j; ++k)
                value -= scatter.lower(j,k)*whitened[k];
            whitened[j] = value/scatter.lower(j,j);
            result[i] += whitened[j]*whitened[j];
        }
    }
    return result;
}

/// @brief Selects the nearest rows, breaking distance ties by original row order.
Rows nearest(const Vector& distance, std::size_t h) {
    Rows rows(distance.size());
    std::iota(rows.begin(),rows.end(),0uz);
    if (h < rows.size())
        std::ranges::nth_element(rows,rows.begin()+h,[&](auto a, auto b) {
            return distance[a] < distance[b] || (distance[a] == distance[b] && a < b);
        });
    rows.resize(h);
    std::ranges::sort(rows);
    return rows;
}

/// @brief Creates a candidate and diagnoses a singular determinant-minimizing subset.
Candidate candidate(const Matrix& x, Rows rows) {
    auto scatter = fit_subset(x,rows);
    if (scatter.lower.rows() == 0)
        throw std::domain_error("minimum covariance determinant: a selected support has singular scatter");
    return {std::move(rows),std::move(scatter),false};
}

/// @brief Orders candidates by determinant with deterministic support ties.
bool better(const Candidate& a, const Candidate& b) {
    if (a.scatter.log_det != b.scatter.log_det)
        return a.scatter.log_det < b.scatter.log_det;
    if (a.rows != b.rows)
        return a.rows < b.rows;
    return a.converged && !b.converged;
}

/// @brief Checks whether the binomial number of subsets fits the enumeration limit.
bool enumerable(std::size_t n, std::size_t h, std::size_t limit) {
    if (limit == 0)
        return false;
    auto count = std::size_t{1};
    const auto k = std::min(h,n-h);
    for (std::size_t i = 1; i <= k; ++i) {
        const auto numerator = n-k+i;
        const auto divisor = std::gcd(numerator,i);
        const auto multiply = numerator/divisor;
        count /= i/divisor;
        if (count > limit/multiply)
            return false;
        count *= multiply;
    }
    return true;
}

/// @brief Advances a sorted subset to the next lexicographic combination.
bool advance(Rows& rows, std::size_t n) {
    for (auto i = rows.size(); i-- > 0;) {
        if (rows[i] < n-(rows.size()-i)) {
            ++rows[i];
            for (std::size_t j = i+1; j < rows.size(); ++j)
                rows[j] = rows[j-1]+1;
            return true;
        }
    }
    return false;
}

/// @brief Searches every support of the requested size.
Candidate exhaustive(const Matrix& x, std::size_t h) {
    Rows rows(h);
    std::iota(rows.begin(),rows.end(),0uz);
    auto best = candidate(x,rows);
    while (advance(rows,x.rows())) {
        auto next = candidate(x,rows);
        if (better(next,best))
            best = std::move(next);
    }
    best.converged = true;
    return best;
}

/// @brief Runs determinant-decreasing concentration steps from one subset.
Candidate concentrate(const Matrix& x, Candidate current, std::size_t steps) {
    for (std::size_t i = 0; i < steps && !current.converged; ++i) {
        auto rows = nearest(distances(x,current.scatter),current.rows.size());
        if (rows == current.rows) {
            current.converged = true;
            break;
        }
        auto next = candidate(x,std::move(rows));
        // Stop on a stable determinant rather than cycling on tied supports.
        if (!better(next,current)) {
            const auto tolerance = 64*std::numeric_limits<long double>::epsilon()*
                std::max(1.0L,std::abs(current.scatter.log_det));
            current.converged = next.scatter.log_det-current.scatter.log_det <= tolerance;
            break;
        }
        current = std::move(next);
    }
    return current;
}

/// @brief Median of a nonempty working vector, which is permuted in place.
long double median(Vector& values) {
    const auto middle = values.begin()+values.size()/2;
    std::ranges::nth_element(values,middle);
    return values.size()%2 ? *middle : std::midpoint(*std::max_element(values.begin(),middle),*middle);
}

/// @brief Produces a deterministic robust start from coordinate medians and absolute deviations.
Rows median_start(const Matrix& x, std::size_t h) {
    Vector squared(x.rows());
    for (std::size_t j = 0; j < x.cols(); ++j) {
        auto values = x.col(j);
        const auto center = median(values);
        for (auto& value : values)
            value = std::abs(value-center);
        auto scale = median(values);
        if (scale == 0)
            scale = *std::ranges::max_element(values);
        if (scale == 0)
            throw std::domain_error("minimum covariance determinant: a feature has no variation");
        for (std::size_t i = 0; i < x.rows(); ++i) {
            const auto deviation = (x(i,j)-center)/scale;
            squared[i] += deviation*deviation;
        }
    }
    return nearest(squared,h);
}

/// @brief Screens seeded elemental starts and refines their best supports.
Candidate approximate(const Matrix& x, std::size_t h, const MCDOptions& options) {
    constexpr auto retained = 10u;
    std::vector<Candidate> finalists;
    /// @brief Screens a support with two concentration steps and retains the best distinct candidates.
    const auto screen = [&](Rows rows) {
        auto trial = concentrate(x,candidate(x,std::move(rows)),2);
        if (std::ranges::any_of(finalists,[&](const auto& fit) { return fit.rows == trial.rows; }))
            return;
        const auto insertion = std::ranges::lower_bound(finalists,trial,better);
        if (finalists.size() < retained || insertion != finalists.end()) {
            finalists.insert(insertion,std::move(trial));
            if (finalists.size() > retained)
                finalists.pop_back();
        }
    };
    Rows order(x.rows()), seed;
    std::iota(order.begin(),order.end(),0uz);
    const auto full = fit_subset(x,order);
    if (full.lower.rows() > 0)
        screen(nearest(distances(x,full),h));
    screen(median_start(x,h));

    std::mt19937_64 random(options.seed);
    seed.reserve(h);
    for (std::size_t start = 0; start < options.random_starts; ++start) {
        std::iota(order.begin(),order.end(),0uz);
        seed.clear();
        Scatter initial;
        for (std::size_t j = 0; j < h; ++j) {
            const auto picked = std::uniform_int_distribution<std::size_t>(j,order.size()-1)(random);
            std::swap(order[j],order[picked]);
            seed.push_back(order[j]);
            if (seed.size() > x.cols()) {
                initial = fit_subset(x,seed);
                if (initial.lower.rows() > 0)
                    break;
            }
        }
        if (initial.lower.rows() == 0)
            throw std::domain_error("minimum covariance determinant: an elemental start reached a singular support");
        screen(nearest(distances(x,initial),h));
    }

    auto best = concentrate(x,std::move(finalists.front()),options.max_c_steps);
    for (std::size_t i = 1; i < finalists.size(); ++i) {
        auto next = concentrate(x,std::move(finalists[i]),options.max_c_steps);
        if (better(next,best))
            best = std::move(next);
    }
    return best;
}

/// @brief Converts a covariance or mean to finite output precision.
double output(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("minimum covariance determinant: fitted value exceeds finite reporting range");
    return static_cast<double>(value);
}

/// @brief Restores location and corrected covariance in original measurement units.
void restore(const Scatter& scatter, const Vector& scales, double correction, std::vector<double>& location,
    linalg::DenseMatrix<double>& covariance) {
    const auto p = scales.size();
    location.resize(p);
    covariance = linalg::DenseMatrix<double>(p,p);
    for (std::size_t j = 0; j < p; ++j) {
        location[j] = output((scatter.origin[j]+scatter.offset[j])*scales[j]);
        for (std::size_t k = 0; k <= j; ++k) {
            long double correlation = 0;
            for (std::size_t a = 0; a <= k; ++a)
                correlation += scatter.lower(j,a)*scatter.lower(k,a);
            const auto value = ((correlation*correction) *
                (scatter.deviation[j]*scales[j]))*(scatter.deviation[k]*scales[k]);
            covariance(j,k) = covariance(k,j) = output(value);
        }
        if (covariance(j,j) == 0)
            throw std::overflow_error("minimum covariance determinant: covariance variance underflows double precision");
    }
}

/// @brief Returns corrected distances, saturating unrepresentably large values at infinity.
std::vector<double> report_distances(const Vector& distances, double correction) {
    std::vector<double> result;
    result.reserve(distances.size());
    for (const auto value : distances) {
        const auto scaled = value/correction;
        result.push_back(scaled > std::numeric_limits<double>::max() ?
            std::numeric_limits<double>::infinity() : static_cast<double>(scaled));
    }
    return result;
}

} // namespace

MCDResult minimum_covariance_determinant(const linalg::DenseMatrix<double>& observations, const MCDOptions& options) {
    const auto n = observations.rows(), p = observations.cols();
    if (p == 0 || n <= p || !(options.reweight_probability > 0 && options.reweight_probability < 1) ||
        options.max_c_steps == 0 ||
        !std::ranges::all_of(observations.data(),[](double value) { return std::isfinite(value); }))
        throw std::invalid_argument("minimum covariance determinant: invalid data or search controls");
    const auto h = options.support_size ? options.support_size : p+(n-p+1)/2;
    if (h <= p || h > n)
        throw std::invalid_argument("minimum covariance determinant: support must exceed dimension and fit the sample");
    if (n > Vector{}.max_size()/p)
        throw std::length_error("minimum covariance determinant: dense workspace exceeds storage capacity");

    Matrix x(n,p);
    Vector scales(p);
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < n; ++i)
            scales[j] = std::max(scales[j],std::abs(static_cast<long double>(observations(i,j))));
        if (scales[j] == 0)
            throw std::domain_error("minimum covariance determinant: a feature has no variation");
        for (std::size_t i = 0; i < n; ++i)
            x(i,j) = observations(i,j)/scales[j];
    }
    const bool exact = h == n || enumerable(n,h,options.exact_subset_limit);
    auto raw = exact ? exhaustive(x,h) : approximate(x,h,options);

    MCDResult result{};
    result.exact_search = exact;
    result.converged = raw.converged;
    result.support = std::move(raw.rows);
    const auto dimension = static_cast<double>(p);
    const auto fraction = static_cast<double>(h)/n;
    const auto raw_correction = h == n ? 1 :
        fraction/random::chi_squared_cdf(random::chi_squared_quantile(fraction,dimension),dimension+2);
    const auto cutoff = random::chi_squared_quantile(options.reweight_probability,dimension);
    const auto correction = options.reweight_probability/random::chi_squared_cdf(cutoff,dimension+2);
    result.raw_consistency_factor = raw_correction;
    result.reweight_consistency_factor = correction;
    result.reweight_cutoff = cutoff;
    restore(raw.scatter,scales,raw_correction,result.raw_location,result.raw_covariance);
    const auto raw_distances = distances(x,raw.scatter);
    result.raw_squared_distances = report_distances(raw_distances,raw_correction);
    for (std::size_t i = 0; i < n; ++i)
        if (raw_distances[i]/raw_correction <= cutoff)
            result.reweighted_rows.push_back(i);
    if (result.reweighted_rows.size() <= p)
        throw std::domain_error("minimum covariance determinant: too few reweighted rows for full scatter");
    const auto reweighted = fit_subset(x,result.reweighted_rows);
    if (reweighted.lower.rows() == 0)
        throw std::domain_error("minimum covariance determinant: reweighted scatter is singular");
    restore(reweighted,scales,correction,result.location,result.covariance);
    result.squared_distances = report_distances(distances(x,reweighted),correction);
    return result;
}

} // namespace datamunge::stats
