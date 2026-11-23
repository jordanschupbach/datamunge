/// @file mean_shift.cpp
/// @brief Gaussian density ascent and clustering by converged endpoints.

#include <datamunge/stats/mean_shift.hpp>

#include <datamunge/linalg/eigen.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Gaussian density and trajectory calculations.
namespace {

/// @brief Coordinates, displacements and Gaussian weights in working precision.
using Vector = std::vector<long double>;
/// @brief Borrowed type of the input coordinates.
using Points = linalg::DenseMatrix<double>;

/// @brief Euclidean distance in original units between a point and arbitrary coordinates.
long double distance(const Points& data, std::size_t row, const Vector& point) {
    long double value = 0;
    for (std::size_t j = 0; j < point.size(); ++j)
        value = std::hypot(value,static_cast<long double>(data(row,j))-point[j]);
    return value;
}

/// @brief Euclidean separation of two iterates in original units.
long double distance(const Vector& a, const Vector& b) {
    long double value = 0;
    for (std::size_t j = 0; j < a.size(); ++j)
        value = std::hypot(value,a[j]-b[j]);
    return value;
}

/// @brief Gaussian log kernel sum and normalized sample weights at one iterate.
struct Density {
    long double log_mass{}; ///< Log of the unnormalized kernel sum.
    Vector weights; ///< Posterior kernel contributions summing to one.
};

/// @brief Evaluates Gaussian weights with an exponent shift by the closest sample.
Density evaluate(const Points& observations, const Vector& point, long double h) {
    Density density{0,Vector(observations.rows())};
    long double minimum = std::numeric_limits<long double>::infinity();
    for (std::size_t i = 0; i < observations.rows(); ++i) {
        const auto radius = distance(observations,i,point)/h;
        density.weights[i] = radius;
        minimum = std::min(minimum,radius);
    }
    if (!std::isfinite(minimum))
        throw std::overflow_error("mean shift: every standardized kernel distance overflowed");
    long double mass = 0;
    for (auto& weight : density.weights) {
        // Difference of squares avoids cancellation when the smallest distance is large.
        weight = weight == minimum ? 1 : std::exp(-.5L*(weight-minimum)*(weight+minimum));
        mass += weight;
    }
    for (auto& weight : density.weights)
        weight /= mass;
    density.log_mass = -.5L*minimum*minimum + std::log(mass);
    return density;
}

/// @brief Calculates the Gaussian posterior mean shift about the current point.
Vector displacement(const Points& data, const Vector& point, const Vector& weights) {
    Vector shift(point.size());
    for (std::size_t i = 0; i < data.rows(); ++i)
        if (weights[i] > 0)
            for (std::size_t j = 0; j < point.size(); ++j)
                shift[j] += weights[i]*(static_cast<long double>(data(i,j))-point[j]);
    return shift;
}

/// @brief Finds a direction of positive log-density curvature at a stationary point.
Vector uphill_direction(const Points& data, const Vector& point, const Vector& weights,
    const Vector& shift, long double h) {
    const auto d = point.size();
    const long double threshold = 1 + 1e-9L;
    long double trace = 0;
    for (std::size_t i = 0; i < data.rows(); ++i)
        if (weights[i] > 0) {
            const auto radius = distance(data,i,point)/h;
            trace += (weights[i]*radius)*radius;
        }
    if (!(trace > threshold))
        return {};
    if (d > Vector{}.max_size()/d)
        throw std::length_error("mean shift: curvature exceeds dense storage capacity");
    linalg::DenseMatrix<long double> second_moment(d,d);
    Vector offset(d);
    for (std::size_t i = 0; i < data.rows(); ++i)
        if (weights[i] > 0) {
            const auto root = std::sqrt(weights[i]);
            for (std::size_t j = 0; j < d; ++j)
                offset[j] = root*((static_cast<long double>(data(i,j))-point[j])/h);
            for (std::size_t j = 0; j < d; ++j)
                for (std::size_t k = j; k < d; ++k)
                    second_moment(j,k) += offset[j]*offset[k];
        }
    for (std::size_t j = 0; j < d; ++j)
        for (std::size_t k = j; k < d; ++k)
            second_moment(j,k) -= (shift[j]/h)*(shift[k]/h);
    const auto eig = linalg::symmetric_eigen(second_moment);
    if (eig.eigenvalues.front() <= threshold)
        return {};
    Vector direction(d);
    for (std::size_t j = 0; j < d; ++j)
        direction[j] = eig.eigenvectors(j,0);
    const auto largest = std::ranges::max_element(direction,{},[](long double x) { return std::abs(x); });
    if (*largest < 0)
        for (auto& value : direction)
            value = -value;
    return direction;
}

/// @brief One terminal position with its density and convergence evidence.
struct Endpoint {
    Vector point; ///< Coordinates in observation units.
    long double log_mass{}; ///< Unnormalized log sum of Gaussian kernels.
    std::size_t updates{}; ///< Number of accepted shifts or stationary-point escapes.
    bool converged{}; ///< True if the trajectory met tolerance without positive curvature.
};

/// @brief Escapes a stationary non-maximum by finding a higher-density perturbation.
bool perturb(const Points& data, Vector& point, const Vector& direction,
    long double h, long double current) {
    Vector candidate(point.size());
    for (long double step = .25L; step >= 1e-6L; step *= .5L)
        for (const int sign : {1,-1}) {
            for (std::size_t j = 0; j < point.size(); ++j)
                candidate[j] = point[j] + sign*step*h*direction[j];
            if (evaluate(data,candidate,h).log_mass > current) {
                point = std::move(candidate);
                return true;
            }
        }
    return false;
}

/// @brief Follows the Gaussian mean shift from one sample until its mode or update limit.
Endpoint follow(const Points& data, std::size_t seed, long double h, const MeanShiftOptions& options) {
    Endpoint endpoint{Vector(data.cols())};
    for (std::size_t j = 0; j < data.cols(); ++j)
        endpoint.point[j] = data(seed,j);
    for (;;) {
        auto density = evaluate(data,endpoint.point,h);
        endpoint.log_mass = density.log_mass;
        const auto shift = displacement(data,endpoint.point,density.weights);
        long double norm = 0;
        for (const auto value : shift)
            norm = std::hypot(norm,value);
        if (norm/h <= options.tolerance) {
            const auto direction = uphill_direction(data,endpoint.point,density.weights,shift,h);
            if (direction.empty()) {
                endpoint.converged = true;
                break;
            }
            if (endpoint.updates == options.max_iterations ||
                !perturb(data,endpoint.point,direction,h,endpoint.log_mass))
                break;
        } else {
            if (endpoint.updates == options.max_iterations)
                break;
            for (std::size_t j = 0; j < endpoint.point.size(); ++j)
                endpoint.point[j] += shift[j];
        }
        ++endpoint.updates;
    }
    return endpoint;
}

/// @brief Converts a mode coordinate or log density to a finite double.
double report_value(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("mean shift: output exceeds finite reporting range");
    return static_cast<double>(value);
}

} // namespace

MeanShiftFit fit_mean_shift(const linalg::DenseMatrix<double>& observations, double bandwidth,
    const MeanShiftOptions& options) {
    const auto n = observations.rows(), d = observations.cols();
    if (n == 0 || d == 0 || !(bandwidth > 0) || !std::isfinite(bandwidth) ||
        options.max_iterations == 0 || !(options.tolerance > 0 && options.tolerance < 1) ||
        !(options.merge_tolerance >= 0) || !std::isfinite(options.merge_tolerance) ||
        !std::ranges::all_of(observations.data(),[](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("mean shift: invalid observations, bandwidth or tolerances");
    MeanShiftFit fit;
    fit.labels.resize(n);
    fit.converged.resize(n);
    fit.iterations.resize(n);
    std::vector<Endpoint> endpoints;
    endpoints.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& endpoint = endpoints.emplace_back(follow(observations,i,bandwidth,options));
        fit.converged[i] = endpoint.converged;
        fit.iterations[i] = endpoint.updates;
    }

    std::vector<std::size_t> order(n), representatives;
    std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](std::size_t left, std::size_t right) {
        return std::pair{endpoints[left].converged,endpoints[left].log_mass} >
            std::pair{endpoints[right].converged,endpoints[right].log_mass};
    });
    for (const auto seed : order) {
        auto closest = representatives.size();
        long double separation = options.merge_tolerance;
        for (std::size_t j = 0; j < representatives.size(); ++j) {
            const auto current = distance(endpoints[seed].point,endpoints[representatives[j]].point)/bandwidth;
            if (current <= separation && (closest == representatives.size() || current < separation)) {
                closest = j;
                separation = current;
            }
        }
        fit.labels[seed] = closest;
        if (closest == representatives.size())
            representatives.push_back(seed);
    }

    fit.modes = linalg::DenseMatrix<double>(representatives.size(),d);
    const long double normalizer = std::log(static_cast<long double>(n)) +
        d*(std::log(static_cast<long double>(bandwidth)) + .5L*std::log(2*std::numbers::pi_v<long double>));
    for (std::size_t i = 0; i < representatives.size(); ++i) {
        const auto& endpoint = endpoints[representatives[i]];
        for (std::size_t j = 0; j < d; ++j)
            fit.modes(i,j) = report_value(endpoint.point[j]);
        fit.log_density.push_back(report_value(endpoint.log_mass-normalizer));
    }
    return fit;
}

} // namespace datamunge::stats
