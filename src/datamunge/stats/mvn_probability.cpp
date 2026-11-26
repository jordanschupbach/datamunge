/// @file mvn_probability.cpp
/// @brief Genz separation of variables with variable reordering and randomized lattice rules.

#include <datamunge/stats/mvn_probability.hpp>

#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <ranges>
#include <random>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Normal tail functions, the reordered Cholesky factor and the lattice integrand.
namespace {

/// @brief Dense double matrix used by the helpers.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Standard normal distribution function, accurate in both tails.
/// @param x Argument, possibly infinite.
/// @return \f$\Phi(x)\f$.
double phi_cdf(double x) {
    return .5*std::erfc(-x/std::numbers::sqrt2);
}

/// @brief Standard normal density.
/// @param x Argument, possibly infinite.
/// @return \f$\varphi(x)\f$, zero at infinity.
double phi_pdf(double x) {
    return std::isinf(x) ? 0.0 : std::exp(-.5*x*x)/std::sqrt(2*std::numbers::pi);
}

/// @brief Standard normal quantile: Acklam's rational approximation refined by one Halley step.
/// @param p Probability in (0, 1).
/// @return \f$\Phi^{-1}(p)\f$.
double phi_quantile(double p) {
    static constexpr double a[]{-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,
                                1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00};
    static constexpr double b[]{-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,
                                6.680131188771972e+01,-1.328068155288572e+01};
    static constexpr double c[]{-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,
                                -2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00};
    static constexpr double d[]{7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,
                                3.754408661907416e+00};
    double x;
    if (p < .02425) {
        const double q = std::sqrt(-2*std::log(p));
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    } else if (p > 1-.02425) {
        const double q = std::sqrt(-2*std::log1p(-p));
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    } else {
        const double q = p-.5, r = q*q;
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
    }
    const double e = phi_cdf(x)-p;
    const double u = e*std::sqrt(2*std::numbers::pi)*std::exp(.5*x*x);
    return x-u/(1+.5*x*u);
}

/// @brief Problem after centering, reordering and Cholesky factorization.
struct Reordered {
    Matrix factor;              ///< Lower-triangular Cholesky factor of the reordered covariance.
    std::vector<double> lower;  ///< Centered, reordered lower bounds.
    std::vector<double> upper;  ///< Centered, reordered upper bounds.
};

/// @brief Cholesky factorization that moves the narrowest expected interval first at each step.
/// @param lower Centered lower bounds.
/// @param upper Centered upper bounds.
/// @param covariance Covariance (copied and permuted).
/// @return Factor and permuted bounds.
/// @throws std::invalid_argument If a pivot is not positive.
Reordered reorder_and_factor(std::vector<double> lower, std::vector<double> upper, Matrix covariance) {
    const std::size_t d = lower.size();
    Matrix factor(d,d,0.0);
    std::vector<double> expected(d,0.0);
    for (std::size_t i = 0; i < d; ++i) {
        std::size_t best = i;
        double best_probability = std::numeric_limits<double>::infinity();
        for (std::size_t j = i; j < d; ++j) {
            double variance = covariance(j,j), shift = 0;
            for (std::size_t k = 0; k < i; ++k) {
                variance -= factor(j,k)*factor(j,k);
                shift += factor(j,k)*expected[k];
            }
            if (!(variance > 0))
                throw std::invalid_argument("mvn_rectangle_probability: covariance must be positive definite");
            const double sd = std::sqrt(variance);
            const double probability = phi_cdf((upper[j]-shift)/sd)-phi_cdf((lower[j]-shift)/sd);
            if (probability < best_probability) {
                best_probability = probability;
                best = j;
            }
        }
        if (best != i) {
            std::swap(lower[i],lower[best]);
            std::swap(upper[i],upper[best]);
            for (std::size_t k = 0; k < d; ++k)
                std::swap(covariance(i,k),covariance(best,k));
            for (std::size_t k = 0; k < d; ++k)
                std::swap(covariance(k,i),covariance(k,best));
            for (std::size_t k = 0; k < i; ++k)
                std::swap(factor(i,k),factor(best,k));
        }
        double pivot = covariance(i,i), shift = 0;
        for (std::size_t k = 0; k < i; ++k) {
            pivot -= factor(i,k)*factor(i,k);
            shift += factor(i,k)*expected[k];
        }
        factor(i,i) = std::sqrt(pivot);
        for (std::size_t j = i+1; j < d; ++j) {
            double s = covariance(j,i);
            for (std::size_t k = 0; k < i; ++k)
                s -= factor(j,k)*factor(i,k);
            factor(j,i) = s/factor(i,i);
        }
        // Mean of the standard normal truncated to this variable's conditional interval.
        const double lo = (lower[i]-shift)/factor(i,i), hi = (upper[i]-shift)/factor(i,i);
        const double mass = phi_cdf(hi)-phi_cdf(lo);
        expected[i] = mass > 1e-300 ? (phi_pdf(lo)-phi_pdf(hi))/mass
                                    : (std::isfinite(lo) ? lo : (std::isfinite(hi) ? hi : 0.0));
    }
    return {std::move(factor),std::move(lower),std::move(upper)};
}

/// @brief Genz's transformed integrand at one point of the unit cube of dimension d-1.
/// @param problem Factor and bounds.
/// @param w Point in \f$[0,1]^{d-1}\f$.
/// @param y Workspace of length d.
/// @param scale Multiplier on both bounds (the chi scale \f$s\f$ of a multivariate t, else 1).
/// @return Product of the conditional interval probabilities.
double integrand(const Reordered& problem, const double* w, std::vector<double>& y, double scale = 1) {
    const std::size_t d = problem.lower.size();
    double dlow = phi_cdf(scale*problem.lower[0]/problem.factor(0,0));
    double eup = phi_cdf(scale*problem.upper[0]/problem.factor(0,0));
    double f = eup-dlow;
    for (std::size_t i = 1; i < d && f > 0; ++i) {
        const double u = std::clamp(dlow+w[i-1]*(eup-dlow),1e-300,1-1e-16);
        y[i-1] = phi_quantile(u);
        double s = 0;
        for (std::size_t k = 0; k < i; ++k)
            s += problem.factor(i,k)*y[k];
        dlow = phi_cdf((scale*problem.lower[i]-s)/problem.factor(i,i));
        eup = phi_cdf((scale*problem.upper[i]-s)/problem.factor(i,i));
        f *= eup-dlow;
    }
    return std::max(f,0.0);
}

/// @brief Chi scale \f$s=\sqrt{\chi^2_\nu/\nu}\f$ at lower-tail probability u.
///
/// Wilson–Hilferty start refined by safeguarded Newton steps on the chi-square distribution function.
/// @param u Probability in (0, 1).
/// @param df Degrees of freedom \f$\nu>0\f$.
/// @return The scale.
double chi_scale(double u, double df) {
    u = std::clamp(u,1e-15,1-1e-15);
    const double z = phi_quantile(u), c = 2/(9*df);
    double x = df*std::pow(std::max(1-c+z*std::sqrt(c),.05),3);
    const double log_norm = (df/2)*std::log(2.0)+std::lgamma(df/2);
    for (int it = 0; it < 8; ++it) {
        const double f = random::chi_squared_cdf(x,df)-u;
        const double density = std::exp((df/2-1)*std::log(x)-x/2-log_norm);
        if (!(density > 0))
            break;
        double next = x-f/density;
        if (!(next > 0))
            next = x/2;
        if (std::abs(next-x) <= 1e-12*x) {
            x = next;
            break;
        }
        x = next;
    }
    return std::sqrt(x/df);
}

/// @brief First n primes.
/// @param n Count.
/// @return Primes 2, 3, 5, ...
std::vector<std::size_t> primes(std::size_t n) {
    std::vector<std::size_t> out;
    for (std::size_t c = 2; out.size() < n; ++c)
        if (std::ranges::none_of(out,[c](std::size_t p) { return c%p == 0; }))
            out.push_back(c);
    return out;
}

/// @brief Shared validation and lattice integration for normal (infinite df) and t laws.
/// @param lower Lower bounds.
/// @param upper Upper bounds.
/// @param mean Location.
/// @param covariance Covariance or scale matrix.
/// @param df Degrees of freedom, positive infinity for the normal law.
/// @param options Accuracy controls.
/// @return Probability and error.
MvnProbability integrate(const std::vector<double>& lower, const std::vector<double>& upper,
                         const std::vector<double>& mean, const linalg::DenseMatrix<double>& covariance, double df,
                         const MvnProbabilityOptions& options) {
    const bool student = std::isfinite(df);
    const std::size_t d = lower.size();
    if (d == 0 || upper.size() != d || mean.size() != d || covariance.rows() != d || covariance.cols() != d)
        throw std::invalid_argument("multivariate rectangle probability: bounds, mean and covariance must have matching nonzero size");
    if (!(options.abs_tol > 0) || options.shifts < 2 || options.max_points == 0)
        throw std::invalid_argument("multivariate rectangle probability: need abs_tol > 0, at least two shifts and max_points > 0");
    for (std::size_t j = 0; j < d; ++j) {
        if (std::isnan(lower[j]) || std::isnan(upper[j]) || lower[j] == std::numeric_limits<double>::infinity() ||
            upper[j] == -std::numeric_limits<double>::infinity() || lower[j] > upper[j])
            throw std::invalid_argument("multivariate rectangle probability: need lower <= upper with -inf/+inf only on the open side");
        if (!std::isfinite(mean[j]))
            throw std::invalid_argument("multivariate rectangle probability: mean must be finite");
    }
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j <= i; ++j)
            if (!std::isfinite(covariance(i,j)) ||
                std::abs(covariance(i,j)-covariance(j,i)) > 1e-10*std::max({1.0,std::abs(covariance(i,j))}))
                throw std::invalid_argument("multivariate rectangle probability: covariance must be finite and symmetric");
    std::vector<double> a(d), b(d);
    for (std::size_t j = 0; j < d; ++j) {
        a[j] = lower[j]-mean[j];
        b[j] = upper[j]-mean[j];
    }
    const auto problem = reorder_and_factor(a,b,covariance);
    MvnProbability result;
    if (std::ranges::any_of(std::views::iota(std::size_t{0},d),[&](std::size_t j) { return a[j] == b[j]; })) {
        result.converged = true;
        return result;
    }
    if (d == 1) {
        const auto cdf = [&](double x) {
            return student ? (std::isinf(x) ? (x > 0 ? 1.0 : 0.0) : random::student_t_cdf(x,df)) : phi_cdf(x);
        };
        result.value = cdf(problem.upper[0]/problem.factor(0,0))-cdf(problem.lower[0]/problem.factor(0,0));
        result.points = 1;
        result.converged = true;
        return result;
    }
    const std::size_t dims = d-1+(student ? 1 : 0);
    std::vector<double> generator(dims);
    const auto prime = primes(dims);
    for (std::size_t j = 0; j < dims; ++j)
        generator[j] = std::fmod(std::sqrt(static_cast<double>(prime[j])),1.0);
    std::mt19937_64 random(options.seed);
    std::uniform_real_distribution<double> unit(0.0,1.0);
    std::vector<double> w(dims), mirrored(dims), shift(dims), y(d);
    double weight_sum = 0, weighted_value = 0;
    std::size_t n = 31;
    while (true) {
        if (result.points+2*n*options.shifts > options.max_points && result.points > 0)
            break;
        double mean_of_shifts = 0, square = 0;
        for (std::size_t m = 0; m < options.shifts; ++m) {
            for (auto& s : shift)
                s = unit(random);
            double total = 0;
            for (std::size_t k = 1; k <= n; ++k) {
                for (std::size_t j = 0; j < dims; ++j) {
                    const double x = std::fmod(static_cast<double>(k)*generator[j]+shift[j],1.0);
                    w[j] = std::abs(2*x-1);
                    mirrored[j] = 1-w[j];
                }
                if (student)
                    total += .5*(integrand(problem,w.data()+1,y,chi_scale(w[0],df))+
                                 integrand(problem,mirrored.data()+1,y,chi_scale(mirrored[0],df)));
                else
                    total += .5*(integrand(problem,w.data(),y)+integrand(problem,mirrored.data(),y));
            }
            const double estimate = total/static_cast<double>(n);
            const double delta = estimate-mean_of_shifts;
            mean_of_shifts += delta/static_cast<double>(m+1);
            square += delta*(estimate-mean_of_shifts);
        }
        result.points += 2*n*options.shifts;
        const double variance = square/static_cast<double>(options.shifts*(options.shifts-1));
        const double weight = 1/std::max(variance,1e-300);
        weight_sum += weight;
        weighted_value += weight*mean_of_shifts;
        result.value = weighted_value/weight_sum;
        result.error = 3*std::sqrt(1/weight_sum);
        if (result.error <= options.abs_tol) {
            result.converged = true;
            break;
        }
        n = n*3/2;
    }
    result.value = std::clamp(result.value,0.0,1.0);
    return result;
}

} // namespace

MvnProbability mvn_rectangle_probability(const std::vector<double>& lower, const std::vector<double>& upper,
                                         const std::vector<double>& mean, const linalg::DenseMatrix<double>& covariance,
                                         const MvnProbabilityOptions& options) {
    return integrate(lower,upper,mean,covariance,std::numeric_limits<double>::infinity(),options);
}

MvnProbability mvt_rectangle_probability(const std::vector<double>& lower, const std::vector<double>& upper,
                                         const std::vector<double>& location, const linalg::DenseMatrix<double>& scale,
                                         double df, const MvnProbabilityOptions& options) {
    if (!(df > 0) || std::isnan(df))
        throw std::invalid_argument("mvt_rectangle_probability: degrees of freedom must be positive");
    return integrate(lower,upper,location,scale,df,options);
}

MvnProbability mvn_cdf(const std::vector<double>& upper, const std::vector<double>& mean, const linalg::DenseMatrix<double>& covariance,
                       const MvnProbabilityOptions& options) {
    return mvn_rectangle_probability(std::vector<double>(upper.size(),-std::numeric_limits<double>::infinity()),
                                     upper,mean,covariance,options);
}

} // namespace datamunge::stats
