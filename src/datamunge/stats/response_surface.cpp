/// @file response_surface.cpp
/// @brief Central composite and Box–Behnken run construction, second-order fits and canonical analysis.

#include <datamunge/stats/response_surface.hpp>

#include <datamunge/linalg/eigen.hpp>
#include <datamunge/linalg/least_squares.hpp>
#include <datamunge/stats/factorial_design.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace datamunge::stats {

CentralCompositeDesign central_composite_design(std::size_t factors, const CentralCompositeOptions& options) {
    if (factors < 2 || factors > 20)
        throw std::invalid_argument("central_composite_design: need 2 to 20 factors");
    const auto cube = options.generators.empty() ? full_factorial_design(factors)
                                                 : fractional_factorial_design(factors,options.generators);
    CentralCompositeDesign design;
    design.factorial_runs = cube.runs.rows();
    switch (options.axial) {
    case AxialDistance::rotatable:
        design.alpha = std::pow(static_cast<double>(design.factorial_runs),.25);
        break;
    case AxialDistance::face_centered:
        design.alpha = 1;
        break;
    case AxialDistance::spherical:
        design.alpha = std::sqrt(static_cast<double>(factors));
        break;
    case AxialDistance::custom:
        if (!(options.alpha > 0) || !std::isfinite(options.alpha))
            throw std::invalid_argument("central_composite_design: custom alpha must be positive and finite");
        design.alpha = options.alpha;
        break;
    }
    const std::size_t n = design.factorial_runs+options.factorial_center_points+2*factors+options.axial_center_points;
    design.runs = linalg::DenseMatrix<double>(n,factors);
    std::size_t row = 0;
    for (std::size_t r = 0; r < design.factorial_runs; ++r, ++row)
        for (std::size_t j = 0; j < factors; ++j)
            design.runs(row,j) = cube.runs(r,j);
    row += options.factorial_center_points;
    for (std::size_t j = 0; j < factors; ++j) {
        design.runs(row++,j) = -design.alpha;
        design.runs(row++,j) = design.alpha;
    }
    return design;
}

linalg::DenseMatrix<double> box_behnken_design(std::size_t factors, std::size_t center_points) {
    if (factors < 3 || factors > 20)
        throw std::invalid_argument("box_behnken_design: need 3 to 20 factors");
    linalg::DenseMatrix<double> runs(2*factors*(factors-1)+center_points,factors);
    std::size_t row = 0;
    for (std::size_t i = 0; i < factors; ++i)
        for (std::size_t j = i+1; j < factors; ++j)
            for (int corner = 0; corner < 4; ++corner, ++row) {
                runs(row,i) = (corner & 1) ? 1.0 : -1.0;
                runs(row,j) = (corner & 2) ? 1.0 : -1.0;
            }
    return runs;
}

QuadraticSurface fit_quadratic_surface(const linalg::DenseMatrix<double>& x, const std::vector<double>& y) {
    const std::size_t n = x.rows(), k = x.cols();
    if (k < 1 || k > 20)
        throw std::invalid_argument("fit_quadratic_surface: need 1 to 20 factors");
    if (y.size() != n)
        throw std::invalid_argument("fit_quadratic_surface: y must have one value per run");
    for (std::size_t r = 0; r < n; ++r) {
        if (!std::isfinite(y[r]))
            throw std::invalid_argument("fit_quadratic_surface: nonfinite response");
        for (std::size_t j = 0; j < k; ++j)
            if (!std::isfinite(x(r,j)))
                throw std::invalid_argument("fit_quadratic_surface: nonfinite factor setting");
    }
    const std::size_t p = (k+1)*(k+2)/2;
    if (n <= p)
        throw std::invalid_argument("fit_quadratic_surface: need more runs than second-order coefficients");
    linalg::DenseMatrix<double> model(n,p);
    for (std::size_t r = 0; r < n; ++r) {
        std::size_t c = 0;
        model(r,c++) = 1;
        for (std::size_t j = 0; j < k; ++j)
            model(r,c++) = x(r,j);
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = i+1; j < k; ++j)
                model(r,c++) = x(r,i)*x(r,j);
        for (std::size_t j = 0; j < k; ++j)
            model(r,c++) = x(r,j)*x(r,j);
    }
    const auto fit = linalg::least_squares(model,y);
    if (fit.rank < p)
        throw std::runtime_error("fit_quadratic_surface: design cannot estimate all second-order terms");
    QuadraticSurface s;
    s.coefficients = fit.coefficients;
    s.intercept = s.coefficients[0];
    s.linear.assign(s.coefficients.begin()+1,s.coefficients.begin()+1+static_cast<std::ptrdiff_t>(k));
    s.quadratic = linalg::DenseMatrix<double>(k,k);
    std::size_t c = 1+k;
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = i+1; j < k; ++j) {
            s.quadratic(i,j) = s.quadratic(j,i) = s.coefficients[c++]/2;
        }
    for (std::size_t j = 0; j < k; ++j)
        s.quadratic(j,j) = s.coefficients[c++];
    double mean = 0, total = 0;
    for (const double v : y)
        mean += v/static_cast<double>(n);
    for (const double v : y)
        total += (v-mean)*(v-mean);
    const double rss = fit.residual_norm*fit.residual_norm;
    s.residual_sd = std::sqrt(rss/static_cast<double>(n-p));
    s.r_squared = total > 0 ? 1-rss/total : 1.0;

    const auto eig = linalg::symmetric_eigen(s.quadratic);
    s.eigenvalues = eig.eigenvalues;
    s.eigenvectors = eig.eigenvectors;
    double largest = 0;
    for (const double l : s.eigenvalues)
        largest = std::max(largest,std::abs(l));
    bool positive = false, negative = false, zero = false;
    s.stationary_point.assign(k,0);
    for (std::size_t m = 0; m < k; ++m) {
        const double l = s.eigenvalues[m];
        if (!(largest > 0) || std::abs(l) <= 1e-8*largest) {
            zero = true;
            continue;
        }
        (l > 0 ? positive : negative) = true;
        double projection = 0;
        for (std::size_t i = 0; i < k; ++i)
            projection += s.eigenvectors(i,m)*s.linear[i];
        for (std::size_t i = 0; i < k; ++i)
            s.stationary_point[i] -= .5*projection/l*s.eigenvectors(i,m);
    }
    s.kind = zero ? StationaryKind::ridge
                  : (positive && negative ? StationaryKind::saddle
                                          : (negative ? StationaryKind::maximum : StationaryKind::minimum));
    s.stationary_response = predict_quadratic_surface(s,s.stationary_point);
    return s;
}

double predict_quadratic_surface(const QuadraticSurface& surface, const std::vector<double>& x) {
    const std::size_t k = surface.linear.size();
    if (x.size() != k)
        throw std::invalid_argument("predict_quadratic_surface: point has the wrong number of factors");
    double value = surface.intercept;
    for (std::size_t i = 0; i < k; ++i) {
        value += surface.linear[i]*x[i];
        for (std::size_t j = 0; j < k; ++j)
            value += x[i]*surface.quadratic(i,j)*x[j];
    }
    return value;
}

} // namespace datamunge::stats
