#include <datamunge/fda/bspline.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace datamunge::fda {
namespace {

void validate_knots(std::size_t degree, const std::vector<double>& knots) {
    if (knots.empty())
        throw std::invalid_argument("BSplineBasis: knot vector must not be empty");
    if (knots.size() < 2 * (degree + 1))
        throw std::invalid_argument("BSplineBasis: knot vector is too short for the degree");
    if (!std::is_sorted(knots.begin(), knots.end()))
        throw std::invalid_argument("BSplineBasis: knot vector must be nondecreasing");
}

} // namespace

BSplineBasis::BSplineBasis(std::size_t degree, std::vector<double> knots)
    : degree_(degree), basis_size_(0), knots_(std::move(knots)) {
    validate_knots(degree_, knots_);
    basis_size_ = knots_.size() - degree_ - 1;
    if (basis_size_ == 0)
        throw std::invalid_argument("BSplineBasis: basis size must be positive");
    if (knots_[degree_] >= knots_[basis_size_])
        throw std::invalid_argument("BSplineBasis: support interval must have positive width");
}

BSplineBasis BSplineBasis::open_uniform(std::size_t degree,
                                        std::size_t basis_size,
                                        double      min_x,
                                        double      max_x) {
    if (basis_size == 0)
        throw std::invalid_argument("BSplineBasis::open_uniform: basis size must be positive");
    if (basis_size < degree + 1)
        throw std::invalid_argument(
            "BSplineBasis::open_uniform: basis size must be at least degree + 1");
    if (!(min_x < max_x))
        throw std::invalid_argument(
            "BSplineBasis::open_uniform: require min_x < max_x");

    std::vector<double> knots;
    knots.reserve(basis_size + degree + 1);

    for (std::size_t i = 0; i <= degree; ++i)
        knots.push_back(min_x);

    const std::size_t interior_count = basis_size - degree - 1;
    for (std::size_t j = 1; j <= interior_count; ++j)
        knots.push_back(min_x + (max_x - min_x) * static_cast<double>(j)
                                    / static_cast<double>(interior_count + 1));

    for (std::size_t i = 0; i <= degree; ++i)
        knots.push_back(max_x);

    return BSplineBasis(degree, std::move(knots));
}

std::pair<double, double> BSplineBasis::domain() const noexcept {
    return {knots_[degree_], knots_[basis_size_]};
}

std::size_t BSplineBasis::find_span(double x) const {
    const std::size_t n = basis_size_ - 1;
    if (x >= knots_[basis_size_])
        return n;

    std::size_t low = degree_;
    std::size_t high = basis_size_;
    std::size_t mid = (low + high) / 2;

    while (x < knots_[mid] || x >= knots_[mid + 1]) {
        if (x < knots_[mid])
            high = mid;
        else
            low = mid;
        mid = (low + high) / 2;
    }

    return mid;
}

LocalBSplineEvaluation BSplineBasis::evaluate_point(double x) const {
    const auto [xmin, xmax] = domain();
    if (x < xmin || x > xmax)
        return {};

    const std::size_t span = find_span(x);
    std::vector<double> left(degree_ + 1, 0.0);
    std::vector<double> right(degree_ + 1, 0.0);
    std::vector<double> basis(degree_ + 1, 0.0);
    basis[0] = 1.0;

    for (std::size_t j = 1; j <= degree_; ++j) {
        left[j] = x - knots_[span + 1 - j];
        right[j] = knots_[span + j] - x;
        double saved = 0.0;
        for (std::size_t r = 0; r < j; ++r) {
            const double denom = right[r + 1] + left[j - r];
            const double temp = (denom == 0.0) ? 0.0 : basis[r] / denom;
            basis[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        basis[j] = saved;
    }

    LocalBSplineEvaluation out;
    out.indices.reserve(degree_ + 1);
    out.values.reserve(degree_ + 1);

    const std::size_t first = span - degree_;
    for (std::size_t local = 0; local <= degree_; ++local) {
        const double value = basis[local];
        if (value == 0.0)
            continue;
        out.indices.push_back(first + local);
        out.values.push_back(value);
    }

    return out;
}

linalg::SparseCOO<double> BSplineBasis::evaluate(const std::vector<double>& points) const {
    linalg::SparseCOO<double> values(points.size(), basis_size_, points.size() * (degree_ + 1));

    for (std::size_t row = 0; row < points.size(); ++row) {
        const auto local = evaluate_point(points[row]);
        for (std::size_t k = 0; k < local.indices.size(); ++k)
            values.set(row, local.indices[k], local.values[k]);
    }

    values.compress();
    return values;
}

linalg::DenseMatrix<double> BSplineBasis::evaluate_dense(const std::vector<double>& points) const {
    linalg::DenseMatrix<double> values(points.size(), basis_size_, 0.0);

    for (std::size_t row = 0; row < points.size(); ++row) {
        const auto local = evaluate_point(points[row]);
        for (std::size_t k = 0; k < local.indices.size(); ++k)
            values(row, local.indices[k]) = local.values[k];
    }

    return values;
}

} // namespace datamunge::fda
