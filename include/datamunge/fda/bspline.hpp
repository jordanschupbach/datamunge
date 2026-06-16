#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/linalg/sparse_coo.hpp>

namespace datamunge::fda {

struct LocalBSplineEvaluation {
    std::vector<std::size_t> indices;
    std::vector<double>      values;
};

class BSplineBasis {
public:
    BSplineBasis(std::size_t degree, std::vector<double> knots);

    static BSplineBasis open_uniform(std::size_t degree,
                                     std::size_t basis_size,
                                     double      min_x = 0.0,
                                     double      max_x = 1.0);

    std::size_t degree() const noexcept { return degree_; }
    std::size_t basis_size() const noexcept { return basis_size_; }
    const std::vector<double>& knots() const noexcept { return knots_; }

    std::pair<double, double> domain() const noexcept;

    LocalBSplineEvaluation evaluate_point(double x) const;

    linalg::SparseCOO<double> evaluate(const std::vector<double>& points) const;

    linalg::DenseMatrix<double> evaluate_dense(const std::vector<double>& points) const;

private:
    std::size_t         degree_{0};
    std::size_t         basis_size_{0};
    std::vector<double> knots_;

    std::size_t find_span(double x) const;
};

} // namespace datamunge::fda
