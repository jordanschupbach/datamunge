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

/// Tensor-product B-spline basis.  A one-dimensional BSpline is a convenient
/// wrapper around BSplineBasis; with two or more bases it evaluates smooth
/// surfaces and higher-dimensional functions.  Design matrices are sparse:
/// each row has at most product(degree_i + 1) non-zero entries.
class BSpline {
public:
    explicit BSpline(BSplineBasis basis);
    explicit BSpline(std::vector<BSplineBasis> bases);

    static BSpline open_uniform(std::size_t degree, std::size_t basis_size,
                                double min_x = 0.0, double max_x = 1.0);
    static BSpline open_uniform(std::vector<std::size_t> degrees,
                                std::vector<std::size_t> basis_sizes,
                                std::vector<double> mins,
                                std::vector<double> maxs);

    std::size_t dimensions() const noexcept { return bases_.size(); }
    std::size_t basis_size() const noexcept { return basis_size_; }
    const std::vector<BSplineBasis>& bases() const noexcept { return bases_; }

    LocalBSplineEvaluation evaluate_point(const std::vector<double>& point) const;
    linalg::SparseCOO<double> evaluate(const std::vector<double>& points) const;
    linalg::SparseCOO<double> evaluate(const std::vector<std::vector<double>>& points) const;
    linalg::DenseMatrix<double> evaluate_dense(const std::vector<double>& points) const;
    linalg::DenseMatrix<double> evaluate_dense(const std::vector<std::vector<double>>& points) const;

private:
    std::vector<BSplineBasis> bases_;
    std::vector<std::size_t> strides_;
    std::size_t basis_size_{0};
};

} // namespace datamunge::fda
