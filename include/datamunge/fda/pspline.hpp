#pragma once

#include <cstddef>
#include <vector>

#include <datamunge/fda/bspline.hpp>
#include <datamunge/linalg/dense_matrix.hpp>

namespace datamunge::fda {

struct PSplineOptions {
    double      lambda{1.0};
    std::size_t penalty_order{2};
};

struct PSplineFitResult {
    BSplineBasis               basis;
    std::vector<double>        coefficients;
    std::vector<double>        fitted;
    std::vector<double>        residuals;
    linalg::DenseMatrix<double> penalty_matrix;
    double                     lambda{1.0};
    std::size_t                penalty_order{2};
    double                     residual_sum_of_squares{0.0};
    double                     total_sum_of_squares{0.0};
    double                     penalty{0.0};
    double                     objective{0.0};
    double                     r_squared{0.0};
    double                     adjusted_r_squared{0.0};
    double                     effective_degrees_of_freedom{0.0};
    double                     residual_degrees_of_freedom{0.0};
    std::size_t                observations{0};

    std::vector<double> predict(const std::vector<double>& points) const;
};

linalg::DenseMatrix<double> pspline_penalty_matrix(std::size_t coefficient_count,
                                                   std::size_t penalty_order);

PSplineFitResult fit_pspline(const std::vector<double>& x,
                             const std::vector<double>& y,
                             const BSplineBasis&        basis,
                             PSplineOptions             opts = {});

PSplineFitResult fit_pspline(const std::vector<double>& x,
                             const std::vector<double>& y,
                             std::size_t                degree,
                             std::size_t                basis_size,
                             PSplineOptions             opts = {});

} // namespace datamunge::fda
