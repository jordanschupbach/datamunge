/// @file d_optimal.cpp
/// @brief Information matrices, greedy starts and Fedorov exchange for exact D-optimal designs.

#include <datamunge/stats/d_optimal.hpp>

#include <datamunge/linalg/eigen.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace datamunge::stats {
/// @brief Inverse information matrices and quadratic forms.
namespace {

/// @brief Dense double matrix used for model and information matrices.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Information matrix of selected candidate rows.
/// @param f Candidate model matrix.
/// @param rows Selected rows.
/// @param ridge Added to the diagonal.
/// @return \f$\sum f_if_i^\top+\text{ridge}\,I\f$.
Matrix information(const Matrix& f, const std::vector<std::size_t>& rows, double ridge) {
    const std::size_t p = f.cols();
    Matrix m(p,p);
    for (const auto r : rows)
        for (std::size_t a = 0; a < p; ++a)
            for (std::size_t b = 0; b < p; ++b)
                m(a,b) += f(r,a)*f(r,b);
    for (std::size_t a = 0; a < p; ++a)
        m(a,a) += ridge;
    return m;
}

/// @brief Inverse and log determinant of a symmetric matrix by eigendecomposition.
/// @param m Symmetric matrix.
/// @param inverse Set to the (pseudo-)inverse over eigenvalues above the rank threshold.
/// @param rank Set to the numerical rank.
/// @return Sum of log eigenvalues when full rank, otherwise minus infinity.
double invert(const Matrix& m, Matrix& inverse, std::size_t& rank) {
    const std::size_t p = m.rows();
    const auto eig = linalg::symmetric_eigen(m);
    const double top = std::max(0.0,eig.eigenvalues.front());
    inverse = Matrix(p,p);
    rank = 0;
    double log_det = 0;
    for (std::size_t k = 0; k < p; ++k) {
        const double l = eig.eigenvalues[k];
        if (!(top > 0) || l <= 1e-10*top)
            continue;
        ++rank;
        log_det += std::log(l);
        for (std::size_t a = 0; a < p; ++a)
            for (std::size_t b = 0; b < p; ++b)
                inverse(a,b) += eig.eigenvectors(a,k)*eig.eigenvectors(b,k)/l;
    }
    return rank == p ? log_det : -std::numeric_limits<double>::infinity();
}

/// @brief \f$f_i^\top Af_j\f$ for candidate rows i and j.
/// @param f Candidate model matrix.
/// @param a Symmetric p by p matrix.
/// @param i First row.
/// @param j Second row.
/// @return The bilinear form.
double form(const Matrix& f, const Matrix& a, std::size_t i, std::size_t j) {
    double s = 0;
    for (std::size_t u = 0; u < f.cols(); ++u) {
        double t = 0;
        for (std::size_t v = 0; v < f.cols(); ++v)
            t += a(u,v)*f(j,v);
        s += f(i,u)*t;
    }
    return s;
}

/// @brief Checks a model matrix for size and finiteness.
/// @param m Matrix.
/// @param what Name for the message.
/// @throws std::invalid_argument For an empty or nonfinite matrix.
void check(const Matrix& m, const char* what) {
    if (m.rows() == 0 || m.cols() == 0)
        throw std::invalid_argument(std::string("d-optimal design: empty ")+what);
    for (std::size_t r = 0; r < m.rows(); ++r)
        for (std::size_t c = 0; c < m.cols(); ++c)
            if (!std::isfinite(m(r,c)))
                throw std::invalid_argument(std::string("d-optimal design: nonfinite entry in ")+what);
}

} // namespace

DesignCriteria design_criteria(const linalg::DenseMatrix<double>& design,
                               const linalg::DenseMatrix<double>& candidates) {
    check(design,"design");
    check(candidates,"candidates");
    if (design.cols() != candidates.cols())
        throw std::invalid_argument("design_criteria: design and candidates need the same columns");
    const std::size_t n = design.rows(), p = design.cols();
    std::vector<std::size_t> all(n);
    for (std::size_t i = 0; i < n; ++i)
        all[i] = i;
    Matrix inverse;
    DesignCriteria c;
    c.runs = n;
    c.parameters = p;
    c.log_det = invert(information(design,all,0),inverse,c.rank);
    c.estimable = c.rank == p;
    if (!c.estimable) {
        c.a_criterion = std::numeric_limits<double>::infinity();
        return c;
    }
    const double nn = static_cast<double>(n), pp = static_cast<double>(p);
    c.d_criterion = std::exp(c.log_det/pp)/nn;
    double trace = 0;
    for (std::size_t a = 0; a < p; ++a)
        trace += inverse(a,a);
    c.a_criterion = nn*trace/pp;
    double worst = 0;
    for (std::size_t r = 0; r < candidates.rows(); ++r)
        worst = std::max(worst,form(candidates,inverse,r,r));
    c.g_efficiency = pp/(nn*worst);
    return c;
}

DOptimalDesign d_optimal_design(const linalg::DenseMatrix<double>& candidates, std::size_t runs,
                                const DOptimalOptions& options) {
    check(candidates,"candidates");
    const std::size_t big_n = candidates.rows(), p = candidates.cols();
    if (runs < p)
        throw std::invalid_argument("d_optimal_design: need at least as many runs as parameters");
    if (!options.allow_repeats && runs > big_n)
        throw std::invalid_argument("d_optimal_design: more runs than candidates without repeats");
    if (options.starts == 0)
        throw std::invalid_argument("d_optimal_design: need at least one start");
    {
        std::vector<std::size_t> all(big_n);
        for (std::size_t i = 0; i < big_n; ++i)
            all[i] = i;
        Matrix unused;
        std::size_t rank;
        (void)invert(information(candidates,all,0),unused,rank);
        if (rank < p)
            throw std::invalid_argument("d_optimal_design: candidate set cannot estimate every parameter");
    }
    double scale = 0;
    for (std::size_t r = 0; r < big_n; ++r)
        for (std::size_t c = 0; c < p; ++c)
            scale = std::max(scale,std::abs(candidates(r,c)));
    const double ridge = 1e-9*scale*scale;

    std::mt19937_64 random(options.seed);
    std::uniform_int_distribution<std::size_t> pick(0,big_n-1);
    DOptimalDesign best;
    double best_log_det = -std::numeric_limits<double>::infinity();
    for (std::size_t start = 0; start < options.starts; ++start) {
        // Random start; a tiny ridge keeps a singular start invertible until exchanges repair it.
        std::vector<std::size_t> rows;
        std::vector<int> used(big_n,0);
        if (options.allow_repeats) {
            for (std::size_t i = 0; i < runs; ++i)
                rows.push_back(pick(random));
        } else {
            std::vector<std::size_t> order(big_n);
            for (std::size_t i = 0; i < big_n; ++i)
                order[i] = i;
            std::ranges::shuffle(order,random);
            rows.assign(order.begin(),order.begin()+static_cast<std::ptrdiff_t>(runs));
        }
        for (const auto r : rows)
            ++used[r];
        Matrix inverse;
        std::size_t rank;
        (void)invert(information(candidates,rows,ridge),inverse,rank);
        std::size_t exchanges = 0;
        for (; exchanges < options.max_exchanges; ++exchanges) {
            std::vector<double> dj(big_n);
            for (std::size_t j = 0; j < big_n; ++j)
                dj[j] = form(candidates,inverse,j,j);
            double gain = 0;
            std::size_t out = 0, in = 0;
            for (std::size_t i = 0; i < runs; ++i) {
                const double di = dj[rows[i]];
                for (std::size_t j = 0; j < big_n; ++j) {
                    if (j == rows[i] || (!options.allow_repeats && used[j]))
                        continue;
                    const double dij = form(candidates,inverse,rows[i],j);
                    const double delta = dj[j]-di-di*dj[j]+dij*dij;
                    if (delta > gain) {
                        gain = delta;
                        out = i;
                        in = j;
                    }
                }
            }
            if (gain <= 1e-10)
                break;
            --used[rows[out]];
            rows[out] = in;
            ++used[in];
            (void)invert(information(candidates,rows,ridge),inverse,rank);
        }
        const double log_det = invert(information(candidates,rows,0),inverse,rank);
        if (rank < p)
            continue;
        if (log_det > best_log_det+1e-10) {
            best_log_det = log_det;
            best.rows = rows;
            best.exchanges = exchanges;
        }
    }
    if (best.rows.empty())
        throw std::runtime_error("d_optimal_design: no start reached a nonsingular design");
    std::ranges::sort(best.rows);
    Matrix chosen(runs,p);
    for (std::size_t i = 0; i < runs; ++i)
        for (std::size_t c = 0; c < p; ++c)
            chosen(i,c) = candidates(best.rows[i],c);
    best.criteria = design_criteria(chosen,candidates);
    return best;
}

} // namespace datamunge::stats
