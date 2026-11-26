/// @file survey_tables.cpp
/// @brief Rao–Scott first- and second-order corrections and Wald tests for two-way survey tables.

#include <datamunge/stats/survey_tables.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <functional>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Internal helpers for survey table tests.
namespace {

/// @brief Dense matrix of doubles used throughout the table computations.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Product of two matrices.
/// @param a Left factor.
/// @param b Right factor.
/// @return a b.
Matrix multiply(const Matrix& a, const Matrix& b) {
    Matrix c(a.rows(),b.cols());
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t k = 0; k < a.cols(); ++k)
            for (std::size_t j = 0; j < b.cols(); ++j)
                c(i,j) += a(i,k)*b(k,j);
    return c;
}

/// @brief Transpose.
/// @param a Matrix.
/// @return Its transpose.
Matrix transpose(const Matrix& a) {
    Matrix t(a.cols(),a.rows());
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < a.cols(); ++j)
            t(j,i) = a(i,j);
    return t;
}

/// @brief Solves the symmetric positive-definite system \f$AX=B\f$ column by column.
/// @param a Symmetric positive-definite matrix.
/// @param b Right-hand sides.
/// @param what Message for a singular matrix.
/// @return The solution X.
/// @throws std::runtime_error If a is not positive definite.
Matrix solve(const Matrix& a, const Matrix& b, const char* what) {
    const auto factor = linalg::cholesky(a);
    if (!factor.ok)
        throw std::runtime_error(what);
    Matrix x(b.rows(),b.cols());
    std::vector<double> column(b.rows());
    for (std::size_t j = 0; j < b.cols(); ++j) {
        for (std::size_t i = 0; i < b.rows(); ++i)
            column[i] = b(i,j);
        const auto s = factor.solve(column);
        for (std::size_t i = 0; i < b.rows(); ++i)
            x(i,j) = s[i];
    }
    return x;
}

/// @brief Eigenvalues of a symmetric matrix by cyclic Jacobi rotations, in decreasing order.
/// @param a Symmetric matrix.
/// @return Eigenvalues.
std::vector<double> symmetric_eigenvalues(Matrix a) {
    const std::size_t n = a.rows();
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0;
        for (std::size_t p = 0; p < n; ++p)
            for (std::size_t q = p+1; q < n; ++q)
                off += a(p,q)*a(p,q);
        if (off < 1e-30)
            break;
        for (std::size_t p = 0; p < n; ++p)
            for (std::size_t q = p+1; q < n; ++q) {
                if (a(p,q) == 0)
                    continue;
                const double theta = (a(q,q)-a(p,p))/(2*a(p,q));
                const double t = (theta >= 0 ? 1.0 : -1.0)/(std::abs(theta)+std::sqrt(theta*theta+1));
                const double c = 1/std::sqrt(t*t+1), s = t*c;
                for (std::size_t k = 0; k < n; ++k) {
                    const double akp = a(k,p), akq = a(k,q);
                    a(k,p) = c*akp-s*akq;
                    a(k,q) = s*akp+c*akq;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    const double apk = a(p,k), aqk = a(q,k);
                    a(p,k) = c*apk-s*aqk;
                    a(q,k) = s*apk+c*aqk;
                }
            }
    }
    std::vector<double> e(n);
    for (std::size_t i = 0; i < n; ++i)
        e[i] = a(i,i);
    std::ranges::sort(e,std::greater<>());
    return e;
}

/// @brief Maps arbitrary category labels to dense indices in increasing label order.
/// @param x Labels.
/// @param levels Receives the number of distinct labels.
/// @return Dense index per unit.
std::vector<std::size_t> dense(const std::vector<std::size_t>& x, std::size_t& levels) {
    std::map<std::size_t,std::size_t> code;
    for (const std::size_t v : x)
        code.emplace(v,0);
    std::size_t next = 0;
    for (auto& [label,index] : code)
        index = next++;
    levels = next;
    std::vector<std::size_t> out;
    for (const std::size_t v : x)
        out.push_back(code[v]);
    return out;
}

/// @brief Upper-tail F probability (chi-square when df2 is infinite).
/// @param x Statistic.
/// @param df1 Numerator degrees of freedom.
/// @param df2 Denominator degrees of freedom.
/// @return p-value.
double upper(double x, double df1, double df2) {
    if (!(x > 0))
        return 1;
    if (std::isinf(df2))
        return 1-random::chi_squared_cdf(x*df1,df1);
    return 1-random::f_cdf(x,df1,df2);
}

} // namespace

SurveyTableTests survey_table_test(const SurveyDesign& design, const std::vector<std::size_t>& rows,
                                   const std::vector<std::size_t>& cols) {
    const std::size_t n = design.weights.size();
    if (rows.size() != n || cols.size() != n)
        throw std::invalid_argument("survey_table_test: need a row and column category per unit");
    SurveyTableTests out;
    std::size_t R = 0, C = 0;
    const auto r = dense(rows,R), c = dense(cols,C);
    if (R < 2 || C < 2)
        throw std::invalid_argument("survey_table_test: need at least two observed categories in each variable");
    out.rows = R;
    out.cols = C;
    out.design_df = survey_degrees_of_freedom(design);
    if (!(out.design_df > 0))
        throw std::invalid_argument("survey_table_test: the design has no degrees of freedom");
    const std::size_t K = R*C, df = (R-1)*(C-1);
    const auto& w = design.weights;
    double total = 0;
    out.proportions.assign(K,0.0);
    for (std::size_t i = 0; i < n; ++i) {
        out.proportions[r[i]+R*c[i]] += w[i];
        total += w[i];
    }
    for (double& p : out.proportions)
        p /= total;
    std::vector<double> row_p(R,0.0), col_p(C,0.0);
    for (std::size_t a = 0; a < R; ++a)
        for (std::size_t b = 0; b < C; ++b) {
            row_p[a] += out.proportions[a+R*b];
            col_p[b] += out.proportions[a+R*b];
        }
    const double nn = static_cast<double>(n);
    for (std::size_t a = 0; a < R; ++a)
        for (std::size_t b = 0; b < C; ++b) {
            const double expected = nn*row_p[a]*col_p[b], observed = nn*out.proportions[a+R*b];
            out.pearson += (observed-expected)*(observed-expected)/expected;
        }
    out.naive_p_value = upper(out.pearson/static_cast<double>(df),static_cast<double>(df),
                              std::numeric_limits<double>::infinity());

    // Linearized covariance of the cell proportions.
    Matrix scores(n,K);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < K; ++k)
            scores(i,k) = w[i]*((r[i]+R*c[i] == k ? 1.0 : 0.0)-out.proportions[k])/total;
    const Matrix V = survey_score_covariance(design,scores);

    // Interaction contrasts: residuals of the interaction indicators on the main-effects design.
    Matrix X1(K,R+C-1), X12(K,df);
    for (std::size_t a = 0; a < R; ++a)
        for (std::size_t b = 0; b < C; ++b) {
            const std::size_t k = a+R*b;
            X1(k,0) = 1;
            if (a > 0)
                X1(k,a) = 1;
            if (b > 0)
                X1(k,R-1+b) = 1;
            if (a > 0 && b > 0)
                X12(k,(a-1)+(R-1)*(b-1)) = 1;
        }
    const Matrix X1t = transpose(X1);
    const Matrix coef = solve(multiply(X1t,X1),multiply(X1t,X12),"survey_table_test: singular main-effects design");
    Matrix Cm = X12;
    const Matrix fitted = multiply(X1,coef);
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < df; ++j)
            Cm(k,j) -= fitted(k,j);
    Matrix iDC = Cm;
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < df; ++j)
            iDC(k,j) = out.proportions[k] > 0 ? Cm(k,j)/out.proportions[k] : 0.0;
    Matrix denominator = multiply(transpose(Cm),iDC);
    for (std::size_t a = 0; a < df; ++a)
        for (std::size_t b = 0; b < df; ++b)
            denominator(a,b) /= nn;
    const Matrix numerator = multiply(transpose(iDC),multiply(V,iDC));
    // Eigenvalues of denominator^{-1} numerator via the symmetric form L^{-1} numerator L^{-T}.
    const auto factor = linalg::cholesky(denominator);
    if (!factor.ok)
        throw std::runtime_error("survey_table_test: singular contrast matrix (empty cells)");
    const Matrix delta = solve(denominator,numerator,"survey_table_test: singular contrast matrix (empty cells)");
    double trace = 0, trace2 = 0;
    for (std::size_t a = 0; a < df; ++a) {
        trace += delta(a,a);
        for (std::size_t b = 0; b < df; ++b)
            trace2 += delta(a,b)*delta(b,a);
    }
    {
        // Symmetrize: with denominator = L L^T, L^{-1} numerator L^{-T} has the same eigenvalues as delta.
        const Matrix& L = factor.L;
        Matrix A(df,df), B(df,df);
        for (std::size_t j = 0; j < df; ++j) // A = L^{-1} numerator by forward substitution
            for (std::size_t i = 0; i < df; ++i) {
                double s = numerator(i,j);
                for (std::size_t k = 0; k < i; ++k)
                    s -= L(i,k)*A(k,j);
                A(i,j) = s/L(i,i);
            }
        for (std::size_t j = 0; j < df; ++j) // B = L^{-1} A^T
            for (std::size_t i = 0; i < df; ++i) {
                double s = A(j,i);
                for (std::size_t k = 0; k < i; ++k)
                    s -= L(i,k)*B(k,j);
                B(i,j) = s/L(i,i);
            }
        out.design_effects = symmetric_eigenvalues(B);
    }
    const double inf = std::numeric_limits<double>::infinity(), dfd = static_cast<double>(df);
    out.first_order.statistic = out.pearson/(trace/dfd);
    out.first_order.df1 = dfd;
    out.first_order.df2 = inf;
    out.first_order.p_value = upper(out.first_order.statistic/dfd,dfd,inf);
    const double d0 = trace*trace/trace2;
    out.second_order.statistic = out.pearson/trace;
    out.second_order.df1 = d0;
    out.second_order.df2 = d0*out.design_df;
    out.second_order.p_value = upper(out.second_order.statistic,d0,d0*out.design_df);

    // Wald test on Y_rc = N_rc - N_r. N_.c / N for r, c >= 1, with linearized influence values.
    Matrix wscores(n,df);
    std::vector<double> Y(df);
    for (std::size_t a = 1; a < R; ++a)
        for (std::size_t b = 1; b < C; ++b) {
            const std::size_t j = (a-1)+(R-1)*(b-1);
            Y[j] = total*(out.proportions[a+R*b]-row_p[a]*col_p[b]);
            for (std::size_t i = 0; i < n; ++i)
                wscores(i,j) = w[i]*((r[i] == a && c[i] == b ? 1.0 : 0.0)-(r[i] == a ? col_p[b] : 0.0)-
                                     (c[i] == b ? row_p[a] : 0.0)+row_p[a]*col_p[b]);
        }
    const Matrix VW = survey_score_covariance(design,wscores);
    Matrix Ym(df,1);
    for (std::size_t j = 0; j < df; ++j)
        Ym(j,0) = Y[j];
    const Matrix sol = solve(VW,Ym,"survey_table_test: singular Wald covariance (too few PSUs for the table)");
    double W = 0;
    for (std::size_t j = 0; j < df; ++j)
        W += Y[j]*sol(j,0);
    out.wald.statistic = W/dfd;
    out.wald.df1 = dfd;
    out.wald.df2 = out.design_df;
    out.wald.p_value = upper(out.wald.statistic,dfd,out.design_df);
    const double adjusted_df = out.design_df-dfd+1;
    out.adjusted_wald.df1 = dfd;
    out.adjusted_wald.df2 = adjusted_df;
    if (adjusted_df > 0) {
        out.adjusted_wald.statistic = W*adjusted_df/(dfd*out.design_df);
        out.adjusted_wald.p_value = upper(out.adjusted_wald.statistic,dfd,adjusted_df);
    } else {
        out.adjusted_wald.statistic = std::numeric_limits<double>::quiet_NaN();
        out.adjusted_wald.p_value = std::numeric_limits<double>::quiet_NaN();
    }
    return out;
}

} // namespace datamunge::stats
