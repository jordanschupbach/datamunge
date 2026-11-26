/// @file cointegration.cpp
/// @brief Johansen eigenproblem, Osterwald-Lenum tables, reduced-rank VECM estimation and forecasts.

#include <datamunge/stats/cointegration.hpp>

#include <datamunge/linalg/eigen.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace datamunge::stats {
/// @brief Small dense-matrix helpers and critical-value tables.
namespace {

/// @brief Matrix alias used throughout.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Osterwald-Lenum (1992) trace critical values (90%, 95%, 99%) for 1..12 free dimensions and
/// det_order -1, 0, 1, as tabulated in statsmodels.
constexpr double trace_table[3][12][3] = {
    {{2.9762, 4.1296, 6.9406},
     {10.4741, 12.3212, 16.364},
     {21.7781, 24.2761, 29.5147},
     {37.0339, 40.1749, 46.5716},
     {56.2839, 60.0627, 67.6367},
     {79.5329, 83.9383, 92.7136},
     {106.7351, 111.7797, 121.7375},
     {137.9954, 143.6691, 154.7977},
     {173.2292, 179.5199, 191.8122},
     {212.4721, 219.4051, 232.8291},
     {255.6732, 263.2603, 277.9962},
     {302.9054, 311.1288, 326.9716}},
    {{2.7055, 3.8415, 6.6349},
     {13.4294, 15.4943, 19.9349},
     {27.0669, 29.7961, 35.4628},
     {44.4929, 47.8545, 54.6815},
     {65.8202, 69.8189, 77.8202},
     {91.109, 95.7542, 104.9637},
     {120.3673, 125.6185, 135.9825},
     {153.6341, 159.529, 171.0905},
     {190.8714, 197.3772, 210.0366},
     {232.103, 239.2468, 253.2526},
     {277.374, 285.1402, 300.2821},
     {326.5354, 334.9795, 351.215}},
    {{2.7055, 3.8415, 6.6349},
     {16.1619, 18.3985, 23.1485},
     {32.0645, 35.0116, 41.0815},
     {51.6492, 55.2459, 62.5202},
     {75.1027, 79.3422, 87.7748},
     {102.4674, 107.3429, 116.9829},
     {133.7852, 139.278, 150.0778},
     {169.0618, 175.1584, 187.1891},
     {208.3582, 215.1268, 228.2226},
     {251.6293, 259.0267, 273.3838},
     {298.8836, 306.8988, 322.4264},
     {350.1125, 358.719, 375.3203}}};

/// @brief Maximum-eigenvalue critical values, laid out as `trace_table`.
constexpr double max_table[3][12][3] = {
    {{2.9762, 4.1296, 6.9406},
     {9.4748, 11.2246, 15.0923},
     {15.7175, 17.7961, 22.2519},
     {21.837, 24.1592, 29.0609},
     {27.916, 30.4428, 35.7359},
     {33.9271, 36.6301, 42.2333},
     {39.9085, 42.7679, 48.6606},
     {45.893, 48.8795, 55.0335},
     {51.8528, 54.9629, 61.3449},
     {57.7954, 61.0404, 67.6415},
     {63.7248, 67.0756, 73.8856},
     {69.6513, 73.0946, 80.0937}},
    {{2.7055, 3.8415, 6.6349},
     {12.2971, 14.2639, 18.52},
     {18.8928, 21.1314, 25.865},
     {25.1236, 27.5858, 32.7172},
     {31.2379, 33.8777, 39.3693},
     {37.2786, 40.0763, 45.8662},
     {43.2947, 46.2299, 52.3069},
     {49.2855, 52.3622, 58.6634},
     {55.2412, 58.4332, 64.996},
     {61.2041, 64.504, 71.2525},
     {67.1307, 70.5392, 77.4877},
     {73.0563, 76.5734, 83.7105}},
    {{2.7055, 3.8415, 6.6349},
     {15.0006, 17.1481, 21.7465},
     {21.8731, 24.2522, 29.2631},
     {28.2398, 30.8151, 36.193},
     {34.4202, 37.1646, 42.8612},
     {40.5244, 43.4183, 49.4095},
     {46.5583, 49.5875, 55.8171},
     {52.5858, 55.7302, 62.1741},
     {58.5316, 61.8051, 68.503},
     {64.5292, 67.904, 74.7434},
     {70.463, 73.9355, 81.0678},
     {76.4081, 79.9878, 87.2395}}};

/// @brief Product a b.
/// @param a Left factor.
/// @param b Right factor.
/// @return a b.
Matrix mul(const Matrix& a, const Matrix& b) {
    Matrix c(a.rows(),b.cols());
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t k = 0; k < a.cols(); ++k) {
            const double v = a(i,k);
            for (std::size_t j = 0; j < b.cols(); ++j)
                c(i,j) += v*b(k,j);
        }
    return c;
}

/// @brief Transpose.
/// @param a Matrix.
/// @return a transposed.
Matrix tr(const Matrix& a) {
    Matrix t(a.cols(),a.rows());
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < a.cols(); ++j)
            t(j,i) = a(i,j);
    return t;
}

/// @brief Function of a symmetric positive definite matrix through its eigen decomposition.
/// @param a Symmetric positive definite matrix.
/// @param power Exponent applied to the eigenvalues (-1 for the inverse, -0.5 for the inverse root).
/// @param logdet When not null, set to the log determinant.
/// @return \f$a^{\text{power}}\f$.
/// @throws std::invalid_argument When a is numerically singular.
Matrix spd_power(const Matrix& a, double power, double* logdet = nullptr) {
    const std::size_t k = a.rows();
    Matrix out(k,k);
    if (logdet)
        *logdet = 0;
    if (k == 0)
        return out;
    const auto eig = linalg::symmetric_eigen(a);
    if (!(eig.eigenvalues.back() > 1e-13*eig.eigenvalues.front()))
        throw std::invalid_argument("cointegration: singular moment matrix (collinear data?)");
    for (std::size_t e = 0; e < k; ++e) {
        const double f = std::pow(eig.eigenvalues[e],power);
        if (logdet)
            *logdet += std::log(eig.eigenvalues[e]);
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j)
                out(i,j) += eig.eigenvectors(i,e)*eig.eigenvectors(j,e)*f;
    }
    return out;
}

/// @brief Residuals of the columns of y after least-squares projection on the columns of x.
/// @param y n by a responses.
/// @param x n by b regressors (b may be zero).
/// @return n by a residuals.
Matrix residualize(const Matrix& y, const Matrix& x) {
    if (x.cols() == 0)
        return y;
    const Matrix xt = tr(x);
    const Matrix coef = mul(spd_power(mul(xt,x),-1),mul(xt,y));
    Matrix r = y;
    const Matrix fit = mul(x,coef);
    for (std::size_t i = 0; i < r.rows(); ++i)
        for (std::size_t j = 0; j < r.cols(); ++j)
            r(i,j) -= fit(i,j);
    return r;
}

/// @brief Polynomial trend regressors in \f$s\in[-1,1]\f$ (statsmodels `np.vander(linspace(-1, 1, n))`).
/// @param n Rows.
/// @param order -1 (none), 0 (constant) or 1 (constant and trend).
/// @return n by (order + 1) matrix.
Matrix trend_basis(std::size_t n, int order) {
    Matrix x(n,static_cast<std::size_t>(order+1));
    for (std::size_t t = 0; t < n; ++t) {
        const double s = n > 1 ? -1+2*static_cast<double>(t)/static_cast<double>(n-1) : 0.0;
        for (int c = 0; c <= order; ++c)
            x(t,static_cast<std::size_t>(c)) = std::pow(s,order-c);
    }
    return x;
}

/// @brief Validates the data matrix.
/// @param y Observations.
/// @param what Caller name for messages.
/// @throws std::invalid_argument For nonfinite values or fewer than two series.
void check_data(const Matrix& y, const char* what) {
    if (y.cols() < 2)
        throw std::invalid_argument(std::string(what)+": need at least two series");
    for (std::size_t i = 0; i < y.rows(); ++i)
        for (std::size_t j = 0; j < y.cols(); ++j)
            if (!std::isfinite(y(i,j)))
                throw std::invalid_argument(std::string(what)+": observations must be finite");
}

/// @brief Solves the symmetric-definite problem \f$S_{k0}S_{00}^{-1}S_{0k}v=\lambda S_{kk}v\f$.
/// @param s00 Moments of the differenced residuals.
/// @param s0k Cross moments.
/// @param skk Moments of the level residuals.
/// @param values Set to the eigenvalues, decreasing.
/// @return Eigenvectors as columns with \f$V^\top S_{kk}V=I\f$.
Matrix canonical(const Matrix& s00, const Matrix& s0k, const Matrix& skk, std::vector<double>& values) {
    const Matrix root = spd_power(skk,-.5);
    const Matrix m = mul(mul(root,mul(mul(tr(s0k),spd_power(s00,-1)),s0k)),root);
    Matrix sym = m;
    for (std::size_t i = 0; i < m.rows(); ++i)
        for (std::size_t j = 0; j < m.cols(); ++j)
            sym(i,j) = .5*(m(i,j)+m(j,i));
    const auto eig = linalg::symmetric_eigen(sym);
    values = eig.eigenvalues;
    return mul(root,eig.eigenvectors);
}

} // namespace

JohansenTest johansen_test(const linalg::DenseMatrix<double>& y, const JohansenOptions& options) {
    check_data(y,"johansen_test");
    const std::size_t n = y.rows(), K = y.cols(), k = options.lagged_differences;
    const int order = options.deterministic == JohansenDeterministic::none ? -1
                      : options.deterministic == JohansenDeterministic::constant ? 0 : 1;
    if (n < k+K+4)
        throw std::invalid_argument("johansen_test: too few observations");
    const Matrix levels = residualize(y,trend_basis(n,order));
    const std::size_t T = n-1-k;
    const int f = order > -1 ? 0 : -1;
    Matrix dx(T,K), lx(T,K), z(T,K*k);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t j = 0; j < K; ++j) {
            dx(t,j) = levels(t+k+1,j)-levels(t+k,j);
            lx(t,j) = levels(t+k,j);
            for (std::size_t l = 1; l <= k; ++l)
                z(t,(l-1)*K+j) = levels(t+k+1-l,j)-levels(t+k-l,j);
        }
    const Matrix basis = trend_basis(T,f);
    const Matrix zd = residualize(z,basis);
    const Matrix r0 = residualize(residualize(dx,basis),zd);
    const Matrix rk = residualize(residualize(lx,basis),zd);
    const double Td = static_cast<double>(T);
    Matrix s00 = mul(tr(r0),r0), sk0 = mul(tr(rk),r0), skk = mul(tr(rk),rk);
    for (std::size_t i = 0; i < K; ++i)
        for (std::size_t j = 0; j < K; ++j) {
            s00(i,j) /= Td;
            sk0(i,j) /= Td;
            skk(i,j) /= Td;
        }
    JohansenTest r;
    r.observations = T;
    r.eigenvectors = canonical(s00,tr(sk0),skk,r.eigenvalues);
    for (std::size_t c = 0; c < K; ++c) {
        std::size_t i = 0;
        while (i+1 < K && r.eigenvectors(i,c) == 0)
            ++i;
        if (r.eigenvectors(i,c) < 0)
            for (std::size_t a = 0; a < K; ++a)
                r.eigenvectors(a,c) = -r.eigenvectors(a,c);
    }
    const auto table = static_cast<std::size_t>(order+1);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    r.trace_rank = r.max_eigenvalue_rank = K;
    for (std::size_t i = 0; i < K; ++i) {
        double s = 0;
        for (std::size_t j = i; j < K; ++j)
            s += std::log1p(-r.eigenvalues[j]);
        r.trace.push_back(-Td*s);
        r.max_eigenvalue.push_back(-Td*std::log1p(-r.eigenvalues[i]));
        const std::size_t free = K-i;
        std::array<double,3> ct{nan,nan,nan}, cm{nan,nan,nan};
        if (free <= 12)
            for (int q = 0; q < 3; ++q) {
                ct[static_cast<std::size_t>(q)] = trace_table[table][free-1][q];
                cm[static_cast<std::size_t>(q)] = max_table[table][free-1][q];
            }
        r.trace_critical.push_back(ct);
        r.max_eigenvalue_critical.push_back(cm);
    }
    for (std::size_t i = 0; i < K; ++i)
        if (!(r.trace[i] > r.trace_critical[i][1])) {
            r.trace_rank = i;
            break;
        }
    for (std::size_t i = 0; i < K; ++i)
        if (!(r.max_eigenvalue[i] > r.max_eigenvalue_critical[i][1])) {
            r.max_eigenvalue_rank = i;
            break;
        }
    return r;
}

VecmModel fit_vecm(const linalg::DenseMatrix<double>& y, const VecmOptions& options) {
    check_data(y,"fit_vecm");
    const std::size_t n = y.rows(), K = y.cols(), k = options.lagged_differences, r = options.rank;
    if (r < 1 || r >= K)
        throw std::invalid_argument("fit_vecm: rank must lie in 1..K-1");
    const std::size_t p = k+1;
    const bool ci = options.constant == VecmTerm::inside, co = options.constant == VecmTerm::outside;
    const bool li = options.trend == VecmTerm::inside, lo = options.trend == VecmTerm::outside;
    const std::size_t din = ci+li, dout = co+lo, m1 = K+din, mx = K*k+dout;
    if (n <= p+mx+m1+1)
        throw std::invalid_argument("fit_vecm: too few observations");
    const std::size_t T = n-p;
    Matrix dy(T,K), y1(T,m1), dxm(T,mx);
    for (std::size_t t = 0; t < T; ++t) {
        const std::size_t now = p+t; // row of y_t
        for (std::size_t j = 0; j < K; ++j) {
            dy(t,j) = y(now,j)-y(now-1,j);
            y1(t,j) = y(now-1,j);
            for (std::size_t l = 1; l <= k; ++l)
                dxm(t,(l-1)*K+j) = y(now-l,j)-y(now-l-1,j);
        }
        std::size_t c = K;
        if (ci)
            y1(t,c++) = 1;
        if (li)
            y1(t,c) = static_cast<double>(t+p);
        c = K*k;
        if (co)
            dxm(t,c++) = 1;
        if (lo)
            dxm(t,c) = static_cast<double>(t+p+1);
    }
    const Matrix r0 = residualize(dy,dxm), r1 = residualize(y1,dxm);
    const double Td = static_cast<double>(T);
    Matrix s00 = mul(tr(r0),r0), s01 = mul(tr(r0),r1), s11 = mul(tr(r1),r1);
    for (auto* s : {&s00,&s01,&s11})
        for (std::size_t i = 0; i < s->rows(); ++i)
            for (std::size_t j = 0; j < s->cols(); ++j)
                (*s)(i,j) /= Td;
    VecmModel v;
    v.rank = r;
    v.lagged_differences = k;
    v.variables = K;
    v.observations = T;
    v.constant = options.constant;
    v.trend = options.trend;
    const Matrix vectors = canonical(s00,s01,s11,v.eigenvalues);
    v.eigenvalues.resize(std::min(v.eigenvalues.size(),K));
    // Cointegrating vectors (with restricted deterministic rows), normalized on the first r rows.
    Matrix b(m1,r), top(r,r);
    for (std::size_t i = 0; i < m1; ++i)
        for (std::size_t j = 0; j < r; ++j)
            b(i,j) = vectors(i,j);
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < r; ++j)
            top(i,j) = b(i,j);
    // Invert the r by r leading block by Gauss-Jordan with partial pivoting.
    Matrix inv(r,r);
    {
        Matrix a = top;
        for (std::size_t i = 0; i < r; ++i)
            inv(i,i) = 1;
        for (std::size_t c = 0; c < r; ++c) {
            std::size_t piv = c;
            for (std::size_t i = c+1; i < r; ++i)
                if (std::abs(a(i,c)) > std::abs(a(piv,c)))
                    piv = i;
            if (std::abs(a(piv,c)) < 1e-14)
                throw std::invalid_argument("fit_vecm: cannot normalize the cointegrating vectors");
            for (std::size_t j = 0; j < r; ++j) {
                std::swap(a(c,j),a(piv,j));
                std::swap(inv(c,j),inv(piv,j));
            }
            const double d = a(c,c);
            for (std::size_t j = 0; j < r; ++j) {
                a(c,j) /= d;
                inv(c,j) /= d;
            }
            for (std::size_t i = 0; i < r; ++i)
                if (i != c) {
                    const double f = a(i,c);
                    for (std::size_t j = 0; j < r; ++j) {
                        a(i,j) -= f*a(c,j);
                        inv(i,j) -= f*inv(c,j);
                    }
                }
        }
    }
    const Matrix beta_full = mul(b,inv);
    v.alpha = mul(mul(s01,beta_full),spd_power(mul(mul(tr(beta_full),s11),beta_full),-1));
    v.beta = Matrix(K,r);
    v.beta_deterministic = Matrix(din,r);
    for (std::size_t i = 0; i < m1; ++i)
        for (std::size_t j = 0; j < r; ++j)
            (i < K ? v.beta(i,j) : v.beta_deterministic(i-K,j)) = beta_full(i,j);
    // Short-run and outside deterministic coefficients given beta: regress dy - alpha beta' y1 on dx.
    const Matrix ecm = mul(y1,beta_full); // T by r
    Matrix target = dy;
    {
        const Matrix a = mul(ecm,tr(v.alpha));
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t j = 0; j < K; ++j)
                target(t,j) -= a(t,j);
    }
    Matrix shortrun(K,mx);
    if (mx > 0)
        shortrun = tr(mul(spd_power(mul(tr(dxm),dxm),-1),mul(tr(dxm),target)));
    v.gamma = Matrix(K,K*k);
    v.deterministic = Matrix(K,dout);
    for (std::size_t i = 0; i < K; ++i)
        for (std::size_t c = 0; c < mx; ++c)
            (c < K*k ? v.gamma(i,c) : v.deterministic(i,c-K*k)) = shortrun(i,c);
    v.residuals = target;
    if (mx > 0) {
        const Matrix fit = mul(dxm,tr(shortrun));
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t j = 0; j < K; ++j)
                v.residuals(t,j) -= fit(t,j);
    }
    v.sigma = mul(tr(v.residuals),v.residuals);
    for (std::size_t i = 0; i < K; ++i)
        for (std::size_t j = 0; j < K; ++j)
            v.sigma(i,j) /= Td;
    double logdet00 = 0;
    (void)spd_power(s00,-1,&logdet00);
    double sum = 0;
    for (std::size_t i = 0; i < r; ++i)
        sum += std::log1p(-v.eigenvalues[i]);
    const double Kd = static_cast<double>(K);
    v.log_likelihood = -.5*Kd*Td*std::log(2*std::numbers::pi)-.5*Td*(logdet00+sum)-.5*Kd*Td;

    // Standard errors of alpha, gamma and outside terms given beta: Omega = W'W, W = [y1 beta, dx].
    Matrix w(T,r+mx);
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t j = 0; j < r; ++j)
            w(t,j) = ecm(t,j);
        for (std::size_t c = 0; c < mx; ++c)
            w(t,r+c) = dxm(t,c);
    }
    const Matrix omega_inv = spd_power(mul(tr(w),w),-1);
    v.alpha_standard_errors = Matrix(K,r);
    v.gamma_standard_errors = Matrix(K,K*k);
    v.deterministic_standard_errors = Matrix(K,dout);
    for (std::size_t i = 0; i < K; ++i) {
        for (std::size_t j = 0; j < r; ++j)
            v.alpha_standard_errors(i,j) = std::sqrt(omega_inv(j,j)*v.sigma(i,i));
        for (std::size_t c = 0; c < mx; ++c) {
            const double se = std::sqrt(omega_inv(r+c,r+c)*v.sigma(i,i));
            (c < K*k ? v.gamma_standard_errors(i,c) : v.deterministic_standard_errors(i,c-K*k)) = se;
        }
    }
    // Beta: Cov(vec beta_2') = (R12 R12')^{-1} (x) (alpha' Sigma^{-1} alpha)^{-1}, R12 = rows r.. of R1.
    v.beta_standard_errors = Matrix(K,r);
    v.beta_deterministic_standard_errors = Matrix(din,r);
    {
        const std::size_t rest = m1-r;
        Matrix r12(T,rest);
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t i = 0; i < rest; ++i)
                r12(t,i) = r1(t,r+i);
        const Matrix a = spd_power(mul(tr(r12),r12),-1);
        const Matrix bb = spd_power(mul(mul(tr(v.alpha),spd_power(v.sigma,-1)),v.alpha),-1);
        for (std::size_t i = 0; i < rest; ++i)
            for (std::size_t j = 0; j < r; ++j) {
                const double se = std::sqrt(a(i,i)*bb(j,j));
                const std::size_t row = r+i;
                (row < K ? v.beta_standard_errors(row,j) : v.beta_deterministic_standard_errors(row-K,j)) = se;
            }
    }
    // Levels-VAR representation for forecasting.
    const Matrix pi = mul(v.alpha,tr(v.beta));
    for (std::size_t l = 0; l < p; ++l) {
        Matrix a(K,K);
        for (std::size_t i = 0; i < K; ++i)
            for (std::size_t j = 0; j < K; ++j) {
                if (l == 0)
                    a(i,j) = (i == j ? 1.0 : 0.0)+pi(i,j)+(k > 0 ? v.gamma(i,j) : 0.0);
                else if (l < k)
                    a(i,j) = v.gamma(i,l*K+j)-v.gamma(i,(l-1)*K+j);
                else
                    a(i,j) = -v.gamma(i,(l-1)*K+j);
            }
        v.var_coefficients.push_back(std::move(a));
    }
    v.history = Matrix(p,K);
    for (std::size_t l = 0; l < p; ++l)
        for (std::size_t j = 0; j < K; ++j)
            v.history(l,j) = y(n-p+l,j);
    return v;
}

linalg::DenseMatrix<double> forecast_vecm(const VecmModel& model, std::size_t horizon) {
    if (horizon == 0)
        throw std::invalid_argument("forecast_vecm: horizon must be positive");
    const std::size_t K = model.variables, p = model.lagged_differences+1, T = model.observations;
    std::vector<std::vector<double>> path;
    for (std::size_t l = 0; l < p; ++l) {
        std::vector<double> row(K);
        for (std::size_t j = 0; j < K; ++j)
            row[j] = model.history(l,j);
        path.push_back(std::move(row));
    }
    Matrix out(horizon,K);
    for (std::size_t h = 0; h < horizon; ++h) {
        std::vector<double> next(K,0.0);
        for (std::size_t i = 0; i < K; ++i) {
            std::size_t c = 0, d = 0;
            if (model.constant == VecmTerm::outside)
                next[i] += model.deterministic(i,c++);
            if (model.trend == VecmTerm::outside)
                next[i] += model.deterministic(i,c)*static_cast<double>(T+p+1+h);
            for (std::size_t q = 0; q < model.rank; ++q) {
                double restricted = 0;
                d = 0;
                if (model.constant == VecmTerm::inside)
                    restricted += model.beta_deterministic(d++,q);
                if (model.trend == VecmTerm::inside)
                    restricted += model.beta_deterministic(d,q)*static_cast<double>(T+p+h);
                next[i] += model.alpha(i,q)*restricted;
            }
            for (std::size_t l = 1; l <= p; ++l)
                for (std::size_t j = 0; j < K; ++j)
                    next[i] += model.var_coefficients[l-1](i,j)*path[path.size()-l][j];
            out(h,i) = next[i];
        }
        path.push_back(std::move(next));
    }
    return out;
}

} // namespace datamunge::stats
