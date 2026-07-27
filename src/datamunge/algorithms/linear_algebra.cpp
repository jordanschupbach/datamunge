#include <datamunge/algorithms/linear_algebra.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

namespace {

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

std::uint64_t splitmix(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z              = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z              = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

double dot(const Vector& a, const Vector& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

Vector matvec(const Matrix& A, const Vector& x) {
    Vector y(A.size(), 0.0);
    for (std::size_t i = 0; i < A.size(); ++i)
        for (std::size_t j = 0; j < x.size(); ++j) y[i] += A[i][j] * x[j];
    return y;
}

// ---- Strassen helpers on square power-of-two matrices ----

Matrix add(const Matrix& A, const Matrix& B) {
    const std::size_t n = A.size();
    Matrix            C(n, Vector(n, 0.0));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) C[i][j] = A[i][j] + B[i][j];
    return C;
}
Matrix sub(const Matrix& A, const Matrix& B) {
    const std::size_t n = A.size();
    Matrix            C(n, Vector(n, 0.0));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) C[i][j] = A[i][j] - B[i][j];
    return C;
}
Matrix naive(const Matrix& A, const Matrix& B) {
    const std::size_t n = A.size();
    Matrix            C(n, Vector(n, 0.0));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < n; ++k) {
            const double a = A[i][k];
            for (std::size_t j = 0; j < n; ++j) C[i][j] += a * B[k][j];
        }
    return C;
}

Matrix strassen_square(const Matrix& A, const Matrix& B) {
    const std::size_t n = A.size();
    if (n <= 64) return naive(A, B);
    const std::size_t m = n / 2;

    Matrix A11(m, Vector(m)), A12(m, Vector(m)), A21(m, Vector(m)), A22(m, Vector(m));
    Matrix B11(m, Vector(m)), B12(m, Vector(m)), B21(m, Vector(m)), B22(m, Vector(m));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j) {
            A11[i][j] = A[i][j];         A12[i][j] = A[i][j + m];
            A21[i][j] = A[i + m][j];     A22[i][j] = A[i + m][j + m];
            B11[i][j] = B[i][j];         B12[i][j] = B[i][j + m];
            B21[i][j] = B[i + m][j];     B22[i][j] = B[i + m][j + m];
        }

    const Matrix M1 = strassen_square(add(A11, A22), add(B11, B22));
    const Matrix M2 = strassen_square(add(A21, A22), B11);
    const Matrix M3 = strassen_square(A11, sub(B12, B22));
    const Matrix M4 = strassen_square(A22, sub(B21, B11));
    const Matrix M5 = strassen_square(add(A11, A12), B22);
    const Matrix M6 = strassen_square(sub(A21, A11), add(B11, B12));
    const Matrix M7 = strassen_square(sub(A12, A22), add(B21, B22));

    const Matrix C11 = add(sub(add(M1, M4), M5), M7);
    const Matrix C12 = add(M3, M5);
    const Matrix C21 = add(M2, M4);
    const Matrix C22 = add(add(sub(M1, M2), M3), M6);

    Matrix C(n, Vector(n, 0.0));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j) {
            C[i][j] = C11[i][j];         C[i][j + m] = C12[i][j];
            C[i + m][j] = C21[i][j];     C[i + m][j + m] = C22[i][j];
        }
    return C;
}

} // namespace

std::vector<std::vector<double>> gram_schmidt(const std::vector<std::vector<double>>& vectors) {
    Matrix Q;
    for (const Vector& v : vectors) {
        Vector u = v;
        for (const Vector& q : Q) {            // modified Gram-Schmidt: subtract one projection at a time
            const double proj = dot(u, q);
            for (std::size_t i = 0; i < u.size(); ++i) u[i] -= proj * q[i];
        }
        const double norm = std::sqrt(dot(u, u));
        if (norm > 1e-12) {                    // drop vectors dependent on the earlier ones
            for (double& x : u) x /= norm;
            Q.push_back(u);
        }
    }
    return Q;
}

EigenPair power_iteration(const std::vector<std::vector<double>>& matrix, int iterations, std::uint64_t seed) {
    const std::size_t n = matrix.size();
    EigenPair         out;
    if (n == 0) return out;

    Vector        v(n);
    std::uint64_t state = seed;
    for (std::size_t i = 0; i < n; ++i)
        v[i] = static_cast<double>(splitmix(state) % 2000) / 1000.0 - 1.0; // in [-1, 1)
    double nrm = std::sqrt(dot(v, v));
    if (nrm < 1e-18) { v.assign(n, 0.0); v[0] = 1.0; nrm = 1.0; }
    for (double& x : v) x /= nrm;

    for (int it = 0; it < iterations; ++it) {
        Vector       w  = matvec(matrix, v);
        const double wn = std::sqrt(dot(w, w));
        if (wn < 1e-18) break;
        for (double& x : w) x /= wn;
        // Keep a consistent sign so the iterate does not flip each step for negative eigenvalues.
        if (dot(w, v) < 0.0)
            for (double& x : w) x = -x;
        v = w;
    }
    out.vector = v;
    out.value  = dot(v, matvec(matrix, v)); // Rayleigh quotient (v is a unit vector)
    return out;
}

std::vector<std::vector<double>> strassen_multiply(const std::vector<std::vector<double>>& a,
                                                   const std::vector<std::vector<double>>& b) {
    const std::size_t m = a.size();
    const std::size_t k = m == 0 ? 0 : a[0].size();
    const std::size_t n = b.empty() ? 0 : b[0].size();
    if (b.size() != k) return {};
    if (m == 0 || k == 0 || n == 0) return Matrix(m, Vector(n, 0.0));

    std::size_t s = 1;
    while (s < m || s < k || s < n) s <<= 1; // next power of two >= max dimension
    Matrix A(s, Vector(s, 0.0)), B(s, Vector(s, 0.0));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < k; ++j) A[i][j] = a[i][j];
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < n; ++j) B[i][j] = b[i][j];

    const Matrix C = strassen_square(A, B);
    Matrix       out(m, Vector(n, 0.0));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) out[i][j] = C[i][j];
    return out;
}

bool freivalds_verify(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b,
                      const std::vector<std::vector<double>>& c, int rounds, std::uint64_t seed) {
    const std::size_t m = a.size();
    const std::size_t k = m == 0 ? 0 : a[0].size();
    const std::size_t n = b.empty() ? 0 : b[0].size();
    if (b.size() != k) return false;
    if (c.size() != m) return false;
    if (m > 0 && c[0].size() != n) return false;

    std::uint64_t state = seed;
    for (int r = 0; r < rounds; ++r) {
        Vector rv(n);
        for (std::size_t j = 0; j < n; ++j) rv[j] = static_cast<double>(splitmix(state) & 1ULL);
        // Compute A(Br) and Cr in O(n^2).
        Vector br(k, 0.0);
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < n; ++j) br[i] += b[i][j] * rv[j];
        Vector abr(m, 0.0), cr(m, 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < k; ++j) abr[i] += a[i][j] * br[j];
            for (std::size_t j = 0; j < n; ++j) cr[i] += c[i][j] * rv[j];
        }
        for (std::size_t i = 0; i < m; ++i)
            if (std::fabs(abr[i] - cr[i]) > 1e-6) return false;
    }
    return true;
}

} // namespace datamunge::algorithms
