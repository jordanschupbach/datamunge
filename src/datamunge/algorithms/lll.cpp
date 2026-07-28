#include <datamunge/algorithms/lll.hpp>

#include <cmath>
#include <vector>

namespace datamunge::algorithms {

namespace {

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// Gram-Schmidt orthogonalisation of the (double) basis rows: fills bstar (the
// orthogonal vectors) and mu (the projection coefficients).
void gram_schmidt(const std::vector<std::vector<double>>& b,
                  std::vector<std::vector<double>>&        bstar,
                  std::vector<std::vector<double>>&        mu) {
    const std::size_t n = b.size();
    bstar.assign(n, {});
    mu.assign(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        bstar[i] = b[i];
        for (std::size_t j = 0; j < i; ++j) {
            mu[i][j]        = dot(b[i], bstar[j]) / dot(bstar[j], bstar[j]);
            for (std::size_t k = 0; k < b[i].size(); ++k) bstar[i][k] -= mu[i][j] * bstar[j][k];
        }
    }
}

} // namespace

std::vector<std::vector<long long>> lll_reduce(std::vector<std::vector<long long>> basis, double delta) {
    const std::size_t n = basis.size();
    if (n < 2) return basis;
    const std::size_t dim = basis[0].size();

    // Work on a double copy for the Gram-Schmidt data; keep `basis` integral.
    auto to_double = [&] {
        std::vector<std::vector<double>> d(n, std::vector<double>(dim));
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k < dim; ++k) d[i][k] = static_cast<double>(basis[i][k]);
        return d;
    };

    std::vector<std::vector<double>> b = to_double(), bstar, mu;
    gram_schmidt(b, bstar, mu);

    std::size_t k = 1;
    while (k < n) {
        // Size-reduce b_k against b_{k-1}, ..., b_0.
        for (std::size_t j = k; j-- > 0;) {
            if (std::fabs(mu[k][j]) > 0.5) {
                const long long q = static_cast<long long>(std::llround(mu[k][j]));
                for (std::size_t c = 0; c < dim; ++c) basis[k][c] -= q * basis[j][c];
                b = to_double();
                gram_schmidt(b, bstar, mu);
            }
            if (j == 0) break;
        }

        // Lovasz condition.
        const double lhs = dot(bstar[k], bstar[k]);
        const double rhs = (delta - mu[k][k - 1] * mu[k][k - 1]) * dot(bstar[k - 1], bstar[k - 1]);
        if (lhs >= rhs) {
            ++k;
        } else {
            std::swap(basis[k], basis[k - 1]);
            b = to_double();
            gram_schmidt(b, bstar, mu);
            k = (k > 1) ? k - 1 : 1;
        }
    }
    return basis;
}

} // namespace datamunge::algorithms
