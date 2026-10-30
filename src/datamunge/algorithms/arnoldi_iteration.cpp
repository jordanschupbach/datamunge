/// @file arnoldi_iteration.cpp
/// @brief Arnoldi iteration (dense and matrix-free), Ritz pairs and diagnostics.

#include <datamunge/algorithms/arnoldi_iteration.hpp>

#include <datamunge/nn/linalg.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace datamunge::algorithms {

/// @brief Small vector helpers for the Arnoldi loop.
namespace {

/// @brief Euclidean dot product.
double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) s += a[k] * b[k];
    return s;
}

/// @brief Euclidean norm.
double norm(const std::vector<double>& a) { return std::sqrt(dot(a, a)); }

}  // namespace

ArnoldiResult arnoldi_iteration(const std::vector<std::vector<double>>& A, const std::vector<double>& b, int m) {
    const std::size_t n = A.size();
    if (b.size() != n) throw std::invalid_argument("arnoldi_iteration: b must have one entry per row of A");
    MatVec matvec = [&A, n](const std::vector<double>& x) {
        std::vector<double> y(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) y[i] = dot(A[i], x);
        return y;
    };
    return arnoldi_iteration(matvec, b, m, false);
}

ArnoldiResult arnoldi_iteration(const MatVec& matvec, const std::vector<double>& b, int m, bool reorthogonalize) {
    if (m <= 0) throw std::invalid_argument("arnoldi_iteration: m must be positive");
    ArnoldiResult res;
    std::vector<double> v0 = b;
    const double        b0 = norm(v0);
    if (b0 < 1e-300) return res;
    for (double& x : v0) x /= b0;
    res.V.push_back(std::move(v0));
    res.H.assign(static_cast<std::size_t>(m) + 1, std::vector<double>(static_cast<std::size_t>(m), 0.0));

    for (int j = 0; j < m; ++j) {
        std::vector<double> w = matvec(res.V[j]);
        if (w.size() != b.size()) throw std::invalid_argument("arnoldi_iteration: matvec returned the wrong length");
        const double scale = norm(w);
        for (int pass = 0; pass < (reorthogonalize ? 2 : 1); ++pass) {
            for (int i = 0; i <= j; ++i) {
                const double h = dot(res.V[i], w);
                res.H[i][j] += h;
                for (std::size_t k = 0; k < w.size(); ++k) w[k] -= h * res.V[i][k];
            }
        }
        const double hn = norm(w);
        res.H[j + 1][j] = hn;
        if (hn <= 1e-12 * std::max(scale, 1e-300)) {  // invariant subspace: stop with a shorter H
            res.H[j + 1][j] = 0.0;
            res.H.resize(static_cast<std::size_t>(j) + 2);
            break;
        }
        for (double& x : w) x /= hn;
        res.V.push_back(std::move(w));
    }
    return res;
}

std::vector<RitzPair> ritz_pairs(const ArnoldiResult& r, bool vectors) {
    if (r.H.empty() || r.H[0].empty()) throw std::invalid_argument("ritz_pairs: empty Arnoldi result");
    const std::size_t m = std::min(r.H.size() - 1, r.H[0].size());
    if (m == 0) throw std::invalid_argument("ritz_pairs: need at least one Arnoldi step");
    std::vector<double> flat;
    flat.reserve(m * m);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j) flat.push_back(r.H[i][j]);
    const auto eig = nn::eig(nn::from_vector<double>(flat, {static_cast<std::int64_t>(m), static_cast<std::int64_t>(m)}),
                             true);
    const auto er = eig.e.real.to_vector<double>(), ei = eig.e.imag.to_vector<double>();
    const auto vr = eig.v.real.to_vector<double>(), vi = eig.v.imag.to_vector<double>();
    const double beta = r.H[m][m - 1];

    std::vector<RitzPair> out;
    out.reserve(m);
    for (std::size_t c = 0; c < m; ++c) {
        RitzPair p;
        p.value = {er[c], ei[c]};
        std::vector<std::complex<double>> y(m);
        double                             ny = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            y[i] = {vr[i * m + c], vi[i * m + c]};
            ny += std::norm(y[i]);
        }
        ny = std::sqrt(ny);
        for (auto& z : y) z /= ny;
        p.residual = std::abs(beta) * std::abs(y[m - 1]);
        if (vectors) {
            p.vector.assign(r.V[0].size(), {0.0, 0.0});
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t k = 0; k < p.vector.size(); ++k) p.vector[k] += r.V[i][k] * y[i];
        }
        out.push_back(std::move(p));
    }
    std::stable_sort(out.begin(), out.end(),
                     [](const RitzPair& a, const RitzPair& b) { return std::abs(a.value) > std::abs(b.value); });
    return out;
}

double arnoldi_orthogonality_loss(const ArnoldiResult& r) {
    double worst = 0.0;
    for (std::size_t a = 0; a < r.V.size(); ++a)
        for (std::size_t c = a; c < r.V.size(); ++c)
            worst = std::max(worst, std::fabs(dot(r.V[a], r.V[c]) - (a == c ? 1.0 : 0.0)));
    return worst;
}

double arnoldi_relation_residual(const MatVec& matvec, const ArnoldiResult& r) {
    if (r.H.empty()) return 0.0;
    const std::size_t cols = std::min(r.H[0].size(), r.V.size());
    double            worst = 0.0;
    for (std::size_t j = 0; j < cols; ++j) {
        std::vector<double> w = matvec(r.V[j]);
        for (std::size_t i = 0; i < r.V.size() && i <= j + 1 && i < r.H.size(); ++i)
            for (std::size_t k = 0; k < w.size(); ++k) w[k] -= r.H[i][j] * r.V[i][k];
        worst = std::max(worst, norm(w));
    }
    return worst;
}

}  // namespace datamunge::algorithms
