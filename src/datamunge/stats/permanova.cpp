/// @file permanova.cpp
/// @brief Gower-matrix PERMANOVA by projections, PCoA-based dispersion, and shared permutation generation.
#include <datamunge/stats/permanova.hpp>

#include <datamunge/linalg/eigen.hpp>
#include <datamunge/random/distributions.hpp>
#include <datamunge/random/random.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Implementation helpers for PERMANOVA and dispersion tests.
namespace {
/// @brief Dense matrix alias.
using Matrix = linalg::DenseMatrix<double>;
/// @brief Quiet NaN.
constexpr double nan = std::numeric_limits<double>::quiet_NaN();
/// @brief Tolerance for counting permuted statistics as at least as large (vegan's sqrt(.Machine$double.eps)).
const double eps = std::sqrt(std::numeric_limits<double>::epsilon());

/// @brief Validates a dissimilarity matrix.
/// @param d Matrix.
/// @param who Caller name.
void check_distances(const Matrix& d, const char* who) {
    if (d.rows() != d.cols() || d.rows() < 2)
        throw std::invalid_argument(std::string(who) + ": distances must be a square matrix with n >= 2");
    for (std::size_t i = 0; i < d.rows(); ++i)
        for (std::size_t j = 0; j < d.cols(); ++j) {
            if (!std::isfinite(d(i, j)) || d(i, j) < -1e-7)
                throw std::invalid_argument(std::string(who) + ": distances must be finite and nonnegative");
            if (std::abs(d(i, j) - d(j, i)) > 1e-10 * std::max(1.0, std::abs(d(i, j))))
                throw std::invalid_argument(std::string(who) + ": distances must be symmetric");
        }
}

/// @brief Double-centered \f$-\frac12 D^{(2)}\f$ (Gower's matrix).
/// @param d Distances.
/// @return G.
Matrix gower(const Matrix& d) {
    const std::size_t n = d.rows();
    Matrix g(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            g(i, j) = -0.5 * d(i, j) * d(i, j);
    std::vector<double> row(n, 0.0);
    double all = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            row[i] += g(i, j) / static_cast<double>(n);
    for (const double r : row)
        all += r / static_cast<double>(n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            g(i, j) += all - row[i] - row[j];
    return g;
}

/// @brief Modified Gram-Schmidt: appends the centered columns of `cols` that are not aliased to `basis`.
/// @param cols Columns to add.
/// @param basis Orthonormal columns so far (each a vector of length n).
/// @return Number of columns added.
std::size_t orthonormalize(const Matrix& cols, std::vector<std::vector<double>>& basis) {
    const std::size_t n = cols.rows();
    std::size_t added = 0;
    for (std::size_t c = 0; c < cols.cols(); ++c) {
        std::vector<double> v(n);
        double mean = 0;
        for (std::size_t i = 0; i < n; ++i)
            mean += cols(i, c) / static_cast<double>(n);
        double before = 0;
        for (std::size_t i = 0; i < n; ++i)
            v[i] = cols(i, c) - mean, before += v[i] * v[i];
        if (!(before > 0))
            continue;
        for (int pass = 0; pass < 2; ++pass)
            for (const auto& q : basis) {
                double dot = 0;
                for (std::size_t i = 0; i < n; ++i)
                    dot += q[i] * v[i];
                for (std::size_t i = 0; i < n; ++i)
                    v[i] -= dot * q[i];
            }
        double after = 0;
        for (const double a : v)
            after += a * a;
        if (std::sqrt(after) <= 1e-7 * std::sqrt(before))
            continue;
        for (auto& a : v)
            a /= std::sqrt(after);
        basis.push_back(std::move(v));
        ++added;
    }
    return added;
}

/// @brief \f$q^\top G_\pi q\f$ with \f$G_\pi=G[\pi,\pi]\f$, computed as \f$w^\top Gw\f$ with \f$w_{\pi_a}=q_a\f$.
/// @param g Gower matrix.
/// @param q Column.
/// @param perm Permutation.
/// @param w Workspace.
/// @return Quadratic form.
double quadratic(const Matrix& g, const std::vector<double>& q, const std::vector<std::size_t>& perm,
                 std::vector<double>& w) {
    const std::size_t n = q.size();
    for (std::size_t a = 0; a < n; ++a)
        w[perm[a]] = q[a];
    double s = 0;
    for (std::size_t i = 0; i < n; ++i) {
        double r = 0;
        for (std::size_t j = 0; j < n; ++j)
            r += g(i, j) * w[j];
        s += w[i] * r;
    }
    return s;
}

/// @brief Permutations: explicit ones, or random ones (within strata if given).
/// @param n Observations.
/// @param count Number of random permutations.
/// @param seed Seed.
/// @param strata Optional blocks.
/// @param explicit_perms Explicit permutations (used if non-empty).
/// @return Permutations.
std::vector<std::vector<std::size_t>> make_permutations(std::size_t n, std::size_t count, std::uint64_t seed,
                                                        const std::vector<std::size_t>& strata,
                                                        const std::vector<std::vector<std::size_t>>& explicit_perms) {
    if (!explicit_perms.empty()) {
        for (const auto& p : explicit_perms) {
            if (p.size() != n)
                throw std::invalid_argument("permutation_matrix rows must have n entries");
            std::vector<char> seen(n, 0);
            for (const std::size_t v : p) {
                if (v >= n || seen[v])
                    throw std::invalid_argument("permutation_matrix rows must be permutations of 0..n-1");
                seen[v] = 1;
            }
        }
        return explicit_perms;
    }
    if (!strata.empty() && strata.size() != n)
        throw std::invalid_argument("strata must have one entry per observation");
    std::map<std::size_t, std::vector<std::size_t>> blocks;
    for (std::size_t i = 0; i < n; ++i)
        blocks[strata.empty() ? 0 : strata[i]].push_back(i);
    random::SplitMix64 rng(seed);
    std::vector<std::vector<std::size_t>> out(count, std::vector<std::size_t>(n));
    for (auto& p : out) {
        std::iota(p.begin(), p.end(), std::size_t{0});
        for (const auto& [key, idx] : blocks) {
            auto shuffled = idx;
            for (std::size_t i = shuffled.size(); i > 1; --i)
                std::swap(shuffled[i - 1], shuffled[rng.uniform_u64(0, i - 1)]);
            for (std::size_t k = 0; k < idx.size(); ++k)
                p[idx[k]] = shuffled[k];
        }
    }
    return out;
}

/// @brief Sorted unique levels and the level index of each label.
/// @param labels Labels.
/// @param index Output level index per observation.
/// @return Levels.
std::vector<std::string> levels_of(const std::vector<std::string>& labels, std::vector<std::size_t>& index) {
    const std::set<std::string> s(labels.begin(), labels.end());
    std::vector<std::string> levels(s.begin(), s.end());
    index.resize(labels.size());
    for (std::size_t i = 0; i < labels.size(); ++i)
        index[i] = static_cast<std::size_t>(std::ranges::lower_bound(levels, labels[i]) - levels.begin());
    return levels;
}

/// @brief Geometric median (Weiszfeld iteration from the coordinate-wise median).
/// @param pts Points (rows), all of the same dimension.
/// @return Median.
std::vector<double> spatial_median(const std::vector<std::vector<double>>& pts) {
    const std::size_t k = pts.front().size();
    std::vector<double> m(k);
    for (std::size_t c = 0; c < k; ++c) {
        std::vector<double> v;
        for (const auto& p : pts)
            v.push_back(p[c]);
        std::ranges::sort(v);
        m[c] = v.size() % 2 ? v[v.size() / 2] : (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
    }
    if (pts.size() <= 2) {
        // Any point on the segment minimizes; use the mean (the minimizer chosen by symmetry).
        for (std::size_t c = 0; c < k; ++c) {
            m[c] = 0;
            for (const auto& p : pts)
                m[c] += p[c] / static_cast<double>(pts.size());
        }
        return m;
    }
    for (int it = 0; it < 100000; ++it) {
        std::vector<double> num(k, 0.0);
        double den = 0;
        bool at_point = false;
        for (const auto& p : pts) {
            double s = 0;
            for (std::size_t c = 0; c < k; ++c)
                s += (p[c] - m[c]) * (p[c] - m[c]);
            const double r = std::sqrt(s);
            if (r < 1e-14) {
                at_point = true;
                continue;
            }
            for (std::size_t c = 0; c < k; ++c)
                num[c] += p[c] / r;
            den += 1 / r;
        }
        std::vector<double> next(k);
        for (std::size_t c = 0; c < k; ++c)
            next[c] = num[c] / den;
        double step = 0, scale = 0;
        for (std::size_t c = 0; c < k; ++c)
            step = std::max(step, std::abs(next[c] - m[c])), scale = std::max(scale, std::abs(m[c]));
        m = next;
        if (step <= 1e-15 * std::max(scale, 1e-300) || (at_point && step == 0))
            break;
    }
    return m;
}
} // namespace

linalg::DenseMatrix<double> bray_curtis_distances(const linalg::DenseMatrix<double>& x) {
    const std::size_t n = x.rows();
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < x.cols(); ++k)
            if (!(x(i, k) >= 0) || !std::isfinite(x(i, k)))
                throw std::invalid_argument("bray_curtis_distances: abundances must be nonnegative and finite");
    Matrix d(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) {
            double num = 0, den = 0;
            for (std::size_t k = 0; k < x.cols(); ++k)
                num += std::abs(x(i, k) - x(j, k)), den += x(i, k) + x(j, k);
            d(i, j) = d(j, i) = den > 0 ? num / den : 0.0;
        }
    return d;
}

PermanovaTerm factor_term(const std::string& name, const std::vector<std::string>& labels) {
    if (labels.empty())
        throw std::invalid_argument("factor_term: no labels");
    std::vector<std::size_t> index;
    const auto levels = levels_of(labels, index);
    Matrix m(labels.size(), levels.size() - 1, 0.0);
    for (std::size_t i = 0; i < labels.size(); ++i)
        if (index[i] > 0)
            m(i, index[i] - 1) = 1;
    return {name, m};
}

PermanovaTerm numeric_term(const std::string& name, const std::vector<double>& values) {
    return {name, Matrix(values.size(), 1, values)};
}

PermanovaTerm interaction_term(const std::string& name, const PermanovaTerm& a, const PermanovaTerm& b) {
    if (a.columns.rows() != b.columns.rows())
        throw std::invalid_argument("interaction_term: terms must have the same rows");
    Matrix m(a.columns.rows(), a.columns.cols() * b.columns.cols(), 0.0);
    for (std::size_t i = 0; i < m.rows(); ++i)
        for (std::size_t p = 0; p < a.columns.cols(); ++p)
            for (std::size_t q = 0; q < b.columns.cols(); ++q)
                m(i, q * a.columns.cols() + p) = a.columns(i, p) * b.columns(i, q);
    return {name, m};
}

Permanova permanova(const linalg::DenseMatrix<double>& distances, const std::vector<PermanovaTerm>& terms,
                    const PermanovaOptions& options) {
    check_distances(distances, "permanova");
    const std::size_t n = distances.rows();
    if (terms.empty())
        throw std::invalid_argument("permanova: need at least one term");
    for (const auto& t : terms)
        if (t.columns.rows() != n)
            throw std::invalid_argument("permanova: every term needs one row per observation");
    const Matrix g = gower(distances);
    double total = 0;
    for (std::size_t i = 0; i < n; ++i)
        total += g(i, i);

    // Sequential orthonormal basis, recording each term's block.
    std::vector<std::vector<double>> basis;
    std::vector<std::pair<std::size_t, std::size_t>> blocks; // [begin, end) in basis
    for (const auto& t : terms) {
        const std::size_t begin = basis.size();
        orthonormalize(t.columns, basis);
        blocks.emplace_back(begin, basis.size());
    }
    const std::size_t rank = basis.size();
    if (rank == 0 || rank + 1 >= n)
        throw std::invalid_argument("permanova: the model needs 1 <= rank <= n - 2");
    const std::size_t r = n - rank - 1;

    // Marginal (reduced-model) bases: all terms but one.
    std::vector<std::vector<std::vector<double>>> reduced;
    if (options.by == PermanovaBy::margin)
        for (std::size_t k = 0; k < terms.size(); ++k) {
            std::vector<std::vector<double>> b;
            for (std::size_t j = 0; j < terms.size(); ++j)
                if (j != k)
                    orthonormalize(terms[j].columns, b);
            reduced.push_back(std::move(b));
        }

    std::vector<std::size_t> identity(n);
    std::iota(identity.begin(), identity.end(), std::size_t{0});
    std::vector<double> w(n);
    // Statistics for one permutation: numerator SS and df of each tested row, residual SS.
    const auto statistics = [&](const std::vector<std::size_t>& perm, std::vector<double>& ss, double& res) {
        std::vector<double> col_ss(rank);
        double model = 0;
        for (std::size_t c = 0; c < rank; ++c)
            model += (col_ss[c] = quadratic(g, basis[c], perm, w));
        res = total - model;
        ss.clear();
        if (options.by == PermanovaBy::overall)
            ss.push_back(model);
        else if (options.by == PermanovaBy::terms)
            for (const auto& [b, e] : blocks)
                ss.push_back(std::accumulate(col_ss.begin() + static_cast<std::ptrdiff_t>(b),
                                             col_ss.begin() + static_cast<std::ptrdiff_t>(e), 0.0));
        else
            for (const auto& red : reduced) {
                double s = 0;
                for (const auto& q : red)
                    s += quadratic(g, q, perm, w);
                ss.push_back(model - s);
            }
    };
    std::vector<std::string> names;
    std::vector<std::size_t> dfs;
    if (options.by == PermanovaBy::overall)
        names.push_back("Model"), dfs.push_back(rank);
    else
        for (std::size_t k = 0; k < terms.size(); ++k) {
            names.push_back(terms[k].name);
            dfs.push_back(options.by == PermanovaBy::terms ? blocks[k].second - blocks[k].first
                                                           : rank - reduced[k].size());
        }
    std::vector<double> ss0;
    double res0 = 0;
    statistics(identity, ss0, res0);
    std::vector<double> f0(ss0.size());
    for (std::size_t k = 0; k < ss0.size(); ++k)
        f0[k] = dfs[k] ? (ss0[k] / static_cast<double>(dfs[k])) / (res0 / static_cast<double>(r)) : nan;

    const auto perms = make_permutations(n, options.permutations, options.seed, options.strata,
                                         options.permutation_matrix);
    std::vector<std::size_t> exceed(ss0.size(), 0);
    std::vector<double> ss;
    double res = 0;
    for (const auto& p : perms) {
        statistics(p, ss, res);
        for (std::size_t k = 0; k < ss.size(); ++k)
            if (dfs[k] && (ss[k] / static_cast<double>(dfs[k])) / (res / static_cast<double>(r)) >= f0[k] - eps)
                ++exceed[k];
    }
    Permanova out;
    out.permutations = perms.size();
    for (std::size_t k = 0; k < ss0.size(); ++k)
        out.rows.push_back({names[k], dfs[k], ss0[k], ss0[k] / total, f0[k],
                            dfs[k] && !perms.empty() ? (static_cast<double>(exceed[k]) + 1) /
                                                           (static_cast<double>(perms.size()) + 1)
                                                     : nan});
    out.rows.push_back({"Residual", r, res0, res0 / total, nan, nan});
    out.rows.push_back({"Total", n - 1, total, 1.0, nan, nan});
    return out;
}

Dispersion dispersion_test(const linalg::DenseMatrix<double>& distances, const std::vector<std::string>& groups,
                           const DispersionOptions& options) {
    check_distances(distances, "dispersion_test");
    const std::size_t n = distances.rows();
    if (groups.size() != n)
        throw std::invalid_argument("dispersion_test: one group label per observation");
    Dispersion out;
    std::vector<std::size_t> gi;
    out.groups = levels_of(groups, gi);
    const std::size_t ng = out.groups.size();
    if (ng < 2 || ng >= n)
        throw std::invalid_argument("dispersion_test: need 2 <= groups < n");

    // PCoA with positive and negative axes.
    const auto eig = linalg::symmetric_eigen(gower(distances));
    const double tol = std::sqrt(std::numeric_limits<double>::epsilon());
    const double cut = std::max(tol, tol * eig.eigenvalues.front());
    std::vector<std::size_t> keep;
    for (std::size_t k = 0; k < eig.eigenvalues.size(); ++k)
        if (std::abs(eig.eigenvalues[k]) > cut)
            keep.push_back(k);
    out.coordinates = Matrix(n, keep.size(), 0.0);
    std::vector<bool> positive;
    for (std::size_t c = 0; c < keep.size(); ++c) {
        const double lambda = eig.eigenvalues[keep[c]];
        out.eigenvalues.push_back(lambda);
        positive.push_back(lambda > 0);
        for (std::size_t i = 0; i < n; ++i)
            out.coordinates(i, c) = eig.eigenvectors(i, keep[c]) * std::sqrt(std::abs(lambda));
    }

    // Group centers, separately on real (positive) and imaginary (negative) axes.
    out.distances.assign(n, 0.0);
    std::vector<double> dpos(n, 0.0), dneg(n, 0.0);
    for (std::size_t g = 0; g < ng; ++g) {
        std::vector<std::size_t> members;
        for (std::size_t i = 0; i < n; ++i)
            if (gi[i] == g)
                members.push_back(i);
        for (const bool sign : {true, false}) {
            std::vector<std::size_t> axes;
            for (std::size_t c = 0; c < keep.size(); ++c)
                if (positive[c] == sign)
                    axes.push_back(c);
            if (axes.empty())
                continue;
            std::vector<std::vector<double>> pts;
            for (const std::size_t i : members) {
                std::vector<double> p;
                for (const std::size_t c : axes)
                    p.push_back(out.coordinates(i, c));
                pts.push_back(std::move(p));
            }
            std::vector<double> center(axes.size(), 0.0);
            if (options.center == DispersionCenter::centroid) {
                for (const auto& p : pts)
                    for (std::size_t c = 0; c < axes.size(); ++c)
                        center[c] += p[c] / static_cast<double>(pts.size());
            } else {
                center = spatial_median(pts);
            }
            for (std::size_t m = 0; m < members.size(); ++m) {
                double s = 0;
                for (std::size_t c = 0; c < axes.size(); ++c)
                    s += (pts[m][c] - center[c]) * (pts[m][c] - center[c]);
                (sign ? dpos : dneg)[members[m]] = s;
            }
        }
    }
    std::vector<double> count(ng, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        count[gi[i]] += 1;
    for (std::size_t i = 0; i < n; ++i) {
        out.distances[i] = std::sqrt(std::max(dpos[i] - dneg[i], 0.0));
        if (options.bias_adjust)
            out.distances[i] *= std::sqrt(count[gi[i]] / (count[gi[i]] - 1));
    }

    // One-way ANOVA of the distances.
    const auto anova = [&](const std::vector<double>& z, double& ssb, double& ssw) {
        std::vector<double> mean(ng, 0.0);
        double grand = 0;
        for (std::size_t i = 0; i < n; ++i)
            mean[gi[i]] += z[i] / count[gi[i]], grand += z[i] / static_cast<double>(n);
        ssb = ssw = 0;
        for (std::size_t i = 0; i < n; ++i)
            ssb += (mean[gi[i]] - grand) * (mean[gi[i]] - grand),
                ssw += (z[i] - mean[gi[i]]) * (z[i] - mean[gi[i]]);
        return (ssb / static_cast<double>(ng - 1)) / (ssw / static_cast<double>(n - ng));
    };
    out.df_groups = ng - 1, out.df_residual = n - ng;
    out.f = anova(out.distances, out.sum_sq_groups, out.sum_sq_residual);
    out.parametric_p_value = 1 - random::f_cdf(out.f, static_cast<double>(ng - 1), static_cast<double>(n - ng));
    out.group_mean_distances.assign(ng, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        out.group_mean_distances[gi[i]] += out.distances[i] / count[gi[i]];

    // Permute residuals from the group means.
    std::vector<double> resid(n);
    for (std::size_t i = 0; i < n; ++i)
        resid[i] = out.distances[i] - out.group_mean_distances[gi[i]];
    const auto perms = make_permutations(n, options.permutations, options.seed, {}, options.permutation_matrix);
    std::size_t exceed = 0;
    std::vector<double> z(n);
    for (const auto& p : perms) {
        for (std::size_t i = 0; i < n; ++i)
            z[i] = resid[p[i]];
        double a = 0, b = 0;
        if (anova(z, a, b) >= out.f - eps)
            ++exceed;
    }
    out.permutations = perms.size();
    out.permutation_p_value =
        perms.empty() ? nan : (static_cast<double>(exceed) + 1) / (static_cast<double>(perms.size()) + 1);
    return out;
}

} // namespace datamunge::stats
