#pragma once

/// \file kabsch.hpp
/// \brief Kabsch algorithm: optimal rigid alignment of two point sets (RMSD minimization).
///
/// Given two ordered sets of corresponding points, the *Kabsch algorithm* (Wolfgang Kabsch,
/// 1976) finds the rotation and translation that overlay one onto the other with the smallest
/// possible root-mean-square deviation -- the standard way to compare protein structures. We
/// use Horn's quaternion formulation: after centering both sets, the optimal rotation is the
/// eigenvector of the largest eigenvalue of a 4x4 symmetric matrix built from the
/// cross-covariance. That eigenvector *is* the rotation quaternion, which also sidesteps the
/// reflection ambiguity that a naive SVD solution must guard against.

#include <datamunge/linalg/eigen.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

using Point3 = std::array<double, 3>;

/// Result of a Kabsch alignment mapping `source` onto `target`.
struct KabschResult {
    std::array<std::array<double, 3>, 3> rotation;     ///< Optimal rotation matrix.
    std::array<double, 3>                translation;  ///< Optimal translation.
    double                               rmsd;         ///< Minimized RMSD after alignment.
};

/// \brief Optimal rigid transform (rotation + translation) aligning `source` onto `target`.
///
/// The two vectors must have equal length with point i in `source` corresponding to point i
/// in `target`. Returns the rotation \c R and translation \c t minimizing
/// \f$\sum_i \lVert R\,s_i + t - q_i\rVert^2\f$, and the resulting RMSD.
inline KabschResult kabsch(const std::vector<Point3>& source, const std::vector<Point3>& target) {
    const std::size_t n = source.size();

    // 1. Centroids.
    Point3 cs{0, 0, 0}, ct{0, 0, 0};
    for (std::size_t i = 0; i < n; ++i)
        for (int d = 0; d < 3; ++d) { cs[d] += source[i][d]; ct[d] += target[i][d]; }
    for (int d = 0; d < 3; ++d) { cs[d] /= n; ct[d] /= n; }

    // 2. Cross-covariance H[a][b] = sum (s_i - cs)_a (q_i - ct)_b.
    double H[3][3] = {{0}};
    for (std::size_t i = 0; i < n; ++i)
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                H[a][b] += (source[i][a] - cs[a]) * (target[i][b] - ct[b]);

    // 3. Horn's 4x4 symmetric key matrix N.
    double Sxx = H[0][0], Sxy = H[0][1], Sxz = H[0][2];
    double Syx = H[1][0], Syy = H[1][1], Syz = H[1][2];
    double Szx = H[2][0], Szy = H[2][1], Szz = H[2][2];
    linalg::DenseMatrix<double> N(4, 4, 0.0);
    N(0, 0) = Sxx + Syy + Szz; N(0, 1) = Syz - Szy;        N(0, 2) = Szx - Sxz;        N(0, 3) = Sxy - Syx;
    N(1, 0) = Syz - Szy;       N(1, 1) = Sxx - Syy - Szz;  N(1, 2) = Sxy + Syx;        N(1, 3) = Szx + Sxz;
    N(2, 0) = Szx - Sxz;       N(2, 1) = Sxy + Syx;        N(2, 2) = -Sxx + Syy - Szz; N(2, 3) = Syz + Szy;
    N(3, 0) = Sxy - Syx;       N(3, 1) = Szx + Sxz;        N(3, 2) = Syz + Szy;        N(3, 3) = -Sxx - Syy + Szz;

    // 4. Largest-eigenvalue eigenvector = optimal rotation quaternion (w, x, y, z).
    auto   eig = linalg::jacobi_eigen(N);
    double w = eig.eigenvectors(0, 0), x = eig.eigenvectors(1, 0),
           y = eig.eigenvectors(2, 0), z = eig.eigenvectors(3, 0);

    // 5. Quaternion -> rotation matrix.
    KabschResult r;
    r.rotation = {{{1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)},
                   {2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
                   {2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)}}};

    // 6. Translation t = ct - R cs.
    for (int a = 0; a < 3; ++a) {
        double Rcs = 0;
        for (int b = 0; b < 3; ++b) Rcs += r.rotation[a][b] * cs[b];
        r.translation[a] = ct[a] - Rcs;
    }

    // 7. RMSD after applying the transform.
    double sq = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (int a = 0; a < 3; ++a) {
            double v = r.translation[a];
            for (int b = 0; b < 3; ++b) v += r.rotation[a][b] * source[i][b];
            double e = v - target[i][a];
            sq += e * e;
        }
    r.rmsd = std::sqrt(sq / n);
    return r;
}

}  // namespace datamunge::algorithms
