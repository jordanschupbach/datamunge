#pragma once

/// @file arnoldi_iteration.hpp
/// @brief Arnoldi iteration: an orthonormal Krylov basis, the Hessenberg projection, and Ritz pairs.
///
/// Arnoldi builds an orthonormal basis of the Krylov subspace
/// \f$K_m(A,b)=\mathrm{span}\{b,Ab,\dots,A^{m-1}b\}\f$ using only products with \f$A\f$.
/// Starting from \f$v_1=b/\lVert b\rVert\f$, each step forms \f$Av_j\f$, removes its components
/// along the earlier basis vectors (modified Gram-Schmidt) and normalizes. The coefficients form
/// an upper Hessenberg matrix \f$H_m\f$ with the Arnoldi relation
/// \f$AV_m=V_mH_m+h_{m+1,m}v_{m+1}e_m^\top\f$, so \f$H_m=V_m^\top AV_m\f$. The eigenvalues of the
/// small matrix \f$H_m\f$ (Ritz values) approximate eigenvalues of \f$A\f$, the extreme ones
/// first. The relation also gives each Ritz pair's residual without touching \f$A\f$:
/// \f$\lVert Ax-\theta x\rVert=h_{m+1,m}\,|e_m^\top y|\f$ for \f$H_my=\theta y\f$, \f$x=V_my\f$.

#include <complex>
#include <cstddef>
#include <functional>
#include <vector>

namespace datamunge::algorithms {

/// Result of an Arnoldi run.
struct ArnoldiResult {
    std::vector<std::vector<double>> V;  ///< Basis vectors: V[k] is \f$v_{k+1}\f$ (up to m+1 of them).
    std::vector<std::vector<double>> H;  ///< (m+1) x m upper Hessenberg matrix, row-major (fewer rows after a breakdown).
};

/// A matrix-vector product \f$x\mapsto Ax\f$.
using MatVec = std::function<std::vector<double>(const std::vector<double>&)>;

/// @brief Runs m steps of Arnoldi on the dense matrix A starting from b.
///
/// Stops early (with a shorter H) when a zero subdiagonal shows that an invariant subspace has
/// been found.
ArnoldiResult arnoldi_iteration(const std::vector<std::vector<double>>& A, const std::vector<double>& b, int m);

/// @brief Matrix-free Arnoldi: only products with A are needed.
///
/// @param matvec          computes \f$Ax\f$; its result must have the length of b.
/// @param b               starting vector (non-zero).
/// @param m               number of steps (Krylov dimension).
/// @param reorthogonalize run a second Gram-Schmidt pass each step ("twice is enough"), which keeps
///                        the basis orthonormal to machine precision; one pass loses orthogonality
///                        as Ritz values converge.
/// @throws std::invalid_argument if m is not positive or matvec returns the wrong length.
ArnoldiResult arnoldi_iteration(const MatVec& matvec, const std::vector<double>& b, int m,
                                bool reorthogonalize = false);

/// An approximate eigenpair from the Krylov subspace.
struct RitzPair {
    std::complex<double>              value;     ///< Ritz value \f$\theta\f$ (eigenvalue of \f$H_m\f$).
    std::vector<std::complex<double>> vector;    ///< Ritz vector \f$x=V_my\f$ (unit length).
    double                            residual;  ///< \f$h_{m+1,m}|e_m^\top y|=\lVert Ax-\theta x\rVert\f$.
};

/// @brief Ritz pairs of an Arnoldi run, sorted by decreasing modulus of the Ritz value.
///
/// Uses the square part \f$H_m\f$ of H and its eigen decomposition.
/// @param r        an Arnoldi result with at least one step.
/// @param vectors  also form the Ritz vectors (otherwise RitzPair::vector is empty).
std::vector<RitzPair> ritz_pairs(const ArnoldiResult& r, bool vectors = true);

/// @brief Largest entry of \f$|V^\top V-I|\f$ over the basis vectors: 0 for an exactly orthonormal basis.
double arnoldi_orthogonality_loss(const ArnoldiResult& r);

/// @brief Largest column residual \f$\lVert Av_j-\sum_i h_{ij}v_i\rVert\f$ of the Arnoldi relation.
double arnoldi_relation_residual(const MatVec& matvec, const ArnoldiResult& r);

}  // namespace datamunge::algorithms
