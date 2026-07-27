#pragma once

#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// @brief The *Gram-Schmidt process*: turns a set of input vectors (the rows of @p vectors) into an
///        *orthonormal* set spanning the same subspace. Each vector has the components already lying
///        along the previously produced orthonormal vectors subtracted off, then is normalized. This
///        implementation uses the *modified* Gram-Schmidt ordering, which subtracts one projection at
///        a time and is markedly more numerically stable than the classical formula. Vectors that are
///        linearly dependent on the earlier ones (near-zero remainder) are dropped, so the output is
///        an orthonormal basis of the span. Runs in @c O(k^2 d) for @c k vectors of dimension @c d.
///
/// @param vectors the input vectors, one per row (all the same length).
/// @return an orthonormal set of vectors (rows) spanning the same subspace.
std::vector<std::vector<double>> gram_schmidt(const std::vector<std::vector<double>>& vectors);

/// @brief A single eigenvalue with its eigenvector, as returned by @ref power_iteration.
struct EigenPair {
    double              value{0.0}; ///< the (dominant) eigenvalue.
    std::vector<double> vector;     ///< a corresponding unit eigenvector.
};

/// @brief *Power iteration*: finds the eigenvalue of largest magnitude of @p matrix and a
///        corresponding eigenvector, by repeatedly multiplying a random start vector by the matrix
///        and renormalizing. The iterate converges to the dominant eigenvector (the direction the
///        matrix stretches most), and the *Rayleigh quotient* @c v^T A v / v^T v gives the
///        eigenvalue. Convergence is linear at rate @c |lambda_2 / lambda_1| (the ratio of the two
///        largest eigenvalues), so it is fast when the dominant eigenvalue is well separated. Intended
///        for symmetric matrices, where the eigenvalues are real. Runs in @c O(iterations * n^2).
///
/// @param matrix the (square, ideally symmetric) matrix.
/// @param iterations the maximum number of iterations.
/// @param seed the seed for the random start vector (so results are reproducible).
/// @return the dominant @ref EigenPair.
EigenPair power_iteration(const std::vector<std::vector<double>>& matrix, int iterations = 300,
                          std::uint64_t seed = 0x9E3779B97F4A7C15ULL);

/// @brief The *Strassen algorithm* for matrix multiplication: computes @c A*B using only *seven*
///        recursive multiplications of half-size blocks instead of the schoolbook eight, at the cost
///        of more additions, giving @c O(n^{log2 7}) ~ @c O(n^{2.807}) instead of @c O(n^3). Operands
///        are zero-padded to a power-of-two square size, multiplied recursively (falling back to the
///        schoolbook method below a small block size), and the result is trimmed. Rectangular
///        operands are supported by padding.
///
/// @param a the left factor (@c m x @c k).
/// @param b the right factor (@c k x @c n).
/// @return the product @c A*B (@c m x @c n); empty if the inner dimensions do not match.
std::vector<std::vector<double>> strassen_multiply(const std::vector<std::vector<double>>& a,
                                                   const std::vector<std::vector<double>>& b);

/// @brief *Freivalds' algorithm*: a randomized *verifier* for a matrix product. Rather than recompute
///        @c A*B (which costs @c O(n^3) or, with Strassen, @c O(n^{2.807})), it checks a claimed
///        product @p c in @c O(n^2) per round by testing @c A(Br) == Cr for a random 0/1 vector @c r.
///        If @c A*B = C the test always passes; if @c A*B != C each round catches the discrepancy with
///        probability at least 1/2, so @p rounds independent rounds reduce the chance of a false
///        "equal" to at most @c 2^{-rounds}.
///
/// @param a,b,c the matrices with @c A (@c m x @c k), @c B (@c k x @c n), and claimed product
///        @c C (@c m x @c n).
/// @param rounds the number of random rounds.
/// @param seed the seed for the random vectors.
/// @return false if a round proves @c A*B != C; true if all rounds pass (probably equal).
bool freivalds_verify(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b,
                      const std::vector<std::vector<double>>& c, int rounds = 20,
                      std::uint64_t seed = 0xD1B54A32D192ED03ULL);

} // namespace datamunge::algorithms
