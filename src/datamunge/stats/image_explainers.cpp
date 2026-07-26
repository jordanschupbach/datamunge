#include <datamunge/stats/image_explainers.hpp>

#include <datamunge/linalg/eigen.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace datamunge::stats {

namespace {

// The superpixel index a pixel (i, j) belongs to, for an R x C patch grid over an H x W image.
inline std::size_t patch_of(std::size_t i, std::size_t j, std::size_t H, std::size_t W, std::size_t R, std::size_t C) {
    const std::size_t pr = i * R / H;
    const std::size_t pc = j * C / W;
    return pr * C + pc;
}

// Build the image with the "off" superpixels (coalition bit 0) replaced by the baseline value.
linalg::DenseMatrix<double> mask_image(const linalg::DenseMatrix<double>& image, const std::vector<char>& on,
                                       std::size_t R, std::size_t C, double baseline) {
    const std::size_t H = image.rows(), W = image.cols();
    linalg::DenseMatrix<double> out(H, W, 0.0);
    for (std::size_t i = 0; i < H; ++i)
        for (std::size_t j = 0; j < W; ++j)
            out(i, j) = on[patch_of(i, j, H, W, R, C)] ? image(i, j) : baseline;
    return out;
}

// Inverse of a small symmetric matrix via its eigendecomposition (floored for stability).
linalg::DenseMatrix<double> symmetric_inverse(const linalg::DenseMatrix<double>& A, double floor = 1e-10) {
    const auto eig = linalg::jacobi_eigen(A);
    const std::size_t n = A.rows();
    linalg::DenseMatrix<double> inv(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t c = 0; c < n; ++c) {
                const double lam = std::max(eig.eigenvalues[c], floor);
                s += eig.eigenvectors(i, c) * eig.eigenvectors(j, c) / lam;
            }
            inv(i, j) = s;
        }
    return inv;
}

linalg::DenseMatrix<double> expand_to_pixels(const linalg::DenseMatrix<double>& patch, std::size_t H, std::size_t W) {
    const std::size_t R = patch.rows(), C = patch.cols();
    linalg::DenseMatrix<double> out(H, W, 0.0);
    for (std::size_t i = 0; i < H; ++i)
        for (std::size_t j = 0; j < W; ++j) {
            const std::size_t p = patch_of(i, j, H, W, R, C);
            out(i, j) = patch(p / C, p % C);
        }
    return out;
}

} // namespace

LimeImageExplainer::LimeImageExplainer(const linalg::DenseMatrix<double>& image, const ImagePredict& predict,
                                       ImageExplanationOptions options)
    : options_(options), height_(image.rows()), width_(image.cols()) {
    const std::size_t R = options_.patch_rows, C = options_.patch_cols, P = R * C;
    if (P == 0) throw std::invalid_argument("LimeImageExplainer: patch grid must be non-empty");
    if (options_.n_samples < P + 1)
        throw std::invalid_argument("LimeImageExplainer: n_samples must exceed the number of superpixels");

    std::mt19937_64 rng(options_.seed);
    std::bernoulli_distribution coin(0.5);
    const std::size_t N = options_.n_samples;

    // Design matrix rows [1, z_1..z_P], model scores f, and proximity weights.
    linalg::DenseMatrix<double> X(N, P + 1, 0.0);
    std::vector<double> f(N, 0.0), w(N, 0.0);
    const double kw2 = options_.lime_kernel_width * options_.lime_kernel_width;
    for (std::size_t s = 0; s < N; ++s) {
        std::vector<char> on(P);
        std::size_t off = 0;
        for (std::size_t p = 0; p < P; ++p) {
            on[p] = coin(rng) ? 1 : 0;
            if (!on[p]) ++off;
        }
        f[s] = predict(mask_image(image, on, R, C, options_.baseline));
        const double frac_off = static_cast<double>(off) / static_cast<double>(P);
        w[s] = std::exp(-frac_off * frac_off / kw2);
        X(s, 0) = 1.0;
        for (std::size_t p = 0; p < P; ++p) X(s, p + 1) = static_cast<double>(on[p]);
    }

    // Weighted ridge normal equations: (X^T W X + lambda I') beta = X^T W f (intercept unpenalized).
    linalg::DenseMatrix<double> A(P + 1, P + 1, 0.0);
    std::vector<double> b(P + 1, 0.0);
    for (std::size_t a = 0; a < P + 1; ++a) {
        for (std::size_t c = 0; c < P + 1; ++c) {
            double sum = 0.0;
            for (std::size_t s = 0; s < N; ++s) sum += w[s] * X(s, a) * X(s, c);
            A(a, c) = sum;
        }
        if (a >= 1) A(a, a) += options_.lime_l2;
        double sb = 0.0;
        for (std::size_t s = 0; s < N; ++s) sb += w[s] * X(s, a) * f[s];
        b[a] = sb;
    }
    const auto Ainv = symmetric_inverse(A);
    std::vector<double> beta(P + 1, 0.0);
    for (std::size_t a = 0; a < P + 1; ++a) {
        double sum = 0.0;
        for (std::size_t c = 0; c < P + 1; ++c) sum += Ainv(a, c) * b[c];
        beta[a] = sum;
    }

    intercept_ = beta[0];
    patch_weights_ = linalg::DenseMatrix<double>(R, C, 0.0);
    for (std::size_t p = 0; p < P; ++p) patch_weights_(p / C, p % C) = beta[p + 1];

    // Weighted R^2 of the local surrogate.
    double wsum = 0.0, fbar = 0.0;
    for (std::size_t s = 0; s < N; ++s) { wsum += w[s]; fbar += w[s] * f[s]; }
    fbar /= (wsum > 0.0 ? wsum : 1.0);
    double sse = 0.0, sst = 0.0;
    for (std::size_t s = 0; s < N; ++s) {
        double pred = beta[0];
        for (std::size_t p = 0; p < P; ++p) pred += beta[p + 1] * X(s, p + 1);
        sse += w[s] * (f[s] - pred) * (f[s] - pred);
        sst += w[s] * (f[s] - fbar) * (f[s] - fbar);
    }
    local_r2_ = (sst > 0.0) ? 1.0 - sse / sst : 0.0;
}

linalg::DenseMatrix<double> LimeImageExplainer::pixel_relevance() const {
    return expand_to_pixels(patch_weights_, height_, width_);
}

ShapImageExplainer::ShapImageExplainer(const linalg::DenseMatrix<double>& image, const ImagePredict& predict,
                                       ImageExplanationOptions options)
    : options_(options), height_(image.rows()), width_(image.cols()) {
    const std::size_t R = options_.patch_rows, C = options_.patch_cols, P = R * C;
    if (P == 0) throw std::invalid_argument("ShapImageExplainer: patch grid must be non-empty");
    if (P > 20)
        throw std::invalid_argument("ShapImageExplainer: exact SHAP enumerates 2^P coalitions; keep "
                                    "patch_rows*patch_cols <= 20");

    // Evaluate the model on every coalition (bitmask over the P superpixels), cached by bitmask.
    const std::size_t M = std::size_t{1} << P;
    std::vector<double> fval(M, 0.0);
    std::vector<char> on(P);
    for (std::size_t bits = 0; bits < M; ++bits) {
        for (std::size_t p = 0; p < P; ++p) on[p] = (bits >> p) & 1u;
        fval[bits] = predict(mask_image(image, on, R, C, options_.baseline));
    }
    base_value_ = fval[0];
    full_value_ = fval[M - 1];

    // Shapley weights by coalition size s: |S|! (P-|S|-1)! / P!.
    std::vector<double> logfact(P + 1, 0.0);
    for (std::size_t i = 1; i <= P; ++i) logfact[i] = logfact[i - 1] + std::log(static_cast<double>(i));
    std::vector<double> weight(P, 0.0);  // weight for a subset of size s (s = 0..P-1)
    for (std::size_t s = 0; s < P; ++s)
        weight[s] = std::exp(logfact[s] + logfact[P - s - 1] - logfact[P]);

    std::vector<double> phi(P, 0.0);
    for (std::size_t p = 0; p < P; ++p) {
        const std::size_t bit = std::size_t{1} << p;
        for (std::size_t S = 0; S < M; ++S) {
            if (S & bit) continue;  // subsets NOT containing p
            const std::size_t s = static_cast<std::size_t>(std::popcount(S));
            phi[p] += weight[s] * (fval[S | bit] - fval[S]);
        }
    }

    patch_shapley_ = linalg::DenseMatrix<double>(R, C, 0.0);
    for (std::size_t p = 0; p < P; ++p) patch_shapley_(p / C, p % C) = phi[p];
}

linalg::DenseMatrix<double> ShapImageExplainer::pixel_relevance() const {
    return expand_to_pixels(patch_shapley_, height_, width_);
}

} // namespace datamunge::stats
