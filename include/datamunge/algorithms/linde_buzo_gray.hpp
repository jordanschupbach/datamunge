#pragma once

/// \file linde_buzo_gray.hpp
/// \brief The Linde-Buzo-Gray (LBG) algorithm for vector-quantization codebook
///        design by successive splitting (Linde, Buzo & Gray 1980).
///
/// *Vector quantization* compresses data by replacing each vector with the nearest
/// entry ("codeword") of a small *codebook*; the design problem is to choose a
/// codebook of a given size that minimizes the average distortion
/// \f$D = \frac1n\sum_i \min_j \|x_i - c_j\|^2\f$. LBG builds such a codebook by
/// *doubling*: start from the single centroid of all the data, then repeatedly
///   1. *split* every codeword \f$c\f$ into two nearby ones \f$c(1\pm\epsilon)\f$, and
///   2. run Lloyd's algorithm (assign to nearest codeword, recompute centroids) to
///      convergence,
/// until the codebook reaches the target size (a power of two). Splitting gives good,
/// data-adapted initializations at each size and yields a nested family of codebooks.
/// LBG is the generalized Lloyd algorithm and is essentially k-means with a principled
/// splitting initialization; it underlies classic speech and image VQ codecs.

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace datamunge::algorithms {

/// A vector-quantization codebook and its distortion on the training data.
struct LBGResult {
    std::vector<std::vector<double>> codebook;    ///< Codewords (centroids).
    std::vector<std::size_t>         assignment;  ///< Nearest codeword index per training point.
    double                           distortion = 0.0;  ///< Mean squared distance to nearest codeword.
    std::vector<double>              distortion_history;  ///< Distortion at each codebook size 1,2,4,...
};

namespace detail {

inline double lbg_sq_dist(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

}  // namespace detail

/// \brief Design a VQ codebook of size \p codebook_size (rounded down to a power of two
///        by the doubling process) with the LBG splitting algorithm.
///
/// \param data           Training vectors.
/// \param codebook_size  Target number of codewords (the doubling stops once reached).
/// \param epsilon        Split perturbation fraction.
/// \param lloyd_iters    Max Lloyd iterations per codebook size.
/// \param tol            Relative distortion-change tolerance to stop Lloyd early.
inline LBGResult linde_buzo_gray(const std::vector<std::vector<double>>& data, std::size_t codebook_size,
                                 double epsilon = 0.01, std::size_t lloyd_iters = 100, double tol = 1e-6) {
    LBGResult out;
    if (data.empty() || codebook_size == 0) return out;
    const std::size_t dim = data.front().size();

    // Initial codebook: the centroid of all data.
    std::vector<std::vector<double>> codebook(1, std::vector<double>(dim, 0.0));
    for (const auto& x : data)
        for (std::size_t d = 0; d < dim; ++d) codebook[0][d] += x[d];
    for (std::size_t d = 0; d < dim; ++d) codebook[0][d] /= static_cast<double>(data.size());

    std::vector<std::size_t> assignment(data.size(), 0);

    auto lloyd = [&]() {
        double prev = std::numeric_limits<double>::infinity();
        double dist = 0.0;
        for (std::size_t it = 0; it < lloyd_iters; ++it) {
            // Assignment step.
            dist = 0.0;
            for (std::size_t i = 0; i < data.size(); ++i) {
                std::size_t best = 0;
                double      bd   = std::numeric_limits<double>::infinity();
                for (std::size_t j = 0; j < codebook.size(); ++j) {
                    const double dd = detail::lbg_sq_dist(data[i], codebook[j]);
                    if (dd < bd) {
                        bd   = dd;
                        best = j;
                    }
                }
                assignment[i] = best;
                dist += bd;
            }
            dist /= static_cast<double>(data.size());
            // Update step.
            std::vector<std::vector<double>> sum(codebook.size(), std::vector<double>(dim, 0.0));
            std::vector<std::size_t>         cnt(codebook.size(), 0);
            for (std::size_t i = 0; i < data.size(); ++i) {
                for (std::size_t d = 0; d < dim; ++d) sum[assignment[i]][d] += data[i][d];
                ++cnt[assignment[i]];
            }
            for (std::size_t j = 0; j < codebook.size(); ++j)
                if (cnt[j] > 0)
                    for (std::size_t d = 0; d < dim; ++d) codebook[j][d] = sum[j][d] / static_cast<double>(cnt[j]);
            if (prev - dist < tol * (prev + 1e-12)) break;
            prev = dist;
        }
        return dist;
    };

    double dist = lloyd();
    out.distortion_history.push_back(dist);

    // Doubling: split then re-optimize until the target size is reached.
    while (codebook.size() < codebook_size) {
        std::vector<std::vector<double>> next;
        next.reserve(codebook.size() * 2);
        for (const auto& c : codebook) {
            std::vector<double> lo = c, hi = c;
            for (std::size_t d = 0; d < dim; ++d) {
                lo[d] *= (1.0 - epsilon);
                hi[d] *= (1.0 + epsilon);
            }
            next.push_back(lo);
            next.push_back(hi);
        }
        codebook.swap(next);
        dist = lloyd();
        out.distortion_history.push_back(dist);
    }

    out.codebook   = codebook;
    out.assignment = assignment;
    out.distortion = dist;
    return out;
}

/// Quantize a vector to the index of its nearest codeword.
inline std::size_t lbg_quantize(const LBGResult& vq, const std::vector<double>& x) {
    std::size_t best = 0;
    double      bd   = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < vq.codebook.size(); ++j) {
        const double d = detail::lbg_sq_dist(x, vq.codebook[j]);
        if (d < bd) {
            bd   = d;
            best = j;
        }
    }
    return best;
}

}  // namespace datamunge::algorithms
