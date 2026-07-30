#pragma once

/// \file pcnn.hpp
/// \brief Pulse-Coupled Neural Network (PCNN): a biomimetic spiking model of the
///        visual cortex used for image processing (Eckhorn 1990; Johnson & Padgett 1999).
///
/// A PCNN lays one neuron over each pixel. Every neuron has a *feeding* input (the
/// pixel stimulus \f$S\f$), a *linking* input from its firing neighbors, a combined
/// *internal activity* \f$U\f$, a *dynamic threshold* \f$\Theta\f$, and a binary
/// *pulse* output \f$Y\f$. A neuron fires when its activity exceeds its threshold;
/// firing then raises that threshold (a refractory jump), while the threshold decays
/// over time. Two effects make it useful for vision:
///   - *intensity ordering in time*: with a decaying threshold, brighter pixels
///     (higher \f$U\f$) fire earlier -- so the *iteration at which a pixel first fires*
///     segments the image by intensity; and
///   - *synchronization by linking*: a fired neuron nudges its neighbors' activity up,
///     so contiguous regions of similar intensity tend to pulse *together*, capturing
///     spatial connectivity.
///
/// This uses the common simplified (spiking-cortical-model) update, per iteration \f$n\f$:
/// \f[
///   U = S\,(1+\beta L),\quad \Theta \leftarrow \alpha_\Theta\,\Theta,\quad
///   Y = [\,U > \Theta\,],\quad \Theta \leftarrow \Theta + V_\Theta\,Y,
/// \f]
/// where \f$L\f$ sums the previous pulses of the 8-connected neighbors.

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// Output of a PCNN run over an image.
struct PCNNResult {
    std::vector<std::vector<std::vector<int>>> pulses;         ///< pulses[n] is the binary pulse map at iteration n.
    std::vector<int>                           time_signature; ///< Number of neurons firing at each iteration.
    std::vector<std::vector<int>>              first_fire;      ///< Iteration (1-based) each pixel first fired (0 = never).
};

/// Hyper-parameters for the PCNN.
struct PCNNParameters {
    std::size_t iterations       = 20;
    double      beta             = 0.2;   ///< Linking strength.
    double      threshold_decay  = 0.8;   ///< Multiplicative threshold decay \f$\alpha_\Theta\f$.
    double      threshold_gain   = 20.0;  ///< Refractory threshold jump \f$V_\Theta\f$ on firing.
    double      initial_threshold = 1.0;  ///< Starting threshold (>= max stimulus so nothing fires at n=0).
};

/// \brief Run a PCNN over a grayscale image (values ideally normalized to [0, 1]).
///
/// \param image   Row-major grayscale intensities \f$S_{ij}\f$.
/// \param params  Dynamics parameters.
inline PCNNResult pcnn_run(const std::vector<std::vector<double>>& image, const PCNNParameters& params) {
    PCNNResult result;
    const std::size_t rows = image.size();
    const std::size_t cols = rows ? image.front().size() : 0;
    if (rows == 0 || cols == 0) return result;

    std::vector<std::vector<double>> theta(rows, std::vector<double>(cols, params.initial_threshold));
    std::vector<std::vector<int>>    Y(rows, std::vector<int>(cols, 0));
    result.first_fire.assign(rows, std::vector<int>(cols, 0));

    for (std::size_t n = 1; n <= params.iterations; ++n) {
        std::vector<std::vector<int>> Ynext(rows, std::vector<int>(cols, 0));
        int                           fired = 0;
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < cols; ++j) {
                // Linking: sum of previous pulses over 8-connected neighbors.
                double L = 0.0;
                for (int di = -1; di <= 1; ++di)
                    for (int dj = -1; dj <= 1; ++dj) {
                        if (di == 0 && dj == 0) continue;
                        const int ni = static_cast<int>(i) + di, nj = static_cast<int>(j) + dj;
                        if (ni >= 0 && ni < static_cast<int>(rows) && nj >= 0 && nj < static_cast<int>(cols))
                            L += Y[static_cast<std::size_t>(ni)][static_cast<std::size_t>(nj)];
                    }
                const double U = image[i][j] * (1.0 + params.beta * L);
                theta[i][j] *= params.threshold_decay;
                const int y = (U > theta[i][j]) ? 1 : 0;
                if (y) {
                    theta[i][j] += params.threshold_gain;
                    ++fired;
                    if (result.first_fire[i][j] == 0) result.first_fire[i][j] = static_cast<int>(n);
                }
                Ynext[i][j] = y;
            }
        Y = Ynext;
        result.pulses.push_back(Y);
        result.time_signature.push_back(fired);
    }
    return result;
}

}  // namespace datamunge::algorithms
