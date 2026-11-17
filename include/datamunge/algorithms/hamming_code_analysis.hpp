#pragma once
/// @file hamming_code_analysis.hpp
/// @brief Fixed exhaustive Hamming(7,4) codebook and actual binary error-pattern audit.
#include <bit>
#include <cmath>
#include <datamunge/algorithms/hamming_code.hpp>
namespace datamunge::algorithms {
/// @brief Fixed exhaustive codebook and decoder outcomes grouped by actual channel error weight.
/// Totals include all16 transmitted codewords and128 masks (2048 decodings).
/// Results describe this deterministic decoder, not an assumed real-channel law.
/// No heap, random draws or caller references; weight arrays have indices0..7.
/// @par Example
/// @code
/// const auto a=dm::algorithms::hamming74_audit();
/// std::print("{} {}\n",a.minimum_distance,a.recovered[1]); // 3 112
/// @endcode
/// @see hamming74_audit()
struct Hamming74Audit {
  std::array<std::array<int, 7>, 16> codewords{};        ///< Nibble IDs0..15, d1 most significant.
  std::array<std::size_t, 8>         codeword_weights{}; ///< Codeword counts by number of set bits.
  int minimum_distance{7}; ///< Smallest differing positions among all120 codeword pairs.
  std::array<std::size_t, 8> patterns{}; ///< Actual transmitted-word/mask pairs per weight.
  std::array<std::size_t, 8> nonzero_syndromes{}; ///< Pairs where received syndrome is nonzero.
  std::array<std::size_t, 8> recovered{}; ///< Pairs whose decoded data equal transmitted data.
};
/// @brief Exhaustively audit the actual fixed codec over its complete binary state space.
/// Builds16 codewords, checks120pair distances, then decodes2048word/mask pairs.
/// There is no sampled approximation or second hand-coded decoder. Constant
/// bounded work/storage for this fixed code; no callbacks, heap or shared state.
/// @return Owned codebook, weight enumerator, minimum distance and actual grouped outcomes.
/// @par Example
/// @code
/// const auto a=dm::algorithms::hamming74_audit();
/// std::print("{} {}\n",a.patterns[2],a.recovered[2]); // 336 0
/// @endcode
/// @par References
/// - R. W. Hamming (1950), https://doi.org/10.1002/j.1538-7305.1950.tb00463.x .
/// @see Hamming74Audit, hamming74_decode_trace()
inline Hamming74Audit hamming74_audit() {
  Hamming74Audit a;
  for (int id = 0; id < 16; ++id) {
    std::array<int, 4> data{(id >> 3) & 1, (id >> 2) & 1, (id >> 1) & 1, id & 1};
    a.codewords[id] = hamming74_encode(data);
    int w           = 0;
    for (int bit : a.codewords[id])
      w += bit;
    ++a.codeword_weights[w];
  }
  for (int i = 0; i < 16; ++i)
    for (int j = i + 1; j < 16; ++j) {
      int d = 0;
      for (int k = 0; k < 7; ++k)
        d += a.codewords[i][k] != a.codewords[j][k];
      if (d < a.minimum_distance)
        a.minimum_distance = d;
    }
  for (int id = 0; id < 16; ++id)
    for (unsigned mask = 0; mask < 128; ++mask) {
      auto c = a.codewords[id];
      for (int k = 0; k < 7; ++k)
        c[k] ^= (mask >> k) & 1;
      const auto t = hamming74_decode_trace(c);
      const auto w = std::popcount(mask);
      ++a.patterns[w];
      a.nonzero_syndromes[w] += t.syndrome != 0;
      a.recovered[w] +=
          t.decoded.data == std::array<int, 4>{(id >> 3) & 1, (id >> 2) & 1, (id >> 1) & 1, id & 1};
    }
  return a;
}
/// @brief Evaluate exact-model block recovery probability on an independent binary symmetric channel.
/// Assumes each of seven bits flips independently with the same probability p,
/// and this bounded-distance decoder: only weights0or1 recover the transmitted
/// data. Evaluates (1-p)^7+7p(1-p)^6 in double; not simulation, confidence bounds
/// or a statement about correlated/unequal real errors. p0and1 permitted.
/// @param p Finite per-bit flip probability in[0,1].
/// @return Floating evaluation of the model's exact recovery probability.
/// @throws std::invalid_argument On nonfinite/out-of-range p.
/// @par Example
/// @code
/// std::print("{:.6f}\n",dm::algorithms::hamming74_bsc_recovery(.1)); // 0.850306
/// @endcode
/// @see hamming74_audit()
inline double hamming74_bsc_recovery(double p) {
  if (!std::isfinite(p) || p < 0 || p > 1)
    throw std::invalid_argument("BSC probability outside [0,1]");
  return std::pow(1 - p, 7) + 7 * p * std::pow(1 - p, 6);
}
} // namespace datamunge::algorithms
