#pragma once

/// \file shannon_fano_elias.hpp
/// \brief Shannon-Fano-Elias coding: prefix codes from cumulative probabilities.
///
/// Shannon-Fano-Elias coding builds a prefix-free code directly from the *cumulative*
/// distribution. For symbol \f$i\f$ with probability \f$p_i\f$ and cumulative probability
/// \f$F_i\f$ (sum of the earlier probabilities), it takes the *midpoint*
/// \f$\bar F_i = F_i + p_i/2\f$ and emits the first \f$\lceil\log_2(1/p_i)\rceil+1\f$ bits of
/// its binary expansion. Those extra bits guarantee the codewords are prefix-free, at a cost of
/// at most 2 bits above the entropy. Historically it is the conceptual bridge from Shannon-Fano
/// coding to *arithmetic* coding, which drops the per-symbol rounding entirely. This module
/// builds the codebook and encodes/decodes with it.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace datamunge::algorithms {

/// \brief Build the Shannon-Fano-Elias codebook from symbol frequencies.
///
/// \return codeword bits for each symbol (in the given order).
inline std::vector<std::vector<bool>> shannon_fano_elias_codebook(const std::vector<std::uint32_t>& freq) {
    double total = 0;
    for (auto f : freq) total += f;

    std::vector<std::vector<bool>> code(freq.size());
    double                         cum = 0;
    for (std::size_t i = 0; i < freq.size(); ++i) {
        double p    = freq[i] / total;
        double fbar = (cum + freq[i] / 2.0) / total;              // midpoint of the symbol's interval
        int    L    = static_cast<int>(std::ceil(std::log2(1.0 / p))) + 1;

        std::vector<bool> cw;
        double            x = fbar;
        for (int j = 0; j < L; ++j) { x *= 2; int bit = static_cast<int>(x); cw.push_back(bit); x -= bit; }
        code[i] = cw;
        cum += freq[i];
    }
    return code;
}

/// \brief Encode a symbol sequence with the Shannon-Fano-Elias codebook.
inline std::vector<bool> shannon_fano_elias_encode(const std::vector<int>&           symbols,
                                                   const std::vector<std::uint32_t>& freq) {
    auto              code = shannon_fano_elias_codebook(freq);
    std::vector<bool> out;
    for (int s : symbols)
        for (bool b : code[s]) out.push_back(b);
    return out;
}

/// \brief Decode \c count symbols from a Shannon-Fano-Elias bit stream (prefix-free).
inline std::vector<int> shannon_fano_elias_decode(const std::vector<bool>&          bits,
                                                  const std::vector<std::uint32_t>& freq, int count) {
    auto code = shannon_fano_elias_codebook(freq);
    // Map each codeword (as a string key) to its symbol.
    std::map<std::vector<bool>, int> lookup;
    for (std::size_t i = 0; i < code.size(); ++i) lookup[code[i]] = static_cast<int>(i);

    std::vector<int> out;
    std::vector<bool> acc;
    for (std::size_t p = 0; p < bits.size() && (int)out.size() < count; ++p) {
        acc.push_back(bits[p]);
        auto it = lookup.find(acc);
        if (it != lookup.end()) { out.push_back(it->second); acc.clear(); }
    }
    return out;
}

}  // namespace datamunge::algorithms
