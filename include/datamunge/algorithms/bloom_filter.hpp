#pragma once

/// \file bloom_filter.hpp
/// \brief Bloom filter: a space-efficient probabilistic set.
///
/// A Bloom filter tests set membership using a bit array and \f$k\f$ hash functions. To *add*
/// an element, set the \f$k\f$ bits its hashes address; to *query*, check whether all \f$k\f$
/// are set. The filter has *no false negatives* -- a stored element always tests positive -- but
/// allows *false positives*, whose rate is tunable by the array size \f$m\f$ and hash count
/// \f$k\f$ relative to the number of elements \f$n\f$. It uses a tiny, fixed number of bits per
/// element regardless of element size, which is why it is used to test k-mer membership in
/// bioinformatics, and to skip disk lookups in databases and caches. This module implements it
/// with Kirsch-Mitzenmacher double hashing.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// A Bloom filter over 64-bit keys (hash strings to 64 bits with \ref bloom_hash64).
class BloomFilter {
  public:
    /// \param num_bits    size of the bit array (m).
    /// \param num_hashes  number of hash probes per element (k).
    BloomFilter(int num_bits, int num_hashes)
        : bits_(num_bits, 0), m_(num_bits), k_(num_hashes) {}

    /// \brief Insert a key.
    void add(std::uint64_t key) {
        auto [h1, h2] = probes(key);
        for (int i = 0; i < k_; ++i) bits_[index(h1, h2, i)] = 1;
    }

    /// \brief Test membership: false means *definitely absent*, true means *probably present*.
    bool maybe_contains(std::uint64_t key) const {
        auto [h1, h2] = probes(key);
        for (int i = 0; i < k_; ++i)
            if (!bits_[index(h1, h2, i)]) return false;   // a clear bit proves absence
        return true;
    }

    /// \brief Fraction of bits currently set.
    double fill_ratio() const {
        std::size_t set = 0;
        for (char b : bits_) set += b;
        return static_cast<double>(set) / static_cast<double>(m_);
    }

    /// \brief Predicted false-positive rate after inserting \c n elements: (1 - e^{-kn/m})^k.
    double expected_false_positive_rate(int n) const {
        double exponent = -static_cast<double>(k_) * n / static_cast<double>(m_);
        return std::pow(1.0 - std::exp(exponent), k_);
    }

    /// \brief Optimal hash count for m bits and n elements: round((m/n) ln 2).
    static int optimal_num_hashes(int m, int n) {
        if (n == 0) return 1;
        int k = static_cast<int>(std::round(static_cast<double>(m) / n * 0.6931471805599453));
        return k < 1 ? 1 : k;
    }

  private:
    std::pair<std::uint64_t, std::uint64_t> probes(std::uint64_t key) const {
        std::uint64_t h1 = splitmix(key);
        std::uint64_t h2 = splitmix(key ^ 0x9E3779B97F4A7C15ull) | 1ull;  // odd, != 0
        return {h1, h2};
    }
    std::size_t index(std::uint64_t h1, std::uint64_t h2, int i) const {
        return static_cast<std::size_t>((h1 + static_cast<std::uint64_t>(i) * h2) % m_);
    }
    static std::uint64_t splitmix(std::uint64_t z) {
        z += 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    std::vector<char> bits_;
    int               m_, k_;
};

/// \brief Hash a string to a 64-bit key (FNV-1a) for use with \ref BloomFilter.
inline std::uint64_t bloom_hash64(const std::string& s) {
    std::uint64_t h = 1469598103934665603ull;   // FNV offset basis
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
}

}  // namespace datamunge::algorithms
