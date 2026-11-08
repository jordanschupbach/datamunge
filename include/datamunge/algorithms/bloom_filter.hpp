#pragma once
/// @file bloom_filter.hpp
/// @brief Packed, checked Bloom membership filter with inspection of its canonical double-hash probes.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::algorithms {
/// @brief Owned approximate set of uint64 keys, using packed bits and deterministic double hashing.
/// Insert/query use exactly the same probes, so inserted keys cannot test negative
/// while the filter remains unmodified except by insertions. Positive queries
/// require an exact backing-set check. There is no erase, resizing or insertion
/// count: duplicates are idempotent and the object retains no original keys.
/// Storage is 64*ceil(m/64) payload bits plus object/allocation bookkeeping.
/// Copies own independent state; moved-from objects support destruction/assignment only.
/// Mutation requires external synchronisation, while
/// concurrent const reads are safe when no thread mutates the object.
/// Probes retain the existing uint64 mapping: (h1+i*h2) wraps modulo 2^64 before
/// reduction modulo m. Odd h2 need not give distinct probes for arbitrary m.
/// @par Example
/// @code
/// dm::algorithms::BloomFilter filter(8000,6);
/// filter.add(42);
/// std::print("{} {}\n",filter.maybe_contains(42),filter.storage_bytes()); // true 1000
/// @endcode
/// @par References
/// - B. H. Bloom (1970), Space/time trade-offs in hash coding with allowable errors,
///   Communications of the ACM 13(7), 422-426, doi:10.1145/362686.362692.
/// - A. Kirsch and M. Mitzenmacher (2008), Less hashing, same performance: Building
///   a better Bloom filter, Random Structures & Algorithms 33(2), 187-218,
///   doi:10.1002/rsa.20208. The finite wrapped probes here are not independent hashes.
/// @see bloom_hash64(), probe_positions(), expected_false_positive_rate()
class BloomFilter {
  public:
    /// @brief Create an empty packed filter with positive logical bit/probe counts.
    /// Validates before allocation; O(ceil(m/64)) zero-initialisation and storage.
    /// @param num_bits Positive logical bit count m, including nonmultiples of 64.
    /// @param num_hashes Positive probe count k; repeated probes are allowed.
    /// @throws std::invalid_argument If either dimension is nonpositive.
    /// @par Example
    /// @code
    /// dm::algorithms::BloomFilter filter(65,3);
    /// std::print("{} {}\n",filter.num_bits(),filter.storage_bytes()); // 65 16
    /// @endcode
    /// @see BloomFilter, num_bits(), num_hashes()
    BloomFilter(int num_bits,int num_hashes)
        : bits_(checked_words(num_bits,num_hashes),0),m_(num_bits),k_(num_hashes) {}

    /// @brief Set every canonical probe bit for a key, leaving all other bits unchanged.
    /// Duplicate keys/probes do not change already-set bits. O(k) work, no allocation;
    /// an insertion cannot remove any prior membership evidence.
    /// @param key Any uint64 value, including zero and UINT64_MAX.
    /// @par Example
    /// @code
    /// dm::algorithms::BloomFilter filter(64,3);filter.add(7);filter.add(7);
    /// std::print("{}\n",filter.maybe_contains(7)); // true
    /// @endcode
    /// @see maybe_contains(), probe_positions(), num_set_bits()
    void add(std::uint64_t key) {
        const auto [h1,h2]=probes(key);
        for (int i=0;i<k_;++i) {
            const auto position=index(h1,h2,i),word=position/64;
            const auto mask=std::uint64_t{1}<<(position%64);
            if (!(bits_[word]&mask)) {bits_[word]|=mask;++set_;}
        }
    }

    /// @brief Return false on the first clear probe, or true when every probe bit is set.
    /// False proves this uint64 key has not been inserted. True permits false positives;
    /// hash collisions between original strings also require exact verification.
    /// Const, allocation-free, O(k) worst-case work with early rejection.
    /// @param key Any uint64 key queried with the same hashing convention as insertion.
    /// @return Possibly present when true, definitely absent when false.
    /// @par Example
    /// @code
    /// dm::algorithms::BloomFilter filter(64,3);
    /// std::print("{}\n",filter.maybe_contains(7)); // false
    /// @endcode
    /// @see add(), probe_positions()
    bool maybe_contains(std::uint64_t key) const {
        const auto [h1,h2]=probes(key);
        for (int i=0;i<k_;++i) if (!bit_unchecked(index(h1,h2,i))) return false;
        return true;
    }

    /// @brief Inspect all canonical probe positions, including repetitions, without mutation.
    /// Uses the same mixer and index function as add()/maybe_contains(). O(k) time
    /// and owned output storage; positions lie in [0,num_bits()).
    /// @param key Any uint64 key.
    /// @return Probe positions in i=0,...,k-1 order; values are not deduplicated.
    /// @par Example
    /// @code
    /// dm::algorithms::BloomFilter filter(1,3);
    /// std::print("{}\n",filter.probe_positions(42)); // [0, 0, 0]
    /// @endcode
    /// @see bit_is_set(), add()
    std::vector<int> probe_positions(std::uint64_t key) const {
        const auto [h1,h2]=probes(key);
        std::vector<int> result;result.reserve(k_);
        for (int i=0;i<k_;++i) result.push_back(static_cast<int>(index(h1,h2,i)));
        return result;
    }

    /// @brief Read one logical bit, rejecting padding and out-of-range coordinates.
    /// Const O(1) access; packed padding is never exposed as membership evidence.
    /// @param position Zero-based logical position in [0,num_bits()).
    /// @return Whether that bit has been set by an insertion.
    /// @throws std::out_of_range If position is negative or >=num_bits().
    /// @par Example
    /// @code
    /// dm::algorithms::BloomFilter filter(1,2);filter.add(42);
    /// std::print("{}\n",filter.bit_is_set(0)); // true
    /// @endcode
    /// @see probe_positions(), num_set_bits()
    bool bit_is_set(int position) const {
        if (position<0 || position>=m_) throw std::out_of_range("Bloom bit position out of range");
        return bit_unchecked(static_cast<std::size_t>(position));
    }

    /// @brief Return the exact fraction of logical bits set, using an insertion-maintained count.
    /// O(1), const, no allocation. Counts each bit once and excludes packed padding.
    /// @return set bits divided by m, in [0,1]; zero for a newly constructed filter.
    /// @par Example
    /// @code
    /// dm::algorithms::BloomFilter filter(1,2);filter.add(42);
    /// std::print("{}\n",filter.fill_ratio()); // 1
    /// @endcode
    /// @see num_set_bits(), expected_false_positive_rate()
    double fill_ratio() const noexcept {return static_cast<double>(set_)/m_;}

    /// @brief Evaluate the approximate independent-hash model (1-exp(-k*n/m))^k.
    /// Uses -expm1 for stability at light load. This model is not an exact finite
    /// probability or a guarantee for deterministic, correlated wrapped probes.
    /// The caller supplies distinct inserted keys; duplicate calls do not add load.
    /// O(1), const; large loads may round to probability one.
    /// @param n Nonnegative number of distinct inserted keys, not stored by the filter.
    /// @return Approximate false-positive probability in [0,1]; zero at n=0.
    /// @throws std::invalid_argument If n<0.
    /// @par Example
    /// @code
    /// dm::algorithms::BloomFilter filter(8000,6);
    /// std::print("{:.6f}\n",filter.expected_false_positive_rate(1000)); // 0.021577
    /// @endcode
    /// @par References
    /// - B. H. Bloom (1970), Space/time trade-offs in hash coding with allowable
    ///   errors, doi:10.1145/362686.362692; independent-placement approximation here.
    /// @see optimal_num_hashes(), fill_ratio()
    double expected_false_positive_rate(int n) const {
        if (n<0) throw std::invalid_argument("Bloom insertion count must be nonnegative");
        const double exponent=-static_cast<double>(k_)*n/m_;
        return std::pow(-std::expm1(exponent),k_);
    }

    /// @brief Return the positive logical bit count fixed at construction, in O(1).
    /// @return Requested m, excluding padding bits.
    [[nodiscard]] int num_bits() const noexcept {return m_;}
    /// @brief Return the positive probe count fixed at construction, in O(1).
    /// @return Requested k, including repeated probe positions.
    [[nodiscard]] int num_hashes() const noexcept {return k_;}
    /// @brief Return the maintained count of distinct set logical bits, in O(1).
    /// @return Number in [0,num_bits()], excluding padding.
    [[nodiscard]] std::size_t num_set_bits() const noexcept {return set_;}
    /// @brief Return the allocated bit-vector payload size, excluding object/allocator bookkeeping.
    /// @return 8*ceil(num_bits()/64) bytes; capacity is fixed after construction.
    [[nodiscard]] std::size_t storage_bytes() const noexcept {return bits_.size()*sizeof(std::uint64_t);}

    /// @brief Round the model's continuous optimum (m/n)*ln(2), clamped to at least one.
    /// This is a sizing rule, not an exact optimiser for finite wrapped probes.
    /// Returns one for n=0. O(1); no filter is constructed or modified.
    /// @param m Positive logical bit budget.
    /// @param n Nonnegative expected distinct key count.
    /// @return Positive rounded probe count; ties round upward.
    /// @throws std::invalid_argument If m<=0 or n<0.
    /// @par Example
    /// @code
    /// std::print("{}\n",dm::algorithms::BloomFilter::optimal_num_hashes(8000,1000)); // 6
    /// @endcode
    /// @par References
    /// - A. Kirsch and M. Mitzenmacher (2008), Less hashing, same performance:
    ///   Building a better Bloom filter, doi:10.1002/rsa.20208; standard continuous sizing model.
    /// @see expected_false_positive_rate(), BloomFilter()
    static int optimal_num_hashes(int m,int n) {
        if (m<=0 || n<0) throw std::invalid_argument("Bloom sizing requires m>0 and n>=0");
        if (n==0) return 1;
        const int k=static_cast<int>(std::round(static_cast<double>(m)/n*0.6931471805599453));
        return k<1?1:k;
    }

  private:
    /// @brief Validate dimensions before calculating the overflow-safe packed word count.
    /// @param m Positive bit count.
    /// @param k Positive probe count.
    /// @return ceil(m/64), with conversion before addition.
    /// @throws std::invalid_argument If m<=0 or k<=0.
    static std::size_t checked_words(int m,int k) {
        if (m<=0 || k<=0) throw std::invalid_argument("Bloom dimensions must be positive");
        return (static_cast<std::size_t>(m)+63)/64;
    }
    /// @brief Derive the two deterministic mix values used by every probe consumer.
    /// @param key Any uint64 value.
    /// @return Mixed base and an odd nonzero step; not a proof of independent uniform hashing.
    static std::pair<std::uint64_t,std::uint64_t> probes(std::uint64_t key) {
        return {splitmix(key),splitmix(key^0x9E3779B97F4A7C15ull)|1ull};
    }
    /// @brief Compute the historical wrapped uint64 affine probe, then reduce modulo m.
    /// @param h1 Mixed base.
    /// @param h2 Odd mixed step.
    /// @param i Nonnegative probe number below k.
    /// @return Logical bit index in [0,m).
    std::size_t index(std::uint64_t h1,std::uint64_t h2,int i) const {
        return static_cast<std::size_t>((h1+static_cast<std::uint64_t>(i)*h2)%static_cast<std::uint64_t>(m_));
    }
    /// @brief Read a valid logical bit from the packed word vector.
    /// @param position Validated logical position.
    /// @return Whether the corresponding mask is set.
    bool bit_unchecked(std::size_t position) const {return (bits_[position/64]>>(position%64))&1ull;}
    /// @brief SplitMix64 output transformation, with all operations modulo 2^64.
    /// @param z Original integer key or salted integer key.
    /// @return Deterministically mixed uint64 value.
    /// @par References
    /// - S. Vigna (2015), public-domain SplitMix64 implementation,
    ///   https://prng.di.unimi.it/splitmix64.c; the stateless output transformation is used here.
    static std::uint64_t splitmix(std::uint64_t z) {
        z+=0x9E3779B97F4A7C15ull;
        z=(z^(z>>30))*0xBF58476D1CE4E5B9ull;
        z=(z^(z>>27))*0x94D049BB133111EBull;
        return z^(z>>31);
    }
    std::vector<std::uint64_t> bits_; ///< Owned zero-initialised words; unused high padding stays zero.
    int m_; ///< Positive logical bit count.
    int k_; ///< Positive probe count.
    std::size_t set_{0}; ///< Distinct set-bit count, maintained on zero-to-one transitions.
};

/// @brief Hash the complete string byte sequence with standard 64-bit FNV-1a.
/// Includes embedded NUL bytes and unsigned high bytes, with modulo-2^64
/// arithmetic and offset basis 14695981039346656037. This corrects the previous
/// nonstandard basis, so hashes of strings change; callers retaining old hashes
/// must rebuild them. O(length) work, O(1) storage, no mutation or shared state.
/// @param s Arbitrary string bytes; empty returns the offset basis.
/// @return Non-cryptographic uint64 key; collisions still require exact set verification.
/// @par Example
/// @code
/// std::print("{:016x}\n",dm::algorithms::bloom_hash64("hello")); // a430d84680aabd0b
/// @endcode
/// @par References
/// - Noll, Vo, Eastlake and Hansen (2026), The FNV Non-Cryptographic Hash Algorithm,
///   RFC 9923, https://www.rfc-editor.org/rfc/rfc9923.html.
/// @see BloomFilter, BloomFilter::add()
inline std::uint64_t bloom_hash64(const std::string& s) {
    std::uint64_t h=14695981039346656037ull;
    for (unsigned char c:s) {h^=c;h*=1099511628211ull;}
    return h;
}
} // namespace datamunge::algorithms
