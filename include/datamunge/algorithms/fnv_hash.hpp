#pragma once

/// \file fnv_hash.hpp
/// \brief The Fowler-Noll-Vo (FNV-1a) non-cryptographic hash function.
///
/// FNV is a fast, simple hash for hash tables and checksums. The FNV-1a variant folds
/// each byte into the hash by *XOR then multiply* by a large prime:
/// \f[
///   h \leftarrow (h \oplus b)\times \text{FNV\_prime},
/// \f]
/// starting from a fixed *offset basis*. The XOR-before-multiply order (1a) gives
/// better avalanche than the original FNV-1 (multiply-before-XOR). It has excellent
/// dispersion and speed for short keys, which is why it is a common default hash for
/// symbol tables, though -- being non-cryptographic and invertible -- it must not be used
/// where collision resistance against an adversary matters.

#include <cstddef>
#include <cstdint>
#include <string>

namespace datamunge::algorithms {

/// \brief 64-bit FNV-1a hash of a byte string.
///
/// Offset basis 14695981039346656037, prime 1099511628211. The empty string hashes to
/// the offset basis.
inline std::uint64_t fnv1a_64(const std::string& data) {
    std::uint64_t h = 14695981039346656037ULL;  // FNV offset basis
    for (unsigned char byte : data) {
        h ^= static_cast<std::uint64_t>(byte);
        h *= 1099511628211ULL;
    }
    return h;
}

/// \brief 32-bit FNV-1a hash of a byte string (offset basis 2166136261, prime 16777619).
inline std::uint32_t fnv1a_32(const std::string& data) {
    std::uint32_t h = 2166136261u;
    for (unsigned char byte : data) {
        h ^= static_cast<std::uint32_t>(byte);
        h *= 16777619u;
    }
    return h;
}

}  // namespace datamunge::algorithms
