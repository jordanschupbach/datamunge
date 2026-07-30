#pragma once

/// \file pearson_hashing.hpp
/// \brief Pearson hashing: an 8-bit hash built from a single 256-byte permutation
///        table (Pearson 1990).
///
/// Pearson hashing produces an 8-bit hash of a byte string using nothing but a fixed
/// *permutation of the 256 byte values* and one table lookup per input byte:
/// \f[
///   h \leftarrow T\big[\,h \oplus b\,\big],
/// \f]
/// starting from \f$h=0\f$. Because \f$T\f$ is a permutation, the hash is
/// well-distributed and, crucially, has the property that changing any input byte
/// changes the output (no trivial collisions from the mixing step). It was designed for
/// 8-bit machines -- no multiplies, no wide arithmetic -- and extends to wider hashes by
/// running it several times with different initial offsets. This implementation ships
/// the widely used reference permutation table and also supports a caller-supplied one.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace datamunge::algorithms {

/// The standard 256-entry Pearson permutation table (from Pearson's 1990 paper).
inline const std::array<std::uint8_t, 256>& pearson_default_table() {
    static const std::array<std::uint8_t, 256> T = {
        98,  6,   85,  150, 36,  23,  112, 164, 135, 207, 169, 5,   26,  64,  165, 219, 61,  20,  68,  89,  130,
        63,  52,  102, 24,  229, 132, 245, 80,  216, 195, 115, 90,  168, 156, 203, 177, 120, 2,   190, 188, 7,
        100, 185, 174, 243, 162, 10,  237, 18,  253, 225, 8,   208, 172, 244, 255, 126, 101, 79,  145, 235, 228,
        121, 123, 251, 67,  250, 161, 0,   107, 97,  241, 111, 181, 82,  249, 33,  69,  55,  59,  153, 29,  9,
        213, 167, 84,  93,  30,  46,  94,  75,  151, 114, 73,  222, 197, 96,  210, 45,  16,  227, 248, 202, 51,
        152, 252, 125, 81,  206, 215, 186, 39,  158, 178, 187, 131, 136, 1,   49,  50,  17,  141, 91,  47,  129,
        60,  99,  154, 35,  86,  171, 105, 34,  38,  200, 147, 58,  77,  118, 173, 246, 76,  254, 133, 232, 196,
        144, 198, 124, 53,  4,   108, 74,  223, 234, 134, 230, 157, 139, 189, 205, 199, 128, 176, 19,  211, 236,
        127, 192, 231, 70,  233, 88,  146, 44,  183, 201, 22,  83,  13,  214, 116, 109, 159, 32,  95,  226, 140,
        220, 57,  12,  221, 31,  209, 182, 143, 92,  149, 184, 148, 62,  113, 65,  37,  27,  106, 166, 3,   14,
        204, 72,  21,  41,  56,  66,  28,  193, 40,  217, 25,  54,  179, 117, 238, 87,  240, 155, 180, 170, 242,
        212, 191, 163, 78,  218, 137, 194, 175, 110, 43,  119, 224, 71,  122, 142, 42,  160, 104, 48,  247, 103,
        15,  11,  138, 239};
    return T;
}

/// \brief 8-bit Pearson hash of a byte string using the given (default: standard) table.
inline std::uint8_t pearson_hash(const std::string& data,
                                 const std::array<std::uint8_t, 256>& table = pearson_default_table()) {
    std::uint8_t h = 0;
    for (unsigned char byte : data) h = table[h ^ byte];
    return h;
}

/// \brief Wider Pearson hash: \p width bytes by re-running with a perturbed first byte.
inline std::uint64_t pearson_hash_wide(const std::string& data, int width,
                                       const std::array<std::uint8_t, 256>& table = pearson_default_table()) {
    std::uint64_t result = 0;
    for (int j = 0; j < width; ++j) {
        std::uint8_t h = data.empty() ? 0 : table[(static_cast<unsigned char>(data[0]) + j) % 256];
        for (std::size_t i = (data.empty() ? 0 : 1); i < data.size(); ++i)
            h = table[h ^ static_cast<unsigned char>(data[i])];
        result = (result << 8) | h;
    }
    return result;
}

}  // namespace datamunge::algorithms
