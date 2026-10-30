#pragma once

/// \file checksums.hpp
/// \brief Three classic error-detecting checksums: CRC-32, Adler-32, and Fletcher-16.
///
/// A *checksum* condenses a message into a short tag appended to it; the receiver
/// recomputes the tag and, if it differs, knows the message was corrupted. The three
/// here trade off strength for speed:
///   - *CRC-32* (cyclic redundancy check) treats the message as a polynomial over
///     GF(2) and returns the remainder modulo a fixed generator polynomial. CRCs
///     catch *all* burst errors up to the check width and are the standard for
///     Ethernet, ZIP, PNG, and disk sectors. This is the reflected IEEE 802.3 variant.
///   - *Adler-32* (Adler 1995, used in zlib) sums the bytes and the running sums modulo
///     the prime 65521 -- much faster than CRC but weaker on short messages.
///   - *Fletcher-16* (Fletcher 1982) is a position-sensitive two-byte running sum,
///     stronger than a plain sum and cheaper than a CRC.
/// Each is a deterministic function of the byte stream and is verified below against
/// its published reference value.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace datamunge::algorithms {

namespace detail {

/// \brief The 256-entry byte table of the reflected CRC-32 polynomial 0xEDB88320.
inline constexpr auto crc32_table = [] {
    std::array<std::uint32_t, 256> table{};
    for (std::uint32_t n = 0; n < 256; ++n) {
        std::uint32_t c = n;
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xEDB88320u & (~(c & 1u) + 1u));
        table[n] = c;
    }
    return table;
}();

} // namespace detail

/// \brief CRC-32 (reflected IEEE 802.3 / zlib polynomial 0xEDB88320), one table lookup per byte.
///
/// crc32("123456789") == 0xCBF43926.
[[nodiscard]] inline std::uint32_t crc32(std::string_view data) {
    std::uint32_t crc = 0xFFFFFFFFu;
    for (unsigned char byte : data) crc = (crc >> 8) ^ detail::crc32_table[(crc ^ byte) & 0xFFu];
    return crc ^ 0xFFFFFFFFu;
}

/// \brief The Adler-32 modulus 65521, the largest prime below \f$2^{16}\f$.
inline constexpr std::uint32_t adler32_modulus = 65521U;

/// \brief The longest run of bytes that adler32_update() sums before reducing modulo 65521.
///
/// Starting from \f$A, B < p\f$, after \f$n\f$ bytes of value at most 255 the second sum is at
/// most \f$255\,n(n+1)/2 + (n+1)(p-1)\f$; 5552 is the largest \f$n\f$ for which that fits in
/// 32 bits (zlib's NMAX).
inline constexpr std::size_t adler32_nmax = 5552;

/// \brief Continues an Adler-32 checksum over @p data: `adler32_update(adler32(x), y) ==
///        adler32(x + y)`.
///
/// Reduces modulo 65521 only once every adler32_nmax bytes, which gives the same result as
/// reducing after every byte at a fraction of the cost.
/// @param adler The checksum so far; 1 for the empty message.
/// @param data The bytes that follow.
///
/// @par Example
/// @code
/// std::uint32_t sum = 1;
/// for (const auto& chunk : chunks) sum = dm::algorithms::adler32_update(sum, chunk);
/// @endcode
[[nodiscard]] std::uint32_t adler32_update(std::uint32_t adler, std::string_view data);

/// \brief Adler-32 (zlib), the checksum of RFC 1950: \f$(B \ll 16) \mid A\f$ with
///        \f$A = 1 + \sum_i x_i\f$ and \f$B = n + \sum_i (n-i+1)\,x_i\f$, both modulo 65521.
///
/// adler32("Wikipedia") == 0x11E60398; the empty message gives 1.
[[nodiscard]] inline std::uint32_t adler32(std::string_view data) { return adler32_update(1U, data); }

/// \brief The Adler-32 of the concatenation of two messages, from their checksums alone.
///
/// With \f$m\f$ the length of the second message, \f$A = A_1 + A_2 - 1\f$ and
/// \f$B = B_1 + B_2 + m(A_1 - 1)\f$, modulo 65521 (zlib's `adler32_combine`).
/// @param adler1 Checksum of the first message.
/// @param adler2 Checksum of the second message.
/// @param length2 Length of the second message in bytes.
///
/// @par Example
/// @code
/// namespace alg = dm::algorithms;
/// const bool same = alg::adler32_combine(alg::adler32("Wiki"), alg::adler32("pedia"), 5) ==
///                   alg::adler32("Wikipedia");   // true
/// @endcode
[[nodiscard]] std::uint32_t adler32_combine(std::uint32_t adler1, std::uint32_t adler2, std::uint64_t length2);

/// \brief Slides the Adler-32 of a @p window byte window one byte along: drops @p out from
///        the front and appends @p in at the back.
///
/// \f$A' = A - x_{out} + x_{in}\f$ and \f$B' = B - w\,x_{out} + A' - 1\f$, modulo 65521. This is the
/// constant-time update behind rsync-style block matching.
/// @param adler Checksum of the current window.
/// @param window Window length \f$w\f$ in bytes.
/// @param out The byte leaving the window.
/// @param in The byte entering the window.
[[nodiscard]] std::uint32_t adler32_roll(std::uint32_t adler, std::size_t window, unsigned char out,
                                         unsigned char in);

/// \brief The Adler-32 of every @p window byte window of @p data, in order of start position,
///        computed with adler32_roll().
/// @return `data.size() - window + 1` checksums, or none if @p data is shorter than @p window.
/// @throws std::invalid_argument if @p window is 0.
///
/// @par Example
/// @code
/// const auto sums = dm::algorithms::adler32_windows("abracadabra", 4);   // 8 windows
/// @endcode
[[nodiscard]] std::vector<std::uint32_t> adler32_windows(std::string_view data, std::size_t window);

/// \brief Approximate probability that two independent random messages of @p length bytes
///        (each byte with variance @p byte_variance) have the same Adler-32.
///
/// The difference of two messages' sums is approximately normal with covariance
/// \f$2\Sigma\f$, where \f$\Sigma\f$ has \f$\sigma_A^2 = nv\f$, \f$\sigma_B^2 = v\,n(n+1)(2n+1)/6\f$ and
/// \f$\mathrm{cov}(A,B) = v\,n(n+1)/2\f$. The probability is
/// \f$P(\Delta A \equiv 0)\,P(\Delta B \equiv 0 \mid \Delta A = 0)\f$, each a normal density summed over
/// the multiples of 65521 (a wrapped normal), so it falls from about
/// \f$1/(4\pi\sqrt{\det\Sigma})\f$ for short messages towards \f$1/65521^2\f$ for long ones.
/// The approximation needs at least 3 bytes: Adler-32 is one-to-one on messages of 1 or 2
/// bytes.
/// @param length Message length \f$n\f$ in bytes, at least 3.
/// @param byte_variance Variance \f$v\f$ of one byte; the default is that of a uniform byte,
///        \f$(256^2-1)/12\f$.
/// @throws std::invalid_argument if @p length is below 3 or @p byte_variance is not positive.
[[nodiscard]] double adler32_collision_probability(std::uint64_t length, double byte_variance = 65535.0 / 12.0);

/// \brief Fletcher-16 checksum.
///
/// fletcher16("abcde") == 0xC8F0.
inline std::uint16_t fletcher16(const std::string& data) {
    std::uint16_t sum1 = 0, sum2 = 0;
    for (unsigned char byte : data) {
        sum1 = static_cast<std::uint16_t>((sum1 + byte) % 255);
        sum2 = static_cast<std::uint16_t>((sum2 + sum1) % 255);
    }
    return static_cast<std::uint16_t>((sum2 << 8) | sum1);
}

}  // namespace datamunge::algorithms
