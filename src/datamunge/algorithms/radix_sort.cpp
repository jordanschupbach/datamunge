#include <datamunge/algorithms/radix_sort.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

void radix_sort(std::vector<std::int64_t>& data) {
    const std::size_t n = data.size();
    if (n < 2) return; // empty or singleton is already sorted

    // Bias signed keys into order-preserving unsigned keys by flipping the sign bit. Adding
    // 2^63 (equivalently XOR-ing with 0x8000000000000000) maps INT64_MIN..INT64_MAX onto
    // 0..UINT64_MAX monotonically, so signed order becomes plain unsigned byte order.
    constexpr std::uint64_t kSignBias = std::uint64_t(1) << 63;

    std::vector<std::uint64_t> keys(n);
    for (std::size_t i = 0; i < n; ++i)
        keys[i] = static_cast<std::uint64_t>(data[i]) ^ kSignBias;

    std::vector<std::uint64_t> buffer(n);
    constexpr int kBytes = 8;   // 8 bytes in a 64-bit key -> 8 passes
    constexpr int kRadix = 256; // base 256: one bucket per possible byte value

    for (int pass = 0; pass < kBytes; ++pass) {
        const int shift = pass * 8;

        // Stable counting sort on the byte selected by `shift`.
        std::array<std::size_t, kRadix + 1> count{}; // zero-initialized; count[b+1] tallies byte b

        for (std::size_t i = 0; i < n; ++i) {
            const auto byte = static_cast<std::uint8_t>((keys[i] >> shift) & 0xFFu);
            ++count[static_cast<std::size_t>(byte) + 1];
        }
        // Prefix sums turn bucket sizes into bucket starting offsets: count[b] = start of byte b.
        for (int b = 0; b < kRadix; ++b)
            count[static_cast<std::size_t>(b) + 1] += count[static_cast<std::size_t>(b)];

        // Stable scatter: keys with an equal byte keep their relative order (this is what makes
        // the successive passes compose into a correct total order).
        for (std::size_t i = 0; i < n; ++i) {
            const auto byte = static_cast<std::uint8_t>((keys[i] >> shift) & 0xFFu);
            buffer[count[static_cast<std::size_t>(byte)]++] = keys[i];
        }
        keys.swap(buffer); // `keys` now holds the array sorted by this byte
    }

    // Undo the bias to recover signed values (8 passes leave the result in `keys`).
    for (std::size_t i = 0; i < n; ++i)
        data[i] = static_cast<std::int64_t>(keys[i] ^ kSignBias);
}

} // namespace datamunge::algorithms
