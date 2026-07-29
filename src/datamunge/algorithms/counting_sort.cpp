#include <datamunge/algorithms/counting_sort.hpp>

#include <algorithm>
#include <stdexcept>

namespace datamunge::algorithms {

void counting_sort(std::vector<std::int64_t>& data, std::int64_t min_value, std::int64_t max_value) {
    if (max_value < min_value)
        throw std::invalid_argument("counting_sort: max_value must be >= min_value");
    if (data.empty()) return;

    // k = max - min + 1. Compute the span in unsigned arithmetic so it is well defined even when
    // min is very negative and max very positive (int64 subtraction could overflow); wrap-around
    // subtraction yields the true difference whenever max >= min and it fits in 64 bits.
    const std::uint64_t span =
        static_cast<std::uint64_t>(max_value) - static_cast<std::uint64_t>(min_value);
    if (span >= static_cast<std::uint64_t>(kCountingSortMaxRange))
        throw std::length_error("counting_sort: key range too large -- use a comparison sort");
    const std::size_t k = static_cast<std::size_t>(span) + 1;

    const auto bucket = [min_value](std::int64_t v) {
        return static_cast<std::size_t>(static_cast<std::uint64_t>(v) - static_cast<std::uint64_t>(min_value));
    };

    // Phase 1: tally the frequency of every key (and validate the range while we are at it).
    std::vector<std::size_t> count(k, 0);
    for (const std::int64_t v : data) {
        if (v < min_value || v > max_value)
            throw std::invalid_argument("counting_sort: value out of [min_value, max_value]");
        ++count[bucket(v)];
    }

    // Phase 2: prefix-sum the counts into the *end* position of each key's block. Iterating the
    // input right-to-left below and pre-decrementing this cursor fills each block back-to-front,
    // which preserves the input order of equal keys -- the source of counting sort's stability.
    std::vector<std::size_t> position(k, 0);
    std::size_t running = 0;
    for (std::size_t v = 0; v < k; ++v) {
        running += count[v];
        position[v] = running; // one past the last slot of key v's block
    }

    // Phase 3: stable placement, scanning right to left.
    std::vector<std::int64_t> output(data.size());
    for (std::size_t i = data.size(); i-- > 0;) {
        const std::size_t b = bucket(data[i]);
        output[--position[b]] = data[i];
    }
    data = std::move(output);
}

void counting_sort(std::vector<std::int64_t>& data) {
    if (data.empty()) return;
    const auto [lo, hi] = std::minmax_element(data.begin(), data.end());
    counting_sort(data, *lo, *hi);
}

} // namespace datamunge::algorithms
