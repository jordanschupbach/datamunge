#pragma once

#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// @brief LSD (least-significant-digit) radix sort for signed 64-bit integers, sorting @p data
///        ascending *in place*. Radix sort is a *non-comparison* sort: instead of asking "is a <
///        b?" it distributes keys by their digits, sidestepping the \(\Omega(n \log n)\) lower
///        bound that binds every comparison sort. This implementation treats each key as eight
///        base-256 "digits" (its bytes) and makes one *stable counting-sort* pass per byte, from
///        least to most significant. Because each pass is stable, keys that agree on the current
///        byte keep the order established by the lower bytes already processed -- and after all
///        eight passes the array is fully ordered. Signed keys are handled by flipping the sign
///        bit (equivalently, biasing by \(2^{63}\)) so that two's-complement order maps onto plain
///        unsigned byte order; the bias is undone at the end. Runs in \(O(d\,(n+b))\) time with
///        \(d = 8\) passes and radix \(b = 256\), i.e. linear in @p data 's size, using scratch
///        buffers of size \(n\). Empty and singleton inputs are returned unchanged.
///
/// @param data the integers to sort; overwritten with the ascending-sorted sequence.
void radix_sort(std::vector<std::int64_t>& data);

} // namespace datamunge::algorithms
