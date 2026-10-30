/// @file word_embeddings.cpp
/// @brief embedding_bag(), cooccurrence_matrix(), pmi(), negative_sampling_loss() and sample_negatives().

#include "datamunge/nn/word_embeddings.hpp"

#include "datamunge/nn/ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::nn {

Tensor embedding_bag(const Tensor& table, const Tensor& ids, const Tensor& offsets, BagMode mode, const Tensor& per_sample_weights) {
    for(const Tensor* value:{&table,&ids,&offsets,&per_sample_weights})
      if(value->defined() && value->device().backend!=compute::Backend::CPU)
        throw std::logic_error("embedding_bag: native numerical GPU kernel is not implemented");
    if (table.dim() != 2) throw std::invalid_argument("embedding_bag: table must be [vocab, dim]");
    if (ids.dim() != 1 || offsets.dim() != 1) throw std::invalid_argument("embedding_bag: ids and offsets must be rank 1");
    if (per_sample_weights.defined() && mode == BagMode::Max) throw std::invalid_argument("embedding_bag: weights are not supported with Max");
    if (per_sample_weights.defined() && (per_sample_weights.dim() != 1 || per_sample_weights.size(0) != ids.size(0)))
        throw std::invalid_argument("embedding_bag: per_sample_weights must be [n]");
    const std::int64_t n = ids.size(0), bags = offsets.size(0), dim = table.size(1);
    const auto         off = cast(offsets, DType::Int64).contiguous().to_vector<std::int64_t>();
    if (bags > 0 && off[0] != 0) throw std::invalid_argument("embedding_bag: offsets[0] must be 0");
    for (std::int64_t b = 0; b < bags; ++b)
        if (off[b] > n || (b > 0 && off[b] < off[b - 1])) throw std::invalid_argument("embedding_bag: offsets must be non-decreasing and <= n");
    std::vector<std::int64_t> seg(static_cast<std::size_t>(n));
    std::vector<double>       count(static_cast<std::size_t>(bags), 0.0);
    for (std::int64_t b = 0; b < bags; ++b) {
        const std::int64_t end = b + 1 < bags ? off[b + 1] : n;
        for (std::int64_t i = off[b]; i < end; ++i) seg[i] = b;
        count[b] = static_cast<double>(end - off[b]);
    }
    if (bags == 0) return zeros({0, dim}, table.dtype());
    const Tensor rows = gather(table, ids, 0);  // [n, dim]
    if (mode == BagMode::Max) {
        // Pad every bag to the longest length, fill padding with -inf, reduce, and zero empty bags.
        const std::int64_t        len = static_cast<std::int64_t>(*std::max_element(count.begin(), count.end()));
        std::vector<std::int64_t> pos(static_cast<std::size_t>(bags * std::max<std::int64_t>(len, 1)), 0);
        std::vector<double>       keep(pos.size(), 0.0);
        for (std::int64_t b = 0; b < bags; ++b)
            for (std::int64_t j = 0; j < static_cast<std::int64_t>(count[b]); ++j) pos[b * len + j] = off[b] + j, keep[b * len + j] = 1;
        if (len == 0) return zeros({bags, dim}, table.dtype());
        const Tensor padded = reshape(gather(rows, from_vector(pos, {bags * len}), 0), {bags, len, dim});
        const Tensor mask   = greater(from_vector(keep, {bags, len, 1}), 0.5);
        const Tensor m      = max(where(mask, padded, -std::numeric_limits<double>::infinity()), {1});
        std::vector<double> nonempty(static_cast<std::size_t>(bags));
        for (std::int64_t b = 0; b < bags; ++b) nonempty[b] = count[b] > 0 ? 1.0 : 0.0;
        return where(greater(from_vector(nonempty, {bags, 1}), 0.5), m, 0.0);
    }
    Tensor weighted = rows;
    Tensor denom    = from_vector(count, {bags, 1});
    if (per_sample_weights.defined()) {
        const Tensor w = cast(per_sample_weights, table.dtype());
        weighted       = rows * reshape(w, {n, 1});
        denom          = reshape(unsorted_segment_sum(reshape(w, {n, 1}), from_vector(seg, {n}), bags), {bags, 1});
    }
    const Tensor s = unsorted_segment_sum(weighted, from_vector(seg, {n}), bags);
    if (mode == BagMode::Sum) return s;
    // Mean: divide by the count (or the total weight); empty bags stay 0.
    const Tensor safe = where(equal(denom, 0.0), 1.0, cast(denom, table.dtype()));
    return s / safe;
}

Tensor cooccurrence_matrix(const std::vector<std::vector<std::int64_t>>& sentences, std::int64_t vocab, std::int64_t window, bool harmonic) {
    if (window < 1) throw std::invalid_argument("cooccurrence_matrix: window must be at least 1");
    std::vector<double> c(static_cast<std::size_t>(vocab * vocab), 0.0);
    for (const auto& s : sentences) {
        for (std::int64_t w : s)
            if (w < 0 || w >= vocab) throw std::invalid_argument("cooccurrence_matrix: id out of range");
        const auto len = static_cast<std::int64_t>(s.size());
        for (std::int64_t i = 0; i < len; ++i)
            for (std::int64_t j = std::max<std::int64_t>(0, i - window); j <= std::min(len - 1, i + window); ++j)
                if (i != j) c[static_cast<std::size_t>(s[i] * vocab + s[j])] += harmonic ? 1.0 / static_cast<double>(std::abs(i - j)) : 1.0;
    }
    return from_vector(c, {vocab, vocab});
}

Tensor pmi(const Tensor& counts, double shift, bool positive, double context_power) {
    if (counts.dim() != 2) throw std::invalid_argument("pmi: counts must be rank 2");
    if (!(context_power > 0)) throw std::invalid_argument("pmi: context_power must be positive");
    const Tensor C = cast(counts.detach(), DType::Float64);
    if (min(C).item() < 0) throw std::invalid_argument("pmi: counts must be non-negative");
    const Tensor rows = sum(C, {1}, true), cols = sum(C, {0}, true);
    const Tensor colp = pow(cols, context_power);
    // log(#(w,c)) - log(#(w)) - log(#(c)^a / sum #(c')^a) - shift; log 0 = -inf.
    const Tensor raw = log(C) - log(rows) - (log(colp) - std::log(sum(colp).item())) - shift;
    const Tensor out = where(equal(C, 0.0), -std::numeric_limits<double>::infinity(), raw);  // also where a row or column is empty
    return positive ? maximum(out, zeros_like(out)) : out;
}

Tensor negative_sampling_loss(const Tensor& target, const Tensor& context, const Tensor& negatives) {
    if (target.dim() != 2 || context.shape() != target.shape()) throw std::invalid_argument("negative_sampling_loss: target and context must be [B, d]");
    if (negatives.dim() != 3 || negatives.size(0) != target.size(0) || negatives.size(2) != target.size(1))
        throw std::invalid_argument("negative_sampling_loss: negatives must be [B, K, d]");
    const Tensor pos = sum(target * context, {1});                                        // [B]
    const Tensor neg = sum(negatives * reshape(target, {target.size(0), 1, target.size(1)}), {2});  // [B, K]
    // -log sigma(x) = softplus(-x).
    return mean(softplus(-pos) + sum(softplus(neg), {1}));
}

Tensor sample_negatives(const Tensor& counts, const Shape& shape, double power, Generator& gen) {
    if (counts.dim() != 1) throw std::invalid_argument("sample_negatives: counts must be rank 1");
    const Tensor w = pow(cast(counts.detach(), DType::Float64), power);
    if (!(sum(w).item() > 0)) throw std::invalid_argument("sample_negatives: counts need a positive entry");
    std::int64_t n = 1;
    for (auto s : shape) n *= s;
    return reshape(multinomial(w, n, true, gen), shape);
}

}  // namespace datamunge::nn
