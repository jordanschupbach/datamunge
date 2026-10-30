/// @file string_search_extra.cpp
/// @brief Compiled digram matching, instrumented scans and record-wise corpus comparisons.
#include <datamunge/algorithms/string_search_extra.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// @brief Private byte tables and search-work baselines.
namespace {

/// @brief Number of possible unsigned-byte values.
constexpr int kAlphabet = 256;

/// @brief Interpret a character as a non-negative byte-table index.
/// @param c Any byte, including high-bit values.
/// @return Its unsigned value.
inline unsigned char uc(char c) { return static_cast<unsigned char>(c); }

/// @brief Build strong good-suffix shifts in linear time via suffix borders.
/// @param x Nonempty pattern of at most INT_MAX bytes.
/// @param bmGs Output shift per mismatch index; index zero is also the full-match shift.
void compute_good_suffix(const std::string& x, std::vector<int>& bmGs) {
    const int        m = static_cast<int>(x.size());
    std::vector<int> suff(m);
    suff[m - 1] = m;
    int f = 0, g = m - 1;
    for (int i = m - 2; i >= 0; --i) {
        if (i > g && suff[i + m - 1 - f] < i - g) {
            suff[i] = suff[i + m - 1 - f];
        } else {
            if (i < g) g = i;
            f = i;
            while (g >= 0 && x[g] == x[g + m - 1 - f]) --g;
            suff[i] = f - g;
        }
    }
    bmGs.assign(m, m);
    int j = 0;
    for (int i = m - 1; i >= 0; --i)
        if (suff[i] == i + 1)
            for (; j < m - 1 - i; ++j)
                if (bmGs[j] == m) bmGs[j] = m - 1 - i;
    for (int i = 0; i <= m - 2; ++i) bmGs[m - 1 - suff[i]] = m - 1 - i;
}

/// @brief Count naive left-to-right comparisons and collect overlapping matches.
/// @param text Borrowed text bytes.
/// @param pattern Borrowed pattern; empty matches nowhere.
/// @return Positions and scan work excluding loop/size checks.
StringSearchScan naive_scan(const std::string& text, const std::string& pattern) {
    StringSearchScan out;
    const auto n = text.size(), m = pattern.size();
    if (m == 0 || m > n) return out;
    for (std::size_t s = 0; s <= n - m; ++s) {
        ++out.alignments;
        std::size_t i = 0;
        while (i < m) { ++out.comparisons; if (text[s+i] != pattern[i]) break; ++i; }
        if (i == m) out.positions.push_back(s);
    }
    return out;
}
/// @brief Precompute the one-byte Horspool shifts once per pattern.
/// @param pattern Borrowed byte pattern, possibly empty.
/// @return Positive distance from the window's final byte to its next candidate alignment.
std::array<std::size_t, kAlphabet> horspool_table(const std::string& pattern) {
    std::array<std::size_t, kAlphabet> shift;
    shift.fill(pattern.size());
    for (std::size_t i = 0; i + 1 < pattern.size(); ++i) shift[uc(pattern[i])] = pattern.size() - 1 - i;
    return shift;
}
/// @brief Run an instrumented Horspool scan with precomputed shifts.
/// @param text Borrowed text bytes.
/// @param pattern Borrowed pattern; empty matches nowhere.
/// @param shift Table constructed for this pattern by horspool_table().
/// @return Positions and scan work, including failed equality checks.
StringSearchScan horspool_scan(const std::string& text, const std::string& pattern,
                              const std::array<std::size_t, kAlphabet>& shift) {
    StringSearchScan out;
    const auto n = text.size(), m = pattern.size();
    if (m == 0 || m > n) return out;
    std::size_t s = 0;
    while (s <= n - m) {
        ++out.alignments;
        std::size_t i = m;
        while (i > 0) { ++out.comparisons; if (text[s+i-1] != pattern[i-1]) break; --i; }
        if (i == 0) out.positions.push_back(s);
        const auto advance = shift[uc(text[s+m-1])];
        if (advance > n - m - s) break;
        s += advance;
    }
    return out;
}
} // namespace

ZhuTakaoka::ZhuTakaoka(std::string pattern) : pattern_(std::move(pattern)) {
    if (pattern_.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("ZhuTakaoka: pattern exceeds shift-table index range");
    const int m = static_cast<int>(pattern_.size());
    if (m < 2) return;
    digram_.assign(kAlphabet * kAlphabet, m);
    for (int a = 0; a < kAlphabet; ++a) digram_[a * kAlphabet + uc(pattern_[0])] = m - 1;
    for (int i = 1; i < m - 1; ++i)
        digram_[uc(pattern_[i-1]) * kAlphabet + uc(pattern_[i])] = m - 1 - i;
    compute_good_suffix(pattern_, suffix_);
}

StringSearchScan ZhuTakaoka::scan_impl(const std::string& text, dstruct::DataFrame* trace) const {
    StringSearchScan out;
    std::vector<double> offsets, matched, comparisons, matches, digrams, suffixes, advances;
    const auto n = text.size(), m = pattern_.size();
    if (m > 0 && m <= n) {
        std::size_t s = 0;
        while (s <= n - m) {
            ++out.alignments;
            const auto start_comparisons = out.comparisons;
            std::size_t i = m;
            while (i > 0) { ++out.comparisons; if (pattern_[i-1] != text[s+i-1]) break; --i; }
            const bool found = i == 0;
            if (found) out.positions.push_back(s);
            const auto good = m == 1 ? std::size_t{1} : static_cast<std::size_t>(suffix_[found ? 0 : i-1]);
            const auto bad = m == 1 || found ? std::size_t{0} : static_cast<std::size_t>(
                digram_[uc(text[s+m-2]) * kAlphabet + uc(text[s+m-1])]);
            const auto advance = std::max(good, bad);
            if (trace) {
                offsets.push_back(static_cast<double>(s)); matched.push_back(static_cast<double>(m-i));
                comparisons.push_back(static_cast<double>(out.comparisons-start_comparisons));
                matches.push_back(found ? 1 : 0); digrams.push_back(static_cast<double>(bad));
                suffixes.push_back(static_cast<double>(good)); advances.push_back(static_cast<double>(advance));
            }
            if (advance > n - m - s) break;
            s += advance;
        }
    }
    if (trace) {
        trace->add_column("offset", offsets); trace->add_column("matched_suffix", matched);
        trace->add_column("comparisons", comparisons); trace->add_column("match", matches);
        trace->add_column("digram_shift", digrams); trace->add_column("suffix_shift", suffixes);
        trace->add_column("advance", advances);
    }
    return out;
}
std::vector<std::size_t> ZhuTakaoka::search(const std::string& text) const { return scan(text).positions; }
StringSearchScan ZhuTakaoka::scan(const std::string& text) const { return scan_impl(text, nullptr); }
dstruct::DataFrame ZhuTakaoka::trace(const std::string& text) const {
    dstruct::DataFrame out;
    (void)scan_impl(text, &out);
    return out;
}
dstruct::DataFrame ZhuTakaoka::digram_table(const std::string& alphabet) const {
    if (pattern_.size() < 2) throw std::invalid_argument("digram_table: pattern must have at least two bytes");
    std::vector<double> first, last, shifts;
    for (unsigned char a : alphabet) for (unsigned char b : alphabet) {
        first.push_back(a); last.push_back(b); shifts.push_back(digram_[a*kAlphabet+b]);
    }
    dstruct::DataFrame out;
    out.add_column("first_byte", first); out.add_column("last_byte", last); out.add_column("shift", shifts);
    return out;
}
std::vector<std::size_t> boyer_moore_horspool_search(const std::string& text, const std::string& pattern) {
    if (pattern.empty() || pattern.size() > text.size()) return {};
    return horspool_scan(text, pattern, horspool_table(pattern)).positions;
}
std::vector<std::size_t> zhu_takaoka_search(const std::string& text, const std::string& pattern) {
    if (pattern.empty() || pattern.size() > text.size()) return {};
    return ZhuTakaoka(pattern).search(text);
}

dstruct::DataFrame string_search_comparison(const dstruct::DataFrame& frame, const std::string& column,
                                            const std::vector<std::string>& patterns) {
    const auto& texts = frame.get_column<std::string>(column);
    std::vector<std::string> labels, methods;
    std::vector<double> records, bytes, matches, alignments, comparisons, agrees;
    for (const auto& pattern : patterns) {
        const ZhuTakaoka zt(pattern);
        const auto hp_table = horspool_table(pattern);
        std::array<std::size_t, 3> nmatch{}, nalign{}, ncompare{};
        std::array<bool, 3> ok{true, true, true};
        std::size_t nbytes = 0;
        for (std::size_t r = 0; r < frame.nrows(); ++r) {
            const auto& text = texts.at(r);
            nbytes += text.size();
            const std::array<StringSearchScan, 3> work{naive_scan(text, pattern), horspool_scan(text, pattern, hp_table), zt.scan(text)};
            for (std::size_t k = 0; k < 3; ++k) {
                nmatch[k] += work[k].positions.size(); nalign[k] += work[k].alignments; ncompare[k] += work[k].comparisons;
                ok[k] = ok[k] && work[k].positions == work[0].positions;
            }
        }
        for (std::size_t k = 0; k < 3; ++k) {
            labels.push_back(pattern); methods.push_back(k == 0 ? "naive" : k == 1 ? "Horspool" : "Zhu-Takaoka");
            records.push_back(static_cast<double>(frame.nrows())); bytes.push_back(static_cast<double>(nbytes));
            matches.push_back(static_cast<double>(nmatch[k])); alignments.push_back(static_cast<double>(nalign[k]));
            comparisons.push_back(static_cast<double>(ncompare[k])); agrees.push_back(ok[k] ? 1 : 0);
        }
    }
    dstruct::DataFrame out;
    out.add_column("pattern", labels); out.add_column("method", methods); out.add_column("records", records);
    out.add_column("bytes", bytes); out.add_column("matches", matches); out.add_column("alignments", alignments);
    out.add_column("comparisons", comparisons); out.add_column("agrees_with_naive", agrees);
    return out;
}
} // namespace datamunge::algorithms
