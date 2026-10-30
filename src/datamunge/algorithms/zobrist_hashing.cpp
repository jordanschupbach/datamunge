/// @file zobrist_hashing.cpp
/// @brief Categorical DataFrame adapters and Zobrist update/collision diagnostics.
#include <datamunge/algorithms/zobrist_hashing.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <unordered_map>

namespace datamunge::algorithms {
/// @brief Private rendering helpers for exact 64-bit diagnostics.
namespace {
/// @brief Format a hash without losing integer precision in a numeric column.
/// @param value Unsigned hash.
/// @return Sixteen hexadecimal digits with a 0x prefix.
std::string hash_hex(std::uint64_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}
/// @brief Count changed hash bits with Kernighan's recurrence.
/// @param value XOR of two hashes.
/// @return Number of set bits, in [0,64].
int changed_bits(std::uint64_t value) {
    int n = 0;
    while (value) { value &= value - 1; ++n; }
    return n;
}
}

std::vector<std::vector<int>> zobrist_boards(const dstruct::DataFrame& frame, std::size_t num_pieces) {
    if (num_pieces == 0 || num_pieces > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("zobrist_boards: invalid category count");
    std::vector<std::vector<int>> boards(frame.nrows(), std::vector<int>(frame.ncols()));
    const auto names = frame.columns();
    for (std::size_t c = 0; c < names.size(); ++c) {
        const auto& column = frame.get_column(names[c]); // validates numeric type, including empty columns
        for (std::size_t r = 0; r < frame.nrows(); ++r) {
            const double value = column.at(r);
            if (!std::isfinite(value) || value < 0 || value >= num_pieces || value != std::floor(value))
                throw std::invalid_argument("zobrist_boards: expected finite integer category");
            boards[r][c] = static_cast<int>(value);
        }
    }
    return boards;
}

dstruct::DataFrame zobrist_transition(const ZobristTable& t, const std::vector<int>& before,
                                      const std::vector<int>& after) {
    auto board = before;
    auto hash = zobrist_hash(t, before);
    (void)zobrist_hash(t, after); // validate all target cells before producing a trace
    std::vector<double> cells, old_values, new_values, flips, verified;
    std::vector<std::string> hashes, full_hashes;
    for (std::size_t c = 0; c < before.size(); ++c) {
        if (before[c] == after[c]) continue;
        const auto next = zobrist_update(t, hash, c, before[c], after[c]);
        board[c] = after[c];
        const auto full = zobrist_hash(t, board);
        cells.push_back(static_cast<double>(c));
        old_values.push_back(before[c]); new_values.push_back(after[c]);
        flips.push_back(changed_bits(hash ^ next)); verified.push_back(next == full ? 1 : 0);
        hashes.push_back(hash_hex(next)); full_hashes.push_back(hash_hex(full));
        hash = next;
    }
    dstruct::DataFrame out;
    out.add_column("cell", cells); out.add_column("old_piece", old_values);
    out.add_column("new_piece", new_values); out.add_column("flipped_bits", flips);
    out.add_column("verified", verified); out.add_column("incremental", hashes);
    out.add_column("full", full_hashes);
    return out;
}

dstruct::DataFrame zobrist_collision_profile(const std::vector<std::vector<int>>& boards,
                                             std::size_t num_pieces, const std::vector<int>& widths,
                                             std::size_t repetitions, std::uint64_t seed) {
    if (repetitions == 0 || num_pieces == 0)
        throw std::invalid_argument("zobrist_collision_profile: counts must be positive");
    for (int b : widths)
        if (b < 1 || b > 64) throw std::invalid_argument("zobrist_collision_profile: width outside [1,64]");
    const std::set<std::vector<int>> distinct(boards.begin(), boards.end());
    const std::size_t cells = boards.empty() ? 0 : boards.front().size();
    std::vector<double> bits, total(widths.size(), 0), minimum(widths.size(), std::numeric_limits<double>::infinity()),
                        maximum(widths.size(), 0), expected;
    for (std::size_t rep = 0; rep < repetitions; ++rep) {
        const auto table = zobrist_init(cells, num_pieces, seed + rep);
        std::vector<std::uint64_t> hashes;
        for (const auto& board : distinct) hashes.push_back(zobrist_hash(table, board));
        for (std::size_t j = 0; j < widths.size(); ++j) {
            const auto mask = widths[j] == 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << widths[j]) - 1;
            std::unordered_map<std::uint64_t, std::uint64_t> counts;
            double pairs = 0;
            for (auto h : hashes) pairs += static_cast<double>(counts[h & mask]++);
            total[j] += pairs; minimum[j] = std::min(minimum[j], pairs); maximum[j] = std::max(maximum[j], pairs);
        }
    }
    const double n = static_cast<double>(distinct.size());
    for (std::size_t j = 0; j < widths.size(); ++j) {
        bits.push_back(widths[j]); total[j] /= static_cast<double>(repetitions);
        expected.push_back(std::ldexp(n * (n - 1) / 2, -widths[j]));
    }
    dstruct::DataFrame out;
    out.add_column("bits", bits); out.add_column("mean_pairs", total);
    out.add_column("min_pairs", minimum); out.add_column("max_pairs", maximum);
    out.add_column("expected_pairs", expected);
    return out;
}
} // namespace datamunge::algorithms
