#pragma once

/// \file zobrist_hashing.hpp
/// \brief Zobrist hashing: a hash of a board/configuration that updates incrementally
///        in O(1) when a single cell changes (Zobrist 1970).
///
/// Game engines need to hash board positions to index a *transposition table* (a cache
/// of previously analyzed positions), and they change the board one move at a time.
/// *Zobrist hashing* makes that update \f$O(1)\f$: precompute an independent random
/// 64-bit key for every (cell, piece) pair, and define a position's hash as the XOR of
/// the keys for the pieces currently on the board. Because XOR is its own inverse,
/// moving a piece is just "XOR out the old (cell,piece) key, XOR in the new one" -- no
/// rehash of the whole board. Collisions are as rare as random 64-bit values allow.
/// It is the standard hashing scheme in chess/Go engines and, more broadly, for hashing
/// sets under incremental element insertion/removal.

#include <datamunge/dstruct/dataframe.hpp>

#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {

/// @brief Owned random keys for a fixed number of categorical cells.
///
/// Piece zero has key zero. Tables and boards are read-only during hashing;
/// concurrent reads are safe. Changing a key invalidates existing hashes.
///
/// @par Example
/// @code
/// const auto table = dm::algorithms::zobrist_init(2, 3, 42);
/// const auto hash = dm::algorithms::zobrist_hash(table, {1, 2});
/// @endcode
/// @par References
/// - A. L. Zobrist (1970), A New Hashing Method With Application for Game Playing,
///   Wisconsin TR88, https://research.cs.wisc.edu/techreports/1970/TR88.pdf
/// @see zobrist_init(), zobrist_hash()
struct ZobristTable {
    std::size_t                             num_cells  = 0;  ///< Number of cells in each board.
    std::size_t                             num_pieces = 0;  ///< Includes the "empty" piece at index 0.
    std::vector<std::vector<std::uint64_t>> keys;            ///< keys[cell][piece].

    /// @brief Read one key without modifying the table.
    /// @param cell Zero-based cell index.
    /// @param piece Non-negative category less than num_pieces.
    /// @return The cell/category key, zero for category zero in an initialised table.
    /// @throws std::out_of_range If either index is outside the stored keys.
    std::uint64_t key(std::size_t cell, int piece) const { return keys.at(cell).at(static_cast<std::size_t>(piece)); }
};

/// \brief Build a Zobrist key table for \p num_cells cells and \p num_pieces piece types.
///
/// The empty square (piece 0) is given a zero key so absent pieces contribute nothing.
/// Construction costs O(num_cells * num_pieces) time and storage. MT19937-64
/// generates reproducible pseudorandom keys; ideal independent-key collision
/// theorems are a model, not a cryptographic guarantee for this generator.
/// @param num_cells Number of cells; zero is permitted.
/// @param num_pieces Number of categories including zero; must be positive.
/// @param seed MT19937-64 seed, default 0.
/// @return A fresh owned table.
/// @throws std::invalid_argument If num_pieces is zero.
/// @par Example
/// @code
/// const auto t = dm::algorithms::zobrist_init(2, 3);
/// const bool empty = dm::algorithms::zobrist_hash(t, {0, 0}) == 0; // true
/// @endcode
/// @par References
/// - A. L. Zobrist (1970), Wisconsin TR88, https://research.cs.wisc.edu/techreports/1970/TR88.pdf
/// @see ZobristTable, zobrist_update()
inline ZobristTable zobrist_init(std::size_t num_cells, std::size_t num_pieces, std::uint64_t seed = 0) {
    if (num_pieces == 0) throw std::invalid_argument("zobrist_init: no categories");
    ZobristTable                             t;
    t.num_cells  = num_cells;
    t.num_pieces = num_pieces;
    std::mt19937_64                          rng(seed);
    std::uniform_int_distribution<std::uint64_t> dist;
    t.keys.assign(num_cells, std::vector<std::uint64_t>(num_pieces, 0));
    for (std::size_t c = 0; c < num_cells; ++c)
        for (std::size_t p = 0; p < num_pieces; ++p) t.keys[c][p] = (p == 0) ? 0ULL : dist(rng);
    return t;
}

/// @brief Full XOR hash of a categorical board in O(board.size()) time.
/// @param t Key table, borrowed and unchanged.
/// @param board Exactly t.num_cells categories, each in [0, t.num_pieces).
/// @return The XOR of the corresponding keys; an all-zero board hashes to zero.
/// @throws std::invalid_argument If board length differs from t.num_cells.
/// @throws std::out_of_range If a category or stored key index is invalid.
/// @par Example
/// @code
/// const auto t = dm::algorithms::zobrist_init(2, 3);
/// const bool same = dm::algorithms::zobrist_hash(t, {1, 2}) == (t.key(0, 1) ^ t.key(1, 2)); // true
/// @endcode
/// @par References
/// - A. L. Zobrist (1970), Wisconsin TR88, https://research.cs.wisc.edu/techreports/1970/TR88.pdf
/// @see zobrist_update(), zobrist_boards()
inline std::uint64_t zobrist_hash(const ZobristTable& t, const std::vector<int>& board) {
    if (board.size() != t.num_cells) throw std::invalid_argument("zobrist_hash: board size mismatch");
    std::uint64_t h = 0;
    for (std::size_t c = 0; c < board.size(); ++c) h ^= t.key(c, board[c]);
    return h;
}

/// \brief Incrementally update a hash when \p cell changes from \p old_piece to \p new_piece.
///
/// Returns the new hash in O(1), without changing a board or table. The caller
/// must supply the actual old category; it cannot be checked from the hash.
/// @param t Borrowed key table.
/// @param hash Hash before the cell change.
/// @param cell Zero-based changed cell.
/// @param old_piece Previous category.
/// @param new_piece Replacement category.
/// @return hash XOR old key XOR new key; unchanged when categories are equal.
/// @throws std::out_of_range If cell or a category is outside the stored keys.
/// @par Example
/// @code
/// const auto t = dm::algorithms::zobrist_init(2, 3);
/// const auto h = dm::algorithms::zobrist_hash(t, {1, 2});
/// const bool same = dm::algorithms::zobrist_update(t, h, 0, 1, 0) ==
///                   dm::algorithms::zobrist_hash(t, {0, 2}); // true
/// @endcode
/// @par References
/// - A. L. Zobrist (1970), Wisconsin TR88, https://research.cs.wisc.edu/techreports/1970/TR88.pdf
/// @see zobrist_hash(), zobrist_transition()
inline std::uint64_t zobrist_update(const ZobristTable& t, std::uint64_t hash, std::size_t cell, int old_piece,
                                    int new_piece) {
    hash ^= t.key(cell, old_piece);  // remove the old piece's contribution
    hash ^= t.key(cell, new_piece);  // add the new piece's contribution
    return hash;
}

/// @brief Convert numeric DataFrame rows to fixed-length categorical boards.
///
/// Each column becomes one cell in the frame's column order. No binning or
/// normalisation is performed; null, non-finite and fractional values are errors.
/// O(rows * columns) time and owned output storage; the frame is unchanged.
/// @param frame Numeric categorical columns only; omit response/metadata first.
/// @param num_pieces Exclusive category upper bound; positive and at most INT_MAX.
/// @return One owned board per row, including repeated rows.
/// @throws std::invalid_argument If the bound or any cell/type is invalid.
/// @par Example
/// @code
/// dm::dstruct::DataFrame frame;
/// frame.add_column("cell", std::vector<double>{0, 2});
/// const auto boards = dm::algorithms::zobrist_boards(frame, 3); // {{0}, {2}}
/// @endcode
/// @see zobrist_hash(), zobrist_collision_profile()
std::vector<std::vector<int>> zobrist_boards(const dstruct::DataFrame& frame, std::size_t num_pieces);

/// @brief Trace each changed cell while transforming one board into another.
///
/// Changes run in cell order. Each row records cell, old_piece, new_piece,
/// flipped_bits, verified (1 on agreement), and exact hexadecimal incremental
/// and full hashes. Recomputing after every change is a diagnostic, O(cells^2)
/// in the worst case, not a timing benchmark. Equal boards return an empty table.
/// @param t Borrowed key table.
/// @param before Starting categories, copied locally.
/// @param after Target categories, same shape.
/// @return Owned diagnostic DataFrame, with no mutation of inputs.
/// @throws std::invalid_argument If either board length is wrong.
/// @throws std::out_of_range If a category is invalid.
/// @par Example
/// @code
/// const auto t = dm::algorithms::zobrist_init(2, 3);
/// const auto trace = dm::algorithms::zobrist_transition(t, {1, 2}, {0, 2});
/// const bool ok = trace.double_at("verified", 0) == 1; // true
/// @endcode
/// @see zobrist_update(), zobrist_boards()
dstruct::DataFrame zobrist_transition(const ZobristTable& t, const std::vector<int>& before,
                                     const std::vector<int>& after);

/// @brief Count colliding pairs of distinct boards across seeded tables and widths.
///
/// Exact duplicate input boards are removed before counting. Low bits of the
/// 64-bit hash are retained. Returns bits, mean_pairs, min_pairs, max_pairs and
/// expected_pairs = n(n-1)/2^(bits+1), an exact mean under ideal independent keys.
/// No assumption of mutually independent board hashes is made. Costs
/// O(repetitions * (n*cells + n*widths)) expected time, plus deduplication.
/// @param boards Fixed-length boards; empty collection is permitted.
/// @param num_pieces Positive category count including zero.
/// @param widths Bit widths in [1,64], preserved in supplied order.
/// @param repetitions Number of tables, positive; default 64.
/// @param seed First table seed; consecutive seeds wrap modulo 2^64; default 2024.
/// @return Owned collision summary with one row per width.
/// @throws std::invalid_argument If counts, widths or board lengths are invalid.
/// @throws std::out_of_range If a board category is invalid.
/// @par Example
/// @code
/// const auto profile = dm::algorithms::zobrist_collision_profile({{0}, {1}}, 2, {8});
/// const double expected = profile.double_at("expected_pairs", 0); // 0.00390625
/// @endcode
/// @par References
/// - A. L. Zobrist (1970), Wisconsin TR88, https://research.cs.wisc.edu/techreports/1970/TR88.pdf
/// @see zobrist_boards(), zobrist_transition()
dstruct::DataFrame zobrist_collision_profile(const std::vector<std::vector<int>>& boards,
                                            std::size_t num_pieces, const std::vector<int>& widths,
                                            std::size_t repetitions = 64, std::uint64_t seed = 2024);

}  // namespace datamunge::algorithms
