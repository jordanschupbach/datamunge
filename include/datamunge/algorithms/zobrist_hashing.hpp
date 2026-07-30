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

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace datamunge::algorithms {

/// A table of random keys, one per (cell, piece) pair.
struct ZobristTable {
    std::size_t                             num_cells  = 0;
    std::size_t                             num_pieces = 0;  ///< Includes the "empty" piece at index 0.
    std::vector<std::vector<std::uint64_t>> keys;            ///< keys[cell][piece].

    std::uint64_t key(std::size_t cell, int piece) const { return keys[cell][static_cast<std::size_t>(piece)]; }
};

/// \brief Build a Zobrist key table for \p num_cells cells and \p num_pieces piece types.
///
/// The empty square (piece 0) is given a zero key so absent pieces contribute nothing.
inline ZobristTable zobrist_init(std::size_t num_cells, std::size_t num_pieces, std::uint64_t seed = 0) {
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

/// \brief Full Zobrist hash of a board (a piece id per cell; 0 = empty).
inline std::uint64_t zobrist_hash(const ZobristTable& t, const std::vector<int>& board) {
    std::uint64_t h = 0;
    for (std::size_t c = 0; c < board.size(); ++c) h ^= t.key(c, board[c]);
    return h;
}

/// \brief Incrementally update a hash when \p cell changes from \p old_piece to \p new_piece.
///
/// Returns the new hash. Equivalent to recomputing the full hash after the change.
inline std::uint64_t zobrist_update(const ZobristTable& t, std::uint64_t hash, std::size_t cell, int old_piece,
                                    int new_piece) {
    hash ^= t.key(cell, old_piece);  // remove the old piece's contribution
    hash ^= t.key(cell, new_piece);  // add the new piece's contribution
    return hash;
}

}  // namespace datamunge::algorithms
