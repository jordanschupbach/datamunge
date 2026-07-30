#pragma once

/// \file blast.hpp
/// \brief BLAST-style seed-and-extend local sequence alignment (heuristic).
///
/// Comparing a query against a large sequence database with full Smith-Waterman alignment is
/// too slow, so BLAST (Altschul et al., 1990) uses a *seed-and-extend* heuristic. It finds
/// short exact matches (*words* / seeds) shared by the query and the database, then *extends*
/// each seed outward without gaps, accumulating a match/mismatch score and keeping the
/// highest-scoring segment (stopping when the score drops too far below the peak -- the
/// *X-drop* rule). The surviving *high-scoring segment pairs* (HSPs) are the reported local
/// alignments. This module implements word seeding and ungapped X-drop extension.

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace datamunge::algorithms {

/// A high-scoring segment pair: an ungapped local alignment.
struct HSP {
    int query_start;  ///< Start offset in the query.
    int db_start;     ///< Start offset in the database sequence.
    int length;       ///< Aligned length.
    int score;        ///< Alignment score.
};

/// \brief Seed-and-extend local alignment of \c query against \c db.
///
/// \param word_size  seed (exact-match word) length.
/// \param match      score for a matching position (>0).
/// \param mismatch   score for a mismatch (<0).
/// \param xdrop      stop extending once the score falls this far below the running best.
/// \param min_score  only report HSPs scoring at least this.
inline std::vector<HSP> blast_search(const std::string& query, const std::string& db,
                                     int word_size, int match = 2, int mismatch = -1,
                                     int xdrop = 5, int min_score = 6) {
    // Index database word -> starting positions.
    std::unordered_map<std::string, std::vector<int>> db_words;
    for (int i = 0; i + word_size <= (int)db.size(); ++i)
        db_words[db.substr(i, word_size)].push_back(i);

    std::vector<HSP>                        hsps;
    std::unordered_map<long long, int>      best_on_diagonal;  // dedup: keep best per diagonal

    for (int qi = 0; qi + word_size <= (int)query.size(); ++qi) {
        auto it = db_words.find(query.substr(qi, word_size));
        if (it == db_words.end()) continue;
        for (int di : it->second) {
            // Ungapped extension from the seed, with X-drop, in both directions.
            int score = word_size * match;               // the exact word matches
            // extend right
            int best_right = 0, run = 0, r = 0;
            for (int k = 0;; ++k) {
                int q = qi + word_size + k, d = di + word_size + k;
                if (q >= (int)query.size() || d >= (int)db.size()) break;
                run += (query[q] == db[d]) ? match : mismatch;
                if (run > best_right) { best_right = run; r = k + 1; }
                if (run < best_right - xdrop) break;
            }
            // extend left
            int best_left = 0, runl = 0, l = 0;
            for (int k = 0;; ++k) {
                int q = qi - 1 - k, d = di - 1 - k;
                if (q < 0 || d < 0) break;
                runl += (query[q] == db[d]) ? match : mismatch;
                if (runl > best_left) { best_left = runl; l = k + 1; }
                if (runl < best_left - xdrop) break;
            }
            int total_score = score + best_right + best_left;
            int qs = qi - l, ds = di - l, len = word_size + l + r;
            if (total_score < min_score) continue;

            long long diag = (long long)ds - qs;          // same diagonal => same alignment family
            auto      bit  = best_on_diagonal.find(diag);
            if (bit == best_on_diagonal.end()) {
                best_on_diagonal[diag] = (int)hsps.size();
                hsps.push_back({qs, ds, len, total_score});
            } else if (total_score > hsps[bit->second].score) {
                hsps[bit->second] = {qs, ds, len, total_score};
            }
        }
    }

    std::sort(hsps.begin(), hsps.end(), [](const HSP& a, const HSP& b) { return a.score > b.score; });
    return hsps;
}

}  // namespace datamunge::algorithms
