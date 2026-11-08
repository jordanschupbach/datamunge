#pragma once
/// @file blast.hpp
/// @brief Checked exact-byte seeding and ungapped X-drop alignment, with canonical inspection APIs.
#include <cstdint>
#include <string>
#include <vector>

namespace datamunge::algorithms {
/// @brief Owned ungapped segment pair with zero-based starts and half-open ranges.
/// The same length applies to both sequences. Score is the sum of match/mismatch
/// contributions; it is not a bit score, probability or E-value. No strings are borrowed.
/// @par Example
/// @code
/// const auto hits=dm::algorithms::blast_search("ACGT","ACGT",2);
/// std::print("{} {}\n",hits.front().length,hits.front().score); // 4 8
/// @endcode
/// @see blast_search(), BlastExtensionTrace
struct HSP {
    int query_start{0}; ///< Inclusive query byte offset.
    int db_start{0}; ///< Inclusive database byte offset.
    int length{0}; ///< Aligned bytes; exclusive ends are start+length.
    int score{0}; ///< Raw integer alignment score, checked before narrowing.
};

/// @brief A pair of equal fixed-length byte words, described by their starting positions.
/// Word length belongs to the enclosing operation. Positions are nonnegative;
/// the seed API orders them by query start, then database start.
/// @par Example
/// @code
/// const auto seeds=dm::algorithms::blast_seeds("AAA","AAAA",2);
/// std::print("{}\n",seeds.size()); // 6
/// @endcode
/// @see blast_seeds(), blast_extend_trace()
struct BlastSeed {
    int query_start{0}; ///< Inclusive query word offset.
    int db_start{0}; ///< Inclusive database word offset.
};

/// @brief One actually inspected byte pair outside an exact seed.
/// Each direction starts running/best scores at zero. Best is updated strictly;
/// equality keeps the shorter extension. A drop stops only when best-running>xdrop.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::blast_extend_trace("AAACCAA","AAATTAA",{0,0},3,2,-1,1);
/// std::print("{} {}\n",trace.steps.back().running_score,trace.steps.back().stopped); // -2 true
/// @endcode
/// @see BlastExtensionTrace
struct BlastExtensionStep {
    bool left{false}; ///< True when walking left from the seed; right events are recorded first.
    int query_position{0}; ///< Inspected query byte index.
    int db_position{0}; ///< Inspected database byte index.
    int delta{0}; ///< Match reward or mismatch penalty at this pair.
    std::int64_t running_score{0}; ///< Direction-local prefix score after this pair.
    std::int64_t best_score{0}; ///< Best direction-local prefix score, including empty prefix.
    bool stopped{false}; ///< Whether this pair crosses the strict X-drop threshold.
};

/// @brief Owned seed, best anchored segment and actual extension events.
/// The HSP contains the seed even if a caller's search threshold would suppress
/// it. Events include discarded negative tails; HSP endpoints use the best prefix
/// in each direction. No input is modified or retained by reference.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::blast_extend_trace("AAACCAA","AAATTAA",{0,0},3,2,-1,2);
/// std::print("{} {}\n",trace.hsp.length,trace.hsp.score); // 7 8
/// @endcode
/// @see blast_extend_trace(), blast_search()
struct BlastExtensionTrace {
    BlastSeed seed; ///< Exact seed supplied by the caller.
    HSP hsp; ///< Highest-scoring inspected seed-containing segment.
    std::vector<BlastExtensionStep> steps; ///< Right-then-left chronological pair inspections.
};

/// @brief Enumerate every shared exact byte word using the search engine's database index.
/// Includes overlapping words and all repeated-word pairs. Case and embedded NUL
/// bytes matter; no alphabet validation, reverse complement or normalisation.
/// Empty/short sequences yield no seeds after validating word_size. Expected
/// O((Q+M)w+S) time and O(Mw+S) storage for lengths Q,M, word length w and S seeds;
/// hash-table collision behavior can be worse. Inputs are borrowed only during the call.
/// @param query Query bytes, at most INT_MAX bytes.
/// @param db Database bytes, at most INT_MAX bytes.
/// @param word_size Positive exact-word length; larger than either sequence yields no seeds.
/// @return Owned seeds ordered by query position, then database position.
/// @throws std::invalid_argument If word_size<=0.
/// @throws std::length_error If either sequence exceeds the int coordinate range.
/// @par Example
/// @code
/// const auto seeds=dm::algorithms::blast_seeds("ACGT","TACGT",3);
/// std::print("{} {} {}\n",seeds.size(),seeds[0].query_start,seeds[0].db_start); // 2 0 1
/// @endcode
/// @par References
/// - S. F. Altschul et al. (1990), Basic local alignment search tool, J. Mol. Biol.
///   215(3), 403-410, doi:10.1016/S0022-2836(05)80360-2. Exact-word-only variant here.
/// @see BlastSeed, blast_search(), blast_extend_trace()
std::vector<BlastSeed> blast_seeds(const std::string& query,const std::string& db,int word_size);

/// @brief Trace the same two-sided ungapped X-drop extension used by blast_search().
/// Right and left prefix gains start independently at zero; the seed score is
/// word_size*match. Extension stops on a boundary or a drop strictly greater
/// than xdrop, in raw score units. Best-prefix ties keep the shorter prefix.
/// O(L) time and owned event storage for L inspected pairs; no shared writes.
/// @param query Query bytes, length<=INT_MAX.
/// @param db Database bytes, length<=INT_MAX.
/// @param seed Nonnegative starts of an in-range exact shared word.
/// @param word_size Positive seed length.
/// @param match Positive match reward, default 2.
/// @param mismatch Negative mismatch penalty, default -1.
/// @param xdrop Nonnegative permitted loss from a direction's peak, default 5.
/// @return Owned seed, best inspected anchored HSP and all inspected pair states.
/// @throws std::invalid_argument For invalid scoring, a nonpositive word size or an invalid/nonexact seed.
/// @throws std::length_error If a sequence exceeds INT_MAX bytes.
/// @throws std::overflow_error If the resulting HSP score exceeds INT_MAX; arithmetic uses int64 internally.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::blast_extend_trace("AAACCAA","AAATTAA",{0,0},3,2,-1,1);
/// std::print("{} {}\n",trace.hsp.length,trace.steps.size()); // 3 2
/// @endcode
/// @par References
/// - S. F. Altschul et al. (1990), Basic local alignment search tool,
///   doi:10.1016/S0022-2836(05)80360-2. Simplified exact-seed, raw-score ungapped variant.
/// @see blast_search(), blast_seeds(), BlastExtensionStep
BlastExtensionTrace blast_extend_trace(const std::string& query,const std::string& db,
    BlastSeed seed,int word_size,int match=2,int mismatch=-1,int xdrop=5);

/// @brief Find exact-seed, ungapped X-drop HSPs, retaining one best segment per diagonal.
/// Visits seeds in query/database order, extending right then left with the same
/// loop as blast_extend_trace(). A higher score replaces the diagonal's prior HSP;
/// equal scores retain the first seed's result, even for distinct separated regions.
/// Results sort by decreasing score, then query start, database start and length.
/// This heuristic can miss unseeded or X-drop-separated optima. It provides no gaps,
/// statistical significance, protein neighbourhood words or reverse complements.
/// Expected time O((Q+M)w+SL+H log(H+1)), index storage O(Mw+H), where S is seed
/// count, L bounds extension length and H<=S is retained diagonal count. Repetitive
/// inputs can cost cubic time; no subquadratic or speedup guarantee is made.
/// @param query Query bytes, length<=INT_MAX; empty yields no HSPs.
/// @param db Database bytes, length<=INT_MAX; empty yields no HSPs.
/// @param word_size Positive exact-word length; longer than either input yields no HSPs.
/// @param match Positive match reward, default 2.
/// @param mismatch Negative mismatch penalty, default -1.
/// @param xdrop Nonnegative direction-local raw-score loss allowance, default 5; stopping is strict.
/// @param min_score Inclusive minimum raw HSP score, default 6; any int is permitted.
/// @return Owned HSPs in deterministic score/coordinate order; no caller data is mutated.
/// @throws std::invalid_argument For word_size<=0, match<=0, mismatch>=0 or xdrop<0.
/// @throws std::length_error If either input exceeds INT_MAX bytes.
/// @throws std::overflow_error If an extended HSP score exceeds INT_MAX.
/// @par Example
/// @code
/// const auto hits=dm::algorithms::blast_search("TTGGATCCAATT","ACGTACGTGGATCCAATTGGCCAAGGTT",4);
/// std::print("{} {}\n",hits.front().score,hits.front().length); // 22 11
/// @endcode
/// @par References
/// - S. F. Altschul, W. Gish, W. Miller, E. W. Myers and D. J. Lipman (1990),
///   Basic local alignment search tool, J. Mol. Biol. 215(3), 403-410,
///   doi:10.1016/S0022-2836(05)80360-2. This is a simplified ungapped exact-seed variant.
/// @see blast_seeds(), blast_extend_trace(), HSP
std::vector<HSP> blast_search(const std::string& query,const std::string& db,
    int word_size,int match=2,int mismatch=-1,int xdrop=5,int min_score=6);
} // namespace datamunge::algorithms
