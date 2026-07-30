#include <gtest/gtest.h>

#include <datamunge/algorithms/cyk_parser.hpp>
#include <datamunge/algorithms/double_dabble.hpp>
#include <datamunge/algorithms/fnv_hash.hpp>
#include <datamunge/algorithms/pearson_hashing.hpp>
#include <datamunge/algorithms/shunting_yard.hpp>
#include <datamunge/algorithms/zobrist_hashing.hpp>

#include <cstdint>
#include <string>
#include <vector>

using namespace datamunge::algorithms;

TEST(ShuntingYard, ConvertsAndEvaluates) {
    EXPECT_EQ(shunting_yard_to_postfix("3 + 4 * 2"), "3 4 2 * +");
    EXPECT_EQ(shunting_yard_to_postfix("( 3 + 4 ) * 2"), "3 4 + 2 *");
    EXPECT_NEAR(shunting_yard_evaluate("3 + 4 * 2 / ( 1 - 5 )"), 1.0, 1e-12);
    EXPECT_NEAR(shunting_yard_evaluate("2 ^ 3 ^ 2"), 512.0, 1e-9);  // right-associative
    EXPECT_NEAR(shunting_yard_evaluate("10 + 2 * 6"), 22.0, 1e-12);
    EXPECT_NEAR(shunting_yard_evaluate("100 * 2 + 12"), 212.0, 1e-12);
}

TEST(CYK, RecognizesAnBn) {
    // S->AB | AC, C->SB, A->a, B->b  generates a^n b^n (n>=1).  Ids: S=0,A=1,B=2,C=3.
    CNFGrammar g;
    g.start = 0;
    g.unary['a'] = {1};
    g.unary['b'] = {2};
    g.binary = {{0, 1, 2}, {0, 1, 3}, {3, 0, 2}};
    EXPECT_TRUE(cyk_recognize(g, "ab"));
    EXPECT_TRUE(cyk_recognize(g, "aabb"));
    EXPECT_TRUE(cyk_recognize(g, "aaabbb"));
    EXPECT_FALSE(cyk_recognize(g, "aab"));
    EXPECT_FALSE(cyk_recognize(g, "abb"));
    EXPECT_FALSE(cyk_recognize(g, "ba"));
    EXPECT_FALSE(cyk_recognize(g, "aabbb"));
}

TEST(FNV, ReferenceValuesAndDispersion) {
    // Empty string hashes to the offset basis.
    EXPECT_EQ(fnv1a_64(""), 14695981039346656037ULL);
    EXPECT_EQ(fnv1a_32(""), 2166136261u);
    // Known reference: FNV-1a 32-bit of "a" is 0xE40C292C.
    EXPECT_EQ(fnv1a_32("a"), 0xE40C292Cu);
    // Distinct strings -> distinct hashes here.
    EXPECT_NE(fnv1a_64("hello"), fnv1a_64("world"));
    EXPECT_NE(fnv1a_64("hello"), fnv1a_64("hellp"));
}

TEST(Pearson, DeterministicAndSensitive) {
    std::uint8_t h1 = pearson_hash("hello");
    EXPECT_EQ(pearson_hash("hello"), h1);          // deterministic
    EXPECT_NE(pearson_hash("hellp"), h1);          // one-byte change flips it
    // Wide hash produces the requested number of bytes of entropy.
    EXPECT_NE(pearson_hash_wide("hello", 4), pearson_hash_wide("world", 4));
}

TEST(Zobrist, IncrementalMatchesFullRecompute) {
    ZobristTable        t = zobrist_init(/*cells=*/64, /*pieces=*/13, /*seed=*/42);
    std::vector<int>    board(64, 0);
    board[0] = 1;  board[10] = 5;  board[63] = 12;
    std::uint64_t       h = zobrist_hash(t, board);

    // Move the piece at cell 10 (piece 5) to cell 20, incrementally.
    std::uint64_t hi = h;
    hi = zobrist_update(t, hi, 10, 5, 0);   // clear cell 10
    hi = zobrist_update(t, hi, 20, 0, 5);   // place piece 5 at cell 20
    board[10] = 0;  board[20] = 5;
    EXPECT_EQ(hi, zobrist_hash(t, board));  // incremental == full recompute
    // Empty board hashes to 0 (all-empty keys are zero).
    EXPECT_EQ(zobrist_hash(t, std::vector<int>(64, 0)), 0ULL);
}

TEST(DoubleDabble, ConvertsBinaryToDecimal) {
    EXPECT_EQ(double_dabble(243, 8), (std::vector<int>{2, 4, 3}));
    EXPECT_EQ(double_dabble(0, 8), (std::vector<int>{0}));
    EXPECT_EQ(double_dabble(255, 8), (std::vector<int>{2, 5, 5}));
    EXPECT_EQ(double_dabble_string(1000000, 20), "1000000");
    EXPECT_EQ(double_dabble_string(4294967295ULL, 32), "4294967295");
}
