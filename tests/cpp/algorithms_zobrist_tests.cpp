#include <datamunge/algorithms/zobrist_hashing.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <cmath>
#include <numeric>

namespace alg = datamunge::algorithms;
TEST(Zobrist, ValidatesDimensionsAndCategories) {
    EXPECT_THROW(alg::zobrist_init(2, 0), std::invalid_argument);
    const auto t = alg::zobrist_init(2, 3);
    EXPECT_THROW(alg::zobrist_hash(t, {1}), std::invalid_argument);
    EXPECT_THROW(alg::zobrist_hash(t, {1, -1}), std::out_of_range);
    EXPECT_THROW(alg::zobrist_update(t, 0, 2, 0, 1), std::out_of_range);
    EXPECT_THROW(alg::zobrist_update(t, 0, 0, 0, 3), std::out_of_range);
    EXPECT_EQ(alg::zobrist_hash(alg::zobrist_init(0, 1), {}), 0);
}
TEST(Zobrist, ConvertsRowsWithoutQuantisingOrIgnoringNulls) {
    datamunge::dstruct::DataFrame frame;
    frame.add_column("a", std::vector<double>{0, 2});
    frame.add_column("b", std::vector<double>{1, 0});
    EXPECT_EQ(alg::zobrist_boards(frame, 3), (std::vector<std::vector<int>>{{0, 1}, {2, 0}}));
    EXPECT_THROW(alg::zobrist_boards(frame, 2), std::invalid_argument);
    for (double invalid : {-1.0, 0.5, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        datamunge::dstruct::DataFrame bad;
        bad.add_column("x", std::vector<double>{invalid});
        EXPECT_THROW(alg::zobrist_boards(bad, 3), std::invalid_argument);
    }
    frame.set_null("a", 0);
    EXPECT_THROW(alg::zobrist_boards(frame, 3), std::invalid_argument);
}
TEST(Zobrist, TraceAgreesWithIndependentXorAndUndo) {
    auto t = alg::zobrist_init(3, 3);
    t.keys = {{0, 1, 2}, {0, 4, 8}, {0, 16, 32}};
    const auto trace = alg::zobrist_transition(t, {1, 2, 0}, {2, 0, 1});
    EXPECT_EQ(trace.nrows(), 3);
    EXPECT_EQ(trace.string_at("incremental", 0), "0x000000000000000a");
    EXPECT_EQ(trace.string_at("incremental", 1), "0x0000000000000002");
    EXPECT_EQ(trace.string_at("incremental", 2), "0x0000000000000012");
    EXPECT_EQ(trace.double_at("flipped_bits", 0), 2);
    for (std::size_t r = 0; r < 3; ++r) EXPECT_EQ(trace.double_at("verified", r), 1);
    EXPECT_EQ(alg::zobrist_transition(t, {1, 2, 0}, {1, 2, 0}).nrows(), 0);
    const auto h = alg::zobrist_hash(t, {1, 2, 0});
    EXPECT_EQ(alg::zobrist_update(t, alg::zobrist_update(t, h, 0, 1, 2), 0, 2, 1), h);
    // A rectangle of boards has a deterministic four-way XOR dependency.
    EXPECT_EQ(alg::zobrist_hash(t, {1, 1, 0}) ^ alg::zobrist_hash(t, {1, 2, 0}) ^
              alg::zobrist_hash(t, {2, 1, 0}) ^ alg::zobrist_hash(t, {2, 2, 0}), 0);
}
TEST(Zobrist, CollisionCountsPairsAndDeduplicatesBoards) {
    std::vector<std::vector<int>> boards;
    for (int v = 0; v < 12; ++v) boards.push_back({v});
    boards.push_back({0});
    const auto profile = alg::zobrist_collision_profile(boards, 12, {1, 8, 64}, 1, 7);
    const auto t = alg::zobrist_init(1, 12, 7);
    int pairs = 0;
    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < i; ++j) pairs += (t.key(0, i) & 1) == (t.key(0, j) & 1);
    EXPECT_EQ(profile.double_at("mean_pairs", 0), pairs);
    EXPECT_EQ(profile.double_at("expected_pairs", 0), 33);
    EXPECT_DOUBLE_EQ(profile.double_at("expected_pairs", 2), std::ldexp(66.0, -64));
    EXPECT_THROW(alg::zobrist_collision_profile({{0}, {0, 1}}, 2, {8}), std::invalid_argument);
    EXPECT_THROW(alg::zobrist_collision_profile(boards, 12, {0}), std::invalid_argument);
    EXPECT_THROW(alg::zobrist_collision_profile(boards, 12, {65}), std::invalid_argument);
    EXPECT_THROW(alg::zobrist_collision_profile(boards, 12, {8}, 0), std::invalid_argument);
    EXPECT_EQ(alg::zobrist_collision_profile({}, 2, {64}, 1).double_at("mean_pairs", 0), 0);
}
