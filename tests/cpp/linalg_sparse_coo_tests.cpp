#include <gtest/gtest.h>

#include <datamunge/linalg/sparse_coo.hpp>

using datamunge::linalg::SparseCOO;
using datamunge::linalg::eval;

// ---- Construction ----

TEST(SparseCOO, default_construction) {
    SparseCOO<double> A;
    EXPECT_EQ(A.rows(), 0u);
    EXPECT_EQ(A.cols(), 0u);
    EXPECT_EQ(A.nnz(), 0u);
}

TEST(SparseCOO, sized_construction) {
    SparseCOO<double> A(3, 4);
    EXPECT_EQ(A.rows(), 3u);
    EXPECT_EQ(A.cols(), 4u);
    EXPECT_EQ(A.nnz(), 0u);
}

TEST(SparseCOO, hinted_construction) {
    SparseCOO<double> A(100, 100, 50);
    EXPECT_EQ(A.nnz(), 0u);
}

// ---- set / set_batch ----

TEST(SparseCOO, set_appends_entries) {
    SparseCOO<double> A(3, 3);
    A.set(0, 0, 1.0);
    A.set(1, 1, 2.0);
    A.set(2, 2, 3.0);
    ASSERT_EQ(A.nnz(), 3u);
    EXPECT_DOUBLE_EQ(A.values()[0], 1.0);
    EXPECT_DOUBLE_EQ(A.values()[1], 2.0);
    EXPECT_DOUBLE_EQ(A.values()[2], 3.0);
}

TEST(SparseCOO, set_batch_appends_entries) {
    SparseCOO<double> A(3, 3);
    A.set_batch({0, 1, 2}, {0, 1, 2}, {1.0, 2.0, 3.0});
    EXPECT_EQ(A.nnz(), 3u);
    EXPECT_EQ(A.row_indices()[2], 2u);
    EXPECT_EQ(A.col_indices()[2], 2u);
    EXPECT_DOUBLE_EQ(A.values()[2], 3.0);
}

// ---- set_zero / set_zero_batch ----

TEST(SparseCOO, set_zero_marks_without_removing) {
    SparseCOO<double> A(2, 2);
    A.set(0, 0, 1.0);
    A.set(0, 1, 2.0);
    A.set_zero(0, 1);
    EXPECT_EQ(A.nnz(), 2u);
    EXPECT_DOUBLE_EQ(A.values()[1], 0.0);
}

TEST(SparseCOO, set_zero_batch_marks_multiple) {
    SparseCOO<double> A(3, 3);
    A.set_batch({0, 1, 2}, {0, 1, 2}, {1.0, 2.0, 3.0});
    A.set_zero_batch({0, 2}, {0, 2});
    A.compress();
    ASSERT_EQ(A.nnz(), 1u);
    EXPECT_DOUBLE_EQ(A.values()[0], 2.0);
}

// ---- compress ----

TEST(SparseCOO, compress_removes_zeros) {
    SparseCOO<double> A(2, 2);
    A.set(0, 0, 1.0);
    A.set(0, 1, 2.0);
    A.set_zero(0, 1);
    A.compress();
    ASSERT_EQ(A.nnz(), 1u);
    EXPECT_DOUBLE_EQ(A.values()[0], 1.0);
}

TEST(SparseCOO, compress_merges_duplicates_by_summing) {
    SparseCOO<double> A(2, 2);
    A.set(0, 0, 1.0);
    A.set(0, 0, 2.0);
    A.set(1, 1, 3.0);
    A.compress();
    ASSERT_EQ(A.nnz(), 2u);
    bool found_00 = false;
    for (std::size_t k = 0; k < A.nnz(); ++k) {
        if (A.row_indices()[k] == 0 && A.col_indices()[k] == 0) {
            EXPECT_DOUBLE_EQ(A.values()[k], 3.0);
            found_00 = true;
        }
    }
    EXPECT_TRUE(found_00);
}

TEST(SparseCOO, compress_cancels_to_zero_and_removes) {
    SparseCOO<double> A(2, 2);
    A.set(0, 0, 5.0);
    A.set(0, 0, -5.0);
    A.compress();
    EXPECT_EQ(A.nnz(), 0u);
}

TEST(SparseCOO, compress_empty_is_noop) {
    SparseCOO<double> A(3, 3);
    A.compress();
    EXPECT_EQ(A.nnz(), 0u);
}

// ---- spmv ----

TEST(SparseCOO, spmv_identity) {
    SparseCOO<double> I(3, 3);
    I.set(0, 0, 1.0);
    I.set(1, 1, 1.0);
    I.set(2, 2, 1.0);
    const std::vector<double> x   = {2.0, 3.0, 5.0};
    const auto                y   = I.spmv(x);
    ASSERT_EQ(y.size(), 3u);
    EXPECT_DOUBLE_EQ(y[0], 2.0);
    EXPECT_DOUBLE_EQ(y[1], 3.0);
    EXPECT_DOUBLE_EQ(y[2], 5.0);
}

TEST(SparseCOO, spmv_general) {
    // [[1,2],[3,4]] * [1,1] = [3,7]
    SparseCOO<double> A(2, 2);
    A.set(0, 0, 1.0);
    A.set(0, 1, 2.0);
    A.set(1, 0, 3.0);
    A.set(1, 1, 4.0);
    const std::vector<double> x = {1.0, 1.0};
    const auto                y = A.spmv(x);
    ASSERT_EQ(y.size(), 2u);
    EXPECT_DOUBLE_EQ(y[0], 3.0);
    EXPECT_DOUBLE_EQ(y[1], 7.0);
}

TEST(SparseCOO, spmv_zero_matrix) {
    SparseCOO<double>         A(3, 3);
    const std::vector<double> x = {1.0, 2.0, 3.0};
    const auto                y = A.spmv(x);
    ASSERT_EQ(y.size(), 3u);
    EXPECT_DOUBLE_EQ(y[0], 0.0);
    EXPECT_DOUBLE_EQ(y[1], 0.0);
    EXPECT_DOUBLE_EQ(y[2], 0.0);
}

// ---- Expression templates: ScaleExpr ----

TEST(SparseCOO, expr_scale_left) {
    SparseCOO<double> A(2, 2);
    A.set(0, 0, 1.0);
    A.set(1, 1, 2.0);
    SparseCOO<double> B = 3.0 * A;
    ASSERT_EQ(B.nnz(), 2u);
    double sum = 0.0;
    for (auto v : B.values()) sum += v;
    EXPECT_DOUBLE_EQ(sum, 9.0);  // 3*1 + 3*2
}

TEST(SparseCOO, expr_scale_right) {
    SparseCOO<double> A(2, 2);
    A.set(0, 0, 4.0);
    SparseCOO<double> B = A * 2.0;
    ASSERT_EQ(B.nnz(), 1u);
    EXPECT_DOUBLE_EQ(B.values()[0], 8.0);
}

// ---- Expression templates: SumExpr ----

TEST(SparseCOO, expr_sum_disjoint) {
    SparseCOO<double> A(2, 2), B(2, 2);
    A.set(0, 0, 1.0);
    A.set(1, 1, 2.0);
    B.set(0, 1, 3.0);
    B.set(1, 0, 4.0);
    SparseCOO<double> C = A + B;
    EXPECT_EQ(C.nnz(), 4u);
}

TEST(SparseCOO, expr_sum_overlapping_merges) {
    SparseCOO<double> A(2, 2), B(2, 2);
    A.set(0, 0, 1.0);
    A.set(1, 1, 2.0);
    B.set(0, 0, 3.0);
    B.set(1, 1, 4.0);
    SparseCOO<double> C = A + B;
    ASSERT_EQ(C.nnz(), 2u);
    for (std::size_t k = 0; k < C.nnz(); ++k) {
        if (C.row_indices()[k] == 0)
            EXPECT_DOUBLE_EQ(C.values()[k], 4.0);
        else
            EXPECT_DOUBLE_EQ(C.values()[k], 6.0);
    }
}

// ---- Expression templates: chained ----

TEST(SparseCOO, expr_chained_scale_sum) {
    SparseCOO<double> A(2, 2), B(2, 2);
    A.set(0, 0, 1.0);
    B.set(1, 1, 1.0);
    SparseCOO<double> C = 2.0 * A + 3.0 * B;
    ASSERT_EQ(C.nnz(), 2u);
    for (std::size_t k = 0; k < C.nnz(); ++k) {
        if (C.row_indices()[k] == 0)
            EXPECT_DOUBLE_EQ(C.values()[k], 2.0);
        else
            EXPECT_DOUBLE_EQ(C.values()[k], 3.0);
    }
}

TEST(SparseCOO, expr_scale_of_sum) {
    SparseCOO<double> A(2, 2), B(2, 2);
    A.set(0, 0, 1.0);
    B.set(1, 1, 1.0);
    SparseCOO<double> C = 5.0 * (A + B);
    ASSERT_EQ(C.nnz(), 2u);
    for (auto v : C.values()) EXPECT_DOUBLE_EQ(v, 5.0);
}

TEST(SparseCOO, explicit_eval) {
    SparseCOO<double> A(2, 2);
    A.set(0, 0, 2.0);
    auto B = eval(A * 4.0);
    ASSERT_EQ(B.nnz(), 1u);
    EXPECT_DOUBLE_EQ(B.values()[0], 8.0);
}

// ---- Typed: float ----

TEST(SparseCOO, float_type) {
    SparseCOO<float> A(2, 2);
    A.set(0, 0, 1.0f);
    A.set(1, 1, 2.0f);
    SparseCOO<float> B = 2.0f * A;
    ASSERT_EQ(B.nnz(), 2u);
    EXPECT_FLOAT_EQ(B.values()[0], 2.0f);
    EXPECT_FLOAT_EQ(B.values()[1], 4.0f);
}

// ---- is_compressed flag ----

TEST(SparseCOO, not_compressed_after_construction) {
    SparseCOO<double> A(3, 3);
    EXPECT_FALSE(A.is_compressed());
}

TEST(SparseCOO, compressed_after_compress) {
    SparseCOO<double> A(3, 3);
    A.set(0, 0, 1.0);
    A.compress();
    EXPECT_TRUE(A.is_compressed());
}

TEST(SparseCOO, compressed_after_compress_empty) {
    SparseCOO<double> A(3, 3);
    A.compress();
    EXPECT_TRUE(A.is_compressed());
}

TEST(SparseCOO, set_clears_compressed) {
    SparseCOO<double> A(3, 3);
    A.compress();
    ASSERT_TRUE(A.is_compressed());
    A.set(0, 0, 1.0);
    EXPECT_FALSE(A.is_compressed());
}

TEST(SparseCOO, set_batch_clears_compressed) {
    SparseCOO<double> A(3, 3);
    A.compress();
    A.set_batch({0}, {0}, {1.0});
    EXPECT_FALSE(A.is_compressed());
}

TEST(SparseCOO, set_zero_clears_compressed) {
    SparseCOO<double> A(3, 3);
    A.set(0, 0, 1.0);
    A.compress();
    A.set_zero(0, 0);
    EXPECT_FALSE(A.is_compressed());
}

TEST(SparseCOO, set_zero_batch_clears_compressed) {
    SparseCOO<double> A(3, 3);
    A.set(0, 0, 1.0);
    A.compress();
    A.set_zero_batch({0}, {0});
    EXPECT_FALSE(A.is_compressed());
}

TEST(SparseCOO, expr_result_is_compressed) {
    SparseCOO<double> A(2, 2), B(2, 2);
    A.set(0, 0, 1.0);
    B.set(1, 1, 2.0);
    SparseCOO<double> C = A + B;  // eval(SumExpr) calls compress()
    EXPECT_TRUE(C.is_compressed());
}

// ---- spmv buffer-reuse overload ----

TEST(SparseCOO, spmv_buffer_reuse_matches_allocating) {
    SparseCOO<double> A(3, 3);
    A.set(0, 0, 1.0);
    A.set(1, 1, 2.0);
    A.set(2, 2, 3.0);
    const std::vector<double> x = {4.0, 5.0, 6.0};

    const auto            y_alloc = A.spmv(x);
    std::vector<double>   y_reuse(A.rows());
    A.spmv(x, y_reuse);

    ASSERT_EQ(y_reuse.size(), y_alloc.size());
    for (std::size_t i = 0; i < y_alloc.size(); ++i)
        EXPECT_DOUBLE_EQ(y_reuse[i], y_alloc[i]);
}

TEST(SparseCOO, spmv_buffer_reuse_zeroes_output) {
    // Previous contents of y must not bleed through
    SparseCOO<double>   A(2, 2);
    A.set(0, 0, 1.0);
    const std::vector<double> x       = {1.0, 1.0};
    std::vector<double>       y       = {999.0, 999.0};
    A.spmv(x, y);
    EXPECT_DOUBLE_EQ(y[0], 1.0);
    EXPECT_DOUBLE_EQ(y[1], 0.0);  // no entry for row 1 → must be zero, not 999
}

// ---- spmv_tiled ----

static SparseCOO<double> make_compressed_2x2() {
    SparseCOO<double> A(2, 2);
    A.set(0, 0, 1.0);
    A.set(0, 1, 2.0);
    A.set(1, 0, 3.0);
    A.set(1, 1, 4.0);
    A.compress();
    return A;
}

TEST(SparseCOO, spmv_tiled_matches_plain_default_block) {
    auto                      A = make_compressed_2x2();
    const std::vector<double> x = {1.0, 1.0};
    const auto                y_plain  = A.spmv(x);
    std::vector<double>       y_tiled(A.rows());
    A.spmv_tiled(x, y_tiled);
    ASSERT_EQ(y_tiled.size(), y_plain.size());
    for (std::size_t i = 0; i < y_plain.size(); ++i)
        EXPECT_DOUBLE_EQ(y_tiled[i], y_plain[i]);
}

TEST(SparseCOO, spmv_tiled_block_rows_1) {
    auto                      A = make_compressed_2x2();
    const std::vector<double> x = {1.0, 1.0};
    const auto                y_plain = A.spmv(x);
    std::vector<double>       y_tiled(A.rows());
    A.spmv_tiled(x, y_tiled, 1);
    for (std::size_t i = 0; i < y_plain.size(); ++i)
        EXPECT_DOUBLE_EQ(y_tiled[i], y_plain[i]);
}

TEST(SparseCOO, spmv_tiled_block_rows_larger_than_matrix) {
    auto                      A = make_compressed_2x2();
    const std::vector<double> x = {1.0, 1.0};
    const auto                y_plain = A.spmv(x);
    std::vector<double>       y_tiled(A.rows());
    A.spmv_tiled(x, y_tiled, 1000);
    for (std::size_t i = 0; i < y_plain.size(); ++i)
        EXPECT_DOUBLE_EQ(y_tiled[i], y_plain[i]);
}

TEST(SparseCOO, spmv_tiled_allocating_overload) {
    auto                      A = make_compressed_2x2();
    const std::vector<double> x = {2.0, 3.0};
    const auto y_plain = A.spmv(x);
    const auto y_tiled = A.spmv_tiled(x);
    ASSERT_EQ(y_tiled.size(), y_plain.size());
    for (std::size_t i = 0; i < y_plain.size(); ++i)
        EXPECT_DOUBLE_EQ(y_tiled[i], y_plain[i]);
}

TEST(SparseCOO, spmv_tiled_zeroes_output_buffer) {
    auto                A = make_compressed_2x2();
    std::vector<double> x      = {0.0, 0.0};
    std::vector<double> y      = {999.0, 999.0};
    A.spmv_tiled(x, y, 1);
    EXPECT_DOUBLE_EQ(y[0], 0.0);
    EXPECT_DOUBLE_EQ(y[1], 0.0);
}

TEST(SparseCOO, spmv_tiled_sparse_rows) {
    // Only rows 0 and 4 have entries; rows 1-3 are empty.
    SparseCOO<double> A(5, 5);
    A.set(0, 0, 2.0);
    A.set(4, 4, 3.0);
    A.compress();
    const std::vector<double> x = {1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<double>       y(5);
    A.spmv_tiled(x, y, 2);  // strips: [0,2), [2,4), [4,5)
    EXPECT_DOUBLE_EQ(y[0], 2.0);
    EXPECT_DOUBLE_EQ(y[1], 0.0);
    EXPECT_DOUBLE_EQ(y[2], 0.0);
    EXPECT_DOUBLE_EQ(y[3], 0.0);
    EXPECT_DOUBLE_EQ(y[4], 3.0);
}
