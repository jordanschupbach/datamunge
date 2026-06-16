#include <gtest/gtest.h>

#include <datamunge/linalg/linalg.hpp>

using namespace datamunge::linalg;

// ============================================================
// Shared helpers
// ============================================================

// 4×4 test matrix:
//   [ 1  0  2  0 ]
//   [ 0  3  0  0 ]
//   [ 4  0  0  5 ]
//   [ 0  0  0  6 ]
static SparseCOO<double> make_A() {
    SparseCOO<double> A(4, 4, 6);
    A.set(0, 0, 1.0);
    A.set(0, 2, 2.0);
    A.set(1, 1, 3.0);
    A.set(2, 0, 4.0);
    A.set(2, 3, 5.0);
    A.set(3, 3, 6.0);
    A.compress();
    return A;
}

// A * [1,2,3,4] = [7, 6, 24, 24]
static const std::vector<double> kX4     = {1.0, 2.0, 3.0, 4.0};
static const std::vector<double> kAX4    = {7.0, 6.0, 24.0, 24.0};

static void expect_vec_eq(const std::vector<double>& a, const std::vector<double>& b) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        EXPECT_DOUBLE_EQ(a[i], b[i]) << " at index " << i;
}

// 5×5 tridiagonal matrix for DIA tests
//   [ 2 -1  0  0  0 ]
//   [-1  2 -1  0  0 ]
//   [ 0 -1  2 -1  0 ]
//   [ 0  0 -1  2 -1 ]
//   [ 0  0  0 -1  2 ]
static SparseCOO<double> make_tridiag() {
    SparseCOO<double> B(5, 5, 13);
    for (std::size_t i = 0; i < 5; ++i) {
        B.set(i, i, 2.0);
        if (i > 0) B.set(i, i - 1, -1.0);
        if (i < 4) B.set(i, i + 1, -1.0);
    }
    B.compress();
    return B;
}

// ============================================================
// CSR
// ============================================================

TEST(SparseCSR, dimensions) {
    auto csr = to_csr(make_A());
    EXPECT_EQ(csr.nrows, 4u);
    EXPECT_EQ(csr.ncols, 4u);
    EXPECT_EQ(csr.nnz(), 6u);
}

TEST(SparseCSR, row_ptr_is_prefix_sum_of_row_counts) {
    auto csr = to_csr(make_A());
    // Row nnz counts: 2, 1, 2, 1
    ASSERT_EQ(csr.row_ptr.size(), 5u);
    EXPECT_EQ(csr.row_ptr[0], 0u);
    EXPECT_EQ(csr.row_ptr[1], 2u);
    EXPECT_EQ(csr.row_ptr[2], 3u);
    EXPECT_EQ(csr.row_ptr[3], 5u);
    EXPECT_EQ(csr.row_ptr[4], 6u);
}

TEST(SparseCSR, spmv_allocating) {
    expect_vec_eq(to_csr(make_A()).spmv(kX4), kAX4);
}

TEST(SparseCSR, spmv_buffer_reuse) {
    auto              csr = to_csr(make_A());
    std::vector<double> y(4);
    csr.spmv(kX4, y);
    expect_vec_eq(y, kAX4);
}

TEST(SparseCSR, spmv_zeroes_output_buffer) {
    auto               csr = to_csr(make_A());
    std::vector<double> y  = {999.0, 999.0, 999.0, 999.0};
    csr.spmv(kX4, y);
    expect_vec_eq(y, kAX4);
}

TEST(SparseCSR, roundtrip_to_coo) {
    auto coo2 = to_coo(to_csr(make_A()));
    EXPECT_EQ(coo2.nnz(), 6u);
    EXPECT_TRUE(coo2.is_compressed());
    expect_vec_eq(coo2.spmv(kX4), kAX4);
}

TEST(SparseCSR, empty_matrix) {
    SparseCOO<double> empty(3, 4);
    empty.compress();
    auto              csr = to_csr(empty);
    EXPECT_EQ(csr.nnz(), 0u);
    ASSERT_EQ(csr.row_ptr.size(), 4u);
    for (auto v : csr.row_ptr) EXPECT_EQ(v, 0u);
    expect_vec_eq(csr.spmv({1.0, 2.0, 3.0, 4.0}), {0.0, 0.0, 0.0});
}

// ============================================================
// CSC
// ============================================================

TEST(SparseCSC, dimensions) {
    auto csc = to_csc(make_A());
    EXPECT_EQ(csc.nrows, 4u);
    EXPECT_EQ(csc.ncols, 4u);
    EXPECT_EQ(csc.nnz(), 6u);
}

TEST(SparseCSC, col_ptr_is_prefix_sum_of_col_counts) {
    auto csc = to_csc(make_A());
    // Col nnz counts: 2, 1, 1, 2
    ASSERT_EQ(csc.col_ptr.size(), 5u);
    EXPECT_EQ(csc.col_ptr[0], 0u);
    EXPECT_EQ(csc.col_ptr[1], 2u);
    EXPECT_EQ(csc.col_ptr[2], 3u);
    EXPECT_EQ(csc.col_ptr[3], 4u);
    EXPECT_EQ(csc.col_ptr[4], 6u);
}

TEST(SparseCSC, spmv_allocating) {
    expect_vec_eq(to_csc(make_A()).spmv(kX4), kAX4);
}

TEST(SparseCSC, spmv_buffer_reuse) {
    auto               csc = to_csc(make_A());
    std::vector<double> y(4);
    csc.spmv(kX4, y);
    expect_vec_eq(y, kAX4);
}

TEST(SparseCSC, entries_within_each_column_are_row_sorted) {
    auto csc = to_csc(make_A());
    for (std::size_t j = 0; j < csc.ncols; ++j)
        for (std::size_t k = csc.col_ptr[j] + 1; k < csc.col_ptr[j + 1]; ++k)
            EXPECT_LT(csc.row_idx[k - 1], csc.row_idx[k]);
}

TEST(SparseCSC, roundtrip_to_coo) {
    auto coo2 = to_coo(to_csc(make_A()));
    EXPECT_EQ(coo2.nnz(), 6u);
    expect_vec_eq(coo2.spmv(kX4), kAX4);
}

TEST(SparseCSC, empty_matrix) {
    SparseCOO<double> empty(3, 4);
    empty.compress();
    auto              csc = to_csc(empty);
    EXPECT_EQ(csc.nnz(), 0u);
    expect_vec_eq(csc.spmv({1.0, 2.0, 3.0, 4.0}), {0.0, 0.0, 0.0});
}

// ============================================================
// ELL
// ============================================================

TEST(SparseELL, dimensions) {
    auto ell = to_ell(make_A());
    EXPECT_EQ(ell.nrows, 4u);
    EXPECT_EQ(ell.ncols, 4u);
    EXPECT_EQ(ell.nnz(), 6u);
}

TEST(SparseELL, max_nnz_per_row) {
    // Rows have 2, 1, 2, 1 entries → max = 2
    EXPECT_EQ(to_ell(make_A()).max_nnz_per_row, 2u);
}

TEST(SparseELL, nnz_padded_equals_rows_times_max) {
    auto ell = to_ell(make_A());
    EXPECT_EQ(ell.nnz_padded(), ell.nrows * ell.max_nnz_per_row);
}

TEST(SparseELL, padded_slots_carry_sentinel) {
    auto ell = to_ell(make_A());
    // Row 1 has 1 entry; slot 1 should be padded (no_entry)
    EXPECT_EQ(ell.col_idx[1 * ell.max_nnz_per_row + 1], SparseELL<double>::no_entry);
    // Row 3 has 1 entry; slot 1 should be padded
    EXPECT_EQ(ell.col_idx[3 * ell.max_nnz_per_row + 1], SparseELL<double>::no_entry);
}

TEST(SparseELL, spmv_allocating) {
    expect_vec_eq(to_ell(make_A()).spmv(kX4), kAX4);
}

TEST(SparseELL, spmv_buffer_reuse) {
    auto               ell = to_ell(make_A());
    std::vector<double> y(4);
    ell.spmv(kX4, y);
    expect_vec_eq(y, kAX4);
}

TEST(SparseELL, roundtrip_to_coo) {
    auto coo2 = to_coo(to_ell(make_A()));
    EXPECT_EQ(coo2.nnz(), 6u);
    expect_vec_eq(coo2.spmv(kX4), kAX4);
}

TEST(SparseELL, empty_matrix) {
    SparseCOO<double> empty(2, 3);
    empty.compress();
    auto              ell = to_ell(empty);
    EXPECT_EQ(ell.max_nnz_per_row, 0u);
    EXPECT_EQ(ell.nnz(), 0u);
    expect_vec_eq(ell.spmv({1.0, 2.0, 3.0}), {0.0, 0.0});
}

// ============================================================
// DIA
// ============================================================

TEST(SparseDIA, tridiag_has_three_diagonals) {
    auto dia = to_dia(make_tridiag());
    EXPECT_EQ(dia.num_diags(), 3u);
    ASSERT_EQ(dia.offsets.size(), 3u);
    EXPECT_EQ(dia.offsets[0], -1);
    EXPECT_EQ(dia.offsets[1],  0);
    EXPECT_EQ(dia.offsets[2],  1);
}

TEST(SparseDIA, dimensions) {
    auto dia = to_dia(make_tridiag());
    EXPECT_EQ(dia.nrows, 5u);
    EXPECT_EQ(dia.ncols, 5u);
    EXPECT_EQ(dia.nnz(), 13u);
}

TEST(SparseDIA, spmv_ones_vector) {
    // B*[1,1,1,1,1] = [1, 0, 0, 0, 1]
    auto                      dia = to_dia(make_tridiag());
    const std::vector<double> x   = {1.0, 1.0, 1.0, 1.0, 1.0};
    const std::vector<double> ref  = {1.0, 0.0, 0.0, 0.0, 1.0};
    expect_vec_eq(dia.spmv(x), ref);
}

TEST(SparseDIA, spmv_matches_coo) {
    auto                      coo = make_tridiag();
    auto                      dia = to_dia(coo);
    const std::vector<double> x   = {1.0, 2.0, 3.0, 4.0, 5.0};
    expect_vec_eq(dia.spmv(x), coo.spmv(x));
}

TEST(SparseDIA, spmv_buffer_reuse) {
    auto                      dia = to_dia(make_tridiag());
    const std::vector<double> x   = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double>       y(5);
    dia.spmv(x, y);
    expect_vec_eq(y, make_tridiag().spmv(x));
}

TEST(SparseDIA, roundtrip_to_coo) {
    auto coo   = make_tridiag();
    auto coo2  = to_coo(to_dia(coo));
    EXPECT_EQ(coo2.nnz(), 13u);
    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    expect_vec_eq(coo2.spmv(x), coo.spmv(x));
}

TEST(SparseDIA, general_matrix_roundtrip) {
    // A is not diagonal-structured; DIA still works, just less efficient.
    auto coo  = make_A();
    auto coo2 = to_coo(to_dia(coo));
    EXPECT_EQ(coo2.nnz(), 6u);
    expect_vec_eq(coo2.spmv(kX4), kAX4);
}

// ============================================================
// SparseMatrix
// ============================================================

TEST(SparseMatrix, construct_from_coo) {
    SparseMatrix<double> mat(make_A());
    EXPECT_EQ(mat.format(), SparseMatrix<double>::Format::COO);
    EXPECT_EQ(mat.rows(), 4u);
    EXPECT_EQ(mat.cols(), 4u);
    EXPECT_EQ(mat.nnz(), 6u);
}

TEST(SparseMatrix, construct_from_csr) {
    SparseMatrix<double> mat(to_csr(make_A()));
    EXPECT_EQ(mat.format(), SparseMatrix<double>::Format::CSR);
}

TEST(SparseMatrix, construct_from_csc) {
    SparseMatrix<double> mat(to_csc(make_A()));
    EXPECT_EQ(mat.format(), SparseMatrix<double>::Format::CSC);
}

TEST(SparseMatrix, construct_from_ell) {
    SparseMatrix<double> mat(to_ell(make_A()));
    EXPECT_EQ(mat.format(), SparseMatrix<double>::Format::ELL);
}

TEST(SparseMatrix, construct_from_dia) {
    SparseMatrix<double> mat(to_dia(make_tridiag()));
    EXPECT_EQ(mat.format(), SparseMatrix<double>::Format::DIA);
}

TEST(SparseMatrix, spmv_dispatches_by_format) {
    using F = SparseMatrix<double>::Format;
    for (auto fmt : {F::COO, F::CSR, F::CSC, F::ELL, F::DIA}) {
        SparseMatrix<double> mat(make_A());
        mat.convert_to(fmt);
        expect_vec_eq(mat.spmv(kX4), kAX4);
    }
}

TEST(SparseMatrix, convert_to_is_noop_for_same_format) {
    SparseMatrix<double> mat(to_csr(make_A()));
    ASSERT_EQ(mat.format(), SparseMatrix<double>::Format::CSR);
    mat.convert_to(SparseMatrix<double>::Format::CSR);  // should not crash
    EXPECT_EQ(mat.format(), SparseMatrix<double>::Format::CSR);
    expect_vec_eq(mat.spmv(kX4), kAX4);
}

TEST(SparseMatrix, convert_csr_to_csc_via_coo) {
    SparseMatrix<double> mat(to_csr(make_A()));
    mat.convert_to(SparseMatrix<double>::Format::CSC);
    EXPECT_EQ(mat.format(), SparseMatrix<double>::Format::CSC);
    expect_vec_eq(mat.spmv(kX4), kAX4);
}

TEST(SparseMatrix, typed_accessor_returns_null_for_wrong_format) {
    SparseMatrix<double> mat(make_A());
    mat.convert_to(SparseMatrix<double>::Format::CSR);
    EXPECT_NE(mat.as_csr(), nullptr);
    EXPECT_EQ(mat.as_coo(), nullptr);
    EXPECT_EQ(mat.as_csc(), nullptr);
    EXPECT_EQ(mat.as_ell(), nullptr);
    EXPECT_EQ(mat.as_dia(), nullptr);
}

TEST(SparseMatrix, spmv_buffer_reuse) {
    SparseMatrix<double> mat(make_A());
    mat.convert_to(SparseMatrix<double>::Format::CSR);
    std::vector<double> y(4);
    mat.spmv(kX4, y);
    expect_vec_eq(y, kAX4);
}

TEST(SparseMatrix, repeated_convert_is_stable) {
    SparseMatrix<double> mat(make_A());
    using F = SparseMatrix<double>::Format;
    // Chain several format conversions; result must stay correct throughout.
    for (auto fmt : {F::CSR, F::CSC, F::ELL, F::DIA, F::COO, F::CSR}) {
        mat.convert_to(fmt);
        expect_vec_eq(mat.spmv(kX4), kAX4);
    }
}
