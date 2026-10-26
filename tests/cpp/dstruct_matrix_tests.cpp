#include <gtest/gtest.h>
#include <datamunge/dstruct/dstruct.hpp>
using namespace datamunge::dstruct;
TEST(MatrixStructures, DenseAndSparseMatrix) { Matrix<int> dense(2, 3); dense.at(0, 2) = 7; EXPECT_EQ(dense.transpose().at(2, 0), 7); SparseMatrix<int> sparse(3, 3); sparse.set(1, 2, 5); EXPECT_EQ(sparse.at(1, 2), 5); EXPECT_EQ(sparse.at(0, 0), 0); sparse.set(1, 2, 0); EXPECT_EQ(sparse.nonzero_count(), 0U); }
TEST(MatrixStructures, ParallelArrayKeepsColumnsAligned) { ParallelArray<int, int> values; values.push_back(2, 4); values.push_back(3, 9); EXPECT_EQ(values.key_at(1), 3); EXPECT_EQ(values.value_at(1), 9); }
