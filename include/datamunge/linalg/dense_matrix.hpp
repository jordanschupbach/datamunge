#pragma once

/// @file dense_matrix.hpp
/// @brief Row-major dense matrix with arithmetic, norms and a printer: the storage type of the
///        dense factorizations (lu(), qr(), cholesky(), svd(), jacobi_eigen()).

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace datamunge::linalg {

/// @brief A dense rows x cols matrix stored row-major in one contiguous `std::vector<T>`.
///
/// Element (i, j) lives at `data()[i * cols() + j]`. Arithmetic operators are plain loops (the
/// product is an i-k-j loop so the inner loop streams along rows); they are meant for the
/// moderate sizes of statistical models, not as a BLAS replacement.
///
/// @tparam T  Element type (usually `double`; `long double` gives extended-precision references).
///
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> A(2, 2, {1, 2, 3, 4});
/// const auto B = A * A.transpose();
/// std::print("{} {} {}\n", B(0, 0), B(0, 1), A.norm_inf());  // 5 11 7
/// @endcode
///
/// @see lu(), qr(), cholesky(), svd()
template <typename T = double>
class DenseMatrix {
public:
    using value_type = T; ///< Element type.

    /// @brief An empty 0 x 0 matrix.
    DenseMatrix() = default;

    /// @brief A rows x cols matrix with every element equal to @p val.
    /// @param rows  Number of rows.
    /// @param cols  Number of columns.
    /// @param val   Fill value (default zero).
    DenseMatrix(std::size_t rows, std::size_t cols, T val = T{})
        : rows_(rows), cols_(cols), data_(rows * cols, val) {}

    /// @brief A rows x cols matrix taking its elements, row by row, from @p data.
    /// @param rows  Number of rows.
    /// @param cols  Number of columns.
    /// @param data  rows*cols values in row-major order.
    /// @throws std::invalid_argument  If `data.size() != rows * cols`.
    DenseMatrix(std::size_t rows, std::size_t cols, std::vector<T> data)
        : rows_(rows), cols_(cols), data_(std::move(data)) {
        if (data_.size() != rows_ * cols_)
            throw std::invalid_argument(
                "DenseMatrix: data size does not match rows*cols");
    }

    /// @brief A rows x cols matrix from a braced list of values in row-major order.
    /// @param rows  Number of rows.
    /// @param cols  Number of columns.
    /// @param il    rows*cols values.
    /// @throws std::invalid_argument  If the list has the wrong length.
    DenseMatrix(std::size_t rows, std::size_t cols, std::initializer_list<T> il)
        : rows_(rows), cols_(cols), data_(il) {
        if (data_.size() != rows_ * cols_)
            throw std::invalid_argument(
                "DenseMatrix: initializer_list size does not match rows*cols");
    }

    /// @brief The rows x cols zero matrix.
    /// @param rows  Number of rows.
    /// @param cols  Number of columns.
    /// @return A matrix of zeros.
    static DenseMatrix zeros(std::size_t rows, std::size_t cols) {
        return DenseMatrix(rows, cols, T{});
    }

    /// @brief The n x n identity matrix.
    /// @param n  Order.
    /// @return \f$I_n\f$.
    static DenseMatrix identity(std::size_t n) {
        DenseMatrix I(n, n, T{});
        for (std::size_t i = 0; i < n; ++i)
            I(i, i) = T{1};
        return I;
    }

    /// @brief A matrix whose columns are the given vectors, in order.
    /// @param columns  Equal-length columns (at least one).
    /// @return A `columns[0].size()` x `columns.size()` matrix.
    /// @throws std::invalid_argument  If @p columns is empty or the lengths differ.
    ///
    /// @par Example
    /// @code
    /// const auto A = dm::linalg::DenseMatrix<double>::from_columns({{1.0, 2.0}, {3.0, 4.0}});
    /// std::print("{} {}\n", A(0, 1), A(1, 0));  // 3 2
    /// @endcode
    static DenseMatrix from_columns(const std::vector<std::vector<T>>& columns) {
        if (columns.empty()) throw std::invalid_argument("DenseMatrix::from_columns: no columns");
        DenseMatrix A(columns[0].size(), columns.size(), T{});
        for (std::size_t j = 0; j < columns.size(); ++j) {
            if (columns[j].size() != A.rows_)
                throw std::invalid_argument("DenseMatrix::from_columns: columns have different lengths");
            for (std::size_t i = 0; i < A.rows_; ++i) A(i, j) = columns[j][i];
        }
        return A;
    }

    /// @brief Number of rows.
    /// @return The row count.
    std::size_t rows() const noexcept { return rows_; }
    /// @brief Number of columns.
    /// @return The column count.
    std::size_t cols() const noexcept { return cols_; }

    /// @brief Element (i, j), unchecked.
    /// @param i  Row index.
    /// @param j  Column index.
    /// @return A reference to the element.
    T& operator()(std::size_t i, std::size_t j) {
        return data_[i * cols_ + j];
    }
    /// @brief Element (i, j), unchecked (read-only).
    /// @param i  Row index.
    /// @param j  Column index.
    /// @return A const reference to the element.
    const T& operator()(std::size_t i, std::size_t j) const {
        return data_[i * cols_ + j];
    }

    /// @brief The row-major element storage.
    /// @return The underlying vector.
    std::vector<T>&       data() noexcept       { return data_; }
    /// @brief The row-major element storage (read-only).
    /// @return The underlying vector.
    const std::vector<T>& data() const noexcept { return data_; }

    /// @brief A copy of row i.
    /// @param i  Row index.
    /// @return The cols() values of the row.
    std::vector<T> row(std::size_t i) const {
        std::vector<T> r(cols_);
        for (std::size_t j = 0; j < cols_; ++j)
            r[j] = (*this)(i, j);
        return r;
    }

    /// @brief A copy of column j.
    /// @param j  Column index.
    /// @return The rows() values of the column.
    std::vector<T> col(std::size_t j) const {
        std::vector<T> c(rows_);
        for (std::size_t i = 0; i < rows_; ++i)
            c[i] = (*this)(i, j);
        return c;
    }

    // ---- Arithmetic ----

    /// @brief Elementwise sum.
    /// @param rhs  Matrix of the same shape.
    /// @return `*this + rhs`.
    /// @throws std::invalid_argument  On a shape mismatch.
    DenseMatrix operator+(const DenseMatrix& rhs) const {
        check_same_shape(rhs, "operator+");
        DenseMatrix result(rows_, cols_);
        for (std::size_t k = 0; k < data_.size(); ++k)
            result.data_[k] = data_[k] + rhs.data_[k];
        return result;
    }

    /// @brief Elementwise difference.
    /// @param rhs  Matrix of the same shape.
    /// @return `*this - rhs`.
    /// @throws std::invalid_argument  On a shape mismatch.
    DenseMatrix operator-(const DenseMatrix& rhs) const {
        check_same_shape(rhs, "operator-");
        DenseMatrix result(rows_, cols_);
        for (std::size_t k = 0; k < data_.size(); ++k)
            result.data_[k] = data_[k] - rhs.data_[k];
        return result;
    }

    /// @brief Elementwise negation.
    /// @return `-*this`.
    DenseMatrix operator-() const {
        DenseMatrix result(rows_, cols_);
        for (std::size_t k = 0; k < data_.size(); ++k)
            result.data_[k] = -data_[k];
        return result;
    }

    /// @brief Multiplication by a scalar.
    /// @param scalar  Factor.
    /// @return `scalar * *this`.
    DenseMatrix operator*(T scalar) const {
        DenseMatrix result(rows_, cols_);
        for (std::size_t k = 0; k < data_.size(); ++k)
            result.data_[k] = data_[k] * scalar;
        return result;
    }

    /// @brief Matrix product (\f$2\,\text{rows}\cdot\text{cols}\cdot\text{rhs.cols}\f$ flops).
    /// @param rhs  Matrix with `rhs.rows() == cols()`.
    /// @return The rows() x rhs.cols() product.
    /// @throws std::invalid_argument  If the inner dimensions differ.
    DenseMatrix operator*(const DenseMatrix& rhs) const {
        if (cols_ != rhs.rows_)
            throw std::invalid_argument(
                "DenseMatrix::operator*: cols != rhs.rows");
        DenseMatrix result(rows_, rhs.cols_, T{});
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t k = 0; k < cols_; ++k)
                for (std::size_t j = 0; j < rhs.cols_; ++j)
                    result(i, j) += (*this)(i, k) * rhs(k, j);
        return result;
    }

    /// @brief Matrix-vector product.
    /// @param x  Vector of length cols().
    /// @return \f$Ax\f$, of length rows().
    /// @throws std::invalid_argument  If `x.size() != cols()`.
    std::vector<T> operator*(const std::vector<T>& x) const {
        if (x.size() != cols_)
            throw std::invalid_argument(
                "DenseMatrix::operator*(vector): x.size() != cols");
        std::vector<T> y(rows_, T{});
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t j = 0; j < cols_; ++j)
                y[i] += (*this)(i, j) * x[j];
        return y;
    }

    /// @brief In-place elementwise sum.
    /// @param rhs  Matrix of the same shape.
    /// @return `*this`.
    /// @throws std::invalid_argument  On a shape mismatch.
    DenseMatrix& operator+=(const DenseMatrix& rhs) {
        check_same_shape(rhs, "operator+=");
        for (std::size_t k = 0; k < data_.size(); ++k)
            data_[k] += rhs.data_[k];
        return *this;
    }

    /// @brief In-place elementwise difference.
    /// @param rhs  Matrix of the same shape.
    /// @return `*this`.
    /// @throws std::invalid_argument  On a shape mismatch.
    DenseMatrix& operator-=(const DenseMatrix& rhs) {
        check_same_shape(rhs, "operator-=");
        for (std::size_t k = 0; k < data_.size(); ++k)
            data_[k] -= rhs.data_[k];
        return *this;
    }

    /// @brief In-place scaling.
    /// @param scalar  Factor.
    /// @return `*this`.
    DenseMatrix& operator*=(T scalar) {
        for (auto& v : data_) v *= scalar;
        return *this;
    }

    /// @brief The transpose.
    /// @return The cols() x rows() matrix \f$A^\top\f$.
    DenseMatrix transpose() const {
        DenseMatrix result(cols_, rows_);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t j = 0; j < cols_; ++j)
                result(j, i) = (*this)(i, j);
        return result;
    }

    // ---- Norms ----

    /// @brief Frobenius norm \f$\lVert A\rVert_F=(\sum_{ij}a_{ij}^2)^{1/2}\f$; satisfies
    ///        \f$\lVert A\rVert_2\le\lVert A\rVert_F\le\sqrt{r}\,\lVert A\rVert_2\f$ for rank r.
    /// @return The norm.
    T norm_frobenius() const {
        T s = T{};
        for (const auto& v : data_) s += v * v;
        return std::sqrt(s);
    }

    /// @brief Induced 1-norm: the largest column sum of absolute values.
    /// @return \f$\max_j\sum_i|a_{ij}|\f$.
    T norm_1() const {
        // max column sum of absolute values
        T best = T{};
        for (std::size_t j = 0; j < cols_; ++j) {
            T s = T{};
            for (std::size_t i = 0; i < rows_; ++i)
                s += std::abs((*this)(i, j));
            if (s > best) best = s;
        }
        return best;
    }

    /// @brief Induced infinity-norm: the largest row sum of absolute values.
    /// @return \f$\max_i\sum_j|a_{ij}|\f$.
    T norm_inf() const {
        // max row sum of absolute values
        T best = T{};
        for (std::size_t i = 0; i < rows_; ++i) {
            T s = T{};
            for (std::size_t j = 0; j < cols_; ++j)
                s += std::abs((*this)(i, j));
            if (s > best) best = s;
        }
        return best;
    }

    /// @brief Largest absolute element (not an induced norm).
    /// @return \f$\max_{ij}|a_{ij}|\f$.
    T norm_max() const {
        T best = T{};
        for (const auto& v : data_) {
            T a = std::abs(v);
            if (a > best) best = a;
        }
        return best;
    }

    /// @brief Sum of the diagonal.
    /// @return \f$\sum_i a_{ii}\f$.
    /// @throws std::invalid_argument  If the matrix is not square.
    T trace() const {
        if (rows_ != cols_)
            throw std::invalid_argument("DenseMatrix::trace: matrix is not square");
        T s = T{};
        for (std::size_t i = 0; i < rows_; ++i) s += (*this)(i, i);
        return s;
    }

    // ---- Solver compatibility ----

    /// @brief \f$y\leftarrow Ax\f$ into a preallocated vector, the interface the iterative solvers expect.
    /// @param x  Input of length cols().
    /// @param y  Output of length rows() (overwritten).
    /// @throws std::invalid_argument  On a size mismatch.
    void spmv(const std::vector<T>& x, std::vector<T>& y) const {
        if (x.size() != cols_)
            throw std::invalid_argument("DenseMatrix::spmv: x.size() != cols");
        if (y.size() != rows_)
            throw std::invalid_argument("DenseMatrix::spmv: y.size() != rows");
        for (std::size_t i = 0; i < rows_; ++i) {
            T s = T{};
            for (std::size_t j = 0; j < cols_; ++j)
                s += (*this)(i, j) * x[j];
            y[i] = s;
        }
    }

private:
    std::size_t    rows_{0}; ///< Number of rows.
    std::size_t    cols_{0}; ///< Number of columns.
    std::vector<T> data_;    ///< Row-major elements.

    /// @brief Throws std::invalid_argument naming @p op unless rhs has this matrix's shape.
    /// @param rhs  Other operand.
    /// @param op   Operator name for the message.
    void check_same_shape(const DenseMatrix& rhs, const char* op) const {
        if (rows_ != rhs.rows_ || cols_ != rhs.cols_)
            throw std::invalid_argument(
                std::string("DenseMatrix::") + op + ": shape mismatch");
    }
};

/// @brief Scalar-times-matrix, `s * m`.
/// @param s  Factor.
/// @param m  Matrix.
/// @return `m * s`.
template <typename T>
DenseMatrix<T> operator*(T s, const DenseMatrix<T>& m) { return m * s; }

/// @brief Prints one row per line, columns right-aligned and separated by a space, with no
///        trailing newline (Eigen's layout).
///
/// Each element is formatted with the stream's own flags and precision, so
/// `std::fixed << std::setprecision(3)` applies; a width set on the stream is consumed, not
/// applied per element.
/// @param os  Output stream.
/// @param m   Matrix to print.
/// @return @p os.
template <typename T>
std::ostream& operator<<(std::ostream& os, const DenseMatrix<T>& m) {
    std::vector<std::string> cells(m.rows() * m.cols());
    std::vector<std::size_t> width(m.cols(), 0);
    std::ostringstream cell;
    cell.flags(os.flags());
    cell.precision(os.precision());
    cell.imbue(os.getloc());
    for (std::size_t i = 0; i < m.rows(); ++i)
        for (std::size_t j = 0; j < m.cols(); ++j) {
            cell.str("");
            cell << m(i, j);
            cells[i * m.cols() + j] = cell.str();
            width[j] = std::max(width[j], cells[i * m.cols() + j].size());
        }
    os.width(0);
    for (std::size_t i = 0; i < m.rows(); ++i) {
        if (i > 0) os << '\n';
        for (std::size_t j = 0; j < m.cols(); ++j) {
            const std::string& s = cells[i * m.cols() + j];
            if (j > 0) os << ' ';
            os << std::string(width[j] - s.size(), ' ') << s;
        }
    }
    return os;
}

} // namespace datamunge::linalg
