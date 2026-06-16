#pragma once

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace datamunge::linalg {

template <typename T = double>
class DenseMatrix {
public:
    using value_type = T;

    DenseMatrix() = default;

    DenseMatrix(std::size_t rows, std::size_t cols, T val = T{})
        : rows_(rows), cols_(cols), data_(rows * cols, val) {}

    DenseMatrix(std::size_t rows, std::size_t cols, std::vector<T> data)
        : rows_(rows), cols_(cols), data_(std::move(data)) {
        if (data_.size() != rows_ * cols_)
            throw std::invalid_argument(
                "DenseMatrix: data size does not match rows*cols");
    }

    DenseMatrix(std::size_t rows, std::size_t cols, std::initializer_list<T> il)
        : rows_(rows), cols_(cols), data_(il) {
        if (data_.size() != rows_ * cols_)
            throw std::invalid_argument(
                "DenseMatrix: initializer_list size does not match rows*cols");
    }

    static DenseMatrix zeros(std::size_t rows, std::size_t cols) {
        return DenseMatrix(rows, cols, T{});
    }

    static DenseMatrix identity(std::size_t n) {
        DenseMatrix I(n, n, T{});
        for (std::size_t i = 0; i < n; ++i)
            I(i, i) = T{1};
        return I;
    }

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }

    T& operator()(std::size_t i, std::size_t j) {
        return data_[i * cols_ + j];
    }
    const T& operator()(std::size_t i, std::size_t j) const {
        return data_[i * cols_ + j];
    }

    std::vector<T>&       data() noexcept       { return data_; }
    const std::vector<T>& data() const noexcept { return data_; }

    std::vector<T> row(std::size_t i) const {
        std::vector<T> r(cols_);
        for (std::size_t j = 0; j < cols_; ++j)
            r[j] = (*this)(i, j);
        return r;
    }

    std::vector<T> col(std::size_t j) const {
        std::vector<T> c(rows_);
        for (std::size_t i = 0; i < rows_; ++i)
            c[i] = (*this)(i, j);
        return c;
    }

    // ---- Arithmetic ----

    DenseMatrix operator+(const DenseMatrix& rhs) const {
        check_same_shape(rhs, "operator+");
        DenseMatrix result(rows_, cols_);
        for (std::size_t k = 0; k < data_.size(); ++k)
            result.data_[k] = data_[k] + rhs.data_[k];
        return result;
    }

    DenseMatrix operator-(const DenseMatrix& rhs) const {
        check_same_shape(rhs, "operator-");
        DenseMatrix result(rows_, cols_);
        for (std::size_t k = 0; k < data_.size(); ++k)
            result.data_[k] = data_[k] - rhs.data_[k];
        return result;
    }

    DenseMatrix operator-() const {
        DenseMatrix result(rows_, cols_);
        for (std::size_t k = 0; k < data_.size(); ++k)
            result.data_[k] = -data_[k];
        return result;
    }

    DenseMatrix operator*(T scalar) const {
        DenseMatrix result(rows_, cols_);
        for (std::size_t k = 0; k < data_.size(); ++k)
            result.data_[k] = data_[k] * scalar;
        return result;
    }

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

    DenseMatrix& operator+=(const DenseMatrix& rhs) {
        check_same_shape(rhs, "operator+=");
        for (std::size_t k = 0; k < data_.size(); ++k)
            data_[k] += rhs.data_[k];
        return *this;
    }

    DenseMatrix& operator-=(const DenseMatrix& rhs) {
        check_same_shape(rhs, "operator-=");
        for (std::size_t k = 0; k < data_.size(); ++k)
            data_[k] -= rhs.data_[k];
        return *this;
    }

    DenseMatrix& operator*=(T scalar) {
        for (auto& v : data_) v *= scalar;
        return *this;
    }

    DenseMatrix transpose() const {
        DenseMatrix result(cols_, rows_);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t j = 0; j < cols_; ++j)
                result(j, i) = (*this)(i, j);
        return result;
    }

    // ---- Norms ----

    T norm_frobenius() const {
        T s = T{};
        for (const auto& v : data_) s += v * v;
        return std::sqrt(s);
    }

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

    T norm_max() const {
        T best = T{};
        for (const auto& v : data_) {
            T a = std::abs(v);
            if (a > best) best = a;
        }
        return best;
    }

    T trace() const {
        if (rows_ != cols_)
            throw std::invalid_argument("DenseMatrix::trace: matrix is not square");
        T s = T{};
        for (std::size_t i = 0; i < rows_; ++i) s += (*this)(i, i);
        return s;
    }

    // ---- Solver compatibility ----

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
    std::size_t    rows_{0}, cols_{0};
    std::vector<T> data_;

    void check_same_shape(const DenseMatrix& rhs, const char* op) const {
        if (rows_ != rhs.rows_ || cols_ != rhs.cols_)
            throw std::invalid_argument(
                std::string("DenseMatrix::") + op + ": shape mismatch");
    }
};

template <typename T>
DenseMatrix<T> operator*(T s, const DenseMatrix<T>& m) { return m * s; }

} // namespace datamunge::linalg
