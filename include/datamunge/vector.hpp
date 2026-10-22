#pragma once

/// @file vector.hpp
/// @brief R-style numeric vectors: element-wise arithmetic, vectorised math, and random draws.
///
/// @code
/// namespace dm = datamunge;
/// dm::Vector x = dm::unif(0, 1, 100);
/// dm::Vector y = dm::sin(20 / (x + 0.25)) + dm::rnorm(0, 1, 100);
/// dm::Plot().points(x, y).to_svg();
/// @endcode

#include <datamunge/plot/plot.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iosfwd>
#include <utility>
#include <vector>

namespace datamunge {

/// @brief A numeric vector with R-like element-wise semantics.
///
/// Binary operators between two vectors require equal lengths (std::invalid_argument otherwise);
/// operators between a vector and a scalar apply the scalar to every element. Converts implicitly
/// to `const std::vector<double>&`, so it can be passed anywhere the library takes one.
class Vector {
 public:
  Vector() = default;
  /// @brief A vector of @p n copies of @p value.
  explicit Vector(std::size_t n, double value = 0.0) : data_(n, value) {}
  Vector(std::initializer_list<double> values) : data_(values) {}
  Vector(std::vector<double> values) : data_(std::move(values)) {}

  std::size_t size() const noexcept { return data_.size(); }
  bool        empty() const noexcept { return data_.empty(); }

  double&       operator[](std::size_t i) { return data_[i]; }
  double        operator[](std::size_t i) const { return data_[i]; }

  auto begin() noexcept { return data_.begin(); }
  auto end() noexcept { return data_.end(); }
  auto begin() const noexcept { return data_.begin(); }
  auto end() const noexcept { return data_.end(); }

  const std::vector<double>& values() const noexcept { return data_; }
  operator const std::vector<double>&() const noexcept { return data_; }

  Vector& operator+=(const Vector& rhs);
  Vector& operator-=(const Vector& rhs);
  Vector& operator*=(const Vector& rhs);
  Vector& operator/=(const Vector& rhs);
  Vector& operator+=(double rhs);
  Vector& operator-=(double rhs);
  Vector& operator*=(double rhs);
  Vector& operator/=(double rhs);

 private:
  std::vector<double> data_;
};

Vector operator-(Vector v);

Vector operator+(Vector lhs, const Vector& rhs);
Vector operator-(Vector lhs, const Vector& rhs);
Vector operator*(Vector lhs, const Vector& rhs);
Vector operator/(Vector lhs, const Vector& rhs);

Vector operator+(Vector lhs, double rhs);
Vector operator-(Vector lhs, double rhs);
Vector operator*(Vector lhs, double rhs);
Vector operator/(Vector lhs, double rhs);

Vector operator+(double lhs, Vector rhs);
Vector operator-(double lhs, Vector rhs);
Vector operator*(double lhs, Vector rhs);
Vector operator/(double lhs, Vector rhs);

/// @brief Prints the vector as `[a, b, c]`.
std::ostream& operator<<(std::ostream& out, const Vector& v);

// Keep the scalar overloads visible alongside the Vector ones, so unqualified calls such as
// `sin(0.5)` inside namespace datamunge still resolve to the <cmath> functions.
using std::abs;
using std::cos;
using std::exp;
using std::log;
using std::pow;
using std::sin;
using std::sqrt;
using std::tan;

/// @name Element-wise math
/// @{
Vector sin(Vector v);
Vector cos(Vector v);
Vector tan(Vector v);
Vector exp(Vector v);
Vector log(Vector v);
Vector sqrt(Vector v);
Vector abs(Vector v);
Vector pow(Vector v, double exponent);
/// @}

/// @brief Evenly spaced values from @p from to @p to inclusive, like R's `seq(from, to, length.out = n)`; `n == 1` gives `{from}`.
Vector seql(double from, double to, std::size_t n);

/// @brief Sum of the elements.
double sum(const Vector& v);
/// @brief Arithmetic mean of the elements (NaN for an empty vector).
double mean(const Vector& v);

/// @brief Reseeds the generator shared by unif() and rnorm(), like R's `set.seed()`.
/// Until this is called, the generator is seeded from std::random_device.
void set_seed(std::uint64_t seed);

/// @brief @p n draws from Uniform[@p lower, @p upper), like R's `runif(n, lower, upper)`.
Vector unif(double lower, double upper, std::size_t n);

/// @brief @p n draws from Normal(@p mean, @p sd), like R's `rnorm(n, mean, sd)`.
Vector rnorm(double mean, double sd, std::size_t n);

/// @brief The R-base-graphics-style plot: `dm::Plot().points(x, y).to_svg()`.
using Plot = plot::RPlot;
using RGB = plot::RGB;
using RGBA = plot::RGBA;

} // namespace datamunge
