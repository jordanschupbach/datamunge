#include <datamunge/random/random.hpp>
#include <datamunge/vector.hpp>

#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>

namespace datamunge {

namespace {

void check_same_size(const Vector& lhs, const Vector& rhs, const char* op) {
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument(std::string("Vector ") + op + ": length mismatch (" +
                                std::to_string(lhs.size()) + " vs " + std::to_string(rhs.size()) +
                                ")");
  }
}

template <typename F>
Vector apply(Vector v, F f) {
  for (double& x : v) x = f(x);
  return v;
}

std::uint64_t random_seed() {
  std::random_device rd;
  return (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
}

random::SplitMix64& generator() {
  static random::SplitMix64 rng(random_seed());
  return rng;
}

} // namespace

Vector& Vector::operator+=(const Vector& rhs) {
  check_same_size(*this, rhs, "+");
  for (std::size_t i = 0; i < size(); ++i) data_[i] += rhs[i];
  return *this;
}

Vector& Vector::operator-=(const Vector& rhs) {
  check_same_size(*this, rhs, "-");
  for (std::size_t i = 0; i < size(); ++i) data_[i] -= rhs[i];
  return *this;
}

Vector& Vector::operator*=(const Vector& rhs) {
  check_same_size(*this, rhs, "*");
  for (std::size_t i = 0; i < size(); ++i) data_[i] *= rhs[i];
  return *this;
}

Vector& Vector::operator/=(const Vector& rhs) {
  check_same_size(*this, rhs, "/");
  for (std::size_t i = 0; i < size(); ++i) data_[i] /= rhs[i];
  return *this;
}

Vector& Vector::operator+=(double rhs) {
  for (double& x : data_) x += rhs;
  return *this;
}

Vector& Vector::operator-=(double rhs) {
  for (double& x : data_) x -= rhs;
  return *this;
}

Vector& Vector::operator*=(double rhs) {
  for (double& x : data_) x *= rhs;
  return *this;
}

Vector& Vector::operator/=(double rhs) {
  for (double& x : data_) x /= rhs;
  return *this;
}

Vector operator-(Vector v) { return apply(std::move(v), [](double x) { return -x; }); }

Vector operator+(Vector lhs, const Vector& rhs) { return lhs += rhs; }
Vector operator-(Vector lhs, const Vector& rhs) { return lhs -= rhs; }
Vector operator*(Vector lhs, const Vector& rhs) { return lhs *= rhs; }
Vector operator/(Vector lhs, const Vector& rhs) { return lhs /= rhs; }

Vector operator+(Vector lhs, double rhs) { return lhs += rhs; }
Vector operator-(Vector lhs, double rhs) { return lhs -= rhs; }
Vector operator*(Vector lhs, double rhs) { return lhs *= rhs; }
Vector operator/(Vector lhs, double rhs) { return lhs /= rhs; }

Vector operator+(double lhs, Vector rhs) { return rhs += lhs; }
Vector operator-(double lhs, Vector rhs) {
  return apply(std::move(rhs), [lhs](double x) { return lhs - x; });
}
Vector operator*(double lhs, Vector rhs) { return rhs *= lhs; }
Vector operator/(double lhs, Vector rhs) {
  return apply(std::move(rhs), [lhs](double x) { return lhs / x; });
}

std::ostream& operator<<(std::ostream& out, const Vector& v) {
  out << '[';
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i > 0) out << ", ";
    out << v[i];
  }
  return out << ']';
}

Vector sin(Vector v) { return apply(std::move(v), [](double x) { return std::sin(x); }); }
Vector cos(Vector v) { return apply(std::move(v), [](double x) { return std::cos(x); }); }
Vector tan(Vector v) { return apply(std::move(v), [](double x) { return std::tan(x); }); }
Vector exp(Vector v) { return apply(std::move(v), [](double x) { return std::exp(x); }); }
Vector log(Vector v) { return apply(std::move(v), [](double x) { return std::log(x); }); }
Vector sqrt(Vector v) { return apply(std::move(v), [](double x) { return std::sqrt(x); }); }
Vector abs(Vector v) { return apply(std::move(v), [](double x) { return std::abs(x); }); }
Vector pow(Vector v, double exponent) {
  return apply(std::move(v), [exponent](double x) { return std::pow(x, exponent); });
}

Vector seql(double from, double to, std::size_t n) {
  Vector v(n);
  if (n == 1) {
    v[0] = from;
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      v[i] = from + (to - from) * static_cast<double>(i) / static_cast<double>(n - 1);
    }
  }
  return v;
}

double sum(const Vector& v) { return std::accumulate(v.begin(), v.end(), 0.0); }

double mean(const Vector& v) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  return sum(v) / static_cast<double>(v.size());
}

void set_seed(std::uint64_t seed) { generator() = random::SplitMix64(seed); }

Vector unif(double lower, double upper, std::size_t n) {
  Vector v(n);
  for (double& x : v) x = generator().uniform(lower, upper);
  return v;
}

Vector rnorm(double mean, double sd, std::size_t n) {
  Vector v(n);
  for (double& x : v) x = generator().normal(mean, sd);
  return v;
}

} // namespace datamunge
