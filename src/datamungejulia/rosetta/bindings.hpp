#pragma once
// Rosetta input for the datamunge Julia package Datamunge. The exported surface is
// the headers listed under "scope" in policy.json (the same headers the former SWIG
// interface %included); this header adds the binding-only helpers that interface
// declared through %extend and its SWIG-only vector.i view.
#include "datamunge/datamunge.hpp"
#include "datamunge/cad/view.hpp"
#include "datamunge/fda/hierarchical_bspline.hpp"
#include "datamunge/linalg/bindings.hpp"
#include "datamunge/nn/api.hpp"
#include "datamunge/sim/tasks.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace datamunge::julia_bindings {

/// Reseeds the generator shared by unif() and rnorm(), like R's set.seed().
inline void set_seed(std::uint64_t seed) { datamunge::set_seed(seed); }
/// n draws from Uniform[lower, upper).
inline std::vector<double> unif(double lower, double upper, std::size_t n) { return datamunge::unif(lower, upper, n); }
/// Alias for unif(lower, upper, n), sharing its generator and validation.
inline std::vector<double> runif(double lower, double upper, std::size_t n) { return datamunge::runif(lower, upper, n); }
/// n draws from Normal(mean, sd).
inline std::vector<double> rnorm(double mean, double sd, std::size_t n) { return datamunge::rnorm(mean, sd, n); }
/// n evenly spaced values from `from` to `to` inclusive.
inline std::vector<double> seql(double from, double to, std::size_t n) { return datamunge::seql(from, to, n); }

/// Render the expression using native grouping and precision; return raw TeX without math delimiters.
inline std::string expr_to_tex(const datamunge::algebra::Expr& self, int precision = 7) {
  return self.to_tex(precision).str();
}
/// Render descending powers using a raw TeX symbol and 1-17 significant digits; return raw TeX.
inline std::string polynomial_to_tex(const datamunge::algebra::Polynomial& self, const std::string& variable = "x",
                                     int precision = 7) {
  return self.to_tex(variable, precision).str();
}
/// Render the native reduced rational function as raw TeX, omitting a unit denominator.
inline std::string rational_function_to_tex(const datamunge::algebra::RationalFunction& self,
                                            const std::string& variable = "x", int precision = 7) {
  return self.to_tex(variable, precision).str();
}

/// Create an owned native CPU task batch. task is InvertedPendulum or Hopper; count and workers
/// must be positive. seed defaults to zero and time_limit to 1000 successful episode steps.
inline datamunge::sim::TaskBatch* task_batch_create(datamunge::sim::Task task, std::size_t count, std::size_t workers = 1,
                                                    std::uint64_t seed = 0, std::uint64_t time_limit = 1000) {
  return new datamunge::sim::TaskBatch(task, count, datamunge::compute::Context({}, workers), seed, time_limit);
}

inline std::string nn_tensor_repr(const datamunge::nn::NnTensor& self) { return self.str(); }

} // namespace datamunge::julia_bindings
