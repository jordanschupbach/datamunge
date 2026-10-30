#pragma once

/// @file format.hpp
/// @brief C++20 std::format and C++23 std::print support for the library's types.
///
/// @code
/// auto model = dm::LM(dm::datasets::iris(), "Sepal.Length ~ Petal.Length");
/// std::print("{}\n", model);   // the model summary, as print_summary() shows it
/// @endcode
///
/// Requires a standard library implementing <format>. The dm umbrella includes
/// this header automatically when that feature is available; C++17 callers can
/// keep using each type's own text methods (to_string(), summary(), info()).
/// Each module's formatters live in <datamunge/format/<module>.hpp>.
/// Ranges such as dm::Vector already print through the standard range formatter.

#include <datamunge/format/algebra.hpp>
#include <datamunge/format/algorithms.hpp>
#include <datamunge/format/autodiff.hpp>
#include <datamunge/format/bayes.hpp>
#include <datamunge/format/cv.hpp>
#include <datamunge/format/dstruct.hpp>
#include <datamunge/format/fda.hpp>
#include <datamunge/format/fem.hpp>
#include <datamunge/format/filter.hpp>
#include <datamunge/format/geometry.hpp>
#include <datamunge/format/gis.hpp>
#include <datamunge/format/image.hpp>
#include <datamunge/format/linalg.hpp>
#include <datamunge/format/nn.hpp>
#include <datamunge/format/ode.hpp>
#include <datamunge/format/optim.hpp>
#include <datamunge/format/plot.hpp>
#include <datamunge/format/random.hpp>
#include <datamunge/format/stats.hpp>
#include <datamunge/format/facade.hpp>
