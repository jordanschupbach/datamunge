#pragma once

/// @file dm.hpp
/// @brief Single C++ entry point for the library's public modules under the dm namespace.
/// Includes the high-level API, bundled datasets, algorithms, linear algebra,
/// neural networks, vectors, and plot composition, plus the standard headers most analyses use
/// (`<iostream>`, `<string>`, `<vector>`, `<filesystem>`, `<cmath>`, `<format>`, and
/// `<print>` under C++23).
/// Link the usual datamunge library; dm names refer to the same types and functions as
/// datamunge names. Works with C++20; the std::format formatters for library types are added
/// automatically under C++23.

#include <datamunge/datamunge.hpp>
#include <datamunge/algorithms/algorithms.hpp>
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/info.hpp>
#include <datamunge/dir.hpp>
#include <datamunge/pipe.hpp>
#include <datamunge/linalg/linalg.hpp>
#include <datamunge/nn/nn.hpp>
#include <datamunge/compute/context.hpp>
#include <datamunge/sim/world.hpp>
#include <datamunge/sim/batch.hpp>
#include <datamunge/sim/tasks.hpp>
#include <datamunge/game/clock.hpp>
#include <datamunge/game/simulation.hpp>
#include <datamunge/game/io.hpp>
#include <datamunge/game/courier.hpp>
#include <datamunge/game/render.hpp>

// Standard headers an analysis script almost always needs, so `#include <dm/dm.hpp>` is the only
// include a short program or report block has to write.
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#if __has_include(<format>)
#include <format>
#endif
#if __has_include(<print>) && __cplusplus > 202002L
#include <print>
#endif

// The std::formatter specializations need C++23 (std::formattable and range formatting), so
// they are only pulled in when the standard library provides those; C++20 builds still get
// everything else.
#if __has_include(<version>)
#include <version>
#endif
#if defined(__cpp_lib_format_ranges)
#include <datamunge/format.hpp>
#endif

/// @brief Short public namespace spelling for the canonical library API.
namespace dm = datamunge;
