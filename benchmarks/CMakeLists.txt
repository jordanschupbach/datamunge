cmake_minimum_required(VERSION 3.14...3.22)

project(DatamungeBenchmarks LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(../cmake/CPM.cmake)

CPMAddPackage(NAME datamunge SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)

# Prefer a preinstalled Google Benchmark; fall back to CPM.
find_package(benchmark QUIET CONFIG)
if(NOT benchmark_FOUND)
  CPMAddPackage(
    NAME benchmark
    GITHUB_REPOSITORY google/benchmark
    GIT_TAG v1.9.4
    OPTIONS
      "BENCHMARK_ENABLE_TESTING OFF"
      "BENCHMARK_ENABLE_GTEST_TESTS OFF"
      "BENCHMARK_INSTALL_DOCS OFF"
  )
endif()

file(GLOB cpp_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cpp)

add_executable(linalg_bench ${cpp_sources})
target_link_libraries(linalg_bench
  datamunge::datamunge
  benchmark::benchmark
  benchmark::benchmark_main
)
set_target_properties(linalg_bench PROPERTIES CXX_STANDARD 17)
# Benchmarks must be built with optimisations; suppress the debug flags that
# the parent preset may inject.
target_compile_options(linalg_bench PRIVATE -O3 -DNDEBUG)
