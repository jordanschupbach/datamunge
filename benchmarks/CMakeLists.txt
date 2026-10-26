cmake_minimum_required(VERSION 3.14...3.22)

project(DatamungeBenchmarks LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(CMAKE_CONFIGURATION_TYPES)
  # Standalone benchmark builds must never select a Debug configuration.
  set(CMAKE_CONFIGURATION_TYPES Release CACHE STRING "Benchmark configurations" FORCE)
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  message(FATAL_ERROR "Benchmarks require Release for the driver and linked core")
endif()

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

# datamunge::nn speed benchmarks (`just bench-nn`).
add_executable(nn_bench ${CMAKE_CURRENT_SOURCE_DIR}/nn/nn_bench.cpp)
target_link_libraries(nn_bench datamunge::datamunge benchmark::benchmark)
set_target_properties(nn_bench PROPERTIES CXX_STANDARD 17)

# Focused, bounded backend cases; the legacy NN epoch remains in nn_bench.
file(GLOB backend_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/backend/*.cpp)
add_executable(backend_bench ${backend_sources})
target_link_libraries(backend_bench PRIVATE datamunge::datamunge benchmark::benchmark ${CMAKE_DL_LIBS})
target_compile_features(backend_bench PRIVATE cxx_std_17)
target_compile_definitions(backend_bench PRIVATE
  DATAMUNGE_BENCH_BUILD="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}/$<CONFIG>")

# Reuse existing correctness tests against exactly the core used by benchmarks.
option(DATAMUNGE_BENCH_CHECKS "Build focused existing backend correctness tests" OFF)
if(DATAMUNGE_BENCH_CHECKS)
  find_package(GTest CONFIG REQUIRED)
  add_executable(backend_checks
    ../tests/cpp/linalg_tensor_tests.cpp
    ../tests/cpp/linalg_sparse_coo_tests.cpp
    ../tests/cpp/linalg_formats_tests.cpp
    ../tests/cpp/dstruct_threadsafe_vector_tests.cpp
    ../tests/cpp/nn_tensor_tests.cpp
    ../tests/cpp/nn_ops_tests.cpp
    ../tests/cpp/nn_autograd_tests.cpp
    ../tests/cpp/nn_linalg_tests.cpp
    ../tests/cpp/nn_fft_tests.cpp
    ../tests/cpp/autodiff_tests.cpp
    ../tests/cpp/optim_tests.cpp
    ../tests/cpp/optim_population_tests.cpp
    ../tests/cpp/optim_multi_objective_tests.cpp
    ../tests/cpp/nn_layers_tests.cpp
    ../tests/cpp/nn_optimizer_fused_tests.cpp
    ../tests/cpp/nn_libgaps_tests.cpp
    ../tests/cpp/nn_sequence_tests.cpp
    ../tests/cpp/nn_attention_tests.cpp
    ../tests/cpp/nn_conv_tests.cpp
    ../tests/cpp/nn_training_tests.cpp
    ../tests/cpp/nn_memory_tests.cpp)
  target_link_libraries(backend_checks PRIVATE datamunge::datamunge GTest::gtest_main)
  target_compile_features(backend_checks PRIVATE cxx_std_17)
endif()
