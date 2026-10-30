cmake_minimum_required(VERSION 3.14...3.22)

project(DatamungeTests LANGUAGES C CXX)

# ---- Options ----

option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)
option(TEST_INSTALLED_VERSION "Test the version found by find_package" OFF)

# --- Import tools ----

# include(../cmake/tools.cmake)

# ---- Dependencies ----

include(../cmake/CPM.cmake)

CPMAddPackage(NAME datamunge SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)

# Prefer a preinstalled GTest (offline-friendly), then fall back to fetching.
find_package(GTest QUIET CONFIG)
if(NOT GTest_FOUND)
  find_package(GTest QUIET)
endif()
if(NOT GTest_FOUND)
  CPMAddPackage(
    NAME googletest
    GITHUB_REPOSITORY google/googletest
    VERSION 1.17.0
    OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt ON"
  )
endif()

# if(TEST_INSTALLED_VERSION) find_package(datamunge REQUIRED) else() CPMAddPackage(NAME datamunge
# SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..) endif()

enable_testing()
# ---- Create binary ----

file(GLOB cpp_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cpp)
list(REMOVE_ITEM cpp_sources ${CMAKE_CURRENT_SOURCE_DIR}/cpp/cpp23_format_tests.cpp)
file(GLOB c_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.c)
message(STATUS "c_sources: ${c_sources}")
message(STATUS "cpp_sources: ${cpp_sources}")
add_executable(run_tests ${c_sources} ${cpp_sources})
target_link_libraries(run_tests datamunge::datamunge GTest::gtest_main)
target_include_directories(run_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../examples/nn)
set_target_properties(run_tests PROPERTIES CXX_STANDARD 17)

include(GoogleTest)
gtest_discover_tests(run_tests)

# Focused engine runner keeps simulation iteration independent of the large NN test suite.
add_executable(run_simulation_tests
  cpp/compute_context_tests.cpp cpp/compute_gpu_tests.cpp cpp/sim_world_tests.cpp
  cpp/sim_batch_tests.cpp cpp/sim_tasks_tests.cpp cpp/game_simulation_tests.cpp cpp/game_io_tests.cpp
  cpp/game_courier_tests.cpp cpp/game_render_tests.cpp cpp/sim_rl_adapter_tests.cpp)
target_link_libraries(run_simulation_tests datamunge::datamunge GTest::gtest_main)
target_include_directories(run_simulation_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../examples/nn)
set_target_properties(run_simulation_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_simulation_tests TEST_PREFIX "simulation.")
add_executable(check_simulation_allocations simulation_allocations.cpp)
target_link_libraries(check_simulation_allocations datamunge::datamunge)
set_target_properties(check_simulation_allocations PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
add_test(NAME simulation.allocations COMMAND check_simulation_allocations)

# Device-storage changes also exercise the existing tensor/gradient/memory/I/O contract.
add_executable(run_nn_device_tests
  cpp/nn_device_tests.cpp cpp/nn_tensor_tests.cpp cpp/nn_views_tests.cpp
  cpp/nn_autograd_tests.cpp cpp/nn_ops_tests.cpp cpp/nn_memory_tests.cpp cpp/nn_io_tests.cpp)
target_link_libraries(run_nn_device_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_device_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_device_tests TEST_PREFIX "nn-device.")

# Model diagrams also exercise the existing layer, parameter, derivative and tape contracts.
add_executable(run_nn_model_plot_tests
  cpp/nn_model_plot_tests.cpp cpp/nn_layers_tests.cpp cpp/nn_derivatives_tape_tests.cpp)
target_link_libraries(run_nn_model_plot_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_model_plot_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_model_plot_tests TEST_PREFIX "nn-model-plot.")

# Confusion SVG rendering shares the native training/metric contract.
add_executable(run_nn_metrics_plot_tests cpp/nn_metrics_plot_tests.cpp cpp/nn_training_tests.cpp)
target_link_libraries(run_nn_metrics_plot_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_metrics_plot_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_metrics_plot_tests TEST_PREFIX "nn-metrics-plot.")

# Formatting/printing is a C++20/23 feature; keep the other binding-contract tests on C++17.
# One file per module under cpp/format/ (each module's std::formatter specializations).
file(GLOB cpp23_format_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/format/*.cpp)
add_executable(run_cpp23_tests cpp/cpp23_format_tests.cpp ${cpp23_format_sources})
target_link_libraries(run_cpp23_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_cpp23_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_cpp23_tests)

# ---- code coverage ----
if(ENABLE_TEST_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    foreach(test_target IN ITEMS run_tests run_cpp23_tests)
      target_compile_options(${test_target} PRIVATE -O0 -g --coverage)
      target_link_options(${test_target} PRIVATE --coverage)
    endforeach()

    # Ensure the datamunge library (built via CPM in this build) is also instrumented.
    if(TARGET datamunge)
      target_compile_options(datamunge PRIVATE -O0 -g --coverage)
      target_link_options(datamunge PRIVATE --coverage)
    endif()
  else()
    message(WARNING "ENABLE_TEST_COVERAGE is ON but coverage flags are only wired for Clang/GNU.")
  endif()
endif()

# # enable compiler warnings if(NOT TEST_INSTALLED_VERSION) if(CMAKE_CXX_COMPILER_ID MATCHES "Clang"
# OR CMAKE_CXX_COMPILER_ID MATCHES "GNU") target_compile_options(datamunge PUBLIC -Wall -Wpedantic
# -Wextra -Werror) elseif(MSVC) target_compile_options(datamunge PUBLIC /W4 /WX)
# target_compile_definitions(${PROJECT_NAME} PUBLIC DOCTEST_CONFIG_USE_STD_HEADERS) endif() endif()

# ---- Add DatamungeTests ----

# enable_testing()
#
# # Note: doctest and similar testing frameworks can automatically configure CMake tests. For other
# # testing frameworks add the tests target instead: add_test(NAME ${PROJECT_NAME} COMMAND #
# ${PROJECT_NAME})
#
# include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake) doctest_discover_tests(${PROJECT_NAME})
#
# # ---- code coverage ----
#
# if(ENABLE_TEST_COVERAGE) target_compile_options(datamunge PUBLIC -O0 -g -fprofile-arcs
# -ftest-coverage) target_link_options(datamunge PUBLIC -fprofile-arcs -ftest-coverage) endif()

if(DATAMUNGE_ENABLE_DATA_ASSETS)
  find_package(nlohmann_json 3.10 REQUIRED)
  add_executable(run_dataset_assets_tests dataset_assets.cpp array_io.cpp cpp/nn_io_tests.cpp)
  target_link_libraries(run_dataset_assets_tests PRIVATE datamunge GTest::gtest_main nlohmann_json::nlohmann_json)
  set_target_properties(run_dataset_assets_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
  gtest_discover_tests(run_dataset_assets_tests)
endif()
