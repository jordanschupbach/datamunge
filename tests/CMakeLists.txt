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
# Ported windowless CAD regression suites retain their independent numerical,
# topology, round-trip and constraint-solving oracles.
file(GLOB cad_regression_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/cad/cad_*_test.cpp")
foreach(cad_test_source IN LISTS cad_regression_sources)
  get_filename_component(cad_test_name "${cad_test_source}" NAME_WE)
  add_executable("run_${cad_test_name}" "${cad_test_source}")
  target_include_directories("run_${cad_test_name}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src/datamunge/cad")
  target_compile_definitions("run_${cad_test_name}" PRIVATE DATAMUNGE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")
  target_link_libraries("run_${cad_test_name}" datamunge::datamunge)
  set_target_properties("run_${cad_test_name}" PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
  add_test(NAME "cad.${cad_test_name}" COMMAND "run_${cad_test_name}")
endforeach()
# Algebra is header-only: this focused runner needs no native shared-library rebuild.
add_executable(run_algebra_tex_tests cpp/algebra_tex_tests.cpp cpp/algebra_expression_tests.cpp
  cpp/algebra_calculus_tests.cpp cpp/algebra_polynomial_core_tests.cpp
  cpp/algebra_roots_and_factorization_tests.cpp cpp/algebra_multivariate_groebner_tests.cpp)
target_include_directories(run_algebra_tex_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(run_algebra_tex_tests GTest::gtest_main)
set_target_properties(run_algebra_tex_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
include(GoogleTest)
gtest_discover_tests(run_algebra_tex_tests TEST_PREFIX "algebra-tex.")
add_executable(run_tex_tests cpp/tex_tests.cpp)
target_link_libraries(run_tex_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_tex_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
include(GoogleTest)
gtest_discover_tests(run_tex_tests TEST_PREFIX "tex.")
# Native concept diagrams have an offline runner independent of dataset/model fitting.
add_executable(run_diagram_tests cpp/diagram_tests.cpp)
target_link_libraries(run_diagram_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_diagram_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
include(GoogleTest)
gtest_discover_tests(run_diagram_tests TEST_PREFIX "diagram.")
add_executable(run_filesystem_tests cpp/filesystem_tests.cpp)
target_link_libraries(run_filesystem_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_filesystem_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_filesystem_tests TEST_PREFIX "filesystem.")
add_executable(run_plot_color_tests cpp/plot_tests.cpp cpp/plot_compose_tests.cpp cpp/ggplot_tests.cpp cpp/dm_api_tests.cpp)
target_link_libraries(run_plot_color_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_plot_color_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_plot_color_tests TEST_PREFIX "plot-colors.")
add_executable(run_vector_expression_tests cpp/vector_expression_tests.cpp cpp/vector_tests.cpp)
target_link_libraries(run_vector_expression_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_vector_expression_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_vector_expression_tests TEST_PREFIX "vector-expression.")
add_executable(check_vector_expression_allocations vector_expression_allocations.cpp)
target_link_libraries(check_vector_expression_allocations datamunge::datamunge)
set_target_properties(check_vector_expression_allocations PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
add_test(NAME vector-expression.allocations COMMAND check_vector_expression_allocations)
# The facade's expression and solver contract plus existing canonical dense/sparse regressions.
add_executable(run_matrix_api_tests cpp/matrix_api_tests.cpp cpp/linalg_dense_tests.cpp
  cpp/linalg_sparse_coo_tests.cpp cpp/linalg_formats_tests.cpp cpp/linalg_solvers_tests.cpp
  cpp/linalg_svd_least_squares_tests.cpp cpp/format/linalg_format_tests.cpp)
target_link_libraries(run_matrix_api_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_matrix_api_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_matrix_api_tests TEST_PREFIX "matrix-api.")
add_executable(run_adaptive_pspline_tests cpp/fda_adaptive_pspline_tests.cpp)
target_link_libraries(run_adaptive_pspline_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_adaptive_pspline_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_adaptive_pspline_tests TEST_PREFIX "adaptive-pspline.")
add_executable(run_bspline_design_tests cpp/bspline_design_tests.cpp cpp/fda_bspline_tests.cpp cpp/stats_lm_tests.cpp)
target_link_libraries(run_bspline_design_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_bspline_design_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_bspline_design_tests TEST_PREFIX "bspline-design.")
add_executable(check_matrix_allocations matrix_allocations.cpp)
target_link_libraries(check_matrix_allocations datamunge::datamunge)
set_target_properties(check_matrix_allocations PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
add_test(NAME matrix-api.allocations COMMAND check_matrix_allocations)
add_executable(check_sparse_lm_allocations sparse_lm_allocations.cpp)
target_link_libraries(check_sparse_lm_allocations datamunge::datamunge)
set_target_properties(check_sparse_lm_allocations PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
add_test(NAME bspline-design.sparse-allocations COMMAND check_sparse_lm_allocations)
# ---- Create binary ----

file(GLOB cpp_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cpp)
list(REMOVE_ITEM cpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/algebra_tex_tests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tex_tests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/cpp23_format_tests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/filesystem_tests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/vector_expression_tests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/bspline_design_tests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/matrix_api_tests.cpp)
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
  cpp/rl_discrete_tests.cpp cpp/sim_articulated_tests.cpp cpp/sim_native_dynamics_tests.cpp
  cpp/compute_context_tests.cpp cpp/compute_gpu_tests.cpp cpp/sim_world_tests.cpp cpp/sim_conveyor_tests.cpp cpp/sim_devarm_pick_tests.cpp cpp/sim_devarm_camera_tests.cpp cpp/sim_fruit_stream_tests.cpp
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

# Proper scores and native CIFAR study helpers have small offline analytic fixtures.
add_executable(run_cifar10_study_tests cpp/nn_classification_analysis_tests.cpp)
target_link_libraries(run_cifar10_study_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_cifar10_study_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_cifar10_study_tests TEST_PREFIX "cifar10-study.")

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

# Small native MobileNet fitting/checkpoint contract, independent of the large NN suite.
add_executable(run_mobile_classifier_tests cpp/nn_mobile_classifier_tests.cpp)
target_link_libraries(run_mobile_classifier_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mobile_classifier_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mobile_classifier_tests TEST_PREFIX "mobile-classifier.")
