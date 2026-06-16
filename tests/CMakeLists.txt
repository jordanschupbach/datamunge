cmake_minimum_required(VERSION 3.14...3.22)

project(OctraTests LANGUAGES C CXX)

# ---- Options ----

option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)
option(TEST_INSTALLED_VERSION "Test the version found by find_package" OFF)

# --- Import tools ----

# include(../cmake/tools.cmake)

# ---- Dependencies ----

include(../cmake/CPM.cmake)

CPMAddPackage(NAME octra SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)

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
file(GLOB c_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.c)
message(STATUS "c_sources: ${c_sources}")
message(STATUS "cpp_sources: ${cpp_sources}")
add_executable(run_tests ${c_sources} ${cpp_sources})
target_link_libraries(run_tests octra::octra GTest::gtest_main)
set_target_properties(run_tests PROPERTIES CXX_STANDARD 17)

include(GoogleTest)
gtest_discover_tests(run_tests)

# ---- code coverage ----
if(ENABLE_TEST_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(run_tests PRIVATE -O0 -g --coverage)
    target_link_options(run_tests PRIVATE --coverage)

    # Ensure the octra library (built via CPM in this build) is also instrumented.
    if(TARGET octra)
      target_compile_options(octra PRIVATE -O0 -g --coverage)
      target_link_options(octra PRIVATE --coverage)
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
