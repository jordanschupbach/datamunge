cmake_minimum_required(VERSION 3.16)

project(
  datamungerust
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)

# The sources come from Rosetta (just prebuild-rust, scripts/prebuild_bindings.py rust):
# src/datamunge_rust_wrap*.cpp (the C ABI library) and the Cargo crate datamunge/ beside this file
# (Cargo.toml, build.rs, src/lib.rs, src/rosetta.rs), generated from rosetta/bindings.hpp and
# rosetta/policy.json.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_rust_wrap*.cpp")
if(NOT GENERATED_SOURCES)
  message(FATAL_ERROR "No generated Rust binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-rust`")
endif()

get_target_property(DATAMUNGE_INSTALLED_INCLUDE_DIRS datamunge::datamunge INTERFACE_INCLUDE_DIRECTORIES)
if(NOT DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  get_filename_component(DATAMUNGE_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH)
  set(DATAMUNGE_INSTALLED_INCLUDE_DIRS "${DATAMUNGE_REPO_INCLUDE_DIR}")
endif()

# The crate links this library as -ldatamunge_rust (libdatamunge is the C++ library itself), found
# through DATAMUNGE_RUST_LIB_DIR (build.rs adds it to the link search path; the dev shells set it).
add_library(datamunge_rust SHARED ${GENERATED_SOURCES})
set_target_properties(
  datamunge_rust
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             OUTPUT_NAME "datamunge_rust"
)

# The generated sources include datamungerust/rosetta/bindings.hpp relative to src/.
target_include_directories(datamunge_rust PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}")
target_link_libraries(datamunge_rust PRIVATE datamunge::datamunge)

# <prefix>/lib/libdatamunge_rust.so, and the crate in <prefix>/share/cargo/datamunge for
# `datamunge = { path = ".../share/cargo/datamunge" }` dependencies.
install(TARGETS datamunge_rust LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/datamunge/" DESTINATION "${CMAKE_INSTALL_DATADIR}/cargo/datamunge"
        PATTERN "target" EXCLUDE)
