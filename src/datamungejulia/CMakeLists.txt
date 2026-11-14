cmake_minimum_required(VERSION 3.16)

project(
  datamungejulia
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)

# The sources come from Rosetta (just prebuild-julia, scripts/prebuild_bindings.py julia):
# src/datamunge_julia_wrap*.cpp (the native library) and the Julia package beside this file
# (Project.toml, src/Datamunge.jl), generated from rosetta/bindings.hpp and rosetta/policy.json.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_julia_wrap*.cpp")
if(NOT GENERATED_SOURCES)
  message(FATAL_ERROR "No generated Julia binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-julia`")
endif()

get_target_property(DATAMUNGE_INSTALLED_INCLUDE_DIRS datamunge::datamunge INTERFACE_INCLUDE_DIRECTORIES)
if(NOT DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  get_filename_component(DATAMUNGE_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH)
  set(DATAMUNGE_INSTALLED_INCLUDE_DIRS "${DATAMUNGE_REPO_INCLUDE_DIR}")
endif()

# A plain shared library (no Julia headers): Datamunge.jl loads it with Libdl and ccall.
add_library(datamunge MODULE ${GENERATED_SOURCES})
set_target_properties(
  datamunge
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             PREFIX ""
             OUTPUT_NAME "datamunge"
)
if(APPLE)
  # Libdl.dlext
  set_target_properties(datamunge PROPERTIES SUFFIX ".dylib")
endif()

# The generated sources include datamungejulia/rosetta/bindings.hpp relative to src/.
target_include_directories(datamunge PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}")
target_link_libraries(datamunge PRIVATE datamunge::datamunge)

# <prefix>/share/julia/Datamunge is a Julia package (the module finds lib/ beside src/);
# put <prefix>/share/julia on JULIA_LOAD_PATH for `using Datamunge`.
set(DATAMUNGE_JULIA_DIR "${CMAKE_INSTALL_DATADIR}/julia/Datamunge")
install(TARGETS datamunge DESTINATION "${DATAMUNGE_JULIA_DIR}/lib")
install(FILES "${CMAKE_CURRENT_LIST_DIR}/Project.toml" DESTINATION "${DATAMUNGE_JULIA_DIR}")
install(FILES "${CMAKE_CURRENT_LIST_DIR}/src/Datamunge.jl" DESTINATION "${DATAMUNGE_JULIA_DIR}/src")
