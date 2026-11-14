cmake_minimum_required(VERSION 3.16)

project(
  datamungefortran
  VERSION 0.0.1
  LANGUAGES CXX Fortran
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)

# The sources come from Rosetta (just prebuild-fortran, scripts/prebuild_bindings.py fortran):
# src/datamunge_fortran_wrap*.cpp (the native library) and the Fortran module source beside this
# file (datamunge.f90), generated from rosetta/bindings.hpp and rosetta/policy.json.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_fortran_wrap*.cpp")
if(NOT GENERATED_SOURCES OR NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/datamunge.f90")
  message(
    FATAL_ERROR
      "No generated Fortran binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-fortran`"
  )
endif()

get_target_property(
  DATAMUNGE_INSTALLED_INCLUDE_DIRS datamunge::datamunge INTERFACE_INCLUDE_DIRECTORIES
)
if(NOT DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  get_filename_component(
    DATAMUNGE_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH
  )
  set(DATAMUNGE_INSTALLED_INCLUDE_DIRS "${DATAMUNGE_REPO_INCLUDE_DIR}")
endif()

# The native library: the C ABI (rosetta_datamunge_call, ...) the Fortran module's bind(C)
# interfaces call. Its name is datamunge_fortran so that it never resolves to libdatamunge, the C++
# library itself.
add_library(datamunge_fortran SHARED ${GENERATED_SOURCES})
set_target_properties(
  datamunge_fortran
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             OUTPUT_NAME "datamunge_fortran"
)
# The generated sources include datamungefortran/rosetta/bindings.hpp relative to src/.
target_include_directories(
  datamunge_fortran PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}"
)
target_link_libraries(datamunge_fortran PRIVATE datamunge::datamunge)

# The Fortran 2018 modules (datamunge, datamunge_rosetta, datamunge_data) as a static library.
# Programs `use datamunge` with -I<prefix>/include/datamunge/fortran and link
# -ldatamunge_fortran_mod -ldatamunge_fortran (just run-fortran shows the full command line).
set(DATAMUNGE_FORTRAN_MODULE_DIR "${CMAKE_CURRENT_BINARY_DIR}/modules")
add_library(datamunge_fortran_mod STATIC "${CMAKE_CURRENT_LIST_DIR}/datamunge.f90")
set_target_properties(
  datamunge_fortran_mod
  PROPERTIES Fortran_MODULE_DIRECTORY "${DATAMUNGE_FORTRAN_MODULE_DIR}"
             Fortran_STANDARD 2018
             Fortran_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
  datamunge_fortran_mod PUBLIC "$<BUILD_INTERFACE:${DATAMUNGE_FORTRAN_MODULE_DIR}>"
)
target_link_libraries(datamunge_fortran_mod PUBLIC datamunge_fortran)

# <prefix>/lib: libdatamunge_fortran.so and libdatamunge_fortran_mod.a;
# <prefix>/include/datamunge/fortran: the .mod files and datamunge.f90 (for editors and FORD).
set(DATAMUNGE_FORTRAN_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/datamunge/fortran")
install(
  TARGETS datamunge_fortran datamunge_fortran_mod
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(
  DIRECTORY "${DATAMUNGE_FORTRAN_MODULE_DIR}/"
  DESTINATION "${DATAMUNGE_FORTRAN_INCLUDE_DIR}"
  FILES_MATCHING
  PATTERN "*.mod"
)
install(FILES "${CMAKE_CURRENT_LIST_DIR}/datamunge.f90"
        DESTINATION "${DATAMUNGE_FORTRAN_INCLUDE_DIR}"
)
