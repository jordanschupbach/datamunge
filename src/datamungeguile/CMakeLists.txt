cmake_minimum_required(VERSION 3.16)

project(
  datamungeguile
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
pkg_check_modules(GUILE REQUIRED IMPORTED_TARGET guile-3.0)

# The sources come from Rosetta (just prebuild-guile, scripts/prebuild_bindings.py guile):
# src/datamunge_guile_wrap*.cpp (the libguile extension libdatamunge_guile) and scheme/ beside this
# file: the module (datamunge) in datamunge.scm and its policy submodule (datamunge data) in
# datamunge/data.scm, whose define/define* docstrings carry the C++ documentation. datamunge.scm
# load-extension's libdatamunge_guile from its own directory.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
set(DATAMUNGE_GUILE_SCHEME_DIR "${CMAKE_CURRENT_LIST_DIR}/scheme")
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_guile_wrap*.cpp")
if(NOT GENERATED_SOURCES OR NOT EXISTS "${DATAMUNGE_GUILE_SCHEME_DIR}/datamunge.scm")
  message(
    FATAL_ERROR
      "No generated Guile binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-guile`"
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

execute_process(
  COMMAND "${PKG_CONFIG_EXECUTABLE}" --variable=effective-version guile-3.0
  OUTPUT_VARIABLE GUILE_EFFECTIVE_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT GUILE_EFFECTIVE_VERSION)
  set(GUILE_EFFECTIVE_VERSION "3.0")
endif()

# The extension libdatamunge_guile.so, built into site/ under the build directory beside a copy of
# the modules, so that site/ mirrors the installed layout.
set(DATAMUNGE_GUILE_BUILD_SITE "${CMAKE_CURRENT_BINARY_DIR}/site")
add_library(datamunge_guile MODULE ${GENERATED_SOURCES})
set_target_properties(
  datamunge_guile
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             PREFIX ""
             OUTPUT_NAME "libdatamunge_guile"
             LIBRARY_OUTPUT_DIRECTORY "${DATAMUNGE_GUILE_BUILD_SITE}"
)
# The generated sources include datamungeguile/rosetta/bindings.hpp relative to src/.
target_include_directories(
  datamunge_guile PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}"
)
target_link_libraries(
  datamunge_guile PRIVATE datamunge::datamunge PkgConfig::GUILE Threads::Threads
)
add_custom_command(
  TARGET datamunge_guile
  POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E copy_directory "${DATAMUNGE_GUILE_SCHEME_DIR}"
          "${DATAMUNGE_GUILE_BUILD_SITE}"
  VERBATIM
)

# Precompiled modules: guild compile writes site-ccache/datamunge.go and site-ccache/datamunge/*.go
# at build time (about 45 s for the 40k-line module), so that (use-modules (datamunge)) needn't
# auto-compile or interpret the source. Compiling the submodules loads (datamunge) and its
# extension, hence the dependencies on datamunge_guile and datamunge.go.
find_program(GUILD_EXECUTABLE NAMES guild3.0 guild REQUIRED)
set(DATAMUNGE_GUILE_BUILD_CCACHE "${CMAKE_CURRENT_BINARY_DIR}/site-ccache")
file(
  GLOB_RECURSE DATAMUNGE_GUILE_SUBMODULES
  RELATIVE "${DATAMUNGE_GUILE_SCHEME_DIR}"
  CONFIGURE_DEPENDS "${DATAMUNGE_GUILE_SCHEME_DIR}/datamunge/*.scm"
)
set(DATAMUNGE_GUILE_GO_FILES)
foreach(module datamunge.scm ${DATAMUNGE_GUILE_SUBMODULES})
  string(REGEX REPLACE "\\.scm$" ".go" go "${module}")
  set(depends "${DATAMUNGE_GUILE_SCHEME_DIR}/${module}" datamunge_guile)
  if(NOT module STREQUAL "datamunge.scm")
    list(APPEND depends "${DATAMUNGE_GUILE_BUILD_CCACHE}/datamunge.go")
  endif()
  add_custom_command(
    OUTPUT "${DATAMUNGE_GUILE_BUILD_CCACHE}/${go}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${DATAMUNGE_GUILE_SCHEME_DIR}/${module}"
            "${DATAMUNGE_GUILE_BUILD_SITE}/${module}"
    COMMAND
      "${CMAKE_COMMAND}" -E env "GUILE_LOAD_PATH=${DATAMUNGE_GUILE_BUILD_SITE}"
      "GUILE_LOAD_COMPILED_PATH=${DATAMUNGE_GUILE_BUILD_CCACHE}" GUILE_AUTO_COMPILE=0
      "${GUILD_EXECUTABLE}" compile -Wshadowed-toplevel -Wunbound-variable -Warity-mismatch -Wformat
      -L "${DATAMUNGE_GUILE_BUILD_SITE}" -o "${DATAMUNGE_GUILE_BUILD_CCACHE}/${go}"
      "${DATAMUNGE_GUILE_BUILD_SITE}/${module}"
    DEPENDS ${depends}
    COMMENT "Compiling Guile module ${module}"
    VERBATIM
  )
  list(APPEND DATAMUNGE_GUILE_GO_FILES "${DATAMUNGE_GUILE_BUILD_CCACHE}/${go}")
endforeach()
add_custom_target(datamunge_guile_go ALL DEPENDS ${DATAMUNGE_GUILE_GO_FILES})

# <prefix>/share/guile/site/<effective-version>: datamunge.scm, datamunge/ and
# libdatamunge_guile.so together. With that directory on GUILE_LOAD_PATH (the dev shells set it),
# (use-modules (datamunge)) finds the module, which loads the extension beside it.
set(DATAMUNGE_GUILE_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/guile/site/${GUILE_EFFECTIVE_VERSION}")
install(TARGETS datamunge_guile LIBRARY DESTINATION "${DATAMUNGE_GUILE_INSTALL_DIR}")
install(
  DIRECTORY "${DATAMUNGE_GUILE_SCHEME_DIR}/"
  DESTINATION "${DATAMUNGE_GUILE_INSTALL_DIR}"
  FILES_MATCHING
  PATTERN "*.scm"
)
# <prefix>/lib/guile/<effective-version>/site-ccache (Guile's siteccachedir layout; lib, not lib64, so
# the path the shells set is fixed): datamunge.go and
# datamunge/*.go. Put it on GUILE_LOAD_COMPILED_PATH (the dev shells and the justfile's GUILE do).
install(
  DIRECTORY "${DATAMUNGE_GUILE_BUILD_CCACHE}/"
  DESTINATION "lib/guile/${GUILE_EFFECTIVE_VERSION}/site-ccache"
  FILES_MATCHING
  PATTERN "*.go"
)
