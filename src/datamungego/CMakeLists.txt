cmake_minimum_required(VERSION 3.16)

project(
  datamungego
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)
find_package(Threads REQUIRED)

# The sources come from Rosetta (just prebuild-go, scripts/prebuild_bindings.py go):
# src/datamunge_go_wrap*.cpp (the shared library libdatamunge_go, the C ABI that cgo calls) and the
# Go module datamunge under go/datamunge beside this file (go.mod, datamunge.go, the policy
# subpackage data/ and internal/rosetta), documented with godoc comments, generated from
# rosetta/bindings.hpp and rosetta/policy.json. The module's cgo preamble links -ldatamunge_go;
# consumers point the linker at this library with CGO_LDFLAGS="-L<dir> -Wl,-rpath,<dir>".
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
set(DATAMUNGE_GO_MODULE_SRC "${CMAKE_CURRENT_LIST_DIR}/go/datamunge")
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_go_wrap*.cpp")
if(NOT GENERATED_SOURCES OR NOT EXISTS "${DATAMUNGE_GO_MODULE_SRC}/datamunge.go")
  message(
    FATAL_ERROR "No generated Go binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-go`"
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

# libdatamunge_go (libdatamunge is the C++ library itself).
add_library(datamunge_go SHARED ${GENERATED_SOURCES})
set_target_properties(
  datamunge_go
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             OUTPUT_NAME "datamunge_go"
             LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
)
# The generated sources include datamungego/rosetta/bindings.hpp relative to src/.
target_include_directories(
  datamunge_go PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}"
)
target_link_libraries(datamunge_go PRIVATE datamunge::datamunge Threads::Threads)

# <prefix>/lib/libdatamunge_go.so and the module <prefix>/share/go/datamunge: a consumer's go.mod
# has `require datamunge v0.0.0` and `replace datamunge => <prefix>/share/go/datamunge`, and builds
# with CGO_LDFLAGS="-L<prefix>/lib -Wl,-rpath,<prefix>/lib" (the dev shells set it).
install(TARGETS datamunge_go LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY "${DATAMUNGE_GO_MODULE_SRC}/" DESTINATION "${CMAKE_INSTALL_DATADIR}/go/datamunge")
