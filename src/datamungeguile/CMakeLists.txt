cmake_minimum_required(VERSION 3.16)

project(
  datamungeguile
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(PkgConfig REQUIRED)
find_package(datamunge CONFIG REQUIRED)
pkg_check_modules(GUILE REQUIRED guile-3.0)

find_program(SWIG_EXECUTABLE swig REQUIRED)

set(SWIG_INTERFACE_FILE "${CMAKE_CURRENT_LIST_DIR}/swig/datamungeguile.i")
set(GENERATED_WRAPPER "${CMAKE_CURRENT_BINARY_DIR}/datamunge_guile_wrap.cxx")
set(GUILE_LOADER_FILE "${CMAKE_CURRENT_LIST_DIR}/datamunge.scm")

get_target_property(DATAMUNGE_INSTALLED_INCLUDE_DIRS datamunge::datamunge INTERFACE_INCLUDE_DIRECTORIES)
if(NOT DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  get_filename_component(DATAMUNGE_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH)
  set(DATAMUNGE_INSTALLED_INCLUDE_DIRS "${DATAMUNGE_REPO_INCLUDE_DIR}")
endif()

set(DATAMUNGE_SWIG_INCLUDE_FLAGS "")
foreach(incdir IN LISTS DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  list(APPEND DATAMUNGE_SWIG_INCLUDE_FLAGS "-I${incdir}")
endforeach()

add_custom_command(
  OUTPUT "${GENERATED_WRAPPER}"
  COMMAND "${SWIG_EXECUTABLE}" -guile -c++ ${DATAMUNGE_SWIG_INCLUDE_FLAGS} -o "${GENERATED_WRAPPER}"
          "${SWIG_INTERFACE_FILE}"
  DEPENDS "${SWIG_INTERFACE_FILE}"
  VERBATIM
)

add_library(datamunge MODULE "${GENERATED_WRAPPER}")
set_target_properties(
  datamunge
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             PREFIX ""
             OUTPUT_NAME "datamunge"
)

target_include_directories(datamunge PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} ${GUILE_INCLUDE_DIRS})

target_link_libraries(datamunge PRIVATE datamunge::datamunge ${GUILE_LIBRARIES})

target_compile_options(datamunge PRIVATE ${GUILE_CFLAGS_OTHER})

execute_process(
  COMMAND "${PKG_CONFIG_EXECUTABLE}" --variable=effective-version guile-3.0
  OUTPUT_VARIABLE GUILE_EFFECTIVE_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT GUILE_EFFECTIVE_VERSION)
  set(GUILE_EFFECTIVE_VERSION "3.0")
endif()

install(TARGETS datamunge
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/guile/${GUILE_EFFECTIVE_VERSION}/extensions"
)

install(FILES "${GUILE_LOADER_FILE}"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/guile/site/${GUILE_EFFECTIVE_VERSION}"
)
