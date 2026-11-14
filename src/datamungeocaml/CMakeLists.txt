cmake_minimum_required(VERSION 3.16)

project(
  datamungeocaml
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)
find_package(Threads REQUIRED)

# The sources come from Rosetta (just prebuild-ocaml, scripts/prebuild_bindings.py ocaml):
# src/datamunge_ocaml_wrap*.cpp (the C ABI library with its OCaml C stubs) and the findlib package
# files ocaml/datamunge/{datamunge.ml,datamunge.mli,META} beside this file. datamunge.mli carries the
# documentation as odoc comments, generated from rosetta/bindings.hpp and rosetta/policy.json.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
set(DATAMUNGE_OCAML_PACKAGE_SRC "${CMAKE_CURRENT_LIST_DIR}/ocaml/datamunge")
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_ocaml_wrap*.cpp")
if(NOT GENERATED_SOURCES OR NOT EXISTS "${DATAMUNGE_OCAML_PACKAGE_SRC}/datamunge.mli")
  message(
    FATAL_ERROR "No generated OCaml binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-ocaml`"
  )
endif()

# OCaml: the compilers, the standard library directory (caml/*.h) and the DLL suffix, as Rosetta's
# cmake/OcamlExample.cmake does.
find_program(DATAMUNGE_OCAMLC NAMES ocamlc.opt ocamlc REQUIRED)
find_program(DATAMUNGE_OCAMLOPT NAMES ocamlopt.opt ocamlopt REQUIRED)
execute_process(
  COMMAND "${DATAMUNGE_OCAMLC}" -where
  OUTPUT_VARIABLE DATAMUNGE_OCAML_STDLIB
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND "${DATAMUNGE_OCAMLC}" -config-var ext_dll
  OUTPUT_VARIABLE DATAMUNGE_OCAML_EXT_DLL
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY
)
if(APPLE)
  set(DATAMUNGE_OCAML_CXX_RUNTIME c++)
else()
  set(DATAMUNGE_OCAML_CXX_RUNTIME stdc++)
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
# The directory of libdatamunge itself: native programs link it (-ldatamunge, with an rpath)
# through the flags recorded in datamunge.cmxa.
get_target_property(DATAMUNGE_LIBRARY_LOCATION datamunge::datamunge LOCATION)
get_filename_component(DATAMUNGE_LIBRARY_DIR "${DATAMUNGE_LIBRARY_LOCATION}" DIRECTORY)

# The findlib package directory datamunge/ in the build tree, so OCAMLPATH=<build dir> works
# without installing: datamunge.{ml,mli,cmi,cmti,cmo,cma,cmx,cmxa,a}, META, the stubs
# dlldatamunge_stubs.so (bytecode) and libdatamunge_stubs.a (native).
set(DATAMUNGE_OCAML_PACKAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/datamunge")

add_library(datamunge_ocaml_objects OBJECT ${GENERATED_SOURCES})
set_target_properties(
  datamunge_ocaml_objects PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON
                                     POSITION_INDEPENDENT_CODE ON
)
# The generated sources include datamungeocaml/rosetta/bindings.hpp relative to src/.
target_include_directories(
  datamunge_ocaml_objects PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}"
)
target_include_directories(datamunge_ocaml_objects SYSTEM PRIVATE "${DATAMUNGE_OCAML_STDLIB}")
target_link_libraries(datamunge_ocaml_objects PRIVATE datamunge::datamunge)

# dlldatamunge_stubs.so: loaded by the bytecode runtime and the toplevel (datamunge.cma -dllib).
add_library(datamunge_ocaml_shared MODULE $<TARGET_OBJECTS:datamunge_ocaml_objects>)
set_target_properties(
  datamunge_ocaml_shared
  PROPERTIES PREFIX "dll"
             SUFFIX "${DATAMUNGE_OCAML_EXT_DLL}"
             OUTPUT_NAME "datamunge_stubs"
             LIBRARY_OUTPUT_DIRECTORY "${DATAMUNGE_OCAML_PACKAGE_DIR}"
)
target_link_libraries(datamunge_ocaml_shared PRIVATE datamunge::datamunge Threads::Threads)
if(APPLE)
  target_link_options(datamunge_ocaml_shared PRIVATE -undefined dynamic_lookup)
endif()

# libdatamunge_stubs.a: linked into native programs (datamunge.cmxa -cclib).
add_library(datamunge_ocaml_static STATIC $<TARGET_OBJECTS:datamunge_ocaml_objects>)
set_target_properties(
  datamunge_ocaml_static PROPERTIES OUTPUT_NAME "datamunge_stubs" ARCHIVE_OUTPUT_DIRECTORY
                                                                 "${DATAMUNGE_OCAML_PACKAGE_DIR}"
)

# The OCaml library: the interface (with -bin-annot, for odoc and editors), the bytecode archive
# datamunge.cma and the native archive datamunge.cmxa, compiled from copies of the committed
# package files in the package directory.
set(pkg "${DATAMUNGE_OCAML_PACKAGE_DIR}")
set(ocaml_outputs
    "${pkg}/datamunge.cmi" "${pkg}/datamunge.cmti" "${pkg}/datamunge.cmo" "${pkg}/datamunge.cma"
    "${pkg}/datamunge.cmx" "${pkg}/datamunge.o" "${pkg}/datamunge.cmxa" "${pkg}/datamunge.a"
)
set(native_link_flags
    -cclib -ldatamunge_stubs -ccopt "-L${DATAMUNGE_LIBRARY_DIR}" -cclib -ldatamunge -ccopt
    "-Wl,-rpath,${DATAMUNGE_LIBRARY_DIR}" -cclib "-l${DATAMUNGE_OCAML_CXX_RUNTIME}"
)
add_custom_command(
  OUTPUT ${ocaml_outputs}
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${pkg}"
  COMMAND "${CMAKE_COMMAND}" -E copy "${DATAMUNGE_OCAML_PACKAGE_SRC}/datamunge.ml"
          "${DATAMUNGE_OCAML_PACKAGE_SRC}/datamunge.mli" "${DATAMUNGE_OCAML_PACKAGE_SRC}/META" "${pkg}"
  COMMAND "${DATAMUNGE_OCAMLC}" -bin-annot -c -o "${pkg}/datamunge.cmi" "${pkg}/datamunge.mli"
  COMMAND "${DATAMUNGE_OCAMLC}" -I "${pkg}" -c -o "${pkg}/datamunge.cmo" "${pkg}/datamunge.ml"
  COMMAND "${DATAMUNGE_OCAMLC}" -a -o "${pkg}/datamunge.cma" "${pkg}/datamunge.cmo" -dllib
          -ldatamunge_stubs ${native_link_flags}
  COMMAND "${DATAMUNGE_OCAMLOPT}" -I "${pkg}" -c -o "${pkg}/datamunge.cmx" "${pkg}/datamunge.ml"
  COMMAND "${DATAMUNGE_OCAMLOPT}" -a -o "${pkg}/datamunge.cmxa" "${pkg}/datamunge.cmx"
          ${native_link_flags}
  DEPENDS "${DATAMUNGE_OCAML_PACKAGE_SRC}/datamunge.ml"
          "${DATAMUNGE_OCAML_PACKAGE_SRC}/datamunge.mli" "${DATAMUNGE_OCAML_PACKAGE_SRC}/META"
  COMMENT "Compiling the OCaml library datamunge"
  VERBATIM
)
add_custom_target(
  datamunge_ocaml ALL DEPENDS ${ocaml_outputs} datamunge_ocaml_shared datamunge_ocaml_static
)

# <prefix>/lib/datamunge: the findlib package (META, the compiled library, its .ml/.mli and both stub
# libraries). `ocamlfind ocamlopt -package datamunge` or dune `(libraries datamunge)` with
# OCAMLPATH=<prefix>/lib (the dev shells set it); `#require "datamunge"` in the toplevel.
install(TARGETS datamunge_ocaml_shared LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/datamunge")
install(TARGETS datamunge_ocaml_static ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/datamunge")
install(
  FILES ${ocaml_outputs} "${pkg}/datamunge.ml"
        "${pkg}/datamunge.mli" "${pkg}/META"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/datamunge"
)
