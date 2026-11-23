cmake_minimum_required(VERSION 3.16)

project(
  datamungeobjc
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)

# The sources come from Rosetta (just prebuild-objc, scripts/prebuild_bindings.py objc):
# src/datamunge_objc_wrap*.cpp (the C ABI library) and the Objective-C classes objc/DM.h and DM.m beside
# this file (/// comments), generated from rosetta/bindings.hpp and rosetta/policy.json.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_objc_wrap*.cpp")
if(NOT GENERATED_SOURCES)
  message(FATAL_ERROR "No generated Objective-C binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-objc`")
endif()

get_target_property(DATAMUNGE_INSTALLED_INCLUDE_DIRS datamunge::datamunge INTERFACE_INCLUDE_DIRECTORIES)
if(NOT DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  get_filename_component(DATAMUNGE_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH)
  set(DATAMUNGE_INSTALLED_INCLUDE_DIRS "${DATAMUNGE_REPO_INCLUDE_DIR}")
endif()

# libDM (the classes) links this library.
add_library(datamunge_objc SHARED ${GENERATED_SOURCES})
set_target_properties(
  datamunge_objc
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             OUTPUT_NAME "datamunge_objc"
)

# The generated sources include datamungeobjc/rosetta/bindings.hpp relative to src/.
target_include_directories(datamunge_objc PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}")
target_link_libraries(datamunge_objc PRIVATE datamunge::datamunge)

# The classes, libDM: clang with ARC and the GNUstep 2.0 runtime (GNUstep Base and libobjc2, found
# through DATAMUNGE_GNUSTEP_BASE_ROOT, DATAMUNGE_GNUSTEP_BASE_LIB and DATAMUNGE_LIBOBJC_ROOT). Programs
# compile with `clang -fobjc-arc -fobjc-runtime=gnustep-2.0 -fblocks -I<prefix>/include/DM program.m
# -L<prefix>/lib -lDM -lgnustep-base -lobjc`.
find_program(DATAMUNGE_OBJC_CC clang REQUIRED)
set(gnustep_base_root "$ENV{DATAMUNGE_GNUSTEP_BASE_ROOT}" CACHE PATH "GNUstep Base headers (prefix)")
set(gnustep_base_lib "$ENV{DATAMUNGE_GNUSTEP_BASE_LIB}" CACHE PATH "GNUstep Base library (prefix)")
set(libobjc_root "$ENV{DATAMUNGE_LIBOBJC_ROOT}" CACHE PATH "libobjc2 (prefix)")
set(objc_flags -fobjc-arc -fobjc-arc-exceptions -fobjc-runtime=gnustep-2.0 -fblocks -Wall -Werror)
foreach(root IN ITEMS "${gnustep_base_root}" "${libobjc_root}")
  if(root)
    list(APPEND objc_flags "-I${root}/include")
  endif()
endforeach()
set(objc_libs "-L$<TARGET_FILE_DIR:datamunge_objc>" -ldatamunge_objc)
foreach(root IN ITEMS "${gnustep_base_lib}" "${libobjc_root}")
  if(root)
    list(APPEND objc_libs "-L${root}/lib" "-Wl,-rpath,${root}/lib")
  endif()
endforeach()
# libDM finds libdatamunge_objc beside it.
list(APPEND objc_libs -lgnustep-base -lobjc "-Wl,-rpath,\$ORIGIN")
set(objc_src "${CMAKE_CURRENT_LIST_DIR}/objc")
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libDM${CMAKE_SHARED_LIBRARY_SUFFIX}"
  COMMAND "${DATAMUNGE_OBJC_CC}" -O2 -fPIC -shared ${objc_flags} "${objc_src}/DM.m"
          -o "${CMAKE_CURRENT_BINARY_DIR}/libDM${CMAKE_SHARED_LIBRARY_SUFFIX}" ${objc_libs}
  DEPENDS "${objc_src}/DM.m" "${objc_src}/DM.h" datamunge_objc
  COMMENT "Compiling the Objective-C classes (libDM)"
  VERBATIM COMMAND_EXPAND_LISTS)
add_custom_target(datamunge_objc_classes ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/libDM${CMAKE_SHARED_LIBRARY_SUFFIX}")

install(TARGETS datamunge_objc LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libDM${CMAKE_SHARED_LIBRARY_SUFFIX}" DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES "${objc_src}/DM.h" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/DM")
