cmake_minimum_required(VERSION 3.16)

project(
  datamungeswift
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)
# swiftc: on PATH, or under DATAMUNGE_SWIFT_ROOT (the dev shells set it and keep the Swift toolchain,
# which ships its own clang, off PATH).
find_program(DATAMUNGE_SWIFTC swiftc HINTS "$ENV{DATAMUNGE_SWIFT_ROOT}/bin" REQUIRED)

# The sources come from Rosetta (just prebuild-swift, scripts/prebuild_bindings.py swift):
# src/datamunge_swift_wrap*.cpp (the C ABI library) and the Swift package Datamunge/ beside this file
# (Package.swift, Sources/Datamunge with the DocC documentation, the C module Sources/CDatamunge),
# generated from rosetta/bindings.hpp and rosetta/policy.json.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_swift_wrap*.cpp")
if(NOT GENERATED_SOURCES)
  message(FATAL_ERROR "No generated Swift binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-swift`")
endif()

get_target_property(DATAMUNGE_INSTALLED_INCLUDE_DIRS datamunge::datamunge INTERFACE_INCLUDE_DIRECTORIES)
if(NOT DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  get_filename_component(DATAMUNGE_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH)
  set(DATAMUNGE_INSTALLED_INCLUDE_DIRS "${DATAMUNGE_REPO_INCLUDE_DIR}")
endif()

# The C module CDatamunge links this library as -ldatamunge_swift (libdatamunge is the C++ library).
add_library(datamunge_swift SHARED ${GENERATED_SOURCES})
set_target_properties(
  datamunge_swift
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             OUTPUT_NAME "datamunge_swift"
)
target_include_directories(datamunge_swift PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}")
target_link_libraries(datamunge_swift PRIVATE datamunge::datamunge)

# The Swift module itself, compiled with swiftc (no SwiftPM, which wants a network-free cache):
# libDatamunge.so plus Datamunge.swiftmodule/.swiftdoc. Programs compile with
# `swiftc -I <prefix>/lib/swift -I <prefix>/share/swift/CDatamunge -L <prefix>/lib -lDatamunge`.
set(package "${CMAKE_CURRENT_LIST_DIR}/Datamunge")
set(swift_out "${CMAKE_CURRENT_BINARY_DIR}/swift")
get_filename_component(swift_root "${DATAMUNGE_SWIFTC}" DIRECTORY)
get_filename_component(swift_root "${swift_root}" DIRECTORY)
# The Swift runtime (libswiftCore.so, ...) lives in the toolchain's lib/ (nixpkgs' Swift); link and
# run against it.
set(swift_link)
if(EXISTS "${swift_root}/lib/libswiftCore.so")
  set(swift_link -L "${swift_root}/lib" -Xlinker -rpath -Xlinker "${swift_root}/lib")
endif()
add_custom_command(
  OUTPUT "${swift_out}/libDatamunge.so" "${swift_out}/Datamunge.swiftmodule"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${swift_out}"
  COMMAND "${DATAMUNGE_SWIFTC}" -swift-version 5 -O -emit-library -emit-module -module-name Datamunge
          -parse-as-library "${package}/Sources/Datamunge/Datamunge.swift" "${package}/Sources/Datamunge/Rosetta.swift"
          -I "${package}/Sources/CDatamunge" -L "$<TARGET_FILE_DIR:datamunge_swift>"
          -emit-module-path "${swift_out}/Datamunge.swiftmodule" -o "${swift_out}/libDatamunge.so"
          -Xlinker -rpath -Xlinker "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" ${swift_link}
  DEPENDS "${package}/Sources/Datamunge/Datamunge.swift" "${package}/Sources/Datamunge/Rosetta.swift"
          "${package}/Sources/CDatamunge/module.modulemap" datamunge_swift
  COMMENT "Compiling the Swift module Datamunge"
  VERBATIM)
add_custom_target(datamunge_swift_module ALL DEPENDS "${swift_out}/libDatamunge.so")

install(TARGETS datamunge_swift LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES "${swift_out}/libDatamunge.so" DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES "${swift_out}/Datamunge.swiftmodule" "${swift_out}/Datamunge.swiftdoc" DESTINATION "${CMAKE_INSTALL_LIBDIR}/swift" OPTIONAL)
install(DIRECTORY "${package}/Sources/CDatamunge" DESTINATION "${CMAKE_INSTALL_DATADIR}/swift")
install(DIRECTORY "${package}/" DESTINATION "${CMAKE_INSTALL_DATADIR}/swift/package/Datamunge")
