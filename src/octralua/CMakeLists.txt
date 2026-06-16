cmake_minimum_required(VERSION 3.16)

project(
  octralua
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(Lua REQUIRED)
find_package(octra CONFIG REQUIRED)

find_program(SWIG_EXECUTABLE swig REQUIRED)

set(SWIG_INTERFACE_FILE "${CMAKE_CURRENT_LIST_DIR}/swig/octralua.i")
set(GENERATED_WRAPPER "${CMAKE_CURRENT_BINARY_DIR}/octra_lua_wrap.cxx")
set(LUA_LOADER_FILE "${CMAKE_CURRENT_LIST_DIR}/octra.lua")

get_target_property(OCTRA_INSTALLED_INCLUDE_DIRS octra::octra INTERFACE_INCLUDE_DIRECTORIES)
if(NOT OCTRA_INSTALLED_INCLUDE_DIRS)
  get_filename_component(OCTRA_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH)
  set(OCTRA_INSTALLED_INCLUDE_DIRS "${OCTRA_REPO_INCLUDE_DIR}")
endif()

set(OCTRA_SWIG_INCLUDE_FLAGS "")
foreach(incdir IN LISTS OCTRA_INSTALLED_INCLUDE_DIRS)
  list(APPEND OCTRA_SWIG_INCLUDE_FLAGS "-I${incdir}")
endforeach()

add_custom_command(
  OUTPUT "${GENERATED_WRAPPER}"
  COMMAND
    "${SWIG_EXECUTABLE}" -lua -c++ ${OCTRA_SWIG_INCLUDE_FLAGS}
    # Note: SWIG's Lua backend may not emit a Lua loader file; we ship our own.
    -outdir "${CMAKE_CURRENT_BINARY_DIR}" -o "${GENERATED_WRAPPER}" "${SWIG_INTERFACE_FILE}"
  DEPENDS "${SWIG_INTERFACE_FILE}"
  VERBATIM
)

add_library(octra MODULE "${GENERATED_WRAPPER}")
set_target_properties(
  octra
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             PREFIX ""
             OUTPUT_NAME "octra"
)

target_include_directories(octra PRIVATE ${OCTRA_INSTALLED_INCLUDE_DIRS} ${LUA_INCLUDE_DIR})

target_link_libraries(octra PRIVATE octra::octra ${LUA_LIBRARIES})

install(TARGETS octra DESTINATION "${CMAKE_INSTALL_LIBDIR}/lua")

install(FILES "${LUA_LOADER_FILE}" DESTINATION "${CMAKE_INSTALL_DATADIR}/lua")
