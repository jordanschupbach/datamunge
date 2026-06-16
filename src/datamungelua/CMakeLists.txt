cmake_minimum_required(VERSION 3.16)

project(
  datamungelua
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(Lua REQUIRED)
find_package(datamunge CONFIG REQUIRED)

find_program(SWIG_EXECUTABLE swig REQUIRED)

set(SWIG_INTERFACE_FILE "${CMAKE_CURRENT_LIST_DIR}/swig/datamungelua.i")
set(GENERATED_WRAPPER "${CMAKE_CURRENT_BINARY_DIR}/datamunge_lua_wrap.cxx")
set(LUA_LOADER_FILE "${CMAKE_CURRENT_LIST_DIR}/datamunge.lua")

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
  COMMAND
    "${SWIG_EXECUTABLE}" -lua -c++ ${DATAMUNGE_SWIG_INCLUDE_FLAGS}
    # Note: SWIG's Lua backend may not emit a Lua loader file; we ship our own.
    -outdir "${CMAKE_CURRENT_BINARY_DIR}" -o "${GENERATED_WRAPPER}" "${SWIG_INTERFACE_FILE}"
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

target_include_directories(datamunge PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} ${LUA_INCLUDE_DIR})

target_link_libraries(datamunge PRIVATE datamunge::datamunge ${LUA_LIBRARIES})

install(TARGETS datamunge DESTINATION "${CMAKE_INSTALL_LIBDIR}/lua")

install(FILES "${LUA_LOADER_FILE}" DESTINATION "${CMAKE_INSTALL_DATADIR}/lua")
