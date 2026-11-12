cmake_minimum_required(VERSION 3.16)

project(
  datamungelua
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(Lua REQUIRED)
find_package(datamunge CONFIG REQUIRED)

# The module sources come from Rosetta (just prebuild-lua, scripts/prebuild_bindings.py lua):
# src/datamunge_lua_wrap*.cpp, generated from rosetta/bindings.hpp and rosetta/policy.json.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_lua_wrap*.cpp")
if(NOT GENERATED_SOURCES)
  message(FATAL_ERROR "No generated Lua binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-lua`")
endif()
set(LUA_LOADER_FILE "${CMAKE_CURRENT_LIST_DIR}/datamunge.lua")
# LuaCATS stub: documentation for editors (lua-language-server, mep), never loaded at run time.
set(LUA_META_FILE "${CMAKE_CURRENT_LIST_DIR}/library/datamunge.lua")

get_target_property(DATAMUNGE_INSTALLED_INCLUDE_DIRS datamunge::datamunge INTERFACE_INCLUDE_DIRECTORIES)
if(NOT DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  get_filename_component(DATAMUNGE_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH)
  set(DATAMUNGE_INSTALLED_INCLUDE_DIRS "${DATAMUNGE_REPO_INCLUDE_DIR}")
endif()

add_library(datamunge MODULE ${GENERATED_SOURCES})
set_target_properties(
  datamunge
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             PREFIX ""
             OUTPUT_NAME "datamunge"
)

# The generated sources include datamungelua/rosetta/bindings.hpp relative to src/.
target_include_directories(datamunge PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}" ${LUA_INCLUDE_DIR})

target_link_libraries(datamunge PRIVATE datamunge::datamunge ${LUA_LIBRARIES})

install(TARGETS datamunge DESTINATION "${CMAKE_INSTALL_LIBDIR}/lua")

install(FILES "${LUA_LOADER_FILE}" DESTINATION "${CMAKE_INSTALL_DATADIR}/lua")
install(FILES "${LUA_META_FILE}" DESTINATION "${CMAKE_INSTALL_DATADIR}/lua/library")
