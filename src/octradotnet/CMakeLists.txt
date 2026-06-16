cmake_minimum_required(VERSION 3.16)
project(
  octradotnet
  VERSION 0.0.1
  LANGUAGES CXX
)

set(OCTRA_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")

# PackageProject Obtained from https://github.com/TheLartians/PackageProject.cmake See
# cmake/justfile
include(${OCTRA_REPO_ROOT}/cmake/PackageProject.cmake)

include(${OCTRA_REPO_ROOT}/cmake/lib_sources.cmake)

include(${OCTRA_REPO_ROOT}/cmake/tools.cmake)
include(${OCTRA_REPO_ROOT}/cmake/CPM.cmake)

CPMAddPackage(NAME octra SOURCE_DIR ${OCTRA_REPO_ROOT})

add_library(${PROJECT_NAME} SHARED octra_csharp_wrap.cpp)

# SWIG-generated C# glue uses `[DllImport("octra_csharp")]` (see `%module(dllimport=...)`).
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "octra_csharp")

# set(SOURCES) foreach(ENTRY ${SOURCES_}) set(FILE_PATH
# "${CMAKE_CURRENT_SOURCE_DIR}/../include/${ENTRY}.hpp") if(EXISTS "${FILE_PATH}") list(APPEND
# SOURCES "${FILE_PATH}") endif() endforeach()
#
# # Check sources foreach(ENTRY ${SOURCES_}) set(FILE_PATH
# "${CMAKE_CURRENT_SOURCE_DIR}/../source/${ENTRY}.cpp") if(EXISTS "${FILE_PATH}") list(APPEND
# SOURCES "${FILE_PATH}") endif() endforeach()

set(FILE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/octra_csharp_wrap.cpp")
if(EXISTS "${FILE_PATH}")
  list(APPEND SOURCES "${FILE_PATH}")
endif()

message(STATUS "COMBINED_FILES: ${SOURCES}")

# add_library(${PROJECT_NAME} STATIC ${HEADERS} ${SOURCES})

# add_library(${PROJECT_NAME} SHARED ${SOURCES}) set_target_properties(${PROJECT_NAME} PROPERTIES
# CXX_STANDARD 23)

set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL ""
)

# for nix environments... TODO: add check for nix, or is this works in general?
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE octra::octra)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC $<BUILD_INTERFACE:${OCTRA_REPO_ROOT}/include>
         $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>
         # ${LIBXML2_INCLUDE_DIRS}
)

string(TOLOWER ${PROJECT_NAME}/version.h VERSION_HEADER_LOCATION)

# # Package config file set(PKG_CONFIG_LIBDIR "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig")
# set(PKG_CONFIG_VERSION "0.0.1") set(PKG_CONFIG_NAME "${PROJECT_NAME}") set(PKG_CONFIG_DESCRIPTION
# "One C to Rule Them All") set(PKG_CONFIG_LIBS "-l${PROJECT_NAME}") set(PKG_CONFIG_LIBDIRS
# "${CMAKE_INSTALL_PREFIX}/lib") set(PKG_CONFIG_INCLUDES "${CMAKE_INSTALL_PREFIX}/include")
# configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.pc.in"
# "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc" @ONLY ) install(FILES
# "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc" DESTINATION ${PKG_CONFIG_LIBDIR} )

packageProject(
  NAME ${PROJECT_NAME}
  VERSION 0.0.1
  NAMESPACE ${PROJECT_NAME}
  BINARY_DIR ${PROJECT_BINARY_DIR}
  INCLUDE_DIR ${OCTRA_REPO_ROOT}/include
  INCLUDE_DESTINATION include/${PROJECT_NAME}-${PROJECT_VERSION} VERSION_HEADER_FILE
                                                                 ${VERSION_HEADER_LOCATION}
  COMPATIBILITY SameMajorVersion
  # DEPENDENCIES "fmt 10.2.1"
)
