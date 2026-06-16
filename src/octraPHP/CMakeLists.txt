cmake_minimum_required(VERSION 3.16)
project(
  octraPHP
  VERSION 0.0.1
  LANGUAGES CXX
)

include(../../cmake/PackageProject.cmake)
include(../../cmake/tools.cmake)
include(../../cmake/CPM.cmake)

CPMAddPackage(NAME octra SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../)

set(SOURCES)
set(FILE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../src/php_octra.h")
if(EXISTS "${FILE_PATH}")
  list(APPEND SOURCES "${FILE_PATH}")
endif()
set(FILE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../src/octra_php_wrap.cpp")
if(EXISTS "${FILE_PATH}")
  list(APPEND SOURCES "${FILE_PATH}")
endif()

message(STATUS "COMBINED_FILES: ${SOURCES}")

add_library(${PROJECT_NAME} SHARED ${SOURCES})

set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
)

execute_process(
  COMMAND php-config --includes
  OUTPUT_VARIABLE PHP_INCLUDES
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Split the output string into a list
string(REPLACE " " ";" PHP_INCLUDES_LIST ${PHP_INCLUDES})

# Remove the '-I' prefix from each path
foreach(PHP_INCLUDE_PATH ${PHP_INCLUDES_LIST})
  string(REPLACE "-I" "" PHP_INCLUDE_PATH_CLEANED ${PHP_INCLUDE_PATH})
  list(APPEND PHP_INCLUDE_LIST_CLEANED ${PHP_INCLUDE_PATH_CLEANED})
endforeach()

message(STATUS "PHP includes cleaned: ${PHP_INCLUDE_LIST_CLEANED}") # For debugging

set(SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../src")

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
         $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>
  PRIVATE $<BUILD_INTERFACE:${SOURCE_DIR}> ${PHP_INCLUDE_LIST_CLEANED}
)

target_link_libraries(${PROJECT_NAME} PRIVATE octra::octra)

string(TOLOWER ${PROJECT_NAME}/version.h VERSION_HEADER_LOCATION)

packageProject(
  NAME ${PROJECT_NAME}
  VERSION 0.0.1
  NAMESPACE ${PROJECT_NAME}
  BINARY_DIR ${PROJECT_BINARY_DIR}
  INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include
  INCLUDE_DESTINATION include/${PROJECT_NAME}-${PROJECT_VERSION} VERSION_HEADER_FILE
                                                                 ${VERSION_HEADER_LOCATION}
  COMPATIBILITY SameMajorVersion
)
