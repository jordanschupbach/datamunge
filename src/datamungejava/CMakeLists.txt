cmake_minimum_required(VERSION 3.16)

project(
  datamungejava
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(Java 17 REQUIRED COMPONENTS Development)
# JNI headers from the JDK whose javac compiles the sources (as Rosetta's cmake/JavaExample.cmake).
get_filename_component(DATAMUNGE_JAVAC "${Java_JAVAC_EXECUTABLE}" REALPATH)
get_filename_component(DATAMUNGE_JAVA_BIN "${DATAMUNGE_JAVAC}" DIRECTORY)
get_filename_component(JAVA_HOME "${DATAMUNGE_JAVA_BIN}" DIRECTORY)
find_package(JNI REQUIRED COMPONENTS JVM)
include(UseJava)

# The sources come from Rosetta (just prebuild-java, scripts/prebuild_bindings.py java):
# src/datamunge_java_wrap*.cpp (the JNI library libdatamunge_java) and the Java package datamunge
# under java/ beside this file (datamunge/*.java, datamunge/data/Data.java), one documented class
# per file, generated from rosetta/bindings.hpp and rosetta/policy.json. The classes load the
# library from -Ddatamunge.library=<file>, else DATAMUNGE_JAVA_LIBRARY, else
# System.loadLibrary("datamunge_java") (java.library.path).
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
set(DATAMUNGE_JAVA_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/java")
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_java_wrap*.cpp")
file(GLOB_RECURSE JAVA_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_JAVA_SOURCE_ROOT}/datamunge/*.java")
if(NOT GENERATED_SOURCES OR NOT JAVA_SOURCES)
  message(
    FATAL_ERROR "No generated Java binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-java`"
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

# libdatamunge_java (libdatamunge is the C++ library itself).
add_library(datamunge_java MODULE ${GENERATED_SOURCES})
set_target_properties(
  datamunge_java
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             OUTPUT_NAME "datamunge_java"
             LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
)
# The generated sources include datamungejava/rosetta/bindings.hpp relative to src/.
target_include_directories(
  datamunge_java PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}"
)
target_include_directories(datamunge_java SYSTEM PRIVATE ${JNI_INCLUDE_DIRS})
target_link_libraries(datamunge_java PRIVATE datamunge::datamunge Threads::Threads)

# datamunge.jar: the compiled package (javac --release 17).
set(CMAKE_JAVA_COMPILE_FLAGS --release 17 -encoding UTF-8 -nowarn)
add_jar(
  datamunge_jar
  SOURCES ${JAVA_SOURCES}
  OUTPUT_NAME datamunge
  OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/share/java"
)

# <prefix>/lib/libdatamunge_java.so and <prefix>/share/java/datamunge.jar: run with
# -cp <prefix>/share/java/datamunge.jar and DATAMUNGE_JAVA_LIBRARY=<prefix>/lib/libdatamunge_java.so
# (the dev shells set it).
install(TARGETS datamunge_java LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install_jar(datamunge_jar DESTINATION "${CMAKE_INSTALL_DATADIR}/java")
