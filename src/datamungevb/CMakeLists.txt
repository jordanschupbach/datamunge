cmake_minimum_required(VERSION 3.16)

project(
  datamungevb
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)

# The sources come from Rosetta (just prebuild-vb, scripts/prebuild_bindings.py vb):
# src/datamunge_vb_wrap*.cpp (the native library) and the Visual Basic project beside this file
# (Datamunge.vb, Datamunge.vbproj), generated from rosetta/bindings.hpp and rosetta/policy.json.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_vb_wrap*.cpp")
if(NOT GENERATED_SOURCES)
  message(FATAL_ERROR "No generated Visual Basic binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-vb`")
endif()

get_target_property(DATAMUNGE_INSTALLED_INCLUDE_DIRS datamunge::datamunge INTERFACE_INCLUDE_DIRECTORIES)
if(NOT DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  get_filename_component(DATAMUNGE_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH)
  set(DATAMUNGE_INSTALLED_INCLUDE_DIRS "${DATAMUNGE_REPO_INCLUDE_DIR}")
endif()

# A plain shared library (no .NET headers): Datamunge.vb loads it with NativeLibrary. Its name is
# datamunge_vb so that it never resolves to libdatamunge, the C++ library itself.
add_library(datamunge_vb MODULE ${GENERATED_SOURCES})
set_target_properties(
  datamunge_vb
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             PREFIX ""
             OUTPUT_NAME "datamunge_vb"
)
if(APPLE)
  set_target_properties(datamunge_vb PROPERTIES SUFFIX ".dylib")
endif()

# The generated sources include datamungevb/rosetta/bindings.hpp relative to src/.
target_include_directories(datamunge_vb PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}")
target_link_libraries(datamunge_vb PRIVATE datamunge::datamunge)

# <prefix>/share/dotnet/DatamungeVB is the Visual Basic project (Datamunge.vbproj, Datamunge.vb) with the native
# library in lib/; `dotnet build` there puts Datamunge.dll and its XML documentation in bin/.
# Applications find the native library through DATAMUNGE_VB_LIBRARY (the dev shells set it).
set(DATAMUNGE_VB_DIR "${CMAKE_INSTALL_DATADIR}/dotnet/DatamungeVB")
install(TARGETS datamunge_vb DESTINATION "${DATAMUNGE_VB_DIR}/lib")
install(FILES "${CMAKE_CURRENT_LIST_DIR}/Datamunge.vbproj" "${CMAKE_CURRENT_LIST_DIR}/Datamunge.vb"
        DESTINATION "${DATAMUNGE_VB_DIR}")
