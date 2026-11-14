cmake_minimum_required(VERSION 3.16)

project(
  datamungecsharp
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)

# The sources come from Rosetta (just prebuild-csharp, scripts/prebuild_bindings.py csharp):
# src/datamunge_csharp_wrap*.cpp (the native library) and the C# project beside this file
# (Datamunge.cs, Datamunge.csproj), generated from rosetta/bindings.hpp and rosetta/policy.json.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_csharp_wrap*.cpp")
if(NOT GENERATED_SOURCES)
  message(FATAL_ERROR "No generated C# binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-csharp`")
endif()

get_target_property(DATAMUNGE_INSTALLED_INCLUDE_DIRS datamunge::datamunge INTERFACE_INCLUDE_DIRECTORIES)
if(NOT DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  get_filename_component(DATAMUNGE_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH)
  set(DATAMUNGE_INSTALLED_INCLUDE_DIRS "${DATAMUNGE_REPO_INCLUDE_DIR}")
endif()

# A plain shared library (no .NET headers): Datamunge.cs loads it with NativeLibrary. Its name is
# datamunge_csharp so that it never resolves to libdatamunge, the C++ library itself.
add_library(datamunge_csharp MODULE ${GENERATED_SOURCES})
set_target_properties(
  datamunge_csharp
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             PREFIX ""
             OUTPUT_NAME "datamunge_csharp"
)
if(APPLE)
  set_target_properties(datamunge_csharp PROPERTIES SUFFIX ".dylib")
endif()

# The generated sources include datamungecsharp/rosetta/bindings.hpp relative to src/.
target_include_directories(datamunge_csharp PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}")
target_link_libraries(datamunge_csharp PRIVATE datamunge::datamunge)

# <prefix>/share/dotnet/Datamunge is the C# project (Datamunge.csproj, Datamunge.cs) with the native
# library in lib/; `dotnet build` there puts Datamunge.dll and its XML documentation in bin/.
# Applications find the native library through DATAMUNGE_CSHARP_LIBRARY (the dev shells set it).
set(DATAMUNGE_CSHARP_DIR "${CMAKE_INSTALL_DATADIR}/dotnet/Datamunge")
install(TARGETS datamunge_csharp DESTINATION "${DATAMUNGE_CSHARP_DIR}/lib")
install(FILES "${CMAKE_CURRENT_LIST_DIR}/Datamunge.csproj" "${CMAKE_CURRENT_LIST_DIR}/Datamunge.cs"
        DESTINATION "${DATAMUNGE_CSHARP_DIR}")
