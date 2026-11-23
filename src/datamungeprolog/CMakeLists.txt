cmake_minimum_required(VERSION 3.16)

project(
  datamungeprolog
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)

# The sources come from Rosetta (just prebuild-prolog, scripts/prebuild_bindings.py prolog):
# src/datamunge_prolog_wrap*.cpp (the foreign library) and the SWI-Prolog module prolog/datamunge.pl
# beside this file (PlDoc), generated from rosetta/bindings.hpp and rosetta/policy.json.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_prolog_wrap*.cpp")
if(NOT GENERATED_SOURCES)
  message(FATAL_ERROR "No generated Prolog binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-prolog`")
endif()

get_target_property(DATAMUNGE_INSTALLED_INCLUDE_DIRS datamunge::datamunge INTERFACE_INCLUDE_DIRECTORIES)
if(NOT DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  get_filename_component(DATAMUNGE_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH)
  set(DATAMUNGE_INSTALLED_INCLUDE_DIRS "${DATAMUNGE_REPO_INCLUDE_DIR}")
endif()

# The module loads this library (DATAMUNGE_PROLOG_LIBRARY, beside it, lib/ or ../lib). It is compiled
# with SWI-Prolog's headers ($PLBASE/include) and does not link libswipl.
find_program(DATAMUNGE_SWIPL swipl HINTS "$ENV{DATAMUNGE_SWIPL_ROOT}/bin" REQUIRED)
execute_process(COMMAND "${DATAMUNGE_SWIPL}" --dump-runtime-variables OUTPUT_VARIABLE swipl_variables COMMAND_ERROR_IS_FATAL ANY)
string(REGEX MATCH "PLBASE=\"([^\"]*)\"" _ "${swipl_variables}")
set(swipl_include "${CMAKE_MATCH_1}/include")
add_library(datamunge_prolog SHARED ${GENERATED_SOURCES})
set_target_properties(
  datamunge_prolog
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             OUTPUT_NAME "datamunge_prolog"
)

# The generated sources include datamungeprolog/rosetta/bindings.hpp relative to src/.
target_include_directories(datamunge_prolog PRIVATE "${swipl_include}" ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}")
target_link_libraries(datamunge_prolog PRIVATE datamunge::datamunge)

install(TARGETS datamunge_prolog LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
# <prefix>/prolog/datamunge.pl finds <prefix>/lib/libdatamunge_prolog through ../lib.
install(FILES "${CMAKE_CURRENT_LIST_DIR}/prolog/datamunge.pl" DESTINATION prolog)
