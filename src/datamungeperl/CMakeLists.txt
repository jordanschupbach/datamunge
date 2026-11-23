cmake_minimum_required(VERSION 3.16)

project(
  datamungeperl
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)
find_program(PERL_EXECUTABLE perl REQUIRED)

# The sources come from Rosetta (just prebuild-perl, scripts/prebuild_bindings.py perl):
# src/datamunge_perl_wrap*.cpp (the C ABI library; the first part also carries the XS glue) and
# lib/Datamunge.pm beside this file (the package Datamunge, documented in POD), generated from
# rosetta/bindings.hpp and rosetta/policy.json.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_perl_wrap*.cpp")
if(NOT GENERATED_SOURCES)
  message(FATAL_ERROR "No generated Perl binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-perl`")
endif()

get_target_property(DATAMUNGE_INSTALLED_INCLUDE_DIRS datamunge::datamunge INTERFACE_INCLUDE_DIRECTORIES)
if(NOT DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  get_filename_component(DATAMUNGE_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH)
  set(DATAMUNGE_INSTALLED_INCLUDE_DIRS "${DATAMUNGE_REPO_INCLUDE_DIR}")
endif()

# The XS glue compiles against this perl's headers and flags.
execute_process(COMMAND "${PERL_EXECUTABLE}" -MExtUtils::Embed -e ccopts OUTPUT_VARIABLE PERL_CCOPTS
                OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
separate_arguments(PERL_CCOPTS UNIX_COMMAND "${PERL_CCOPTS}")

# Datamunge.pm loads this library with DynaLoader: from DATAMUNGE_PERL_LIBRARY, beside the module, or
# from auto/Datamunge/ under @INC (where it is installed).
add_library(datamunge_perl MODULE ${GENERATED_SOURCES})
set_target_properties(
  datamunge_perl
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             PREFIX "lib"
             OUTPUT_NAME "datamunge_perl"
)
target_compile_options(datamunge_perl PRIVATE ${PERL_CCOPTS})
# The generated sources include datamungeperl/rosetta/bindings.hpp relative to src/.
target_include_directories(datamunge_perl PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}")
target_link_libraries(datamunge_perl PRIVATE datamunge::datamunge)

# <prefix>/lib/perl5/Datamunge.pm and <prefix>/lib/perl5/auto/Datamunge/libdatamunge_perl.so:
# PERL5LIB=<prefix>/lib/perl5 is all a program needs.
install(FILES "${CMAKE_CURRENT_LIST_DIR}/lib/Datamunge.pm" DESTINATION "lib/perl5")
install(TARGETS datamunge_perl LIBRARY DESTINATION "lib/perl5/auto/Datamunge")
