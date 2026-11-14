cmake_minimum_required(VERSION 3.16)

project(
  datamungeruby
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)
find_package(Ruby 3.1 REQUIRED)

# The sources come from Rosetta (just prebuild-ruby, scripts/prebuild_bindings.py ruby):
# src/datamunge_ruby_wrap*.cpp (the native extension) and lib/datamunge.rb beside this file, which
# require_relative's the extension (datamunge/datamunge.<DLEXT>) and carries the YARD documentation,
# generated from rosetta/bindings.hpp and rosetta/policy.json.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_ruby_wrap*.cpp")
if(NOT GENERATED_SOURCES OR NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/lib/datamunge.rb")
  message(
    FATAL_ERROR
      "No generated Ruby binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-ruby`"
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

# The interpreter's extension suffix (so on Linux, bundle on macOS).
execute_process(
  COMMAND "${Ruby_EXECUTABLE}" -rrbconfig -e "print RbConfig::CONFIG['DLEXT']"
  OUTPUT_VARIABLE RUBY_DLEXT
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY
)

# The MRI extension datamunge/datamunge.<DLEXT> (Init_datamunge), in the build tree beside nothing
# else, so that lib/ under the build directory mirrors the installed layout.
add_library(datamunge_ruby MODULE ${GENERATED_SOURCES})
set_target_properties(
  datamunge_ruby
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             PREFIX ""
             OUTPUT_NAME "datamunge"
             SUFFIX ".${RUBY_DLEXT}"
             LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib/datamunge"
)
# The generated sources include datamungeruby/rosetta/bindings.hpp relative to src/.
target_include_directories(
  datamunge_ruby PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}"
)
target_include_directories(datamunge_ruby SYSTEM PRIVATE ${Ruby_INCLUDE_DIRS})
target_link_libraries(datamunge_ruby PRIVATE datamunge::datamunge ${Ruby_LIBRARIES})

# <prefix>/lib: datamunge.rb and datamunge/datamunge.<DLEXT>. `require 'datamunge'` with
# `ruby -I <prefix>/lib` or RUBYLIB=<prefix>/lib (the dev shells set it).
install(TARGETS datamunge_ruby LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/datamunge")
install(FILES "${CMAKE_CURRENT_LIST_DIR}/lib/datamunge.rb" DESTINATION "${CMAKE_INSTALL_LIBDIR}")
