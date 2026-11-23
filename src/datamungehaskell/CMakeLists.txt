cmake_minimum_required(VERSION 3.16)

project(
  datamungehaskell
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)

# The sources come from Rosetta (just prebuild-haskell, scripts/prebuild_bindings.py haskell):
# src/datamunge_haskell_wrap*.cpp (the C ABI library) and the GHC modules haskell/Datamunge.hs and
# haskell/Datamunge/Rosetta.hs beside this file (Haddock), generated from rosetta/bindings.hpp and
# rosetta/policy.json.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_haskell_wrap*.cpp")
if(NOT GENERATED_SOURCES)
  message(FATAL_ERROR "No generated Haskell binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-haskell`")
endif()

get_target_property(DATAMUNGE_INSTALLED_INCLUDE_DIRS datamunge::datamunge INTERFACE_INCLUDE_DIRECTORIES)
if(NOT DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  get_filename_component(DATAMUNGE_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH)
  set(DATAMUNGE_INSTALLED_INCLUDE_DIRS "${DATAMUNGE_REPO_INCLUDE_DIR}")
endif()

# The GHC package datamunge links this library (extra-libraries in datamunge.cabal).
add_library(datamunge_haskell SHARED ${GENERATED_SOURCES})
set_target_properties(
  datamunge_haskell
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             OUTPUT_NAME "datamunge_haskell"
)

# The generated sources include datamungehaskell/rosetta/bindings.hpp relative to src/.
target_include_directories(datamunge_haskell PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}")
target_link_libraries(datamunge_haskell PRIVATE datamunge::datamunge)

install(TARGETS datamunge_haskell LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")

# The GHC package datamunge (datamunge.cabal, built with the Cabal library GHC ships), registered in
# <prefix>/lib/haskell/package.db at install. Programs compile with
# `ghc -threaded -package-db <prefix>/lib/haskell/package.db -package datamunge program.hs
#  -optl-Wl,-rpath,<prefix>/lib`. Its prefix is CMAKE_INSTALL_PREFIX at configure time.
find_program(DATAMUNGE_GHC ghc HINTS "$ENV{DATAMUNGE_GHC_ROOT}/bin" REQUIRED)
get_filename_component(ghc_bin "${DATAMUNGE_GHC}" DIRECTORY)
find_program(DATAMUNGE_GHC_PKG NAMES ghc-pkg HINTS "${ghc_bin}" REQUIRED NO_DEFAULT_PATH)
set(setup "${CMAKE_CURRENT_BINARY_DIR}/setup")
set(dist "${CMAKE_CURRENT_BINARY_DIR}/dist")
set(hs_dir "${CMAKE_CURRENT_LIST_DIR}/haskell")
set(package_lib "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
add_custom_command(
  OUTPUT "${setup}"
  COMMAND "${DATAMUNGE_GHC}" -v0 -O0 -outputdir "${CMAKE_CURRENT_BINARY_DIR}/setup-obj" -o "${setup}" "${CMAKE_CURRENT_LIST_DIR}/Setup.hs"
  DEPENDS "${CMAKE_CURRENT_LIST_DIR}/Setup.hs"
  COMMENT "Compiling the Cabal setup program"
  VERBATIM)
add_custom_command(
  OUTPUT "${dist}/stamp"
  COMMAND "${CMAKE_COMMAND}" -E chdir "${CMAKE_CURRENT_LIST_DIR}" "${setup}" configure -v0 "--builddir=${dist}"
          "--with-compiler=${DATAMUNGE_GHC}" "--with-hc-pkg=${DATAMUNGE_GHC_PKG}" "--prefix=${CMAKE_INSTALL_PREFIX}"
          "--libdir=${package_lib}/haskell" --disable-shared --disable-library-profiling
          "--extra-lib-dirs=${package_lib}" "--extra-lib-dirs=$<TARGET_FILE_DIR:datamunge_haskell>"
  COMMAND "${CMAKE_COMMAND}" -E chdir "${CMAKE_CURRENT_LIST_DIR}" "${setup}" build -v0 "--builddir=${dist}"
  COMMAND "${CMAKE_COMMAND}" -E touch "${dist}/stamp"
  DEPENDS "${setup}" "${CMAKE_CURRENT_LIST_DIR}/datamunge.cabal" "${hs_dir}/Datamunge.hs" "${hs_dir}/Datamunge/Rosetta.hs" datamunge_haskell
  COMMENT "Building the GHC package datamunge"
  VERBATIM)
add_custom_target(datamunge_haskell_package ALL DEPENDS "${dist}/stamp")

install(CODE "
  set(db \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/haskell/package.db\")
  if(NOT \"\${CMAKE_INSTALL_PREFIX}\" STREQUAL \"${CMAKE_INSTALL_PREFIX}\")
    message(FATAL_ERROR \"the GHC package was configured for ${CMAKE_INSTALL_PREFIX}: configure with -DCMAKE_INSTALL_PREFIX=\${CMAKE_INSTALL_PREFIX}\")
  endif()
  execute_process(COMMAND \"${setup}\" copy -v0 \"--builddir=${dist}\" WORKING_DIRECTORY \"${CMAKE_CURRENT_LIST_DIR}\" COMMAND_ERROR_IS_FATAL ANY)
  execute_process(COMMAND \"${setup}\" register -v0 \"--builddir=${dist}\" \"--gen-pkg-config=${dist}/datamunge.conf\"
                  WORKING_DIRECTORY \"${CMAKE_CURRENT_LIST_DIR}\" COMMAND_ERROR_IS_FATAL ANY)
  file(REMOVE_RECURSE \"\${db}\")
  execute_process(COMMAND \"${DATAMUNGE_GHC_PKG}\" init \"\${db}\" COMMAND_ERROR_IS_FATAL ANY)
  execute_process(COMMAND \"${DATAMUNGE_GHC_PKG}\" -v0 \"--package-db=\${db}\" register \"${dist}/datamunge.conf\" COMMAND_ERROR_IS_FATAL ANY)
")
