cmake_minimum_required(VERSION 3.21)

project(Documentation LANGUAGES NONE)

# XML extraction needs only Doxygen. Executable website reports additionally use
# the installed library supplied by the cpp shell (or an explicit pkg-config override).
find_package(Doxygen REQUIRED)
get_filename_component(DOXYGEN_PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

# Read canonical metadata without configuring the core's dependencies. This also
# follows the template's rename script when it changes the root project.
file(READ "${DOXYGEN_PROJECT_ROOT}/CMakeLists.txt" root_cmake)
if(NOT root_cmake MATCHES "project\\([ \t\r\n]*([^ \t\r\n)]+)[ \t\r\n]+VERSION[ \t\r\n]+([^ \t\r\n)]+)")
  message(FATAL_ERROR "Cannot read the root project's name and version")
endif()
set(DOXYGEN_PROJECT_NAME "${CMAKE_MATCH_1}")
set(DOXYGEN_PROJECT_VERSION "${CMAKE_MATCH_2}")

# Doxygen reads the headers' documentation into XML (GenerateDocsXml, also what
# the documentation audits read); the website is the project's own, built from
# that XML, the mepml reports and the Markdown guides by scripts/website/build.py
# (configured by docs/website.toml) into ${CMAKE_CURRENT_BINARY_DIR}/site.
set(DOXYGEN_GENERATE_HTML NO)
set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/audit")
configure_file(Doxyfile "Doxyfile.audit" @ONLY)
add_custom_target(
  GenerateDocsXml
  COMMAND ${CMAKE_COMMAND} -E remove_directory "${DOXYGEN_OUTPUT_DIRECTORY}/xml"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DOXYGEN_OUTPUT_DIRECTORY}"
  COMMAND Doxygen::doxygen "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.audit"
  WORKING_DIRECTORY "${DOXYGEN_PROJECT_ROOT}"
  COMMENT "Reading the headers' documentation with Doxygen (XML)"
  VERBATIM
)

# The mepml reports (docs/mepml/examples/**/*.mepml) are pages of the site. With
# DOCS_MEPML_RUN (and a mep that has `mep-mepml run`), mep first runs every report
# that changed since its last run -- the editor's C-c C-c, headless -- writing its
# results and figures back into the report; each run's per-block outcome is kept
# in DOCS_MEPML_STATUS_DIR and shown on the site's "Report build status" page, and
# the build fails if a block failed (unless DOCS_MEPML_ALLOW_FAILURES). mep's HTML
# export of each report is what the site shows. Without mep-mepml the site is
# built without the reports.
option(DOCS_MEPML_RUN "Run the mepml reports' code blocks with mep when building the docs" ON)
option(DOCS_MEPML_ALLOW_FAILURES "Build the docs even when a report's code block fails" OFF)
option(DOCS_MEPML_FORCE "Rerun every mepml report, not only the ones that changed" OFF)
set(DOCS_MEPML_TIMEOUT 1800 CACHE STRING "Longest a report's code block may run, in seconds (0: no limit)")
set(DOCS_MEPML_JOBS 0 CACHE STRING "How many reports mep runs at once (0: one per core, as memory allows)")
# Nearly all of a C++ block's time is parsing <dm/dm.hpp>: with clang++ on PATH the
# blocks that include it compile against a precompiled dm.hpp, about six times
# faster (scripts/website/pch_cxx.py); everything else still goes to g++.
option(DOCS_MEPML_PCH "Compile the reports' C++ blocks with clang++ and a precompiled dm.hpp when possible" ON)
set(DOCS_MEPML_PKG_CONFIG_PATH "" CACHE STRING
    "PKG_CONFIG_PATH for the reports' blocks (which find datamunge with pkg-config); empty: the caller's")
set(DOCS_MEPML_STATUS_DIR "${CMAKE_CURRENT_BINARY_DIR}/mepml-status" CACHE PATH
    "Where each report's last run status is kept between builds")
set(DOCS_SITE_DIR "${CMAKE_CURRENT_BINARY_DIR}/site")

# The shell's python3, looked up on every configure (not cached): the dev shell's
# Python environment changes when its packages do.
find_program(DOCS_PYTHON NAMES python3 python NO_CACHE REQUIRED)
execute_process(COMMAND "${DOCS_PYTHON}" -c "import jinja2, markdown_it, pygments"
                RESULT_VARIABLE docs_python_rc OUTPUT_QUIET ERROR_QUIET)
if(NOT docs_python_rc EQUAL 0)
  message(FATAL_ERROR "${DOCS_PYTHON} lacks jinja2, markdown-it-py or pygments, which the website needs; "
                      "configure inside `nix develop .#cpp-tools`")
endif()
find_program(MEP_MEPML_EXECUTABLE mep-mepml)
set(site_args --xml "${DOXYGEN_OUTPUT_DIRECTORY}/xml" --out "${DOCS_SITE_DIR}"
              --status-dir "${DOCS_MEPML_STATUS_DIR}")
if(MEP_MEPML_EXECUTABLE)
  list(APPEND site_args --mep-mepml "${MEP_MEPML_EXECUTABLE}")
  if(DOCS_MEPML_RUN)
    # An older mep can export but not run blocks: then the stored results are used.
    execute_process(COMMAND "${MEP_MEPML_EXECUTABLE}" run --help RESULT_VARIABLE mepml_run_rc
                    OUTPUT_QUIET ERROR_QUIET)
    if(mepml_run_rc EQUAL 0)
      list(APPEND site_args --run --timeout "${DOCS_MEPML_TIMEOUT}" --jobs "${DOCS_MEPML_JOBS}")
      if(NOT DOCS_MEPML_PCH)
        list(APPEND site_args --no-pch)
      endif()
      if(DOCS_MEPML_FORCE)
        list(APPEND site_args --force)
      endif()
    else()
      message(WARNING "${MEP_MEPML_EXECUTABLE} cannot run code blocks (no `mep-mepml run`; update mep): "
                      "the site uses the results stored in the mepml reports")
    endif()
  endif()
else()
  message(STATUS "mep-mepml not found: building the website without the mepml reports")
  list(APPEND site_args --no-reports)
endif()
if(DOCS_MEPML_ALLOW_FAILURES)
  list(APPEND site_args --allow-failures)
endif()
set(site_env)
if(DOCS_MEPML_PKG_CONFIG_PATH)
  set(site_env "PKG_CONFIG_PATH=${DOCS_MEPML_PKG_CONFIG_PATH}")
endif()
add_custom_target(
  Website
  COMMAND ${CMAKE_COMMAND} -E env ${site_env}
          "${DOCS_PYTHON}" "${DOXYGEN_PROJECT_ROOT}/scripts/website/build.py" ${site_args}
  WORKING_DIRECTORY "${DOXYGEN_PROJECT_ROOT}"
  COMMENT "Building the website (running changed mepml reports with mep) in ${DOCS_SITE_DIR}"
  USES_TERMINAL
  VERBATIM
)
add_dependencies(Website GenerateDocsXml)
# The documentation, by its older name.
add_custom_target(GenerateDocs)
add_dependencies(GenerateDocs Website)
