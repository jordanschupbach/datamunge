# if(NOT BUILD_GOLANG) return() endif()

message("Building Go library...")

cmake_minimum_required(VERSION 3.18)
if(POLICY CMP0122)
  cmake_policy(SET CMP0122 NEW)
endif()

project(godatamunge VERSION 0.1)

# Copy source files into working directory
file(COPY ${PROJECT_SOURCE_DIR}/../../include/datamunge DESTINATION ${PROJECT_BINARY_DIR}/datamunge/)
file(COPY ${PROJECT_SOURCE_DIR}/../../src/datamunge DESTINATION ${PROJECT_BINARY_DIR}/datamunge/)

# Grab headers to put into go.mod later
file(GLOB_RECURSE C_HEADERS "${PROJECT_BINARY_DIR}/*.h")
file(GLOB_RECURSE CXX_HEADERS "${PROJECT_BINARY_DIR}/*.hpp")
file(GLOB_RECURSE C_SOURCES "${PROJECT_BINARY_DIR}/*.c")
file(GLOB_RECURSE CXX_SOURCES "${PROJECT_BINARY_DIR}/*.cpp")
set(DATAMUNGE_SOURCES ${C_HEADERS} ${CXX_HEADERS} ${C_SOURCES} ${CXX_SOURCES})

# Copy source files into working directory and transform includes from system to local
foreach(filepath ${DATAMUNGE_SOURCES})
  file(READ ${filepath} input_content)
  string(REGEX REPLACE "#include<datamunge/([^>]*)>" "#include \"datamunge/\\1\"" output_content
                       "${input_content}"
  )
  string(REGEX REPLACE "#include <datamunge/([^>]*)>" "#include \"datamunge/\\1\"" output_content
                       "${input_content}"
  )
  file(WRITE ${filepath} "${output_content}")
endforeach()

message("Building Go SWIG bindings...")

execute_process(
  COMMAND swig -go -c++ -intgosize 64 -o ${PROJECT_BINARY_DIR}/godatamunge_wrap.cxx -outdir
          ${PROJECT_BINARY_DIR}/datamunge/godatamunge ../swig_src/src/godatamunge.i
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

# Create go.mod file
file(
  WRITE ${PROJECT_BINARY_DIR}/godatamunge/go.mod
  "module datamunge

go 1.21
"
)

file(COPY ${PROJECT_BINARY_DIR}/godatamunge/ DESTINATION ${PROJECT_SOURCE_DIR}/../../godatamunge/)

# For now, we'll just note that Go bindings need special handling Go builds are typically done with
# 'go build' rather than cmake
message("Go SWIG interface generated. You'll need to run 'go build' separately.")
