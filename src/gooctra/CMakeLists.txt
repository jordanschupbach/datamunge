# if(NOT BUILD_GOLANG) return() endif()

message("Building Go library...")

cmake_minimum_required(VERSION 3.18)
if(POLICY CMP0122)
  cmake_policy(SET CMP0122 NEW)
endif()

project(gooctra VERSION 0.1)

# Copy source files into working directory
file(COPY ${PROJECT_SOURCE_DIR}/../../include/octra DESTINATION ${PROJECT_BINARY_DIR}/octra/)
file(COPY ${PROJECT_SOURCE_DIR}/../../src/octra DESTINATION ${PROJECT_BINARY_DIR}/octra/)

# Grab headers to put into go.mod later
file(GLOB_RECURSE C_HEADERS "${PROJECT_BINARY_DIR}/*.h")
file(GLOB_RECURSE CXX_HEADERS "${PROJECT_BINARY_DIR}/*.hpp")
file(GLOB_RECURSE C_SOURCES "${PROJECT_BINARY_DIR}/*.c")
file(GLOB_RECURSE CXX_SOURCES "${PROJECT_BINARY_DIR}/*.cpp")
set(OCTRA_SOURCES ${C_HEADERS} ${CXX_HEADERS} ${C_SOURCES} ${CXX_SOURCES})

# Copy source files into working directory and transform includes from system to local
foreach(filepath ${OCTRA_SOURCES})
  file(READ ${filepath} input_content)
  string(REGEX REPLACE "#include<octra/([^>]*)>" "#include \"octra/\\1\"" output_content
                       "${input_content}"
  )
  string(REGEX REPLACE "#include <octra/([^>]*)>" "#include \"octra/\\1\"" output_content
                       "${input_content}"
  )
  file(WRITE ${filepath} "${output_content}")
endforeach()

message("Building Go SWIG bindings...")

execute_process(
  COMMAND swig -go -c++ -intgosize 64 -o ${PROJECT_BINARY_DIR}/gooctra_wrap.cxx -outdir
          ${PROJECT_BINARY_DIR}/octra/gooctra ../swig_src/src/gooctra.i
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

# Create go.mod file
file(
  WRITE ${PROJECT_BINARY_DIR}/gooctra/go.mod
  "module octra

go 1.21
"
)

file(COPY ${PROJECT_BINARY_DIR}/gooctra/ DESTINATION ${PROJECT_SOURCE_DIR}/../../gooctra/)

# For now, we'll just note that Go bindings need special handling Go builds are typically done with
# 'go build' rather than cmake
message("Go SWIG interface generated. You'll need to run 'go build' separately.")
