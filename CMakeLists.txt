cmake_minimum_required(VERSION 3.10)

project(
  datamunge
  VERSION 0.0.1
  LANGUAGES CXX
)

# An unspecified single-configuration build should be usable for production and
# performance comparisons. Explicit Debug builds and multi-config generators
# retain their normal configuration selection.
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(DATAMUNGE_ENABLE_DATA_ASSETS "Enable native report dataset acquisition and preparation" ON)

option(DATAMUNGE_ENABLE_MUJOCO "Enable optional MuJoCo reference backend (native articulated physics is always available)" OFF)

option(DATAMUNGE_ENABLE_ARROW "Enable Apache Arrow integration when available" ON)
option(DATAMUNGE_USE_BLAS "Use an external CBLAS (Accelerate/OpenBLAS) for nn GEMM instead of the in-house kernels (comparison only)" OFF)
option(DATAMUNGE_USE_LAPACK "Use an external LAPACK (Accelerate, OpenBLAS's or another) for large nn decompositions instead of the built-in kernels" OFF)
option(DATAMUNGE_ENABLE_CUDA "Enable native CUDA Driver API compute (runtime NVIDIA driver required)" OFF)
option(DATAMUNGE_ENABLE_CUBLAS "cuBLAS comparison path for Context::matmul (in-house kernels are the default; CUDA toolkit required)" OFF)
option(DATAMUNGE_ENABLE_VULKAN "Enable native Vulkan compute (Vulkan 1.3 headers/loader required)" OFF)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

include(GNUInstallDirs)

include(cmake/CPM.cmake)
include(cmake/PackageProject.cmake)

set(DATAMUNGE_PACKAGE_DEPENDENCIES "")

# CPMAddPackage("gh:TheLartians/PackageProject.cmake@1.8.0")

# TODO: find somewhere proper for these two lines
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL ""
)
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

# Gather source files
file(GLOB_RECURSE c_headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
file(GLOB_RECURSE c_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/datamunge/*.c")
file(GLOB_RECURSE cxx_headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
file(GLOB_RECURSE cxx_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/datamunge/*.cpp")

# message(DEBUG "c_headers: ${c_headers}") message(DEBUG "c_sources: ${c_sources}") message(DEBUG
# "cxx_headers: ${cxx_headers}") message(DEBUG "cxx_sources: ${cxx_sources}")

# Add C shared library (TODO: add static library)
add_library(${PROJECT_NAME} SHARED ${c_headers} ${c_sources} ${cxx_headers} ${cxx_sources})
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 23)
if(DATAMUNGE_ENABLE_MUJOCO)
  # Only the headless C API is needed; upstream's viewer config also requires OpenGL.
  find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h)
  find_library(MUJOCO_LIBRARY mujoco)
  if(NOT MUJOCO_INCLUDE_DIR OR NOT MUJOCO_LIBRARY)
    message(FATAL_ERROR "DATAMUNGE_ENABLE_MUJOCO requires MuJoCo >=3.3 headers and library")
  endif()
  target_include_directories(${PROJECT_NAME} PRIVATE ${MUJOCO_INCLUDE_DIR})
  target_link_libraries(${PROJECT_NAME} PRIVATE ${MUJOCO_LIBRARY})
  target_compile_definitions(${PROJECT_NAME} PRIVATE DATAMUNGE_ENABLE_MUJOCO=1)
endif()
# CAD predicates and geometric tolerances require individually rounded operations.
# Preserve the source kernel's arithmetic on ARM as well as x86.
file(GLOB cad_kernel_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/datamunge/cad/*.cpp")
set_source_files_properties(${cad_kernel_sources} PROPERTIES COMPILE_OPTIONS
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-ffp-contract=off>;$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/fp:strict>")
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")

# The nn kernels never read floating-point exception flags or math errno. Under
# GCC's default trapping-math semantics a float compare may trap, so the
# compiler refuses to if-convert the selects in ReLU masks, pooling argmax and
# similar loops and they run as scalar branches (10-15x slower than the
# vectorised form). Results are bit-identical either way.
option(DATAMUNGE_NN_FAST_FP "Compile the nn kernels with -fno-trapping-math -fno-math-errno" ON)
if(DATAMUNGE_NN_FAST_FP)
  file(GLOB nn_kernel_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/datamunge/nn/*.cpp")
  set_source_files_properties(${nn_kernel_sources} PROPERTIES COMPILE_OPTIONS
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-fno-trapping-math;-fno-math-errno>")
endif()
# The in-house BLAS (src/datamunge/nn/blas_kernels.hpp) is compiled once per
# instruction set; blas.cpp picks a unit at run time from the CPU features, so
# the library itself stays buildable for the baseline ISA. The -m flags apply
# to those units only (never -march=native, which Nix strips anyway).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/datamunge/nn/blas_avx512.cpp" APPEND PROPERTY COMPILE_OPTIONS
    -mavx512f -mavx512vl -mavx512dq -mavx512bw -mfma -ffp-contract=fast)
  set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/datamunge/nn/blas_avx2.cpp" APPEND PROPERTY COMPILE_OPTIONS
    -mavx2 -mfma -ffp-contract=fast)
  set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/datamunge/nn/blas_generic.cpp" APPEND PROPERTY COMPILE_OPTIONS
    -ffp-contract=fast)
endif()
target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>
)

# Native dataset adapters are optional for minimal library consumers. The docs
# shell enables them explicitly; a disabled build keeps a diagnostic API stub.
if(DATAMUNGE_ENABLE_DATA_ASSETS)
  find_package(LibArchive REQUIRED)
  find_package(OpenSSL REQUIRED COMPONENTS Crypto)
  find_package(nlohmann_json 3.10 REQUIRED)
  find_package(HDF5 REQUIRED COMPONENTS CXX)
  target_compile_definitions(${PROJECT_NAME} PRIVATE DATAMUNGE_ENABLE_DATA_ASSETS=1)
  target_link_libraries(${PROJECT_NAME} PRIVATE LibArchive::LibArchive OpenSSL::Crypto nlohmann_json::nlohmann_json ${HDF5_CXX_LIBRARIES})
  target_include_directories(${PROJECT_NAME} PRIVATE ${HDF5_INCLUDE_DIRS})
else()
  list(FILTER cxx_sources EXCLUDE REGEX "/datasets/preparation/")
  get_target_property(_dm_sources ${PROJECT_NAME} SOURCES)
  list(FILTER _dm_sources EXCLUDE REGEX "/datasets/preparation/.*\\.cpp$")
  set_property(TARGET ${PROJECT_NAME} PROPERTY SOURCES ${_dm_sources})
endif()

# GPU support is explicit and optional; ordinary core builds require neither SDK.
if(DATAMUNGE_ENABLE_CUBLAS AND NOT DATAMUNGE_ENABLE_CUDA)
  message(FATAL_ERROR "DATAMUNGE_ENABLE_CUBLAS requires DATAMUNGE_ENABLE_CUDA=ON")
endif()
if(DATAMUNGE_ENABLE_CUDA)
  target_compile_definitions(${PROJECT_NAME} PRIVATE DATAMUNGE_ENABLE_CUDA=1)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_DL_LIBS})
endif()
if(DATAMUNGE_ENABLE_CUBLAS)
  if(CMAKE_VERSION VERSION_LESS 3.17)
    message(FATAL_ERROR "The cuBLAS adapter requires CMake 3.17 or newer")
  endif()
  find_package(CUDAToolkit 12 REQUIRED)
  target_compile_definitions(${PROJECT_NAME} PRIVATE DATAMUNGE_ENABLE_CUBLAS=1)
  target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cublas)
  message(STATUS "datamunge CUDA matrix adapter: cuBLAS ${CUDAToolkit_VERSION}")
endif()
if(DATAMUNGE_ENABLE_VULKAN)
  find_package(Vulkan 1.3 REQUIRED)
  target_compile_definitions(${PROJECT_NAME} PRIVATE DATAMUNGE_ENABLE_VULKAN=1)
  target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan)
endif()

if(DATAMUNGE_ENABLE_ARROW)
  find_package(Arrow CONFIG QUIET)
  if(Arrow_FOUND)
    if(TARGET Arrow::arrow_shared)
      target_link_libraries(${PROJECT_NAME} PUBLIC Arrow::arrow_shared)
    elseif(TARGET Arrow::arrow_static)
      target_link_libraries(${PROJECT_NAME} PUBLIC Arrow::arrow_static)
    elseif(TARGET Arrow::arrow)
      target_link_libraries(${PROJECT_NAME} PUBLIC Arrow::arrow)
    else()
      message(WARNING "Arrow was found, but no supported Arrow target was exported.")
    endif()
    target_compile_definitions(${PROJECT_NAME} PUBLIC DATAMUNGE_HAVE_ARROW=1)
    set(DATAMUNGE_PACKAGE_DEPENDENCIES "Arrow CONFIG")
  else()
    message(STATUS "Apache Arrow not found; building without Arrow integration.")
  endif()
endif()
find_package(X11 QUIET)
if(X11_FOUND)
  target_include_directories(${PROJECT_NAME} PRIVATE ${X11_INCLUDE_DIR})
  target_link_libraries(${PROJECT_NAME} PRIVATE ${X11_LIBRARIES})
  target_compile_definitions(${PROJECT_NAME} PRIVATE DATAMUNGE_HAVE_X11=1)
endif()

# datamunge::nn kernels use a std::thread pool and the in-house BLAS above.
# DATAMUNGE_USE_BLAS swaps the GEMM for a CBLAS (Accelerate on macOS, OpenBLAS
# or another CBLAS elsewhere) so the two can be compared; it is not needed.
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)
if(DATAMUNGE_USE_BLAS)
  if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
      target_link_libraries(${PROJECT_NAME} PRIVATE ${ACCELERATE_FRAMEWORK})
      target_compile_definitions(${PROJECT_NAME} PRIVATE DATAMUNGE_HAVE_CBLAS=1 ACCELERATE_NEW_LAPACK=1)
      message(STATUS "datamunge::nn GEMM: Accelerate")
    endif()
  else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_search_module(DATAMUNGE_OPENBLAS QUIET IMPORTED_TARGET openblas)
    endif()
    if(DATAMUNGE_OPENBLAS_FOUND)
      target_link_libraries(${PROJECT_NAME} PRIVATE PkgConfig::DATAMUNGE_OPENBLAS)
      target_compile_definitions(${PROJECT_NAME} PRIVATE DATAMUNGE_HAVE_CBLAS=1 DATAMUNGE_HAVE_OPENBLAS=1)
      message(STATUS "datamunge::nn GEMM: OpenBLAS")
    else()
      find_package(BLAS QUIET)
      find_path(DATAMUNGE_CBLAS_INCLUDE_DIR cblas.h)
      if(BLAS_FOUND AND DATAMUNGE_CBLAS_INCLUDE_DIR)
        target_include_directories(${PROJECT_NAME} PRIVATE ${DATAMUNGE_CBLAS_INCLUDE_DIR})
        target_link_libraries(${PROJECT_NAME} PRIVATE ${BLAS_LIBRARIES})
        target_compile_definitions(${PROJECT_NAME} PRIVATE DATAMUNGE_HAVE_CBLAS=1)
        message(STATUS "datamunge::nn GEMM: CBLAS (${BLAS_LIBRARIES})")
      else()
        message(STATUS "datamunge::nn GEMM: in-house (no CBLAS found)")
      endif()
    endif()
  endif()
endif()

# datamunge::nn linear algebra (SVD, eigen, QR, Cholesky, LU, eig) has built-in
# kernels for every size; DATAMUNGE_USE_LAPACK optionally sends larger matrices
# to an external LAPACK instead. On Linux OpenBLAS usually bundles LAPACK;
# otherwise any LAPACK found by FindLAPACK is used.
if(DATAMUNGE_USE_LAPACK)
  set(DATAMUNGE_LAPACK_BACKEND "")
  if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
      target_link_libraries(${PROJECT_NAME} PRIVATE ${ACCELERATE_FRAMEWORK})
      target_compile_definitions(${PROJECT_NAME} PRIVATE DATAMUNGE_HAVE_LAPACK=1 ACCELERATE_NEW_LAPACK=1)
      set(DATAMUNGE_LAPACK_BACKEND "Accelerate")
    endif()
  else()
    if(DATAMUNGE_OPENBLAS_FOUND)
      include(CheckCXXSourceCompiles)
      set(CMAKE_REQUIRED_LIBRARIES ${DATAMUNGE_OPENBLAS_LINK_LIBRARIES})
      check_cxx_source_compiles("extern \"C\" void dgesdd_(); int main() { dgesdd_(); return 0; }"
                                DATAMUNGE_OPENBLAS_HAS_LAPACK)
      unset(CMAKE_REQUIRED_LIBRARIES)
      if(DATAMUNGE_OPENBLAS_HAS_LAPACK)
        target_compile_definitions(${PROJECT_NAME} PRIVATE DATAMUNGE_HAVE_LAPACK=1)
        set(DATAMUNGE_LAPACK_BACKEND "OpenBLAS")
        set(DATAMUNGE_LAPACK_LINK ${DATAMUNGE_OPENBLAS_LINK_LIBRARIES})
      endif()
    endif()
    if(NOT DATAMUNGE_LAPACK_BACKEND)
      find_package(LAPACK QUIET)
      if(LAPACK_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${LAPACK_LIBRARIES})
        target_compile_definitions(${PROJECT_NAME} PRIVATE DATAMUNGE_HAVE_LAPACK=1)
        set(DATAMUNGE_LAPACK_BACKEND "${LAPACK_LIBRARIES}")
        set(DATAMUNGE_LAPACK_LINK ${LAPACK_LIBRARIES})
      endif()
    endif()
    # The integer width of the LAPACK entry points is part of their ABI: OpenBLAS
    # built with INTERFACE64 (nixpkgs' default on 64-bit platforms) takes 64-bit
    # integers. Ask the linked OpenBLAS; -DDATAMUNGE_LAPACK_ILP64=ON/OFF overrides
    # the probe (another ILP64 LAPACK such as MKL's, or a cross build).
    if(DATAMUNGE_LAPACK_BACKEND)
      if(DEFINED DATAMUNGE_LAPACK_ILP64)
        set(DATAMUNGE_LAPACK_IS_ILP64 ${DATAMUNGE_LAPACK_ILP64})
      elseif(NOT CMAKE_CROSSCOMPILING)
        include(CheckCXXSourceRuns)
        set(CMAKE_REQUIRED_LIBRARIES ${DATAMUNGE_LAPACK_LINK})
        check_cxx_source_runs([=[
#include <cstring>
extern "C" char* openblas_get_config();
int main() { return std::strstr(openblas_get_config(), "USE64BITINT") ? 0 : 1; }
]=] DATAMUNGE_LAPACK_OPENBLAS_ILP64)
        unset(CMAKE_REQUIRED_LIBRARIES)
        set(DATAMUNGE_LAPACK_IS_ILP64 ${DATAMUNGE_LAPACK_OPENBLAS_ILP64})
      endif()
      if(DATAMUNGE_LAPACK_IS_ILP64)
        target_compile_definitions(${PROJECT_NAME} PRIVATE DATAMUNGE_LAPACK_ILP64=1)
        string(APPEND DATAMUNGE_LAPACK_BACKEND ", 64-bit integers")
      endif()
    endif()
  endif()
  if(DATAMUNGE_LAPACK_BACKEND)
    message(STATUS "datamunge::nn linalg: LAPACK (${DATAMUNGE_LAPACK_BACKEND})")
  else()
    message(STATUS "datamunge::nn linalg: built-in kernels (no LAPACK found)")
  endif()
endif()

string(TOLOWER ${PROJECT_NAME}/version.h VERSION_HEADER_LOCATION)
# Package the C library
packageProject(
  NAME ${PROJECT_NAME}
  VERSION ${PROJECT_VERSION}
  NAMESPACE ${PROJECT_NAME}
  BINARY_DIR ${PROJECT_BINARY_DIR}
  INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include
  INCLUDE_DESTINATION include/${PROJECT_NAME}-${PROJECT_VERSION}
  VERSION_HEADER "${VERSION_HEADER_LOCATION}"
  COMPATIBILITY SameMajorVersion
  DEPENDENCIES "${DATAMUNGE_PACKAGE_DEPENDENCIES}"
)

set(PKG_CONFIG_VERSION "${PROJECT_VERSION}")
set(PKG_CONFIG_LIBS "-l${PROJECT_NAME}")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/datamunge.pc.in" "${CMAKE_CURRENT_BINARY_DIR}/datamunge.pc" @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/datamunge.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
)
install(FILES "${PROJECT_SOURCE_DIR}/third_party_licenses/mep-cad-LICENSE"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/${PROJECT_NAME}")

if(DATAMUNGE_ENABLE_DATA_ASSETS)
  add_executable(datamunge-prepare-data tools/prepare_example_data.cpp)
  target_compile_features(datamunge-prepare-data PRIVATE cxx_std_23)
  target_link_libraries(datamunge-prepare-data PRIVATE ${PROJECT_NAME})
  # The core installs in a versioned library directory; keep the CLI usable
  # after installation and when the complete installation prefix is moved.
  if(APPLE)
    set_property(TARGET datamunge-prepare-data PROPERTY INSTALL_RPATH
      "@loader_path/../${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}-${PROJECT_VERSION}")
  elseif(UNIX)
    set_property(TARGET datamunge-prepare-data PROPERTY INSTALL_RPATH
      "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}-${PROJECT_VERSION}")
  endif()
  install(TARGETS datamunge-prepare-data RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
