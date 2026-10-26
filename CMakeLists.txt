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

option(DATAMUNGE_ENABLE_ARROW "Enable Apache Arrow integration when available" ON)
option(DATAMUNGE_USE_BLAS "Use the platform BLAS (Accelerate or OpenBLAS) for nn GEMM when found" ON)
option(DATAMUNGE_USE_LAPACK "Use LAPACK (Accelerate, OpenBLAS's or another) for nn linear algebra when found" ON)

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
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")
target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>
)

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

# datamunge::nn kernels use a std::thread pool and, when available, a CBLAS
# GEMM (Accelerate on macOS, OpenBLAS or another CBLAS elsewhere). Without
# one they fall back to a built-in blocked GEMM.
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
        message(STATUS "datamunge::nn GEMM: built-in (no CBLAS found)")
      endif()
    endif()
  endif()
endif()

# datamunge::nn linear algebra (SVD, eigen, QR, Cholesky, LU, eig) calls
# LAPACK for larger matrices when available; the built-in kernels remain for
# small matrices and as the fallback. On Linux OpenBLAS usually bundles LAPACK;
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
      endif()
    endif()
    if(NOT DATAMUNGE_LAPACK_BACKEND)
      find_package(LAPACK QUIET)
      if(LAPACK_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${LAPACK_LIBRARIES})
        target_compile_definitions(${PROJECT_NAME} PRIVATE DATAMUNGE_HAVE_LAPACK=1)
        set(DATAMUNGE_LAPACK_BACKEND "${LAPACK_LIBRARIES}")
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
