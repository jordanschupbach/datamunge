set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86)

find_program(CMAKE_C_COMPILER clang)
find_program(CMAKE_CXX_COMPILER clang++)
find_program(CMAKE_MAKE_PROGRAM make)

if(NOT CMAKE_C_COMPILER)
  message(
    FATAL_ERROR
      "Clang C compiler not found (are you using the proper toolchain file for your system?)"
  )
endif()

if(NOT CMAKE_CXX_COMPILER)
  message(
    FATAL_ERROR
      "Clang C++ compiler not found (are you using the proper toolchain file for your system?)"
  )
endif()

if(NOT CMAKE_CXX_COMPILER)
  message(
    FATAL_ERROR "Make program not found  (are you using the proper toolchain file for your system?)"
  )
endif()

execute_process(COMMAND ${CMAKE_C_COMPILER} --version OUTPUT_VARIABLE C_COMPILER_VERSION)
message(STATUS "C Compiler: ${CMAKE_C_COMPILER} (version: ${C_COMPILER_VERSION})")

execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE CXX_COMPILER_VERSION)
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER} (version: ${CXX_COMPILER_VERSION})")
