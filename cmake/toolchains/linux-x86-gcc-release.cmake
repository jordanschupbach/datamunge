# Define the target system
set(CMAKE_SYSTEM_NAME "Linux")

# Set compilers
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set warnings
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g0 -flto")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g0 -flto")
