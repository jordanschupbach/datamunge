# Define the target system set(CMAKE_SYSTEM_NAME "Linux")
#
# # Set compilers set(CMAKE_C_COMPILER "gcc") set(CMAKE_CXX_COMPILER "g++")

# set(CMAKE_C_COMPILER "cl.exe") set(CMAKE_CXX_COMPILER "cl.exe")

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set warnings set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror  -fPIC -g -fprofile-arcs
# -ftest-coverage") set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -fPIC -g -fprofile-arcs
# -ftest-coverage")
