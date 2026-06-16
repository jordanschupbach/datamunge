# Define the target system
set(CMAKE_SYSTEM_NAME "Linux")

# Set compilers
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set warnings -Wall -Werror -Wall -Werror
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -g -fprofile-arcs -ftest-coverage")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g -fprofile-arcs -ftest-coverage")
