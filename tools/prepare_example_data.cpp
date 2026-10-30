/// @file prepare_example_data.cpp
/// @brief Native command-line entrypoint for the installed report data adapters.
#include <datamunge/datasets/assets.hpp>
#include <iostream>
/// @brief Prepare one dataset, fetching missing original source files automatically.
/// @param argc Program name, adapter name and optional destination directory.
/// @param argv Exact command-line arguments; paths containing spaces are supported.
/// @return Zero on success, two for usage errors, one for data errors.
int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: datamunge-prepare-data ADAPTER [DIRECTORY]\n";
    return 2;
  }
  try {
    std::cout << datamunge::datasets::prepare_example_data(argv[1], argc == 3 ? argv[2] : "")
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
