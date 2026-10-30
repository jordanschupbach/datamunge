/// @file nn_style_prepare.cpp
/// @brief Compatibility CLI; native data acquisition and preparation live in the library.
#include <dm/dm.hpp>
#include <iostream>
int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("expected dataset directory");
    dm::datasets::prepare_example_data("style", argv[1]);
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
