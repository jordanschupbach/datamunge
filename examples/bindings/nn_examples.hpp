#pragma once
#include <string>
#include <vector>

// Optional example support, separate from the core library's public API.
namespace datamunge::examples {
std::vector<std::string> nn_example_names();
int run_nn_example(const std::string& name, const std::vector<std::string>& arguments);
}
