/// @file nn_vqa_prepare.cpp
/// @brief Thin native VQA dataset preparation entrypoint.
#include <dm/dm.hpp>
#include <iostream>
int main(int argc,char** argv) { try { if(argc!=2) return 1;
 dm::datasets::prepare_example_data("vqa",argv[1]); return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;} }
