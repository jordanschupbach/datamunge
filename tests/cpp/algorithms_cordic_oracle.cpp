#include <datamunge/algorithms/cordic.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
int main(int argc,char** argv) {
    if(argc!=3)return 2;std::ifstream input(argv[1],std::ios::binary);std::size_t count=0;
    std::array<double,7> v;std::int8_t flip;double max_model=0,max_moderate=0,max_original=0;
    while(input.read(reinterpret_cast<char*>(v.data()),v.size()*sizeof(v[0]))) {
        if(!input.read(reinterpret_cast<char*>(&flip),sizeof(flip)))return 3;
        const auto a=datamunge::algorithms::cordic_sincos_analyze(v[0],44);
        const double model=std::max(std::abs(a.result.sin-v[5]),std::abs(a.result.cos-v[6]));
        const double original=std::max(std::abs(a.result.sin-v[3]),std::abs(a.result.cos-v[4]));
        if(a.reduction.principal!=v[1]||a.reduction.reduced!=v[2]||a.reduction.flipped!=bool(flip)||model>5e-13){
            std::cerr<<"CORDIC Decimal mismatch at record "<<count<<", model error "<<model<<'\n';return 1;}
        max_model=std::max(max_model,model);max_original=std::max(max_original,original);
        if(std::abs(v[0])<=10){max_moderate=std::max(max_moderate,original);if(original>5e-13)return 4;}
        ++count;
    }
    if(count!=std::stoull(argv[2])||max_original<.1)return 5;
    std::cout<<"All "<<count<<" Decimal/Machin/Taylor controls match the rounded-period model.\n"
        <<"Maximum folded-model component error: "<<max_model<<"; moderate original-angle error: "<<max_moderate
        <<"; maximum original-angle error including huge inputs: "<<max_original<<".\n";
}
