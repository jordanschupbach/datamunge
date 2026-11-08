#include <string>
#include <datamunge/geometry/orientation.hpp>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
int main(int argc,char** argv) {
    if(argc!=3)return 2;std::ifstream input(argv[1],std::ios::binary);std::size_t count=0;
    std::array<double,6> v;std::int8_t expected;
    while(input.read(reinterpret_cast<char*>(v.data()),v.size()*sizeof(v[0]))) {
        if(!input.read(reinterpret_cast<char*>(&expected),sizeof(expected)))return 3;
        const int actual=datamunge::geometry::orientation_sign({v[0],v[1]},{v[2],v[3]},{v[4],v[5]});
        if(actual!=expected){std::cerr<<"mismatch "<<count<<' '<<actual<<' '<<int(expected)<<'\n';return 1;}++count;
    }
    if(count!=std::stoull(argv[2]))return 4;
    std::cout<<"All "<<count<<" exact Fraction orientation controls match.\n";
}
