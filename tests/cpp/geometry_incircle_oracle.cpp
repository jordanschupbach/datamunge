#include <datamunge/geometry/incircle.hpp>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
int main(int argc,char** argv) {
  if(argc!=3)return 2;std::ifstream file(argv[1],std::ios::binary);const auto count=std::stoull(argv[2]);
  for(std::size_t i=0;i<count;++i) {
    std::array<double,8> v;std::int8_t expected;file.read(reinterpret_cast<char*>(v.data()),sizeof(v));file.read(reinterpret_cast<char*>(&expected),1);if(!file)return 3;
    const int actual=datamunge::geometry::incircle_sign({v[0],v[1]},{v[2],v[3]},{v[4],v[5]},{v[6],v[7]});if(actual!=expected) {std::cerr<<"incircle fraction mismatch at "<<i<<'\n';return 1;}
  }
  std::cout<<count<<" exact incircle Fraction controls passed\n";
}
