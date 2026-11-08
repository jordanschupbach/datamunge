#include <datamunge/algorithms/primality.hpp>
#include <cstdint>
#include <iostream>
int main(){std::uint64_t n,a;while(std::cin>>n>>a){const auto r=datamunge::algorithms::fermat_base_test(n,a);std::cout<<r.common_divisor<<' '<<r.residue<<' '<<r.passes<<'\n';}}
