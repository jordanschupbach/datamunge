#include <datamunge/algorithms/integer_arithmetic.hpp>
#include <cstdint>
#include <iostream>
int main(){std::uint64_t hi,lo;while(std::cin>>hi>>lo)std::cout<<datamunge::algorithms::integer_sqrt_u128((static_cast<unsigned __int128>(hi)<<64)|lo)<<'\n';}
