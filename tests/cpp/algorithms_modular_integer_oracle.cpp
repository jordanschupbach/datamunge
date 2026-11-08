#include <datamunge/algorithms/diffie_hellman.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
int main(){
 std::string op;std::uint64_t a,b,m;
 while(std::cin>>op>>a>>b>>m){
  try{std::cout<<(op=="mul"?datamunge::algorithms::mul_mod(a,b,m):datamunge::algorithms::mod_pow(a,b,m))<<'\n';}
  catch(const std::invalid_argument&){std::cout<<"invalid\n";}
 }
 return std::cin.eof()?0:2;
}
