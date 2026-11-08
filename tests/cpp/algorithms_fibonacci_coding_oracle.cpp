#include <datamunge/algorithms/fibonacci_coding.hpp>
#include <iostream>
#include <string>
int main(){char op;std::string token;while(std::cin>>op>>token){try{if(op=='e'){const auto b=datamunge::algorithms::fibonacci_encode({std::stoull(token)});for(bool x:b)std::cout<<x;std::cout<<'\n';}else{std::vector<bool>b;for(char c:token)b.push_back(c=='1');const auto v=datamunge::algorithms::fibonacci_decode(b);for(auto x:v)std::cout<<x<<' ';std::cout<<'\n';}}catch(const std::invalid_argument&){std::cout<<"invalid\n";}catch(const std::overflow_error&){std::cout<<"overflow\n";}}}
