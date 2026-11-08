/// @file diffie_hellman.cpp
/// @brief Toy arithmetic exchange composition and bounded exact prime-field power cycles.
#include <datamunge/algorithms/diffie_hellman.hpp>
#include <datamunge/algorithms/miller_rabin.hpp>
#include <stdexcept>
namespace datamunge::algorithms {
std::uint64_t dh_public_key(std::uint64_t g,std::uint64_t secret,std::uint64_t p){return mod_pow(g,secret,p);}
std::uint64_t dh_shared_secret(std::uint64_t peer_public,std::uint64_t secret,std::uint64_t p){return mod_pow(peer_public,secret,p);}
DhExchange dh_exchange(std::uint64_t g,std::uint64_t a,std::uint64_t b,std::uint64_t p) {
 const auto A=dh_public_key(g,a,p),B=dh_public_key(g,b,p);
 return {g,p,A,B,dh_shared_secret(B,a,p),dh_shared_secret(A,b,p)};
}
DhPowerCycle dh_power_cycle(std::uint64_t g,std::uint64_t p,std::size_t max_order) {
 if(!is_probable_prime(p) || g==0 || g>=p)throw std::invalid_argument("dh_power_cycle: prime p and 1<=g<p required");
 DhPowerCycle cycle;cycle.modulus=p;cycle.generator=g;std::uint64_t value=1;
 do {
  if((max_order && cycle.values.size()>=max_order) || cycle.values.size()==cycle.values.max_size())throw std::length_error("dh_power_cycle: actual order cap or capacity");
  cycle.values.push_back(value);value=mul_mod(value,g,p);
 }while(value!=1);
 cycle.order=cycle.values.size();return cycle;
}
} // namespace datamunge::algorithms
