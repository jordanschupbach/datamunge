#include <datamunge/algorithms/extended_euclidean.hpp>
#include <datamunge/algorithms/integer_arithmetic.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
int main(){using namespace datamunge::algorithms;char op;std::int64_t a,b;while(std::cin>>op>>a>>b){
 try{EuclidTrace t;std::int64_t inverse=0;if(op=='i'){auto s=modular_inverse_trace(a,b);t=std::move(s.euclid);inverse=s.inverse;}else t=extended_gcd_trace(a,b);
 const auto& c=t.outcome.certificate;std::cout<<"ok "<<c.gcd<<' '<<c.x<<' '<<c.y<<' '<<t.outcome.work.divisions<<' '<<t.outcome.work.coefficient_updates<<' '<<t.outcome.work.sign_normalized<<' '<<t.captured_entries<<' '<<inverse;
 for(const auto& r:t.rows)std::cout<<'|'<<int128_text(r.remainder)<<','<<int128_text(r.s)<<','<<int128_text(r.t)<<','<<int128_text(r.quotient)<<','<<r.divided;std::cout<<'\n';}
 catch(const std::invalid_argument&){std::cout<<"invalid\n";}
 }return std::cin.eof()?0:2;}
