#include <datamunge/algorithms/cubic_spline.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
namespace {
bool close(double actual,double expected) {
  return std::isfinite(actual) && std::fabs(actual-expected)<=5e-11*(1+std::fabs(expected));
}
}
int main() {
  std::size_t n,cases=0,queries=0;
  while(std::cin>>n) {
    if(n<2 || n>1000)return 1;
    std::vector<double> x(n),y(n);for(auto& v:x)if(!(std::cin>>v))return 1;for(auto& v:y)if(!(std::cin>>v))return 1;
    const auto trace=datamunge::algorithms::natural_cubic_spline_trace(x,y);
    const datamunge::algorithms::PreparedCubicSpline p(trace.spline);
    for(std::size_t i=0;i+1<n;++i)for(const auto* values:{&trace.spline.a,&trace.spline.b,&trace.spline.c,&trace.spline.d}) {
      double expected=0;if(!(std::cin>>expected))return 1;if(!close((*values)[i],expected))return 2;
    }
    for(const auto value:trace.moments) {double expected=0;if(!(std::cin>>expected))return 1;if(!close(value,expected))return 3;}
    std::size_t count=0;if(!(std::cin>>count))return 1;
    for(std::size_t i=0;i<count;++i) {
      double query,expected;unsigned order;std::cin>>query>>order>>expected;
      if(!std::cin || !close(p.evaluate(query,order),expected))return 4;
      if(order==0 && !close(datamunge::algorithms::cubic_spline_eval(trace.spline,query),expected))return 5;
      ++queries;
    }
    ++cases;
  }
  if(!std::cin.eof() || !cases)return 6;
  std::cout<<"Exact Fraction full-constraint oracle: "<<cases<<" fits, "<<queries<<" value/derivative controls passed\n";
}
