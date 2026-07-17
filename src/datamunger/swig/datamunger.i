%module(directors="1") datamunger

%include <stdint.i>
%include <std_vector.i>
%include <std_string.i>
%include <std_pair.i>

%ignore datamunge::dstruct::NullableColumn;
%ignore datamunge::dstruct::DataFrame;
%template(IPair) std::pair<int, int>;
%template(DPair) std::pair<double, double>;
%template(IVector) std::vector<int>;
%template(DVector) std::vector<double>;
%template(SizeVector) std::vector<size_t>;
%template(SVector) std::vector<std::string>;

%feature("director") datamunge::Callback;

%{
  #include "datamunge/datamunge.hpp"
%}
%include "datamunge/datamunge.hpp"
