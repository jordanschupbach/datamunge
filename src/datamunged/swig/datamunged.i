%module(directors="1") datamunge

%include <stdint.i>
%include <std_vector.i>
%include <std_string.i>
%include <std_pair.i>

%ignore datamunge::dstruct::NullableColumn;
%ignore datamunge::dstruct::DataFrame;
%template(DPair) std::pair<double, double>;
%template(DVector) std::vector<double>;
%template(IVector) std::vector<int>;
%template(SizeVector) std::vector<size_t>;
%template(SVector) std::vector<std::string>;

%feature("director") datamunge::Callback;

%{
#include "datamunge/datamunge.hpp"
%}

%include "datamunge/datamunge.hpp"
