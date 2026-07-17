%module datamunge

// {{{ core

// {{{ stl

%include <stdint.i>
%include <std_vector.i>
%include <std_string.i>
%include <std_pair.i>

%ignore datamunge::dstruct::NullableColumn;
%ignore datamunge::dstruct::DataFrame;
%template(IPair) std::pair<int, int>;
%template(DPair) std::pair<double, double>;
%template(SPair) std::pair<std::string, std::string>;
%template(IVector) std::vector<int>;
%template(DVector) std::vector<double>;
%template(SizeVector) std::vector<size_t>;
%template(SVector) std::vector<std::string>;

// }}} stl

// }}} core

%{
  #include "datamunge/datamunge.hpp"
%}
%include "datamunge/datamunge.hpp"
