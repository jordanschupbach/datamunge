// BUG: so far there are issues with swig/php and shared_ptr
// Can posibly remedy by wrapping in another class?

%module datamunge

// NOTE: no working shared_ptr support in php swig?
%include <stdint.i>
%include <std_vector.i>
%include <std_string.i>
%include <std_pair.i>

%ignore datamunge::dstruct::NullableColumn;
%ignore datamunge::dstruct::DataFrame;

%template(IPair) std::pair<int, int>;
%template(DPair) std::pair<double, double>;
%template(DVector) std::vector<double>;
%template(IVector) std::vector<int>;
%template(SizeVector) std::vector<size_t>;

%{
  #include "datamunge/datamunge.hpp"
%}
%include "datamunge/datamunge.hpp"
