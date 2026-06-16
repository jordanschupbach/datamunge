%module(directors="1") Datamunge

%include <stdint.i>
%include <std_vector.i>
%include <std_pair.i>

%template(DPair) std::pair<double, double>;
%template(DVector) std::vector<double>;

%feature("director") datamunge::Callback;

%{
#include "datamunge/datamunge.hpp"
%}

%include <datamunge/datamunge.hpp>
