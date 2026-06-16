%module(directors="1") Octra

%include <stdint.i>
%include <std_vector.i>
%include <std_pair.i>

%template(DPair) std::pair<double, double>;
%template(DVector) std::vector<double>;

%feature("director") octra::Callback;

%{
#include "octra/octra.hpp"
%}

%include <octra/octra.hpp>
