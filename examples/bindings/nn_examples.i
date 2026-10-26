%module nn_examples
%include <std_string.i>
%include <std_vector.i>
%template(ExampleNames) std::vector<std::string>;
%include <exception.i>
%exception {
  try { $action }
  catch (const std::exception& e) { SWIG_exception(SWIG_RuntimeError, e.what()); }
}
%{
#include "nn_examples.hpp"
%}
#ifdef SWIGR
%rename(native_nn_example_names) datamunge::examples::nn_example_names;
#endif
%include "nn_examples.hpp"
