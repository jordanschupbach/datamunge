// Shared thin language adapter. Physics and rollout semantics remain in the C++ core.
%{
#include <datamunge/sim/tasks.hpp>
%}
namespace std { typedef ::uint8_t uint8_t; }
%naturalvar;
#ifdef SWIGPYTHON
%{
static PyObject* simulation_flags(const std::vector<std::uint8_t>& values) {
  PyObject* result=PyTuple_New(values.size());if(!result)return nullptr;
  for(std::size_t i=0;i<values.size();++i){PyObject* v=PyLong_FromLong(values[i]);if(!v){Py_DECREF(result);return nullptr;}PyTuple_SET_ITEM(result,i,v);}
  return result;
}
static PyObject* simulation_flags(const std::vector<std::uint8_t>* values) {return simulation_flags(*values);}
%}
%typemap(out) std::vector<std::uint8_t>, std::vector<std::uint8_t>*, const std::vector<std::uint8_t>& {
  $result=simulation_flags($1);
}
%typemap(typecheck,precedence=SWIG_TYPECHECK_POINTER) const std::vector<std::uint8_t>& { $1=PySequence_Check($input); }
%typemap(in) const std::vector<std::uint8_t>& (std::vector<std::uint8_t> values) {
  PyObject* seq=PySequence_Fast($input,"mask must be a sequence of binary integers");
  if(!seq)SWIG_exception_fail(SWIG_TypeError,"mask must be a sequence");
  const auto count=PySequence_Fast_GET_SIZE(seq);values.reserve(count);
  for(Py_ssize_t i=0;i<count;++i){int value=0;
    if(!SWIG_IsOK(SWIG_AsVal_int(PySequence_Fast_GET_ITEM(seq,i),&value))||value<0||value>1){Py_DECREF(seq);SWIG_exception_fail(SWIG_ValueError,"mask values must be zero or one");}
    values.push_back(static_cast<std::uint8_t>(value));
  }
  Py_DECREF(seq);$1 = &values$argnum;
}
#else
%{
static SEXP simulation_flags(const std::vector<std::uint8_t>& values) {
  SEXP result=PROTECT(Rf_allocVector(INTSXP,values.size()));
  for(std::size_t i=0;i<values.size();++i)INTEGER(result)[i]=values[i];
  UNPROTECT(1);return result;
}
static SEXP simulation_flags(const std::vector<std::uint8_t>* values) {return simulation_flags(*values);}
%}
%typemap(out) std::vector<std::uint8_t>, std::vector<std::uint8_t>*, const std::vector<std::uint8_t>& { $result=simulation_flags($1); }
%typemap(rtype) std::vector<std::uint8_t>, std::vector<std::uint8_t>*, const std::vector<std::uint8_t>& "integer";
%typemap(scoerceout) std::vector<std::uint8_t>, std::vector<std::uint8_t>*, const std::vector<std::uint8_t>& "";
%typemap(scoercein) const std::vector<std::uint8_t>& "$input=as.integer($input);";
%typemap(rtypecheck) const std::vector<std::uint8_t>& "is.numeric($arg)";
%typemap(in) const std::vector<std::uint8_t>& (std::vector<std::uint8_t> values) {
  if(!Rf_isInteger($input))SWIG_exception_fail(SWIG_TypeError,"mask must be binary integers");
  values.reserve(Rf_xlength($input));
  for(R_xlen_t i=0;i<Rf_xlength($input);++i){const int v=INTEGER($input)[i];if(v<0||v>1)SWIG_exception_fail(SWIG_ValueError,"mask values must be zero or one");values.push_back(static_cast<std::uint8_t>(v));}
  $1 = &values$argnum;
}
#endif
%typemap(freearg) const std::vector<std::uint8_t>& "";
// Language results are owned snapshots so temporary tasks and garbage collection
// cannot leave dangling references to C++'s reusable output arrays.
%typemap(out) const datamunge::sim::TaskOutput& {
  $result = SWIG_NewPointerObj(new datamunge::sim::TaskOutput(*$1),
                             $descriptor(datamunge::sim::TaskOutput *),SWIG_POINTER_OWN);
}
%ignore datamunge::sim::TaskBatch::world;
%ignore datamunge::sim::TaskBatch::TaskBatch;
%feature("docstring") datamunge::sim::TaskBatch::output "Return an owned snapshot of current native rollout outputs without advancing physics. Numeric arrays and binary end flags remain valid after the task is collected.";
%feature("docstring") datamunge::sim::TaskBatch::step "Advance selected native CPU task worlds with row-major actions. An empty mask selects all rows; otherwise use one binary integer per environment. Completed selected rows require explicit reset. Return an owned TaskOutput snapshot, including separate terminated/truncated flags and retained final observations.";
%feature("docstring") datamunge::sim::TaskBatch::reset "Restore selected native worlds and sample independent initial perturbations. An empty mask selects all rows; otherwise use one binary integer per environment. Random streams continue and final observations remain available. Return an owned TaskOutput snapshot.";
%immutable;
%include "datamunge/sim/tasks.hpp"
%mutable;
%newobject datamunge::sim::TaskBatch::create;
%feature("docstring") datamunge::sim::TaskBatch::create "Create an owned native CPU task batch. task is InvertedPendulum or Hopper; count and workers must be positive. seed defaults to zero and time_limit to 1000 successful episode steps. Physics runs in the canonical C++ engine. No rendering or external simulator is required.";
%extend datamunge::sim::TaskBatch {
  // Language callers select CPU worker count without needing a native Context wrapper.
  static datamunge::sim::TaskBatch* create(datamunge::sim::Task task, std::size_t count, std::size_t workers=1,
            std::uint64_t seed=0, std::uint64_t time_limit=1000) {
    return new datamunge::sim::TaskBatch(task,count,datamunge::compute::Context({},workers),seed,time_limit);
  }
}
