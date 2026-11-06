// Thin native CAD facade shared by Python and R; advanced kernel types stay C++.
%{
#include <datamunge/cad/view.hpp>
%}
%include <exception.i>
%rename(CadSolid) datamunge::cad::Solid;
%rename(CadScene) datamunge::cad::View;
%ignore datamunge::cad::Solid::Solid;
%ignore datamunge::cad::Solid::model;
%ignore datamunge::cad::Solid::body;
%ignore datamunge::cad::Solid::transformed;
%ignore datamunge::cad::Solid::tessellate;
%ignore datamunge::cad::Solid::boolean_with;
%ignore datamunge::cad::add_solid;
%ignore datamunge::cad::load_scene;
%ignore datamunge::cad::View::scene;
%exception {
  try { $action }
  catch (const std::out_of_range& error) { SWIG_exception_fail(SWIG_IndexError, error.what()); }
  catch (const std::invalid_argument& error) { SWIG_exception_fail(SWIG_ValueError, error.what()); }
  catch (const std::exception& error) { SWIG_exception_fail(SWIG_RuntimeError, error.what()); }
}
%include "datamunge/cad/solid.hpp"
%include "datamunge/cad/view.hpp"
%exception;
