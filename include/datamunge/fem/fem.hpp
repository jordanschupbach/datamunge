#pragma once

// Convenience umbrella header -- basic finite element method (FEM) support: a 1D two-point
// boundary value problem solver (Dirichlet/Neumann/Robin BCs) with an additional transient
// (parabolic, heat-equation-style) mode, and a 2D Poisson equation solver on a triangular
// mesh, both using piecewise-linear Galerkin (P1) elements. Director-enabled
// ScalarField1D/ScalarField2D let bindings with SWIG director support supply a live
// source-term callback; every binding can instead use each solver's solve_builtin() fixed set
// of named source terms.

#include <datamunge/fem/fem1d.hpp>
#include <datamunge/fem/fem2d.hpp>
#include <datamunge/fem/mesh1d.hpp>
#include <datamunge/fem/mesh2d.hpp>
#include <datamunge/fem/source_function.hpp>
