classdef FEM1D < datamunge.NativeObject
  % FEM1D  Solve steady or transient one-dimensional divergence-form problems with P1 Galerkin FEM. Models -(p*u')'+q*u=f and u_t-(p*u')'+q*u=f on a caller-supplied mesh. Requires at least two finite strictly increasing nodes, finite coefficient/ boundary/source data and finite assembled arithmetic with nonzero Thomas pivots; those mathematical/geometry preconditions are not fully validated. Positive p, nonnegative q/Robin data and adequate constraints give the usual coercive model. Piecewise-constant p/q can be scalar or one per element. Results own their values; input mesh/data are unchanged. Calls share no mutable global state, while callback side effects/concurrency belong to the caller.
  %
  %   obj = datamunge.FEM1D ()
  %
  %   Example:
  %     const auto r=dm::fem::FEM1D::solve_builtin(dm::fem::make_uniform_mesh1d(0,1,2),{1},{0},"constant",{1},{},{});
  %     std::print("{}\n",r.value_at(1)); // 0.125
  %
  %   References:
  %   - MFEM, Weak Formulations: https://mfem.org/fem_weak_form/ . This native implementation
  %     uses P1 intervals, three-point Gauss-Legendre loads and a tridiagonal direct solve.
  %
  %   Static methods:
  %     solve  Solve the steady P1 system for -(p*u')'+q*u=f with endpoint boundary data. Follows FEM1D's mesh/coefficient/finite-arithmetic/nonsingular-pivot preconditions. Uses three-point Gauss-Legendre quadrature of f times each local basis, natural boundary terms and symmetric Dirichlet elimination, then Thomas elimination. O(N) work/storage for N nodes; invokes the source three times per element. No callback pointer is retained. Independent calls share no mutable state.
  %     solve_builtin  Solve the steady system using an owned named spatial source, without a director callback. Shares the live-source solver and its mathematical/precondition/storage contracts. Sources: zero=0 (params unused); constant=c ({1}); linear=m*x+c ({1,0}); polynomial=sum(params[i]*x^i) (empty gives0); sine=A*sin(w*x+phase) ({1,1,0}). Missing trailing parameters use those defaults; extra parameters are ignored except polynomial, which uses all entries. Name matching is exact/case-sensitive.
  %     solve_transient  Integrate u_t-(p*u')'+q*u=f with P1 space and actual backward-Euler time intervals. Shares FEM1D's spatial/finite-arithmetic/nonzero-pivot preconditions. Source and boundary data represent a time-independent model; the live callback is actually reevaluated three times per element at every time advance, not cached. Stores initial u0 as supplied, then imposes boundary data at later solves. Nominal step dt is shortened at t_end; actual positive time differences enter the mass term. Stops once t_end is reached, avoiding a duplicate rounded last time even when ceil(t_end/dt) overestimates the number of advances. O(S*N) work/output storage for S advances/N nodes, O(N) assembly temporaries; no optional trace or quota. Backward Euler is unconditionally stable for the coercive semidiscrete model, not a guarantee of arbitrary-step accuracy or finite arithmetic for arbitrary parameters. Input arrays remain unchanged; callback exceptions/side effects/concurrent safety belong to caller.
  %     solve_transient_builtin  Integrate the same backward-Euler model with a named source and no callback subclass. Uses the same private source construction as solve_builtin() and the same actual time loop as solve_transient(); no second solver or per-language glue. Names/defaults: zero=0; constant=c ({1}); linear=m*x+c ({1,0}); polynomial=sum(params[i]*x^i) (empty0); sine=A*sin(w*x+phase) ({1,1,0}). Extra entries ignored except polynomial; exact case-sensitive names. Source objects are local and synchronous; results own values, not callback aliases. All spatial/time/initial-state, complexity and stability contracts follow solve_transient(); ordinary output has no optional history/quota.
  %
  %   C++: class datamunge::fem::FEM1D
  %
  %   See also: datamunge.Mesh1D, datamunge.FEM1DResult, datamunge.FEM1DTimeSeries
  methods
    function obj = FEM1D (varargin)
      % FEM1D  Construct a C++ datamunge::fem::FEM1D.
      %
      %   obj = datamunge.FEM1D ()
      %
      %   C++: datamunge::fem::FEM1D::FEM1D()
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 669, false, varargin{:});
    end
  end
  methods (Static)
    function r = solve (mesh, p, q, f, left, right)
      % solve  Solve the steady P1 system for -(p*u')'+q*u=f with endpoint boundary data. Follows FEM1D's mesh/coefficient/finite-arithmetic/nonsingular-pivot preconditions. Uses three-point Gauss-Legendre quadrature of f times each local basis, natural boundary terms and symmetric Dirichlet elimination, then Thomas elimination. O(N) work/storage for N nodes; invokes the source three times per element. No callback pointer is retained. Independent calls share no mutable state.
      %
      %   r = datamunge.FEM1D.solve (mesh, p, q, f, left, right)
      %
      %   Example:
      %     struct Unit final:dm::fem::ScalarField1D{double evaluate(double) override{return 1;}} source;
      %     const auto r=dm::fem::FEM1D::solve(dm::fem::make_uniform_mesh1d(0,1,2),{1},{0},source,{},{});
      %     std::print("{}\n",r.value_at(1)); // 0.125
      %
      %   References:
      %   - https://mfem.org/fem_weak_form/ ; native interval P1 and quadrature choices as FEM1D.
      %
      %   Inputs:
      %     mesh  datamunge.Mesh1D: Borrowed mesh with at least two finite strictly increasing nodes.
      %     p  double vector: Borrowed diffusion coefficient: length1 or one per element.
      %     q  double vector: Borrowed reaction coefficient: length1 or one per element.
      %     f  datamunge.ScalarField1D: Borrowed spatial source; evaluate() may have caller-owned side effects.
      %     left  datamunge.BoundaryCondition1D: Left endpoint condition, with outward normal -1.
      %     right  datamunge.BoundaryCondition1D: Right endpoint condition, with outward normal +1.
      %
      %   Output:
      %     r  datamunge.FEM1DResult: Owned original nodes and computed nodal values.
      %
      %   Errors:
      %     datamunge:invalid_argument  If p/q are empty or have the wrong lengths. (C++ std::invalid_argument)
      %     datamunge:allocation  If assembly/result storage cannot be allocated; callback exceptions propagate. (C++ std::bad_alloc)
      %
      %   C++: FEM1DResult datamunge::fem::FEM1D::solve(const Mesh1D & mesh, const std::vector<double> & p, const std::vector<double> & q, ScalarField1D & f, const BoundaryCondition1D & left, const BoundaryCondition1D & right)
      %
      %   See also: datamunge.solve_builtin(), datamunge.solve_transient()
      r = __datamunge__(665, mesh, p, q, f, left, right);
    end
    function r = solve_builtin (mesh, p, q, source, source_params, left, right)
      % solve_builtin  Solve the steady system using an owned named spatial source, without a director callback. Shares the live-source solver and its mathematical/precondition/storage contracts. Sources: zero=0 (params unused); constant=c ({1}); linear=m*x+c ({1,0}); polynomial=sum(params[i]*x^i) (empty gives0); sine=A*sin(w*x+phase) ({1,1,0}). Missing trailing parameters use those defaults; extra parameters are ignored except polynomial, which uses all entries. Name matching is exact/case-sensitive.
      %
      %   r = datamunge.FEM1D.solve_builtin (mesh, p, q, source, source_params, left, right)
      %
      %   Example:
      %     const auto r=dm::fem::FEM1D::solve_builtin(dm::fem::make_uniform_mesh1d(0,1,2),{1},{0},"constant",{}, {},{});
      %     std::print("{}\n",r.value_at(1)); // 0.125
      %
      %   References:
      %   - https://mfem.org/fem_weak_form/ ; source definitions are native convenience choices.
      %
      %   Inputs:
      %     mesh  datamunge.Mesh1D: Borrowed valid finite interval mesh; unchanged.
      %     p  double vector: Borrowed scalar/per-element diffusion coefficient.
      %     q  double vector: Borrowed scalar/per-element reaction coefficient.
      %     source  char row: Exact name zero, constant, linear, polynomial or sine.
      %     source_params  double vector: Borrowed positional source parameters, with defaults described above.
      %     left  datamunge.BoundaryCondition1D: Left endpoint data using outward normal -1.
      %     right  datamunge.BoundaryCondition1D: Right endpoint data using outward normal +1.
      %
      %   Output:
      %     r  datamunge.FEM1DResult: Owned computed nodal result, retaining no source or input reference.
      %
      %   Errors:
      %     datamunge:invalid_argument  For an unknown source or malformed p/q lengths. (C++ std::invalid_argument)
      %     datamunge:allocation  If owned parameter/assembly/result storage cannot be allocated. (C++ std::bad_alloc)
      %
      %   C++: FEM1DResult datamunge::fem::FEM1D::solve_builtin(const Mesh1D & mesh, const std::vector<double> & p, const std::vector<double> & q, const std::string & source, const std::vector<double> & source_params, const BoundaryCondition1D & left, const BoundaryCondition1D & right)
      %
      %   See also: datamunge.solve(), datamunge.solve_transient_builtin()
      r = __datamunge__(666, mesh, p, q, source, source_params, left, right);
    end
    function r = solve_transient (mesh, p, q, f, left, right, u0, t_end, dt)
      % solve_transient  Integrate u_t-(p*u')'+q*u=f with P1 space and actual backward-Euler time intervals. Shares FEM1D's spatial/finite-arithmetic/nonzero-pivot preconditions. Source and boundary data represent a time-independent model; the live callback is actually reevaluated three times per element at every time advance, not cached. Stores initial u0 as supplied, then imposes boundary data at later solves. Nominal step dt is shortened at t_end; actual positive time differences enter the mass term. Stops once t_end is reached, avoiding a duplicate rounded last time even when ceil(t_end/dt) overestimates the number of advances. O(S*N) work/output storage for S advances/N nodes, O(N) assembly temporaries; no optional trace or quota. Backward Euler is unconditionally stable for the coercive semidiscrete model, not a guarantee of arbitrary-step accuracy or finite arithmetic for arbitrary parameters. Input arrays remain unchanged; callback exceptions/side effects/concurrent safety belong to caller.
      %
      %   r = datamunge.FEM1D.solve_transient (mesh, p, q, f, left, right, u0, t_end, dt)
      %
      %   Example:
      %     struct Zero final:dm::fem::ScalarField1D{double evaluate(double) override{return 0;}} source;
      %     const auto t=dm::fem::FEM1D::solve_transient(dm::fem::make_uniform_mesh1d(0,1,2),{1},{0},source,{},{},{0,1,0},.5,.5);
      %     std::print("{} {:.6f}\n",t.num_steps(),t.value_at(1,1)); // 2 0.142857
      %
      %   References:
      %   - Arnold (2012), Numerical Analysis of Partial Differential Equations, chapter5 section2.
      %     https://www-users.cse.umn.edu/~arnold/8445.f12/notes.pdf . Native load quadrature,
      %     fixed data and rounded/shortened time grid are implementation choices.
      %
      %   Inputs:
      %     mesh  datamunge.Mesh1D: Borrowed valid mesh with at least two finite increasing nodes.
      %     p  double vector: Borrowed scalar/per-element diffusion coefficient.
      %     q  double vector: Borrowed scalar/per-element reaction coefficient.
      %     f  datamunge.ScalarField1D: Borrowed spatial callback, reevaluated at every advance.
      %     left  datamunge.BoundaryCondition1D: Fixed left condition, with outward normal -1.
      %     right  datamunge.BoundaryCondition1D: Fixed right condition, with outward normal +1.
      %     u0  double vector: Borrowed finite initial nodal values, exactly one per mesh node.
      %     t_end  double: Positive finite final time, in time units.
      %     dt  double: Positive finite nominal time step in the same units; may exceed t_end.
      %
      %   Output:
      %     r  datamunge.FEM1DTimeSeries: Owned nodes, initial state and strictly advancing slices ending at t_end.
      %
      %   Errors:
      %     datamunge:invalid_argument  For fewer than two nodes, wrong coefficient/u0 lengths or nonpositive/nonfinite time controls. (C++ std::invalid_argument)
      %     datamunge:length_error  If frame counts, vector capacities or total scalar count cannot be represented. (C++ std::length_error)
      %     datamunge:arithmetic_error  If rounded time products cannot advance before t_end. (C++ std::overflow_error)
      %     datamunge:allocation  If output/assembly storage cannot be allocated; callback exceptions propagate. (C++ std::bad_alloc)
      %
      %   C++: FEM1DTimeSeries datamunge::fem::FEM1D::solve_transient(const Mesh1D & mesh, const std::vector<double> & p, const std::vector<double> & q, ScalarField1D & f, const BoundaryCondition1D & left, const BoundaryCondition1D & right, const std::vector<double> & u0, double t_end, double dt)
      %
      %   See also: datamunge.solve_transient_builtin(), datamunge.FEM1DTimeSeries
      r = __datamunge__(667, mesh, p, q, f, left, right, u0, t_end, dt);
    end
    function r = solve_transient_builtin (mesh, p, q, source, source_params, left, right, u0, t_end, dt)
      % solve_transient_builtin  Integrate the same backward-Euler model with a named source and no callback subclass. Uses the same private source construction as solve_builtin() and the same actual time loop as solve_transient(); no second solver or per-language glue. Names/defaults: zero=0; constant=c ({1}); linear=m*x+c ({1,0}); polynomial=sum(params[i]*x^i) (empty0); sine=A*sin(w*x+phase) ({1,1,0}). Extra entries ignored except polynomial; exact case-sensitive names. Source objects are local and synchronous; results own values, not callback aliases. All spatial/time/initial-state, complexity and stability contracts follow solve_transient(); ordinary output has no optional history/quota.
      %
      %   r = datamunge.FEM1D.solve_transient_builtin (mesh, p, q, source, source_params, left, right, u0, t_end, dt)
      %
      %   Example:
      %     const auto t=dm::fem::FEM1D::solve_transient_builtin(dm::fem::make_uniform_mesh1d(0,1,2),{1},{0},"constant",{1},{},{},{0,0,0},.5,.5);
      %     std::print("{} {:.6f}\n",t.num_steps(),t.value_at(1,1)); // 2 0.107143
      %
      %   References:
      %   - https://www-users.cse.umn.edu/~arnold/8445.f12/notes.pdf ; same native discretization as solve_transient().
      %
      %   Inputs:
      %     mesh  datamunge.Mesh1D: Borrowed valid finite increasing mesh, with at least two nodes.
      %     p  double vector: Borrowed scalar/per-element diffusion coefficient.
      %     q  double vector: Borrowed scalar/per-element reaction coefficient.
      %     source  char row: Exact name zero, constant, linear, polynomial or sine.
      %     source_params  double vector: Borrowed positional parameters; omitted trailing values use the defaults above.
      %     left  datamunge.BoundaryCondition1D: Fixed left boundary condition using outward normal -1.
      %     right  datamunge.BoundaryCondition1D: Fixed right boundary condition using outward normal +1.
      %     u0  double vector: Borrowed finite initial values, exactly one per spatial node.
      %     t_end  double: Positive finite final time.
      %     dt  double: Positive finite nominal time step; final interval is shortened as needed.
      %
      %   Output:
      %     r  datamunge.FEM1DTimeSeries: Owned initial state and actual strictly advancing backward-Euler slices.
      %
      %   Errors:
      %     datamunge:invalid_argument  For an unknown source, malformed coefficient/u0 lengths, fewer than two nodes or nonpositive/nonfinite time controls. (C++ std::invalid_argument)
      %     datamunge:length_error  If frame/container/total scalar counts cannot be represented. (C++ std::length_error)
      %     datamunge:arithmetic_error  If rounded time products cannot advance. (C++ std::overflow_error)
      %     datamunge:allocation  If parameters, output or assembly storage cannot be allocated. (C++ std::bad_alloc)
      %
      %   C++: FEM1DTimeSeries datamunge::fem::FEM1D::solve_transient_builtin(const Mesh1D & mesh, const std::vector<double> & p, const std::vector<double> & q, const std::string & source, const std::vector<double> & source_params, const BoundaryCondition1D & left, const BoundaryCondition1D & right, const std::vector<double> & u0, double t_end, double dt)
      %
      %   See also: datamunge.solve_builtin(), datamunge.solve_transient()
      r = __datamunge__(668, mesh, p, q, source, source_params, left, right, u0, t_end, dt);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::fem::FEM1D. Do not edit.
