classdef HierarchicalBSpline < datamunge.NativeObject
  % HierarchicalBSpline  Classical (overlay) hierarchical B-spline in the style of Forsey and Bartels: every cell of a dyadic tree may carry its own local "detail" TensorBSpline.
  %
  %   obj = datamunge.HierarchicalBSpline (domain_min, domain_max)
  %
  %   evaluate(x) sums the patch of every cell on the root-to-leaf path containing x. Each
  %   patch is evaluated in its cell's local coordinates, (x - cell_min) / (cell_max -
  %   cell_min) in [0, 1]^d, so patch bases should span [0, 1]. Patches are independent
  %   (no shared nested basis and no truncation); see TruncatedHierarchicalBSpline for the
  %   THB-spline construction. Cells are identified by integer ids (root 0).
  %
  %   Example:
  %     dm::fda::HierarchicalBSpline h({0.0}, {1.0});
  %     auto b = dm::fda::BSplineBasis::open_uniform(0, 1);              // constant patch
  %     h.set_spline(h.root(), dm::fda::TensorBSpline(b, {1.0}));
  %     h.refine(h.root(), 1, dm::fda::TensorBSpline(b, {2.0}));          // upper half
  %     std::print("{} {}\n", h.evaluate({0.25}), h.evaluate({0.75}));    // 1 3
  %
  %   References:
  %   - D. R. Forsey and R. H. Bartels (1988). Hierarchical B-spline refinement. ACM SIGGRAPH
  %     Computer Graphics 22(4), 205-212. https://doi.org/10.1145/378456.378512
  %
  %   Methods:
  %     dimensions  Number of dimensions d.
  %     cell_count  Number of cells created so far.
  %     root  Root cell id.
  %     parent  Parent id.
  %     child  Child id at `slot`.
  %     depth  Depth of `cell` (root 0).
  %     cell_min  Lower corner of `cell`.
  %     cell_max  Upper corner of `cell`.
  %     locate  Deepest existing cell containing `point`.
  %     has_spline  Whether `cell` carries a patch.
  %     spline  The patch of `cell`.
  %     set_spline  Installs (or replaces) the patch of `cell`.
  %     clear_spline  Removes the patch of `cell` (no-op if none).
  %     subdivide  Creates all 2^d children of `cell` (existing ones are kept), without patches.
  %     refine  Creates (if needed) only the child of `cell` at `slot`.
  %     refine_region  Subdivides, down to depth `depth`, every cell overlapping the open box (region_min, region_max).
  %     fit_cell  Least-squares fits a new patch on `cell` so the whole hierarchy matches `values` at `points`.
  %     fit_cell_penalized  Penalized (P-spline) variant of fit_cell() with per-axis `lambdas` and difference `order`; all-zero lambdas give fit_cell().
  %     evaluate  Sum of the patches along the path of cells containing `point`.
  %     evaluate_points  Evaluates at many points.
  %     evaluate_grid  Evaluates on a Cartesian grid.
  %
  %   C++: class datamunge::fda::HierarchicalBSpline
  %
  %   See also: datamunge.TruncatedHierarchicalBSpline
  methods
    function obj = HierarchicalBSpline (varargin)
      % HierarchicalBSpline  Construct a C++ datamunge::fda::HierarchicalBSpline.
      %
      %   obj = datamunge.HierarchicalBSpline (domain_min, domain_max)
      %
      %   Example:
      %     dm::fda::HierarchicalBSpline h({0.0, 0.0}, {1.0, 2.0});
      %     std::print("{} {}\n", h.dimensions(), h.cell_count()); // 2 1
      %
      %   Creates a hierarchy whose root cell is [domain_min, domain_max], with no patches.
      %
      %   Inputs:
      %     domain_min  double vector: Lower corner (length d >= 1).
      %     domain_max  double vector: Upper corner, with domain_min[k] < domain_max[k].
      %
      %   Errors:
      %     datamunge:invalid_argument  On empty, mismatched or non-finite bounds, min >= max, or more than 20 dimensions. (C++ std::invalid_argument)
      %
      %   C++: datamunge::fda::HierarchicalBSpline::HierarchicalBSpline(std::vector<double> domain_min, std::vector<double> domain_max)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 3049, false, varargin{:});
    end
    function r = dimensions (obj)
      % dimensions  Number of dimensions d.
      %
      %   r = obj.dimensions ()
      %
      %   Output:
      %     r  integer: d.
      %
      %   C++: std::size_t datamunge::fda::HierarchicalBSpline::dimensions() const
      r = __datamunge__(3028, obj);
    end
    function r = cell_count (obj)
      % cell_count  Number of cells created so far.
      %
      %   r = obj.cell_count ()
      %
      %   Output:
      %     r  integer: Cell count.
      %
      %   C++: int datamunge::fda::HierarchicalBSpline::cell_count() const
      r = __datamunge__(3029, obj);
    end
    function r = root (obj)
      % root  Root cell id.
      %
      %   r = obj.root ()
      %
      %   Output:
      %     r  integer: Always 0.
      %
      %   C++: int datamunge::fda::HierarchicalBSpline::root() const
      r = __datamunge__(3030, obj);
    end
    function r = parent (obj, cell)
      % parent  Parent id.
      %
      %   r = obj.parent (cell)
      %
      %   Inputs:
      %     cell  integer: Cell id.
      %
      %   Output:
      %     r  integer: Parent id, -1 for the root.
      %
      %   Errors:
      %     datamunge:out_of_range  For an invalid id. (C++ std::out_of_range)
      %
      %   C++: int datamunge::fda::HierarchicalBSpline::parent(int cell) const
      r = __datamunge__(3031, obj, cell);
    end
    function r = child (obj, cell, slot)
      % child  Child id at `slot`.
      %
      %   r = obj.child (cell, slot)
      %
      %   Inputs:
      %     cell  integer: Cell id.
      %     slot  integer: Child slot in [0, 2^d).
      %
      %   Output:
      %     r  integer: Child id, or -1 when not created.
      %
      %   Errors:
      %     datamunge:out_of_range  For invalid ids/slots. (C++ std::out_of_range)
      %
      %   C++: int datamunge::fda::HierarchicalBSpline::child(int cell, int slot) const
      r = __datamunge__(3032, obj, cell, slot);
    end
    function r = depth (obj, cell)
      % depth  Depth of `cell` (root 0).
      %
      %   r = obj.depth (cell)
      %
      %   Inputs:
      %     cell  integer: Cell id.
      %
      %   Output:
      %     r  integer: Depth.
      %
      %   Errors:
      %     datamunge:out_of_range  For an invalid id. (C++ std::out_of_range)
      %
      %   C++: int datamunge::fda::HierarchicalBSpline::depth(int cell) const
      r = __datamunge__(3033, obj, cell);
    end
    function r = cell_min (obj, cell)
      % cell_min  Lower corner of `cell`.
      %
      %   r = obj.cell_min (cell)
      %
      %   Inputs:
      %     cell  integer: Cell id.
      %
      %   Output:
      %     r  double vector: d coordinates.
      %
      %   Errors:
      %     datamunge:out_of_range  For an invalid id. (C++ std::out_of_range)
      %
      %   C++: std::vector<double> datamunge::fda::HierarchicalBSpline::cell_min(int cell) const
      r = __datamunge__(3034, obj, cell);
    end
    function r = cell_max (obj, cell)
      % cell_max  Upper corner of `cell`.
      %
      %   r = obj.cell_max (cell)
      %
      %   Inputs:
      %     cell  integer: Cell id.
      %
      %   Output:
      %     r  double vector: d coordinates.
      %
      %   Errors:
      %     datamunge:out_of_range  For an invalid id. (C++ std::out_of_range)
      %
      %   C++: std::vector<double> datamunge::fda::HierarchicalBSpline::cell_max(int cell) const
      r = __datamunge__(3035, obj, cell);
    end
    function r = locate (obj, point)
      % locate  Deepest existing cell containing `point`.
      %
      %   r = obj.locate (point)
      %
      %   Example:
      %     dm::fda::HierarchicalBSpline h({0.0}, {1.0});
      %     int c = h.refine(h.root(), 1);
      %     std::print("{} {}\n", h.locate({0.9}) == c, h.locate({2.0})); // true -1
      %
      %   Inputs:
      %     point  double vector: d coordinates.
      %
      %   Output:
      %     r  integer: Cell id, or -1 if `point` is outside the domain.
      %
      %   Errors:
      %     datamunge:invalid_argument  If point.size() != dimensions(). (C++ std::invalid_argument)
      %
      %   C++: int datamunge::fda::HierarchicalBSpline::locate(const std::vector<double> & point) const
      r = __datamunge__(3036, obj, point);
    end
    function r = has_spline (obj, cell)
      % has_spline  Whether `cell` carries a patch.
      %
      %   r = obj.has_spline (cell)
      %
      %   Inputs:
      %     cell  integer: Cell id.
      %
      %   Output:
      %     r  logical: Flag.
      %
      %   Errors:
      %     datamunge:out_of_range  For an invalid id. (C++ std::out_of_range)
      %
      %   C++: bool datamunge::fda::HierarchicalBSpline::has_spline(int cell) const
      r = __datamunge__(3037, obj, cell);
    end
    function r = spline (obj, cell)
      % spline  The patch of `cell`.
      %
      %   r = obj.spline (cell)
      %
      %   Inputs:
      %     cell  integer: Cell id.
      %
      %   Output:
      %     r  datamunge.TensorBSpline: Reference valid until the patch is replaced or cleared.
      %
      %   Errors:
      %     datamunge:out_of_range  For an invalid id. (C++ std::out_of_range)
      %     datamunge:invalid_argument  If no patch. (C++ std::invalid_argument)
      %
      %   C++: const TensorBSpline & datamunge::fda::HierarchicalBSpline::spline(int cell) const
      r = __datamunge__(3038, obj, cell);
    end
    function set_spline (obj, cell, spline)
      % set_spline  Installs (or replaces) the patch of `cell`.
      %
      %   obj.set_spline (cell, spline)
      %
      %   Inputs:
      %     cell  integer: Cell id.
      %     spline  datamunge.TensorBSpline: Patch in cell-local [0, 1]^d coordinates.
      %
      %   Errors:
      %     datamunge:out_of_range  For an invalid id. (C++ std::out_of_range)
      %     datamunge:invalid_argument  If the patch dimension differs from dimensions(). (C++ std::invalid_argument)
      %
      %   C++: void datamunge::fda::HierarchicalBSpline::set_spline(int cell, TensorBSpline spline)
      __datamunge__(3039, obj, cell, spline);
    end
    function clear_spline (obj, cell)
      % clear_spline  Removes the patch of `cell` (no-op if none).
      %
      %   obj.clear_spline (cell)
      %
      %   Inputs:
      %     cell  integer: Cell id.
      %
      %   Errors:
      %     datamunge:out_of_range  For an invalid id. (C++ std::out_of_range)
      %
      %   C++: void datamunge::fda::HierarchicalBSpline::clear_spline(int cell)
      __datamunge__(3040, obj, cell);
    end
    function r = subdivide (obj, cell)
      % subdivide  Creates all 2^d children of `cell` (existing ones are kept), without patches.
      %
      %   r = obj.subdivide (cell)
      %
      %   Example:
      %     dm::fda::HierarchicalBSpline h({0.0, 0.0}, {1.0, 1.0});
      %     std::print("{}\n", h.subdivide(h.root()).size()); // 4
      %
      %   Inputs:
      %     cell  integer: Cell id.
      %
      %   Output:
      %     r  integer vector: The 2^d child ids in slot order.
      %
      %   Errors:
      %     datamunge:out_of_range  For an invalid id. (C++ std::out_of_range)
      %
      %   C++: std::vector<int> datamunge::fda::HierarchicalBSpline::subdivide(int cell)
      r = __datamunge__(3041, obj, cell);
    end
    function r = refine (obj, varargin)
      % refine  Creates (if needed) only the child of `cell` at `slot`.
      %
      %   r = obj.refine (cell, slot)
      %   r = obj.refine (cell, slot, spline)
      %
      %   Example:
      %     dm::fda::HierarchicalBSpline h({0.0}, {1.0});
      %     int c = h.refine(h.root(), 0);
      %     std::print("{}\n", h.cell_max(c)); // [0.5]
      %
      %   Errors:
      %     datamunge:out_of_range  For an invalid id or slot. (C++ std::out_of_range)
      %
      %   Overloads are chosen at run time from the argument types (C++):
      %   C++: int datamunge::fda::HierarchicalBSpline::refine(int cell, int slot)
      %   C++: int datamunge::fda::HierarchicalBSpline::refine(int cell, int slot, TensorBSpline spline)
      r = __datamunge__(3042, obj, varargin{:});
    end
    function refine_region (obj, varargin)
      % refine_region  Subdivides, down to depth `depth`, every cell overlapping the open box (region_min, region_max).
      %
      %   obj.refine_region (region_min, region_max, depth)
      %   obj.refine_region (region_min, region_max, depth, factory)
      %
      %   Example:
      %     dm::fda::HierarchicalBSpline h({0.0}, {1.0});
      %     h.refine_region({0.0}, {0.3}, 2);
      %     std::print("{}\n", h.cell_count()); // 4
      %
      %   Errors:
      %     datamunge:invalid_argument  On mismatched sizes or min >= max. (C++ std::invalid_argument)
      %
      %   Overloads are chosen at run time from the argument types (C++):
      %   C++: void datamunge::fda::HierarchicalBSpline::refine_region(const std::vector<double> & region_min, const std::vector<double> & region_max, int depth)
      %   C++: void datamunge::fda::HierarchicalBSpline::refine_region(const std::vector<double> & region_min, const std::vector<double> & region_max, int depth, const SplineFactory & factory)
      __datamunge__(3043, obj, varargin{:});
    end
    function fit_cell (obj, varargin)
      % fit_cell  Least-squares fits a new patch on `cell` so the whole hierarchy matches `values` at `points`.
      %
      %   obj.fit_cell (cell, bases, points, values)
      %   obj.fit_cell (cell, bases, points, values, options)
      %
      %   The patch explains only what is left after every other patch on the path (the
      %   residual values - evaluate(points)), in the spirit of Forsey-Bartels editing, so pass
      %   raw data values rather than residuals. Points are global coordinates inside the cell.
      %
      %   Example:
      %     dm::fda::HierarchicalBSpline h({0.0}, {1.0});
      %     auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
      %     h.fit_cell(h.root(), {b}, {{0.0}, {1.0}}, {1.0, 3.0});
      %     std::print("{}\n", h.evaluate({0.5})); // 2
      %
      %   Inputs:
      %     cell  integer: Cell id; must not already have a patch (clear_spline() first to refit).
      %     bases  cell array of datamunge.BSplineBasis: Patch bases on [0, 1], one per dimension.
      %     points  double matrix (rows): Sample locations inside the cell.
      %     values  double vector: One raw data value per point.
      %     options  datamunge.SplineSolveOptions, default the C++ default: Solver settings.
      %
      %   Errors:
      %     datamunge:out_of_range  For an invalid id. (C++ std::out_of_range)
      %     datamunge:invalid_argument  If the cell has a patch, inputs are empty/mismatched, or a point lies outside the cell. (C++ std::invalid_argument)
      %     datamunge:native  If the local normal equations are singular. (C++ std::runtime_error)
      %
      %   C++: void datamunge::fda::HierarchicalBSpline::fit_cell(int cell, std::vector<BSplineBasis> bases, const std::vector<std::vector<double>> & points, const std::vector<double> & values, const SplineSolveOptions & options = {})
      __datamunge__(3044, obj, varargin{:});
    end
    function fit_cell_penalized (obj, varargin)
      % fit_cell_penalized  Penalized (P-spline) variant of fit_cell() with per-axis `lambdas` and difference `order`; all-zero lambdas give fit_cell().
      %
      %   obj.fit_cell_penalized (cell, bases, points, values, lambdas)
      %   obj.fit_cell_penalized (cell, bases, points, values, lambdas, order)
      %   obj.fit_cell_penalized (cell, bases, points, values, lambdas, order, options)
      %
      %   Example:
      %     dm::fda::HierarchicalBSpline h({0.0}, {1.0});
      %     auto b = dm::fda::BSplineBasis::open_uniform(3, 10);
      %     h.fit_cell_penalized(h.root(), {b}, {{0.0}, {1.0}}, {0.0, 2.0}, {1e3});
      %     std::print("{:.3f}\n", h.evaluate({0.5})); // 1.000
      %
      %   Inputs:
      %     cell  integer: Cell id without a patch.
      %     bases  cell array of datamunge.BSplineBasis: Patch bases on [0, 1].
      %     points  double matrix (rows): Sample locations inside the cell.
      %     values  double vector: Raw data values.
      %     lambdas  double vector: Nonnegative per-axis weights.
      %     order  integer, default 2: Difference order (default 2).
      %     options  datamunge.SplineSolveOptions, default the C++ default: Solver settings.
      %
      %   Errors:
      %     datamunge:out_of_range  For an invalid id. (C++ std::out_of_range)
      %     datamunge:invalid_argument  As for fit_cell() and fit_tensor_pspline(). (C++ std::invalid_argument)
      %     datamunge:native  If the penalized normal equations are singular. (C++ std::runtime_error)
      %
      %   C++: void datamunge::fda::HierarchicalBSpline::fit_cell_penalized(int cell, std::vector<BSplineBasis> bases, const std::vector<std::vector<double>> & points, const std::vector<double> & values, const std::vector<double> & lambdas, int order = 2, const SplineSolveOptions & options = {})
      __datamunge__(3045, obj, varargin{:});
    end
    function r = evaluate (obj, point)
      % evaluate  Sum of the patches along the path of cells containing `point`.
      %
      %   r = obj.evaluate (point)
      %
      %   Example:
      %   See the class example.
      %
      %   Inputs:
      %     point  double vector: d coordinates inside the domain.
      %
      %   Output:
      %     r  double: Hierarchy value (0 where no patch applies).
      %
      %   Errors:
      %     datamunge:invalid_argument  On a size mismatch or a point outside the domain. (C++ std::invalid_argument)
      %
      %   C++: double datamunge::fda::HierarchicalBSpline::evaluate(const std::vector<double> & point) const
      r = __datamunge__(3046, obj, point);
    end
    function r = evaluate_points (obj, points)
      % evaluate_points  Evaluates at many points.
      %
      %   r = obj.evaluate_points (points)
      %
      %   Example:
      %     dm::fda::HierarchicalBSpline h({0.0}, {1.0});
      %     std::print("{}\n", h.evaluate_points({{0.1}, {0.9}})); // [0, 0]
      %
      %   Inputs:
      %     points  double matrix (rows): Points of length d.
      %
      %   Output:
      %     r  double vector: One value per point.
      %
      %   Errors:
      %     datamunge:invalid_argument  As for evaluate(). (C++ std::invalid_argument)
      %
      %   C++: std::vector<double> datamunge::fda::HierarchicalBSpline::evaluate_points(const std::vector<std::vector<double>> & points) const
      r = __datamunge__(3047, obj, points);
    end
    function r = evaluate_grid (obj, axes)
      % evaluate_grid  Evaluates on a Cartesian grid.
      %
      %   r = obj.evaluate_grid (axes)
      %
      %   Example:
      %     dm::fda::HierarchicalBSpline h({0.0, 0.0}, {1.0, 1.0});
      %     std::print("{}\n", h.evaluate_grid({{0.0, 1.0}, {0.5}}).size()); // 2
      %
      %   Inputs:
      %     axes  double matrix (rows): d non-empty coordinate lists.
      %
      %   Output:
      %     r  double vector: Row-major values (last axis fastest).
      %
      %   Errors:
      %     datamunge:invalid_argument  On a dimension mismatch, empty axis, or outside point. (C++ std::invalid_argument)
      %
      %   C++: std::vector<double> datamunge::fda::HierarchicalBSpline::evaluate_grid(const std::vector<std::vector<double>> & axes) const
      r = __datamunge__(3048, obj, axes);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::fda::HierarchicalBSpline. Do not edit.
