classdef TruncatedHierarchicalBSpline < datamunge.NativeObject
  % TruncatedHierarchicalBSpline  Truncated hierarchical B-spline (THB-spline) of Giannelli, Juttler and Speleers.
  %
  %   obj = datamunge.TruncatedHierarchicalBSpline (base)
  %
  %   Level 0 is the caller's TensorBSpline; level l + 1 is the dyadic refinement of level l.
  %   A dyadic cell tree defines the nested domains Omega^0 (whole domain) contains Omega^1
  %   contains ..., where Omega^l is the union of cells of depth >= l. A level-l basis
  %   function is active when its support lies in Omega^l but not in Omega^(l+1). The THB
  %   basis consists of the active functions, each truncated: its representation on finer
  %   levels drops the finer functions whose support lies in the finer refined domain. The
  %   resulting basis is a partition of unity, preserves the continuity of the underlying
  %   splines, and spans the hierarchical spline space.
  %
  %   Internally each level stores a full coefficient net whose entries outside the next
  %   level's refined domain always equal the refinement of the coarser level; coefficient
  %   edits are pushed down through all finer levels with truncation, so evaluation needs
  %   only the finest level covering a point. Refinement never changes the represented
  %   function. Bases must be clamped. Cells are integer ids (root 0).
  %
  %   Example:
  %     auto b = dm::fda::BSplineBasis::open_uniform(2, 4);
  %     dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline({b, b}, std::vector<double>(16, 1.0)));
  %     thb.refine_region({0.0, 0.0}, {0.5, 0.5}, 1);
  %     std::print("{} {:.3f}\n", thb.level_count(), thb.evaluate({0.2, 0.2})); // 2 1.000
  %
  %   References:
  %   - C. Giannelli, B. Juttler and H. Speleers (2012). THB-splines: The truncated basis for
  %     hierarchical splines. Computer Aided Geometric Design 29(7), 485-498.
  %     https://doi.org/10.1016/j.cagd.2012.03.025
  %   - D. R. Forsey and R. H. Bartels (1988). Hierarchical B-spline refinement.
  %     https://doi.org/10.1145/378456.378512
  %
  %   Methods:
  %     dimensions  Number of dimensions d.
  %     level_count  Number of materialized levels (1 + deepest cell depth).
  %     level  The level- `l` spline on the whole domain; it equals the hierarchy wherever Omega^l but not Omega^(l+1) applies.
  %     cell_count  Number of cells.
  %     root  Root cell id.
  %     parent  Parent id.
  %     child  Child id.
  %     depth  Depth (= level used there if it is a leaf).
  %     cell_min  Lower corner.
  %     cell_max  Upper corner.
  %     locate  Deepest cell containing `point` (its depth is the finest level there).
  %     refine  Creates (if needed) the child of `cell` at `slot`, materializing the next level when first reached. The represented function is unchanged.
  %     subdivide  Creates all 2^d children of `cell`.
  %     refine_region  Subdivides, down to depth `depth`, every cell overlapping the open box (region_min, region_max). The represented function is unchanged.
  %     support_min  Lower corner of the support box of basis function `index` at `level`.
  %     support_max  Upper corner of the support box.
  %     is_active  Whether level- `level` function `index` is in the THB basis (support in Omega^level but not in Omega^(level+1)).
  %     active_indices  Every active multi-index at `level`, in row-major order.
  %     basis_size  Total number of THB basis functions over all levels.
  %     coefficient  THB coefficient of a basis function (for active functions; inactive entries are internal bookkeeping).
  %     set_coefficient  Sets the coefficient of an active function and propagates the change, with truncation, to every finer level. The edit is local to the function's support.
  %     truncated_basis  Value of the truncated basis function (level, index) at `point`.
  %     fit_level  Least-squares update of the active coefficients at `level` only.
  %     fit_level_penalized  Penalized variant of fit_level(): adds the level's full-grid tensor difference penalty restricted to its active functions, applied to the coefficient change.
  %     fit  Least-squares fit of every active coefficient on every level at once (replaces the current function).
  %     fit_penalized  Penalized all-level fit: adds lambda_k times the squared order-q differences along axis k of every run of q + 1 neighbouring active coefficients of a level.
  %     evaluate  Evaluates the hierarchical spline (finest level covering `point`).
  %     evaluate_points  Evaluates at many points.
  %     evaluate_grid  Evaluates on a Cartesian grid.
  %
  %   C++: class datamunge::fda::TruncatedHierarchicalBSpline
  %
  %   See also: datamunge.HierarchicalBSpline, datamunge.dyadic_refine()
  methods
    function obj = TruncatedHierarchicalBSpline (varargin)
      % TruncatedHierarchicalBSpline  Construct a C++ datamunge::fda::TruncatedHierarchicalBSpline.
      %
      %   obj = datamunge.TruncatedHierarchicalBSpline (base)
      %
      %   Example:
      %   See the class example.
      %
      %   Starts a hierarchy whose level 0 (and root cell / domain) is `base`.
      %
      %   Inputs:
      %     base  datamunge.TensorBSpline: Level-0 spline; every axis basis must be clamped.
      %
      %   Errors:
      %     datamunge:invalid_argument  If an axis basis is not clamped or d > 20. (C++ std::invalid_argument)
      %
      %   C++: datamunge::fda::TruncatedHierarchicalBSpline::TruncatedHierarchicalBSpline(TensorBSpline base)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 3079, false, varargin{:});
    end
    function r = dimensions (obj)
      % dimensions  Number of dimensions d.
      %
      %   r = obj.dimensions ()
      %
      %   Output:
      %     r  integer: d.
      %
      %   C++: std::size_t datamunge::fda::TruncatedHierarchicalBSpline::dimensions() const
      r = __datamunge__(3050, obj);
    end
    function r = level_count (obj)
      % level_count  Number of materialized levels (1 + deepest cell depth).
      %
      %   r = obj.level_count ()
      %
      %   Output:
      %     r  integer: Level count.
      %
      %   C++: int datamunge::fda::TruncatedHierarchicalBSpline::level_count() const
      r = __datamunge__(3051, obj);
    end
    function r = level (obj, level)
      % level  The level- `l` spline on the whole domain; it equals the hierarchy wherever Omega^l but not Omega^(l+1) applies.
      %
      %   r = obj.level (level)
      %
      %   Inputs:
      %     level  integer: Level in [0, level_count()).
      %
      %   Output:
      %     r  datamunge.TensorBSpline: Reference valid until the next refinement or coefficient edit.
      %
      %   Errors:
      %     datamunge:out_of_range  For an invalid level. (C++ std::out_of_range)
      %
      %   C++: const TensorBSpline & datamunge::fda::TruncatedHierarchicalBSpline::level(int level) const
      r = __datamunge__(3052, obj, level);
    end
    function r = cell_count (obj)
      % cell_count  Number of cells.
      %
      %   r = obj.cell_count ()
      %
      %   Output:
      %     r  integer: Cell count.
      %
      %   C++: int datamunge::fda::TruncatedHierarchicalBSpline::cell_count() const
      r = __datamunge__(3053, obj);
    end
    function r = root (obj)
      % root  Root cell id.
      %
      %   r = obj.root ()
      %
      %   Output:
      %     r  integer: Always 0.
      %
      %   C++: int datamunge::fda::TruncatedHierarchicalBSpline::root() const
      r = __datamunge__(3054, obj);
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
      %   C++: int datamunge::fda::TruncatedHierarchicalBSpline::parent(int cell) const
      r = __datamunge__(3055, obj, cell);
    end
    function r = child (obj, cell, slot)
      % child  Child id.
      %
      %   r = obj.child (cell, slot)
      %
      %   Inputs:
      %     cell  integer: Cell id.
      %     slot  integer: Child slot.
      %
      %   Output:
      %     r  integer: Child id or -1.
      %
      %   Errors:
      %     datamunge:out_of_range  For invalid ids/slots. (C++ std::out_of_range)
      %
      %   C++: int datamunge::fda::TruncatedHierarchicalBSpline::child(int cell, int slot) const
      r = __datamunge__(3056, obj, cell, slot);
    end
    function r = depth (obj, cell)
      % depth  Depth (= level used there if it is a leaf).
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
      %   C++: int datamunge::fda::TruncatedHierarchicalBSpline::depth(int cell) const
      r = __datamunge__(3057, obj, cell);
    end
    function r = cell_min (obj, cell)
      % cell_min  Lower corner.
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
      %   C++: std::vector<double> datamunge::fda::TruncatedHierarchicalBSpline::cell_min(int cell) const
      r = __datamunge__(3058, obj, cell);
    end
    function r = cell_max (obj, cell)
      % cell_max  Upper corner.
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
      %   C++: std::vector<double> datamunge::fda::TruncatedHierarchicalBSpline::cell_max(int cell) const
      r = __datamunge__(3059, obj, cell);
    end
    function r = locate (obj, point)
      % locate  Deepest cell containing `point` (its depth is the finest level there).
      %
      %   r = obj.locate (point)
      %
      %   Inputs:
      %     point  double vector: d coordinates.
      %
      %   Output:
      %     r  integer: Cell id, or -1 outside the domain.
      %
      %   Errors:
      %     datamunge:invalid_argument  If point.size() != dimensions(). (C++ std::invalid_argument)
      %
      %   C++: int datamunge::fda::TruncatedHierarchicalBSpline::locate(const std::vector<double> & point) const
      r = __datamunge__(3060, obj, point);
    end
    function r = refine (obj, cell, slot)
      % refine  Creates (if needed) the child of `cell` at `slot`, materializing the next level when first reached. The represented function is unchanged.
      %
      %   r = obj.refine (cell, slot)
      %
      %   Example:
      %     auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
      %     dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, {0.0, 1.0}));
      %     thb.refine(thb.root(), 0);
      %     std::print("{} {}\n", thb.level_count(), thb.evaluate({0.25})); // 2 0.25
      %
      %   Inputs:
      %     cell  integer: Cell id.
      %     slot  integer: Child slot.
      %
      %   Output:
      %     r  integer: Child id.
      %
      %   Errors:
      %     datamunge:out_of_range  For an invalid id or slot. (C++ std::out_of_range)
      %
      %   C++: int datamunge::fda::TruncatedHierarchicalBSpline::refine(int cell, int slot)
      r = __datamunge__(3061, obj, cell, slot);
    end
    function r = subdivide (obj, cell)
      % subdivide  Creates all 2^d children of `cell`.
      %
      %   r = obj.subdivide (cell)
      %
      %   Example:
      %   See refine().
      %
      %   Inputs:
      %     cell  integer: Cell id.
      %
      %   Output:
      %     r  integer vector: Child ids in slot order.
      %
      %   Errors:
      %     datamunge:out_of_range  For an invalid id. (C++ std::out_of_range)
      %
      %   C++: std::vector<int> datamunge::fda::TruncatedHierarchicalBSpline::subdivide(int cell)
      r = __datamunge__(3062, obj, cell);
    end
    function refine_region (obj, region_min, region_max, depth)
      % refine_region  Subdivides, down to depth `depth`, every cell overlapping the open box (region_min, region_max). The represented function is unchanged.
      %
      %   obj.refine_region (region_min, region_max, depth)
      %
      %   Example:
      %   See the class example.
      %
      %   Inputs:
      %     region_min  double vector: Lower corner.
      %     region_max  double vector: Upper corner (min < max per axis).
      %     depth  integer: Deepest level to create.
      %
      %   Errors:
      %     datamunge:invalid_argument  On mismatched sizes or min >= max. (C++ std::invalid_argument)
      %
      %   C++: void datamunge::fda::TruncatedHierarchicalBSpline::refine_region(const std::vector<double> & region_min, const std::vector<double> & region_max, int depth)
      __datamunge__(3063, obj, region_min, region_max, depth);
    end
    function r = support_min (obj, level, index)
      % support_min  Lower corner of the support box of basis function `index` at `level`.
      %
      %   r = obj.support_min (level, index)
      %
      %   Inputs:
      %     level  integer: Level.
      %     index  integer vector: Multi-index into that level's net.
      %
      %   Output:
      %     r  double vector: d coordinates.
      %
      %   Errors:
      %     datamunge:out_of_range  For invalid levels/indices. (C++ std::out_of_range)
      %     datamunge:invalid_argument  On a dimension mismatch. (C++ std::invalid_argument)
      %
      %   C++: std::vector<double> datamunge::fda::TruncatedHierarchicalBSpline::support_min(int level, const std::vector<int> & index) const
      r = __datamunge__(3064, obj, level, index);
    end
    function r = support_max (obj, level, index)
      % support_max  Upper corner of the support box.
      %
      %   r = obj.support_max (level, index)
      %
      %   Inputs:
      %     level  integer: Level.
      %     index  integer vector: Multi-index.
      %
      %   Output:
      %     r  double vector: d coordinates.
      %
      %   Errors:
      %     datamunge:out_of_range  For invalid levels/indices. (C++ std::out_of_range)
      %     datamunge:invalid_argument  On a dimension mismatch. (C++ std::invalid_argument)
      %
      %   C++: std::vector<double> datamunge::fda::TruncatedHierarchicalBSpline::support_max(int level, const std::vector<int> & index) const
      r = __datamunge__(3065, obj, level, index);
    end
    function r = is_active (obj, level, index)
      % is_active  Whether level- `level` function `index` is in the THB basis (support in Omega^level but not in Omega^(level+1)).
      %
      %   r = obj.is_active (level, index)
      %
      %   Example:
      %     auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
      %     dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, {0.0, 1.0}));
      %     thb.subdivide(thb.root());
      %     std::print("{} {}\n", thb.is_active(0, {0}), thb.is_active(1, {0})); // false true
      %
      %   Inputs:
      %     level  integer: Level.
      %     index  integer vector: Multi-index.
      %
      %   Output:
      %     r  logical: Activity flag.
      %
      %   Errors:
      %     datamunge:out_of_range  For invalid levels/indices. (C++ std::out_of_range)
      %     datamunge:invalid_argument  On a dimension mismatch. (C++ std::invalid_argument)
      %
      %   C++: bool datamunge::fda::TruncatedHierarchicalBSpline::is_active(int level, const std::vector<int> & index) const
      r = __datamunge__(3066, obj, level, index);
    end
    function r = active_indices (obj, level)
      % active_indices  Every active multi-index at `level`, in row-major order.
      %
      %   r = obj.active_indices (level)
      %
      %   Example:
      %     auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
      %     dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, {0.0, 1.0}));
      %     std::print("{}\n", thb.active_indices(0).size()); // 2
      %
      %   Inputs:
      %     level  integer: Level.
      %
      %   Output:
      %     r  cell array of integer vector: Multi-indices.
      %
      %   Errors:
      %     datamunge:out_of_range  For an invalid level. (C++ std::out_of_range)
      %
      %   C++: std::vector<std::vector<int>> datamunge::fda::TruncatedHierarchicalBSpline::active_indices(int level) const
      r = __datamunge__(3067, obj, level);
    end
    function r = basis_size (obj)
      % basis_size  Total number of THB basis functions over all levels.
      %
      %   r = obj.basis_size ()
      %
      %   Example:
      %     auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
      %     dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, {0.0, 1.0}));
      %     thb.refine(thb.root(), 0);
      %     std::print("{}\n", thb.basis_size()); // 3
      %
      %   Output:
      %     r  integer: Count.
      %
      %   C++: std::size_t datamunge::fda::TruncatedHierarchicalBSpline::basis_size() const
      r = __datamunge__(3068, obj);
    end
    function r = coefficient (obj, level, index)
      % coefficient  THB coefficient of a basis function (for active functions; inactive entries are internal bookkeeping).
      %
      %   r = obj.coefficient (level, index)
      %
      %   Inputs:
      %     level  integer: Level.
      %     index  integer vector: Multi-index.
      %
      %   Output:
      %     r  double: Coefficient.
      %
      %   Errors:
      %     datamunge:out_of_range  For invalid levels/indices. (C++ std::out_of_range)
      %     datamunge:invalid_argument  On a dimension mismatch. (C++ std::invalid_argument)
      %
      %   C++: double datamunge::fda::TruncatedHierarchicalBSpline::coefficient(int level, const std::vector<int> & index) const
      r = __datamunge__(3069, obj, level, index);
    end
    function set_coefficient (obj, level, index, value_)
      % set_coefficient  Sets the coefficient of an active function and propagates the change, with truncation, to every finer level. The edit is local to the function's support.
      %
      %   obj.set_coefficient (level, index, value_)
      %
      %   Example:
      %     auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
      %     dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, {0.0, 0.0}));
      %     thb.set_coefficient(0, {1}, 2.0);
      %     std::print("{}\n", thb.evaluate({0.5})); // 1
      %
      %   Inputs:
      %     level  integer: Level.
      %     index  integer vector: Active multi-index.
      %     value_  double: New coefficient.
      %
      %   Errors:
      %     datamunge:out_of_range  For invalid levels/indices. (C++ std::out_of_range)
      %     datamunge:invalid_argument  On a dimension mismatch or an inactive index. (C++ std::invalid_argument)
      %
      %   C++: void datamunge::fda::TruncatedHierarchicalBSpline::set_coefficient(int level, const std::vector<int> & index, double value)
      __datamunge__(3070, obj, level, index, value_);
    end
    function r = truncated_basis (obj, level, index, point)
      % truncated_basis  Value of the truncated basis function (level, index) at `point`.
      %
      %   r = obj.truncated_basis (level, index, point)
      %
      %   Example:
      %     auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
      %     dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, {0.0, 1.0}));
      %     std::print("{}\n", thb.truncated_basis(0, {1}, {0.25})); // 0.25
      %
      %   Inputs:
      %     level  integer: Level.
      %     index  integer vector: Multi-index (inactive functions give 0).
      %     point  double vector: d coordinates inside the domain.
      %
      %   Output:
      %     r  double: Truncated basis value.
      %
      %   Errors:
      %     datamunge:out_of_range  For invalid levels/indices. (C++ std::out_of_range)
      %     datamunge:invalid_argument  On a size mismatch or a point outside the domain. (C++ std::invalid_argument)
      %
      %   C++: double datamunge::fda::TruncatedHierarchicalBSpline::truncated_basis(int level, const std::vector<int> & index, const std::vector<double> & point) const
      r = __datamunge__(3071, obj, level, index, point);
    end
    function fit_level (obj, varargin)
      % fit_level  Least-squares update of the active coefficients at `level` only.
      %
      %   obj.fit_level (level, points, values)
      %   obj.fit_level (level, points, values, options)
      %
      %   Solves for the change of the level's active coefficients that best explains the
      %   residual values - evaluate(points), using the exact truncated basis, so pass raw
      %   data values. Points shallower than `level` are unaffected by these functions and
      %   are ignored. Typical use is coarse-to-fine: fit level 0, refine where residuals are
      %   large, then fit the new level.
      %
      %   Example:
      %     auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
      %     dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, {0.0, 0.0}));
      %     thb.fit_level(0, {{0.0}, {1.0}}, {1.0, 3.0});
      %     std::print("{}\n", thb.evaluate({0.5})); // 2
      %
      %   Inputs:
      %     level  integer: Level with at least one active function.
      %     points  double matrix (rows): Sample locations inside the domain.
      %     values  double vector: One raw value per point.
      %     options  datamunge.SplineSolveOptions, default the C++ default: Solver settings.
      %
      %   Errors:
      %     datamunge:out_of_range  For an invalid level. (C++ std::out_of_range)
      %     datamunge:invalid_argument  On empty/mismatched inputs, points outside the domain, or a level with no active functions. (C++ std::invalid_argument)
      %     datamunge:native  If the normal equations are singular. (C++ std::runtime_error)
      %
      %   C++: void datamunge::fda::TruncatedHierarchicalBSpline::fit_level(int level, const std::vector<std::vector<double>> & points, const std::vector<double> & values, const SplineSolveOptions & options = {})
      __datamunge__(3072, obj, varargin{:});
    end
    function fit_level_penalized (obj, varargin)
      % fit_level_penalized  Penalized variant of fit_level(): adds the level's full-grid tensor difference penalty restricted to its active functions, applied to the coefficient change.
      %
      %   obj.fit_level_penalized (level, points, values, lambdas)
      %   obj.fit_level_penalized (level, points, values, lambdas, order)
      %   obj.fit_level_penalized (level, points, values, lambdas, order, options)
      %
      %   Example:
      %     auto b = dm::fda::BSplineBasis::open_uniform(3, 8);
      %     dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, std::vector<double>(8, 0.0)));
      %     thb.fit_level_penalized(0, {{0.0}, {1.0}}, {0.0, 1.0}, {1e3});
      %     std::print("{:.3f}\n", thb.evaluate({0.5})); // 0.500
      %
      %   Inputs:
      %     level  integer: Level.
      %     points  double matrix (rows): Sample locations.
      %     values  double vector: Raw values.
      %     lambdas  double vector: Nonnegative per-axis weights.
      %     order  integer, default 2: Difference order (default 2).
      %     options  datamunge.SplineSolveOptions, default the C++ default: Solver settings.
      %
      %   Errors:
      %     datamunge:out_of_range  For an invalid level. (C++ std::out_of_range)
      %     datamunge:invalid_argument  As for fit_level() and tensor_difference_penalty(). (C++ std::invalid_argument)
      %     datamunge:native  If the normal equations are singular. (C++ std::runtime_error)
      %
      %   C++: void datamunge::fda::TruncatedHierarchicalBSpline::fit_level_penalized(int level, const std::vector<std::vector<double>> & points, const std::vector<double> & values, const std::vector<double> & lambdas, int order = 2, const SplineSolveOptions & options = {})
      __datamunge__(3073, obj, varargin{:});
    end
    function fit (obj, varargin)
      % fit  Least-squares fit of every active coefficient on every level at once (replaces the current function).
      %
      %   obj.fit (points, values)
      %   obj.fit (points, values, options)
      %
      %   Example:
      %     auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
      %     dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, {0.0, 0.0}));
      %     thb.refine(thb.root(), 1);                      // adds a knot at 0.5
      %     thb.fit({{0.0}, {0.25}, {0.5}, {0.75}, {1.0}}, {1.0, 0.5, 0.0, 0.5, 1.0}); // |2x - 1|
      %     std::print("{:.3f}\n", thb.evaluate({0.75})); // 0.500
      %
      %   Inputs:
      %     points  double matrix (rows): Sample locations inside the domain.
      %     values  double vector: One value per point.
      %     options  datamunge.SplineSolveOptions, default the C++ default: Solver settings.
      %
      %   Errors:
      %     datamunge:invalid_argument  On empty/mismatched inputs or points outside the domain. (C++ std::invalid_argument)
      %     datamunge:native  If some active function has too little data (singular). (C++ std::runtime_error)
      %
      %   C++: void datamunge::fda::TruncatedHierarchicalBSpline::fit(const std::vector<std::vector<double>> & points, const std::vector<double> & values, const SplineSolveOptions & options = {})
      __datamunge__(3074, obj, varargin{:});
    end
    function fit_penalized (obj, varargin)
      % fit_penalized  Penalized all-level fit: adds lambda_k times the squared order-q differences along axis k of every run of q + 1 neighbouring active coefficients of a level.
      %
      %   obj.fit_penalized (points, values, lambdas)
      %   obj.fit_penalized (points, values, lambdas, order)
      %   obj.fit_penalized (points, values, lambdas, order, options)
      %
      %   Differences that would involve an inactive coefficient are omitted, so for q >= 1
      %   constant coefficients (and, generally, polynomials of degree < q in the coefficient
      %   index on each level) are unpenalized; q = 0 is ridge regression on all coefficients.
      %
      %   Example:
      %     auto b = dm::fda::BSplineBasis::open_uniform(2, 6);
      %     dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, std::vector<double>(6, 0.0)));
      %     thb.refine_region({0.0}, {0.5}, 1);
      %     thb.fit_penalized({{0.0}, {1.0}}, {2.0, 2.0}, {1.0}, 1);
      %     std::print("{:.3f}\n", thb.evaluate({0.3})); // 2.000
      %
      %   Inputs:
      %     points  double matrix (rows): Sample locations.
      %     values  double vector: One value per point.
      %     lambdas  double vector: Nonnegative per-axis weights.
      %     order  integer, default 2: Difference order (default 2).
      %     options  datamunge.SplineSolveOptions, default the C++ default: Solver settings.
      %
      %   Errors:
      %     datamunge:invalid_argument  As for fit() and tensor_difference_penalty(). (C++ std::invalid_argument)
      %     datamunge:native  If the normal equations are singular. (C++ std::runtime_error)
      %
      %   C++: void datamunge::fda::TruncatedHierarchicalBSpline::fit_penalized(const std::vector<std::vector<double>> & points, const std::vector<double> & values, const std::vector<double> & lambdas, int order = 2, const SplineSolveOptions & options = {})
      __datamunge__(3075, obj, varargin{:});
    end
    function r = evaluate (obj, point)
      % evaluate  Evaluates the hierarchical spline (finest level covering `point`).
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
      %     r  double: Value.
      %
      %   Errors:
      %     datamunge:invalid_argument  On a size mismatch or a point outside the domain. (C++ std::invalid_argument)
      %
      %   C++: double datamunge::fda::TruncatedHierarchicalBSpline::evaluate(const std::vector<double> & point) const
      r = __datamunge__(3076, obj, point);
    end
    function r = evaluate_points (obj, points)
      % evaluate_points  Evaluates at many points.
      %
      %   r = obj.evaluate_points (points)
      %
      %   Example:
      %     auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
      %     dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, {0.0, 1.0}));
      %     std::print("{}\n", thb.evaluate_points({{0.0}, {0.5}})); // [0, 0.5]
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
      %   C++: std::vector<double> datamunge::fda::TruncatedHierarchicalBSpline::evaluate_points(const std::vector<std::vector<double>> & points) const
      r = __datamunge__(3077, obj, points);
    end
    function r = evaluate_grid (obj, axes)
      % evaluate_grid  Evaluates on a Cartesian grid.
      %
      %   r = obj.evaluate_grid (axes)
      %
      %   Example:
      %     auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
      %     dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, {0.0, 1.0}));
      %     std::print("{}\n", thb.evaluate_grid({{0.0, 1.0}})); // [0, 1]
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
      %   C++: std::vector<double> datamunge::fda::TruncatedHierarchicalBSpline::evaluate_grid(const std::vector<std::vector<double>> & axes) const
      r = __datamunge__(3078, obj, axes);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::fda::TruncatedHierarchicalBSpline. Do not edit.
