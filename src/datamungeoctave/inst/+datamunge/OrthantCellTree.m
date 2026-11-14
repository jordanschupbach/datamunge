classdef OrthantCellTree < datamunge.NativeObject
  % OrthantCellTree  Dyadic 2^d-ary tree of axis-aligned boxes shared by the hierarchical splines.
  %
  %   obj = datamunge.OrthantCellTree (lo, hi)
  %
  %   Cells live in a flat vector and are addressed by id (root 0). Children are created on
  %   demand by halving every axis; slot bit k selects the upper half of axis k.
  %
  %   Methods:
  %     dimensions  Number of dimensions d.
  %     arity  Children per cell, 2^d.
  %     size_  Number of cells created so far.
  %     max_depth  Greatest depth of any cell.
  %     check  Throws std::out_of_range unless `cell` is a valid id.
  %     check_slot  Throws std::out_of_range unless 0 <= `slot` < arity().
  %     lo  Cell lower corner.
  %     hi  Cell upper corner.
  %     depth  Cell depth.
  %     parent  Parent id.
  %     child  Child id.
  %     is_leaf  True when no child of `cell` exists.
  %     ensure_child  Returns the child at `slot`, creating it if needed.
  %     child_box  Box of the (possibly not yet created) child at `slot`.
  %     slot_for  Slot of `cell's` child that contains `point` (upper half on ties at the max).
  %     contains  Whether `point` lies in the closed box of `cell`.
  %     locate  Deepest existing cell containing `point`.
  %     cells_at_depth  Ids of every cell at `depth`.
  %     covered_to_depth  Whether the part of [blo, bhi] inside the root is covered by cells of depth >= `target_depth` (i.e. the box lies in the refined domain Omega^target_depth).
  %
  %   Static methods:
  %     overlaps  Whether the open box (lo, hi) intersects the box [blo, bhi].
  %
  %   C++: class datamunge::fda::detail::OrthantCellTree
  methods
    function obj = OrthantCellTree (varargin)
      % OrthantCellTree  Construct a C++ datamunge::fda::detail::OrthantCellTree.
      %
      %   obj = datamunge.OrthantCellTree (lo, hi)
      %
      %   Creates the root cell [lo, hi].
      %
      %   Inputs:
      %     lo  double vector: Lower corner.
      %     hi  double vector: Upper corner, with lo[k] < hi[k].
      %
      %   Errors:
      %     datamunge:invalid_argument  On empty/mismatched/non-finite bounds, lo >= hi, or more than 20 dimensions. (C++ std::invalid_argument)
      %
      %   C++: datamunge::fda::detail::OrthantCellTree::OrthantCellTree(std::vector<double> lo, std::vector<double> hi)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 3011, false, varargin{:});
    end
    function r = dimensions (obj)
      % dimensions  Number of dimensions d.
      %
      %   r = obj.dimensions ()
      %
      %   Output:
      %     r  integer: d.
      %
      %   C++: std::size_t datamunge::fda::detail::OrthantCellTree::dimensions() const
      r = __datamunge__(2991, obj);
    end
    function r = arity (obj)
      % arity  Children per cell, 2^d.
      %
      %   r = obj.arity ()
      %
      %   Output:
      %     r  integer: 2^d.
      %
      %   C++: int datamunge::fda::detail::OrthantCellTree::arity() const
      r = __datamunge__(2992, obj);
    end
    function r = size_ (obj)
      % size_  Number of cells created so far.
      %
      %   r = obj.size_ ()
      %
      %   Output:
      %     r  integer: Cell count.
      %
      %   C++: int datamunge::fda::detail::OrthantCellTree::size() const
      r = __datamunge__(2993, obj);
    end
    function r = max_depth (obj)
      % max_depth  Greatest depth of any cell.
      %
      %   r = obj.max_depth ()
      %
      %   Output:
      %     r  integer: Maximum depth (root is 0).
      %
      %   C++: int datamunge::fda::detail::OrthantCellTree::max_depth() const
      r = __datamunge__(2994, obj);
    end
    function check (obj, cell)
      % check  Throws std::out_of_range unless `cell` is a valid id.
      %
      %   obj.check (cell)
      %
      %   Inputs:
      %     cell  integer: Cell id.
      %
      %   C++: void datamunge::fda::detail::OrthantCellTree::check(int cell) const
      __datamunge__(2995, obj, cell);
    end
    function check_slot (obj, slot)
      % check_slot  Throws std::out_of_range unless 0 <= `slot` < arity().
      %
      %   obj.check_slot (slot)
      %
      %   Inputs:
      %     slot  integer: Child slot.
      %
      %   C++: void datamunge::fda::detail::OrthantCellTree::check_slot(int slot) const
      __datamunge__(2996, obj, slot);
    end
    function r = lo (obj, cell)
      % lo  Cell lower corner.
      %
      %   r = obj.lo (cell)
      %
      %   Inputs:
      %     cell  integer: Cell id.
      %
      %   Output:
      %     r  double vector: d coordinates.
      %
      %   C++: const std::vector<double> & datamunge::fda::detail::OrthantCellTree::lo(int cell) const
      r = __datamunge__(2997, obj, cell);
    end
    function r = hi (obj, cell)
      % hi  Cell upper corner.
      %
      %   r = obj.hi (cell)
      %
      %   Inputs:
      %     cell  integer: Cell id.
      %
      %   Output:
      %     r  double vector: d coordinates.
      %
      %   C++: const std::vector<double> & datamunge::fda::detail::OrthantCellTree::hi(int cell) const
      r = __datamunge__(2998, obj, cell);
    end
    function r = depth (obj, cell)
      % depth  Cell depth.
      %
      %   r = obj.depth (cell)
      %
      %   Inputs:
      %     cell  integer: Cell id.
      %
      %   Output:
      %     r  integer: Depth (root 0).
      %
      %   C++: int datamunge::fda::detail::OrthantCellTree::depth(int cell) const
      r = __datamunge__(2999, obj, cell);
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
      %     r  integer: Parent id, or -1 for the root.
      %
      %   C++: int datamunge::fda::detail::OrthantCellTree::parent(int cell) const
      r = __datamunge__(3000, obj, cell);
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
      %     r  integer: Id or -1 if absent.
      %
      %   C++: int datamunge::fda::detail::OrthantCellTree::child(int cell, int slot) const
      r = __datamunge__(3001, obj, cell, slot);
    end
    function r = is_leaf (obj, cell)
      % is_leaf  True when no child of `cell` exists.
      %
      %   r = obj.is_leaf (cell)
      %
      %   Inputs:
      %     cell  integer: Cell id.
      %
      %   Output:
      %     r  logical: Leaf flag.
      %
      %   C++: bool datamunge::fda::detail::OrthantCellTree::is_leaf(int cell) const
      r = __datamunge__(3002, obj, cell);
    end
    function r = ensure_child (obj, cell, slot)
      % ensure_child  Returns the child at `slot`, creating it if needed.
      %
      %   r = obj.ensure_child (cell, slot)
      %
      %   Inputs:
      %     cell  integer: Parent id.
      %     slot  integer: Child slot.
      %
      %   Output:
      %     r  integer: Child id.
      %
      %   C++: int datamunge::fda::detail::OrthantCellTree::ensure_child(int cell, int slot)
      r = __datamunge__(3003, obj, cell, slot);
    end
    function r = child_box (obj, cell, slot)
      % child_box  Box of the (possibly not yet created) child at `slot`.
      %
      %   r = obj.child_box (cell, slot)
      %
      %   Inputs:
      %     cell  integer: Parent id.
      %     slot  integer: Child slot.
      %
      %   Output:
      %     r  cell {double vector, double vector}: Pair (lo, hi) of the child box.
      %
      %   C++: std::pair<std::vector<double>, std::vector<double>> datamunge::fda::detail::OrthantCellTree::child_box(int cell, int slot) const
      r = __datamunge__(3004, obj, cell, slot);
    end
    function r = slot_for (obj, cell, point)
      % slot_for  Slot of `cell's` child that contains `point` (upper half on ties at the max).
      %
      %   r = obj.slot_for (cell, point)
      %
      %   Inputs:
      %     cell  integer: Cell id.
      %     point  double vector: d coordinates.
      %
      %   Output:
      %     r  integer: Child slot.
      %
      %   C++: int datamunge::fda::detail::OrthantCellTree::slot_for(int cell, const std::vector<double> & point) const
      r = __datamunge__(3005, obj, cell, point);
    end
    function r = contains (obj, cell, point)
      % contains  Whether `point` lies in the closed box of `cell`.
      %
      %   r = obj.contains (cell, point)
      %
      %   Inputs:
      %     cell  integer: Cell id.
      %     point  double vector: d coordinates.
      %
      %   Output:
      %     r  logical: Containment flag.
      %
      %   C++: bool datamunge::fda::detail::OrthantCellTree::contains(int cell, const std::vector<double> & point) const
      r = __datamunge__(3006, obj, cell, point);
    end
    function r = locate (obj, point)
      % locate  Deepest existing cell containing `point`.
      %
      %   r = obj.locate (point)
      %
      %   Inputs:
      %     point  double vector: d coordinates.
      %
      %   Output:
      %     r  integer: Cell id, or -1 outside the root box.
      %
      %   C++: int datamunge::fda::detail::OrthantCellTree::locate(const std::vector<double> & point) const
      r = __datamunge__(3007, obj, point);
    end
    function r = cells_at_depth (obj, depth)
      % cells_at_depth  Ids of every cell at `depth`.
      %
      %   r = obj.cells_at_depth (depth)
      %
      %   Inputs:
      %     depth  integer: Depth.
      %
      %   Output:
      %     r  integer vector: Ids in creation order.
      %
      %   C++: const std::vector<int> & datamunge::fda::detail::OrthantCellTree::cells_at_depth(int depth) const
      r = __datamunge__(3008, obj, depth);
    end
    function r = covered_to_depth (obj, target_depth, blo, bhi)
      % covered_to_depth  Whether the part of [blo, bhi] inside the root is covered by cells of depth >= `target_depth` (i.e. the box lies in the refined domain Omega^target_depth).
      %
      %   r = obj.covered_to_depth (target_depth, blo, bhi)
      %
      %   Inputs:
      %     target_depth  integer: Depth to test.
      %     blo  double vector: Box lower corner.
      %     bhi  double vector: Box upper corner.
      %
      %   Output:
      %     r  logical: Coverage flag.
      %
      %   C++: bool datamunge::fda::detail::OrthantCellTree::covered_to_depth(int target_depth, const std::vector<double> & blo, const std::vector<double> & bhi) const
      r = __datamunge__(3010, obj, target_depth, blo, bhi);
    end
  end
  methods (Static)
    function r = overlaps (lo, hi, blo, bhi)
      % overlaps  Whether the open box (lo, hi) intersects the box [blo, bhi].
      %
      %   r = datamunge.OrthantCellTree.overlaps (lo, hi, blo, bhi)
      %
      %   Inputs:
      %     lo  double vector: Box A lower corner.
      %     hi  double vector: Box A upper corner.
      %     blo  double vector: Box B lower corner.
      %     bhi  double vector: Box B upper corner.
      %
      %   Output:
      %     r  logical: True when the interiors overlap.
      %
      %   C++: bool datamunge::fda::detail::OrthantCellTree::overlaps(const std::vector<double> & lo, const std::vector<double> & hi, const std::vector<double> & blo, const std::vector<double> & bhi)
      r = __datamunge__(3009, lo, hi, blo, bhi);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::fda::detail::OrthantCellTree. Do not edit.
