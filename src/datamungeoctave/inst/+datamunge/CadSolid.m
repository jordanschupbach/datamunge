classdef CadSolid < datamunge.NativeObject
  % CadSolid  An owned B-rep solid with value-returning modelling and export operations. Geometry uses doubles in caller-selected consistent units. Copies own topology; immutable surfaces may be shared. Operations return new solids, preserving inputs. Unsupported kernel operations throw std::runtime_error with the kernel diagnostic. Concurrent const access is safe; do not mutate a shared object concurrently.
  %
  %   Example:
  %     auto block = dm::cad::Solid::box(4, 3, 2);
  %     auto hole = dm::cad::Solid::cylinder(.4, 4).translated(2, 1.5, -1);
  %     auto part = block.subtracted(hole);
  %     part.save("part.step");
  %
  %   Methods:
  %     united  Return the union with another solid.
  %     subtracted  Subtract another solid, preserving cavities and trimmed faces.
  %     intersected  Return the common material of two solids.
  %     translated  Return a translated copy.
  %     rotated  Rotate about an axis through the origin, using the right-hand rule.
  %     filleted  Fillet selected edge IDs in sequence using persistent face-pair references.
  %     chamfered  Chamfer selected edge IDs by a common setback.
  %     shelled  Hollow a solid, optionally opening specified faces.
  %     volume  Measure volume from the trimmed tessellated boundary.
  %     valid  Validate topology, orientations and geometric boundary consistency.
  %     edges  List the source edge IDs usable by filleted() and chamfered().
  %     faces  List face IDs usable by shelled() and the low-level modifiers.
  %     save  Save STEP, binary STL, OBJ, glTF or an SVG drawing by extension.
  %
  %   Static methods:
  %     box  Construct an axis-aligned box with minimum corner at the origin.
  %     cylinder  Construct a cylinder with base at the origin and axis +Z.
  %     sphere  Construct a sphere centred at the origin.
  %     torus  Construct a ring torus centred at the origin around +Z.
  %
  %   C++: class datamunge::cad::Solid
  %
  %   See also: datamunge.Model, datamunge.FeatureTree, datamunge.load_scene()
  methods
    function obj = CadSolid (varargin)
      % CadSolid  Construct a C++ datamunge::cad::Solid.
      %
      %   It has no public constructor: C++ functions return its objects.
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      error ('datamunge:binding', '%s', 'datamunge.CadSolid has no public constructor');
    end
    function r = united (obj, other)
      % united  Return the union with another solid.
      %
      %   r = obj.united (other)
      %
      %   Example:
      %     auto a = dm::cad::Solid::box(2, 2, 2);
      %     auto b = a.translated(.5, .5, .5);
      %     auto joined = a.united(b);
      %
      %   Inputs:
      %     other  datamunge.CadSolid: Borrowed operand, in the same units/frame.
      %
      %   Output:
      %     r  datamunge.CadSolid: New solid; inputs unchanged.
      %
      %   Errors:
      %     datamunge:native  Kernel failure. (C++ std::runtime_error)
      %
      %   C++: Solid datamunge::cad::Solid::united(const Solid & other) const
      r = __datamunge__(2932, obj, other);
    end
    function r = subtracted (obj, other)
      % subtracted  Subtract another solid, preserving cavities and trimmed faces.
      %
      %   r = obj.subtracted (other)
      %
      %   Example:
      %   See the class example.
      %
      %   Inputs:
      %     other  datamunge.CadSolid: Borrowed tool solid.
      %
      %   Output:
      %     r  datamunge.CadSolid: New solid.
      %
      %   Errors:
      %     datamunge:native  Kernel failure or empty result. (C++ std::runtime_error)
      %
      %   C++: Solid datamunge::cad::Solid::subtracted(const Solid & other) const
      r = __datamunge__(2933, obj, other);
    end
    function r = intersected (obj, other)
      % intersected  Return the common material of two solids.
      %
      %   r = obj.intersected (other)
      %
      %   Example:
      %     auto a = dm::cad::Solid::box(2, 2, 2);
      %     auto common = a.intersected(a.translated(.5, .5, .5));
      %
      %   Inputs:
      %     other  datamunge.CadSolid: Borrowed operand.
      %
      %   Output:
      %     r  datamunge.CadSolid: New solid.
      %
      %   Errors:
      %     datamunge:native  Kernel failure or empty result. (C++ std::runtime_error)
      %
      %   C++: Solid datamunge::cad::Solid::intersected(const Solid & other) const
      r = __datamunge__(2934, obj, other);
    end
    function r = translated (obj, x, y, z)
      % translated  Return a translated copy.
      %
      %   r = obj.translated (x, y, z)
      %
      %   Example:
      %   See the class example.
      %
      %   Inputs:
      %     x  double: Finite X displacement in model units.
      %     y  double: Finite Y displacement.
      %     z  double: Finite Z displacement.
      %
      %   Output:
      %     r  datamunge.CadSolid: New solid.
      %
      %   Errors:
      %     datamunge:invalid_argument  Nonfinite displacement. (C++ std::invalid_argument)
      %     datamunge:native  Transform failure. (C++ std::runtime_error)
      %
      %   C++: Solid datamunge::cad::Solid::translated(double x, double y, double z) const
      r = __datamunge__(2935, obj, x, y, z);
    end
    function r = rotated (obj, x, y, z, radians)
      % rotated  Rotate about an axis through the origin, using the right-hand rule.
      %
      %   r = obj.rotated (x, y, z, radians)
      %
      %   Example:
      %     auto turned = dm::cad::Solid::box(2, 3, 4).rotated(0, 0, 1, dm::cad::kHalfPi);
      %
      %   Inputs:
      %     x  double: Axis X component.
      %     y  double: Axis Y component.
      %     z  double: Axis Z component; axis must be finite and nonzero.
      %     radians  double: Finite rotation angle.
      %
      %   Output:
      %     r  datamunge.CadSolid: New solid.
      %
      %   Errors:
      %     datamunge:invalid_argument  Invalid axis or angle. (C++ std::invalid_argument)
      %     datamunge:native  Kernel failure. (C++ std::runtime_error)
      %
      %   C++: Solid datamunge::cad::Solid::rotated(double x, double y, double z, double radians) const
      r = __datamunge__(2936, obj, x, y, z, radians);
    end
    function r = filleted (obj, edges, radius)
      % filleted  Fillet selected edge IDs in sequence using persistent face-pair references.
      %
      %   r = obj.filleted (edges, radius)
      %
      %   Example:
      %     auto a = dm::cad::Solid::box(4, 4, 4);
      %     auto rounded = a.filleted({a.edges().front()}, .2);
      %
      %   Inputs:
      %     edges  integer vector: Edge IDs from this solid's model; empty list leaves a copy unchanged.
      %     radius  double: Positive finite rolling-ball radius.
      %
      %   Output:
      %     r  datamunge.CadSolid: New solid. Source kernel supports straight edges between planar faces.
      %
      %   Errors:
      %     datamunge:invalid_argument  Invalid radius. (C++ std::invalid_argument)
      %     datamunge:native  Unsupported edge/blend geometry. (C++ std::runtime_error)
      %
      %   C++: Solid datamunge::cad::Solid::filleted(const std::vector<int> & edges, double radius) const
      r = __datamunge__(2937, obj, edges, radius);
    end
    function r = chamfered (obj, edges, setback)
      % chamfered  Chamfer selected edge IDs by a common setback.
      %
      %   r = obj.chamfered (edges, setback)
      %
      %   Example:
      %     auto a = dm::cad::Solid::box(4, 4, 4);
      %     auto bevel = a.chamfered({a.edges().front()}, .2);
      %
      %   Inputs:
      %     edges  integer vector: Source edge IDs; empty list returns a copy.
      %     setback  double: Positive finite distance along adjacent faces.
      %
      %   Output:
      %     r  datamunge.CadSolid: New solid.
      %
      %   Errors:
      %     datamunge:invalid_argument  Invalid setback. (C++ std::invalid_argument)
      %     datamunge:native  Kernel failure. (C++ std::runtime_error)
      %
      %   C++: Solid datamunge::cad::Solid::chamfered(const std::vector<int> & edges, double setback) const
      r = __datamunge__(2938, obj, edges, setback);
    end
    function r = shelled (obj, varargin)
      % shelled  Hollow a solid, optionally opening specified faces.
      %
      %   r = obj.shelled (thickness)
      %   r = obj.shelled (thickness, open_faces)
      %
      %   Example:
      %     auto hollow = dm::cad::Solid::box(4, 4, 4).shelled(.2);
      %
      %   Inputs:
      %     thickness  double: Positive finite wall thickness.
      %     open_faces  integer vector, default the C++ default: Face IDs to open; empty creates a sealed cavity.
      %
      %   Output:
      %     r  datamunge.CadSolid: New solid, subject to ShellBody's supported surface combinations.
      %
      %   Errors:
      %     datamunge:invalid_argument  Invalid thickness. (C++ std::invalid_argument)
      %     datamunge:native  Kernel failure. (C++ std::runtime_error)
      %
      %   C++: Solid datamunge::cad::Solid::shelled(double thickness, const std::vector<int> & open_faces = {}) const
      r = __datamunge__(2939, obj, varargin{:});
    end
    function r = volume (obj, varargin)
      % volume  Measure volume from the trimmed tessellated boundary.
      %
      %   r = obj.volume ()
      %   r = obj.volume (chord)
      %
      %   Example:
      %     auto volume = dm::cad::Solid::box(2, 3, 4).volume(); // 24
      %
      %   Inputs:
      %     chord  double, default 0.01: Positive tessellation tolerance; lower improves curved-face accuracy.
      %
      %   Output:
      %     r  double: Signed volume in cubic model units; honours imported STEP trimming.
      %
      %   Errors:
      %     datamunge:invalid_argument  Invalid tolerance. (C++ std::invalid_argument)
      %     datamunge:native  Tessellation failure. (C++ std::runtime_error)
      %
      %   C++: double datamunge::cad::Solid::volume(double chord = 0.01) const
      r = __datamunge__(2940, obj, varargin{:});
    end
    function r = valid (obj)
      % valid  Validate topology, orientations and geometric boundary consistency.
      %
      %   r = obj.valid ()
      %
      %   Example:
      %     bool valid = dm::cad::Solid::box(2, 3, 4).valid(); // true
      %
      %   Output:
      %     r  logical: True when the kernel validation report has no errors.
      %
      %   C++: bool datamunge::cad::Solid::valid() const
      r = __datamunge__(2941, obj);
    end
    function r = edges (obj)
      % edges  List the source edge IDs usable by filleted() and chamfered().
      %
      %   r = obj.edges ()
      %
      %   Output:
      %     r  integer vector: Owned vector of unique edge IDs.
      %
      %   C++: std::vector<int> datamunge::cad::Solid::edges() const
      r = __datamunge__(2942, obj);
    end
    function r = faces (obj)
      % faces  List face IDs usable by shelled() and the low-level modifiers.
      %
      %   r = obj.faces ()
      %
      %   Output:
      %     r  integer vector: Owned vector of face IDs.
      %
      %   C++: std::vector<int> datamunge::cad::Solid::faces() const
      r = __datamunge__(2943, obj);
    end
    function save (obj, varargin)
      % save  Save STEP, binary STL, OBJ, glTF or an SVG drawing by extension.
      %
      %   obj.save (path)
      %   obj.save (path, chord)
      %
      %   Example:
      %   See the class example.
      %
      %   Inputs:
      %     path  char row: Output path; parent directory must exist; existing files are replaced.
      %     chord  double, default 0.01: Positive tessellation tolerance for mesh/drawing output; default .01.
      %
      %   Errors:
      %     datamunge:invalid_argument  Unsupported extension or tolerance. (C++ std::invalid_argument)
      %     datamunge:native  Export or file I/O failure. (C++ std::runtime_error)
      %
      %   C++: void datamunge::cad::Solid::save(const std::string & path, double chord = 0.01) const
      __datamunge__(2944, obj, varargin{:});
    end
  end
  methods (Static)
    function r = box (width, depth, height)
      % box  Construct an axis-aligned box with minimum corner at the origin.
      %
      %   r = datamunge.CadSolid.box (width, depth, height)
      %
      %   Example:
      %   See the class example.
      %
      %   Inputs:
      %     width  double: Positive finite X extent.
      %     depth  double: Positive finite Y extent.
      %     height  double: Positive finite Z extent.
      %
      %   Output:
      %     r  datamunge.CadSolid: Owned solid.
      %
      %   Errors:
      %     datamunge:invalid_argument  Nonpositive/nonfinite size. (C++ std::invalid_argument)
      %
      %   C++: Solid datamunge::cad::Solid::box(double width, double depth, double height)
      r = __datamunge__(2928, width, depth, height);
    end
    function r = cylinder (radius, height)
      % cylinder  Construct a cylinder with base at the origin and axis +Z.
      %
      %   r = datamunge.CadSolid.cylinder (radius, height)
      %
      %   Example:
      %   See the class example.
      %
      %   Inputs:
      %     radius  double: Positive finite radius.
      %     height  double: Positive finite height.
      %
      %   Output:
      %     r  datamunge.CadSolid: Owned solid.
      %
      %   Errors:
      %     datamunge:invalid_argument  Invalid dimensions. (C++ std::invalid_argument)
      %
      %   C++: Solid datamunge::cad::Solid::cylinder(double radius, double height)
      r = __datamunge__(2929, radius, height);
    end
    function r = sphere (radius)
      % sphere  Construct a sphere centred at the origin.
      %
      %   r = datamunge.CadSolid.sphere (radius)
      %
      %   Example:
      %     auto ball = dm::cad::Solid::sphere(2).translated(0, 0, 2);
      %
      %   Inputs:
      %     radius  double: Positive finite radius.
      %
      %   Output:
      %     r  datamunge.CadSolid: Owned solid.
      %
      %   Errors:
      %     datamunge:invalid_argument  Invalid radius. (C++ std::invalid_argument)
      %
      %   C++: Solid datamunge::cad::Solid::sphere(double radius)
      r = __datamunge__(2930, radius);
    end
    function r = torus (major_radius, minor_radius)
      % torus  Construct a ring torus centred at the origin around +Z.
      %
      %   r = datamunge.CadSolid.torus (major_radius, minor_radius)
      %
      %   Example:
      %     auto ring = dm::cad::Solid::torus(3, 1);
      %
      %   Inputs:
      %     major_radius  double: Positive finite distance from axis to tube centre.
      %     minor_radius  double: Positive finite tube radius smaller than major_radius.
      %
      %   Output:
      %     r  datamunge.CadSolid: Owned solid.
      %
      %   Errors:
      %     datamunge:invalid_argument  Invalid radii. (C++ std::invalid_argument)
      %
      %   C++: Solid datamunge::cad::Solid::torus(double major_radius, double minor_radius)
      r = __datamunge__(2931, major_radius, minor_radius);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::cad::Solid. Do not edit.
