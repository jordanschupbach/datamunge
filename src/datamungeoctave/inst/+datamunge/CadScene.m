classdef CadScene < datamunge.NativeObject
  % CadScene  A small code-driven scene facade, exposed as CadScene in Python/R. Owns scene/camera/render controls. All operations delegate to the canonical CAD, polygon and renderer APIs; copying retains independent scene state.
  %
  %   obj = datamunge.CadScene ()
  %
  %   Example:
  %     dm::cad::View view;
  %     view.add(dm::cad::Solid::box(2, 3, 4), "#3b82f6", "block");
  %     view.size(640, 480); view.render("block.png");
  %
  %   Methods:
  %     set_joint  Set a movable independent joint on a view loaded from URDF.
  %     joint_names  List independent movable joints of a URDF view.
  %     add  Tessellate a solid and append it as a scene object.
  %     primitive  Append a source polygon modeller primitive.
  %     size_  Set the output dimensions used by render() and picking.
  %     samples  Set regular-grid antialiasing samples per axis.
  %     style  Select shaded, unlit, wireframe or shaded-wireframe output.
  %     shadows  Enable/disable shadows cast by the first visible light.
  %     camera  Set explicit camera position and target, with +Z as up.
  %     fit  Resume automatic perspective framing after an explicit camera.
  %     light  Add a directional light; the first explicit light replaces the fallback light.
  %     render  Render synchronously and save a PNG/JPEG/PPM color image.
  %     save_gltf  Save the scene as self-contained glTF with meshes, textures and lights.
  %     pick_object  Pick a source object using the current camera and image dimensions.
  %     object_count  Count scene objects, including empty groups.
  %     triangle_count  Count triangles over all object mesh references, including hidden objects.
  %
  %   Static methods:
  %     from_file  Import any format supported by load_scene().
  %     from_urdf  Load an articulated URDF visual model and its asset files.
  %
  %   C++: class datamunge::cad::View
  methods
    function obj = CadScene (varargin)
      % CadScene  Construct a C++ datamunge::cad::View.
      %
      %   obj = datamunge.CadScene ()
      %
      %   Example:
      %   See the class example.
      %
      %   Create an empty scene with automatic camera framing.
      %
      %   C++: datamunge::cad::View::View()
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2963, false, varargin{:});
    end
    function set_joint (obj, name, value_)
      % set_joint  Set a movable independent joint on a view loaded from URDF.
      %
      %   obj.set_joint (name, value_)
      %
      %   Example:
      %   See from_urdf().
      %
      %   Inputs:
      %     name  char row: Joint name.
      %     value_  double: Coordinate in radians/metres, within limits.
      %
      %   Errors:
      %     datamunge:invalid_argument  View has no robot or joint value is invalid. (C++ std::invalid_argument)
      %
      %   C++: void datamunge::cad::View::set_joint(const std::string & name, double value)
      __datamunge__(2947, obj, name, value_);
    end
    function r = joint_names (obj)
      % joint_names  List independent movable joints of a URDF view.
      %
      %   r = obj.joint_names ()
      %
      %   Output:
      %     r  cell array of char: Owned names; empty for ordinary scenes.
      %
      %   C++: std::vector<std::string> datamunge::cad::View::joint_names() const
      r = __datamunge__(2948, obj);
    end
    function r = add (obj, varargin)
      % add  Tessellate a solid and append it as a scene object.
      %
      %   r = obj.add (solid)
      %   r = obj.add (solid, color)
      %   r = obj.add (solid, color, name)
      %
      %   Example:
      %   See the class example.
      %
      %   Inputs:
      %     solid  datamunge.CadSolid: Borrowed source, unchanged.
      %     color  char row, default '#4080d9': Hex RGB/RGBA tint, default #4080d9.
      %     name  char row, default 'Solid': Object name, default Solid.
      %
      %   Output:
      %     r  integer: New object ID.
      %
      %   Errors:
      %     datamunge:invalid_argument  Invalid color. (C++ std::invalid_argument)
      %     datamunge:native  Tessellation failure. (C++ std::runtime_error)
      %
      %   C++: int datamunge::cad::View::add(const Solid & solid, const std::string & color = "#4080d9", const std::string & name = "Solid")
      r = __datamunge__(2949, obj, varargin{:});
    end
    function r = primitive (obj, kind)
      % primitive  Append a source polygon modeller primitive.
      %
      %   r = obj.primitive (kind)
      %
      %   Example:
      %     dm::cad::View view; view.primitive("torus"); view.render("ring.png");
      %
      %   Inputs:
      %     kind  char row: cube, sphere, cylinder, cone, plane, torus or wedge (lowercase).
      %
      %   Output:
      %     r  integer: New object ID; editable via scene() in C++.
      %
      %   Errors:
      %     datamunge:invalid_argument  Unknown kind. (C++ std::invalid_argument)
      %
      %   C++: int datamunge::cad::View::primitive(const std::string & kind)
      r = __datamunge__(2950, obj, kind);
    end
    function size_ (obj, width, height)
      % size_  Set the output dimensions used by render() and picking.
      %
      %   obj.size_ (width, height)
      %
      %   Example:
      %   See the class example.
      %
      %   Inputs:
      %     width  integer: Pixel width, 1..8192.
      %     height  integer: Pixel height, 1..8192.
      %
      %   Errors:
      %     datamunge:invalid_argument  Invalid dimensions. (C++ std::invalid_argument)
      %
      %   C++: void datamunge::cad::View::size(int width, int height)
      __datamunge__(2951, obj, width, height);
    end
    function samples (obj, count)
      % samples  Set regular-grid antialiasing samples per axis.
      %
      %   obj.samples (count)
      %
      %   Example:
      %     dm::cad::View view; view.samples(2);
      %
      %   Inputs:
      %     count  integer: 1..4, i.e. 1..16 rays/pixel.
      %
      %   Errors:
      %     datamunge:invalid_argument  Invalid count. (C++ std::invalid_argument)
      %
      %   C++: void datamunge::cad::View::samples(int count)
      __datamunge__(2952, obj, count);
    end
    function style (obj, name)
      % style  Select shaded, unlit, wireframe or shaded-wireframe output.
      %
      %   obj.style (name)
      %
      %   Example:
      %     dm::cad::View view; view.style("shaded-wireframe");
      %
      %   Inputs:
      %     name  char row: One of those exact lowercase names.
      %
      %   Errors:
      %     datamunge:invalid_argument  Unknown style. (C++ std::invalid_argument)
      %
      %   C++: void datamunge::cad::View::style(const std::string & name)
      __datamunge__(2953, obj, name);
    end
    function shadows (obj, enabled)
      % shadows  Enable/disable shadows cast by the first visible light.
      %
      %   obj.shadows (enabled)
      %
      %   Example:
      %     dm::cad::View view; view.shadows(false);
      %
      %   Inputs:
      %     enabled  logical: True enables triangle-ray shadow tests.
      %
      %   C++: void datamunge::cad::View::shadows(bool enabled)
      __datamunge__(2954, obj, enabled);
    end
    function camera (obj, x, y, z, target_x, target_y, target_z)
      % camera  Set explicit camera position and target, with +Z as up.
      %
      %   obj.camera (x, y, z, target_x, target_y, target_z)
      %
      %   Example:
      %     dm::cad::View view; view.camera(6,-8,5,0,0,0);
      %
      %   Inputs:
      %     x  double: Camera X position.
      %     y  double: Camera Y position.
      %     z  double: Camera Z position.
      %     target_x  double: Target X.
      %     target_y  double: Target Y.
      %     target_z  double: Target Z; all values finite, target distinct and view not parallel to +Z.
      %
      %   Errors:
      %     datamunge:invalid_argument  Invalid camera. (C++ std::invalid_argument)
      %
      %   C++: void datamunge::cad::View::camera(double x, double y, double z, double target_x, double target_y, double target_z)
      __datamunge__(2955, obj, x, y, z, target_x, target_y, target_z);
    end
    function fit (obj)
      % fit  Resume automatic perspective framing after an explicit camera.
      %
      %   obj.fit ()
      %
      %   Example:
      %     dm::cad::View view; view.fit();
      %
      %   C++: void datamunge::cad::View::fit()
      __datamunge__(2956, obj);
    end
    function r = light (obj, varargin)
      % light  Add a directional light; the first explicit light replaces the fallback light.
      %
      %   r = obj.light (x, y, z)
      %   r = obj.light (x, y, z, color)
      %   r = obj.light (x, y, z, color, intensity)
      %
      %   Example:
      %     dm::cad::View view; view.light(1,-2,3,"#ffffff",1.2);
      %
      %   Inputs:
      %     x  double: Direction X from surface toward light.
      %     y  double: Direction Y.
      %     z  double: Direction Z; finite nonzero vector.
      %     color  char row, default '#ffffff': Hex RGB/RGBA light color, default white; alpha unused.
      %     intensity  double, default 1.0: Nonnegative finite multiplier, default 1.
      %
      %   Output:
      %     r  integer: New light ID.
      %
      %   Errors:
      %     datamunge:invalid_argument  Invalid direction, color or intensity. (C++ std::invalid_argument)
      %
      %   C++: int datamunge::cad::View::light(double x, double y, double z, const std::string & color = "#ffffff", double intensity = 1.0)
      r = __datamunge__(2957, obj, varargin{:});
    end
    function render (obj, path)
      % render  Render synchronously and save a PNG/JPEG/PPM color image.
      %
      %   obj.render (path)
      %
      %   Example:
      %   See the class example.
      %
      %   Inputs:
      %     path  char row: Output filename, parent directory must exist.
      %
      %   Errors:
      %     datamunge:invalid_argument  Invalid scene or extension. (C++ std::invalid_argument)
      %     datamunge:native  File I/O failure. (C++ std::runtime_error)
      %
      %   C++: void datamunge::cad::View::render(const std::string & path) const
      __datamunge__(2958, obj, path);
    end
    function save_gltf (obj, path)
      % save_gltf  Save the scene as self-contained glTF with meshes, textures and lights.
      %
      %   obj.save_gltf (path)
      %
      %   Example:
      %     dm::cad::View view; view.primitive("cube"); view.save_gltf("cube.gltf");
      %
      %   Inputs:
      %     path  char row: Output .gltf filename; parent must exist.
      %
      %   Errors:
      %     datamunge:native  Export/I/O failure. (C++ std::runtime_error)
      %
      %   C++: void datamunge::cad::View::save_gltf(const std::string & path) const
      __datamunge__(2959, obj, path);
    end
    function r = pick_object (obj, x, y)
      % pick_object  Pick a source object using the current camera and image dimensions.
      %
      %   r = obj.pick_object (x, y)
      %
      %   Example:
      %     dm::cad::View view; view.primitive("cube"); int id=view.pick_object(400,300);
      %
      %   Inputs:
      %     x  double: Pixel X from left; continuous coordinates.
      %     y  double: Pixel Y from top.
      %
      %   Output:
      %     r  integer: Object ID or -1 for background.
      %
      %   Errors:
      %     datamunge:invalid_argument  Invalid ray or geometry. (C++ std::invalid_argument)
      %
      %   C++: int datamunge::cad::View::pick_object(double x, double y) const
      r = __datamunge__(2960, obj, x, y);
    end
    function r = object_count (obj)
      % object_count  Count scene objects, including empty groups.
      %
      %   r = obj.object_count ()
      %
      %   Output:
      %     r  integer: Object count.
      %
      %   C++: int datamunge::cad::View::object_count() const
      r = __datamunge__(2961, obj);
    end
    function r = triangle_count (obj)
      % triangle_count  Count triangles over all object mesh references, including hidden objects.
      %
      %   r = obj.triangle_count ()
      %
      %   Output:
      %     r  integer: Triangle count.
      %
      %   C++: int datamunge::cad::View::triangle_count() const
      r = __datamunge__(2962, obj);
    end
  end
  methods (Static)
    function r = from_file (path)
      % from_file  Import any format supported by load_scene().
      %
      %   r = datamunge.CadScene.from_file (path)
      %
      %   Example:
      %     auto view = dm::cad::View::from_file("part.step");
      %     view.render("part.png");
      %
      %   Inputs:
      %     path  char row: Input filename.
      %
      %   Output:
      %     r  datamunge.CadScene: Owned view with automatic framing.
      %
      %   Errors:
      %     datamunge:invalid_argument  Unsupported format. (C++ std::invalid_argument)
      %     datamunge:native  Import/I/O failure. (C++ std::runtime_error)
      %
      %   C++: View datamunge::cad::View::from_file(const std::string & path)
      r = __datamunge__(2945, path);
    end
    function r = from_urdf (path)
      % from_urdf  Load an articulated URDF visual model and its asset files.
      %
      %   r = datamunge.CadScene.from_urdf (path)
      %
      %   Example:
      %     auto view = dm::cad::View::from_urdf("robot.urdf");
      %     view.set_joint("joint2", -.6); view.render("robot.png");
      %
      %   Inputs:
      %     path  char row: URDF file; relative meshes resolve beside it.
      %
      %   Output:
      %     r  datamunge.CadScene: Owned view at the loader's initial joint coordinates.
      %
      %   Errors:
      %     datamunge:invalid_argument  Invalid URDF/joint data. (C++ std::invalid_argument)
      %     datamunge:native  Import failure. (C++ std::runtime_error)
      %
      %   C++: View datamunge::cad::View::from_urdf(const std::string & path)
      r = __datamunge__(2946, path);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::cad::View. Do not edit.
