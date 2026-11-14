classdef ShapeLayer < datamunge.NativeObject
  % ShapeLayer  SWIG-friendly facade for datamunge::gis::ShapeLayer — reads a shapefile (.shp geometry + .dbf attributes) and draws it as a map. Geometry access is flattened into per-shape/   per-part coordinate vectors rather than exposing gis::Shape's nested   vector<vector<Point2D>> directly, matching this facade layer's usual convention for   types SWIG can't bind cleanly.
  %
  %   Methods:
  %     size_  C++ datamunge::ShapeLayer::size.
  %     shape_type  One of "point", "polyline", "polygon", "multipoint", or "null" (an empty layer).
  %     bounds  {xmin, ymin, xmax, ymax}, from the shapefile's own declared bounding box.
  %     attributes  C++ datamunge::ShapeLayer::attributes.
  %     shape_kind  C++ datamunge::ShapeLayer::shape_kind.
  %     num_parts  Number of parts (rings for Polygon, lines for PolyLine; 0 for Point/MultiPoint, which use point_x()/point_y() instead).
  %     part_x  C++ datamunge::ShapeLayer::part_x.
  %     part_y  C++ datamunge::ShapeLayer::part_y.
  %     point_x  Every point's x/y in a Point or MultiPoint shape (a single element for Point).
  %     point_y  C++ datamunge::ShapeLayer::point_y.
  %     plot  C++ datamunge::ShapeLayer::plot.
  %
  %   Static methods:
  %     read  Reads "<path>.shp" and "<path>.dbf" (`path` may already end in one of those extensions, or in neither).
  %
  %   C++: class datamunge::ShapeLayer
  methods
    function obj = ShapeLayer (varargin)
      % ShapeLayer  Construct a C++ datamunge::ShapeLayer.
      %
      %   It has no public constructor: C++ functions return its objects.
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      error ('datamunge:binding', '%s', 'datamunge.ShapeLayer has no public constructor');
    end
    function r = size_ (obj)
      % size_  C++ datamunge::ShapeLayer::size.
      %
      %   r = obj.size_ ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::ShapeLayer::size() const
      r = __datamunge__(2108, obj);
    end
    function r = shape_type (obj)
      % shape_type  One of "point", "polyline", "polygon", "multipoint", or "null" (an empty layer).
      %
      %   r = obj.shape_type ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::ShapeLayer::shape_type() const
      r = __datamunge__(2109, obj);
    end
    function r = bounds (obj)
      % bounds  {xmin, ymin, xmax, ymax}, from the shapefile's own declared bounding box.
      %
      %   r = obj.bounds ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::ShapeLayer::bounds() const
      r = __datamunge__(2110, obj);
    end
    function r = attributes (obj)
      % attributes  C++ datamunge::ShapeLayer::attributes.
      %
      %   r = obj.attributes ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::ShapeLayer::attributes() const
      r = __datamunge__(2111, obj);
    end
    function r = shape_kind (obj, shape_index)
      % shape_kind  C++ datamunge::ShapeLayer::shape_kind.
      %
      %   r = obj.shape_kind (shape_index)
      %
      %   Inputs:
      %     shape_index  integer
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::ShapeLayer::shape_kind(std::size_t shape_index) const
      r = __datamunge__(2112, obj, shape_index);
    end
    function r = num_parts (obj, shape_index)
      % num_parts  Number of parts (rings for Polygon, lines for PolyLine; 0 for Point/MultiPoint, which use point_x()/point_y() instead).
      %
      %   r = obj.num_parts (shape_index)
      %
      %   Inputs:
      %     shape_index  integer
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::ShapeLayer::num_parts(std::size_t shape_index) const
      r = __datamunge__(2113, obj, shape_index);
    end
    function r = part_x (obj, shape_index, part_index)
      % part_x  C++ datamunge::ShapeLayer::part_x.
      %
      %   r = obj.part_x (shape_index, part_index)
      %
      %   Inputs:
      %     shape_index  integer
      %     part_index  integer
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::ShapeLayer::part_x(std::size_t shape_index, std::size_t part_index) const
      r = __datamunge__(2114, obj, shape_index, part_index);
    end
    function r = part_y (obj, shape_index, part_index)
      % part_y  C++ datamunge::ShapeLayer::part_y.
      %
      %   r = obj.part_y (shape_index, part_index)
      %
      %   Inputs:
      %     shape_index  integer
      %     part_index  integer
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::ShapeLayer::part_y(std::size_t shape_index, std::size_t part_index) const
      r = __datamunge__(2115, obj, shape_index, part_index);
    end
    function r = point_x (obj, shape_index)
      % point_x  Every point's x/y in a Point or MultiPoint shape (a single element for Point).
      %
      %   r = obj.point_x (shape_index)
      %
      %   Inputs:
      %     shape_index  integer
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::ShapeLayer::point_x(std::size_t shape_index) const
      r = __datamunge__(2116, obj, shape_index);
    end
    function r = point_y (obj, shape_index)
      % point_y  C++ datamunge::ShapeLayer::point_y.
      %
      %   r = obj.point_y (shape_index)
      %
      %   Inputs:
      %     shape_index  integer
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::ShapeLayer::point_y(std::size_t shape_index) const
      r = __datamunge__(2117, obj, shape_index);
    end
    function r = plot (obj, varargin)
      % plot  C++ datamunge::ShapeLayer::plot.
      %
      %   r = obj.plot ()
      %   r = obj.plot (fill_color)
      %   r = obj.plot (fill_color, border_color)
      %   r = obj.plot (fill_color, border_color, width)
      %   r = obj.plot (fill_color, border_color, width, height)
      %
      %   Inputs:
      %     fill_color  datamunge.RGB, default the C++ default
      %     border_color  datamunge.RGB, default the C++ default
      %     width  integer, default 800
      %     height  integer, default 800
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::ShapeLayer::plot(datamunge::graphics::RGB fill_color = ..., datamunge::graphics::RGB border_color = ..., std::size_t width = 800, std::size_t height = 800) const
      r = __datamunge__(2118, obj, varargin{:});
    end
  end
  methods (Static)
    function r = read (path)
      % read  Reads "<path>.shp" and "<path>.dbf" (`path` may already end in one of those extensions, or in neither).
      %
      %   r = datamunge.ShapeLayer.read (path)
      %
      %   Inputs:
      %     path  char row
      %
      %   Output:
      %     r  datamunge.ShapeLayer or []
      %
      %   C++: ShapeLayer * datamunge::ShapeLayer::read(const std::string & path)
      r = __datamunge__(2107, path);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::ShapeLayer. Do not edit.
