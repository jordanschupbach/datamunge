classdef DataSeries < datamunge.NativeObject
  % DataSeries  One drawn layer of a plot (points, a line, bars, a box, a polygon, a label, or segments), as stored by RPlot's layer verbs and read back through Plot::series().
  %
  %   obj = datamunge.DataSeries ()
  %
  %   A plain aggregate holding data coordinates and styling. Which fields matter depends on
  %   kind; each Kind value documents how it reads x, y, and the other fields. Layers are drawn
  %   in insertion order, after the reference lines, and every layer's coordinates widen the
  %   automatic axis ranges (bars also pull the y range down or up to 0). A nonempty label puts
  %   the layer in the automatic legend unless it is a Text layer. User code normally creates
  %   layers through RPlot verbs (points(), lines(), bars(), ...) and only reads this struct.
  %
  %   Example:
  %     auto plt = dm::Plot().points(dm::c(1.0, 2.0), dm::c(3.0, 4.0), "pts");
  %     const dm::graphics::DataSeries& s = plt.series()[0];
  %     std::print("{} {} {} {}\n", s.kind == dm::graphics::DataSeries::Kind::Scatter, s.x.size(),
  %                s.label, s.marker_size);  // true 2 pts 4
  %
  %   Properties:
  %     kind  datamunge.DataSeries_Kind value: How the layer is drawn (default Line).
  %     x  double vector: Horizontal data coordinates; meaning depends on kind.
  %     y  double vector: Vertical data coordinates; meaning depends on kind.
  %     label  char row: Legend text (or the drawn text for Kind::Text); empty omits it.
  %     color  datamunge.RGB: Layer color, including alpha (default RGBA{}, dark slate).
  %     stroke_width  double: Line, Segment, and outline-Polygon width in pixels (default 2).
  %     marker_size  double: Scatter marker radius, or Text font size, in pixels (default 4).
  %     bar_width  double: Bar or Box width in data units (default 0.8).
  %     filled  logical: Polygon only: true fills it, false draws its outline (default false).
  %     point_colors  cell array of datamunge.RGB: Optional scatter palette, recycled in input order; empty uses color for every point.
  %
  %   C++: class datamunge::graphics::DataSeries
  %
  %   See also: datamunge.series(), datamunge.points(), datamunge.lines()
  properties (Dependent)
    kind  % How the layer is drawn (default Line).
    x  % Horizontal data coordinates; meaning depends on kind.
    y  % Vertical data coordinates; meaning depends on kind.
    label  % Legend text (or the drawn text for Kind::Text); empty omits it.
    color  % Layer color, including alpha (default RGBA{}, dark slate).
    stroke_width  % Line, Segment, and outline-Polygon width in pixels (default 2).
    marker_size  % Scatter marker radius, or Text font size, in pixels (default 4).
    bar_width  % Bar or Box width in data units (default 0.8).
    filled  % Polygon only: true fills it, false draws its outline (default false).
    point_colors  % Optional scatter palette, recycled in input order; empty uses color for every point.
  end
  methods
    function obj = DataSeries (varargin)
      % DataSeries  Construct a C++ datamunge::graphics::DataSeries.
      %
      %   obj = datamunge.DataSeries ()
      %
      %   C++: datamunge::graphics::DataSeries::DataSeries()
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 942, false, varargin{:});
    end
    function v = get.kind (obj)
      v = __datamunge__(922, obj);
    end
    function set.kind (obj, v)
      __datamunge__(923, obj, v);
    end
    function v = get.x (obj)
      v = __datamunge__(924, obj);
    end
    function set.x (obj, v)
      __datamunge__(925, obj, v);
    end
    function v = get.y (obj)
      v = __datamunge__(926, obj);
    end
    function set.y (obj, v)
      __datamunge__(927, obj, v);
    end
    function v = get.label (obj)
      v = __datamunge__(928, obj);
    end
    function set.label (obj, v)
      __datamunge__(929, obj, v);
    end
    function v = get.color (obj)
      v = __datamunge__(930, obj);
    end
    function set.color (obj, v)
      __datamunge__(931, obj, v);
    end
    function v = get.stroke_width (obj)
      v = __datamunge__(932, obj);
    end
    function set.stroke_width (obj, v)
      __datamunge__(933, obj, v);
    end
    function v = get.marker_size (obj)
      v = __datamunge__(934, obj);
    end
    function set.marker_size (obj, v)
      __datamunge__(935, obj, v);
    end
    function v = get.bar_width (obj)
      v = __datamunge__(936, obj);
    end
    function set.bar_width (obj, v)
      __datamunge__(937, obj, v);
    end
    function v = get.filled (obj)
      v = __datamunge__(938, obj);
    end
    function set.filled (obj, v)
      __datamunge__(939, obj, v);
    end
    function v = get.point_colors (obj)
      v = __datamunge__(940, obj);
    end
    function set.point_colors (obj, v)
      __datamunge__(941, obj, v);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::graphics::DataSeries. Do not edit.
