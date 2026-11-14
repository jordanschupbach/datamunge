classdef GGPlot < datamunge.NativeObject
  % GGPlot  SWIG-friendly facade for datamunge::graphics::GGPlot — a ggplot2-style grammar-of-graphics builder. Geom/theme/scale/facet calls are chainable, mirroring ggplot2's own layered `ggplot(df, aes(...)) + geom_point() + ...` style as closely as C++ method chaining allows.
  %
  %   obj = datamunge.GGPlot (data, x_column)
  %   obj = datamunge.GGPlot (data, x_column, y_column)
  %   obj = datamunge.GGPlot (data, x_column, y_column, color_column)
  %   obj = datamunge.GGPlot (data, x_column, y_column, color_column, fill_column)
  %   obj = datamunge.GGPlot (data, x_column, y_column, color_column, fill_column, group_column)
  %
  %   Methods:
  %     geom_point  C++ datamunge::GGPlot::geom_point.
  %     geom_line  C++ datamunge::GGPlot::geom_line.
  %     geom_bar  C++ datamunge::GGPlot::geom_bar.
  %     geom_col  C++ datamunge::GGPlot::geom_col.
  %     geom_histogram  C++ datamunge::GGPlot::geom_histogram.
  %     geom_boxplot  C++ datamunge::GGPlot::geom_boxplot.
  %     geom_smooth  C++ datamunge::GGPlot::geom_smooth.
  %     geom_area  C++ datamunge::GGPlot::geom_area.
  %     geom_ribbon  C++ datamunge::GGPlot::geom_ribbon.
  %     geom_density  C++ datamunge::GGPlot::geom_density.
  %     facet_wrap  C++ datamunge::GGPlot::facet_wrap.
  %     theme_minimal  C++ datamunge::GGPlot::theme_minimal.
  %     theme_bw  C++ datamunge::GGPlot::theme_bw.
  %     theme_classic  C++ datamunge::GGPlot::theme_classic.
  %     scale_color_manual  C++ datamunge::GGPlot::scale_color_manual.
  %     labs  C++ datamunge::GGPlot::labs.
  %     to_svg  Renders the plot as a standalone SVG document and returns it as a string.
  %     save  C++ datamunge::GGPlot::save.
  %     save_svg  C++ datamunge::GGPlot::save_svg.
  %     show  C++ datamunge::GGPlot::show.
  %     rows  Number of data rows being plotted.
  %     x_column  The column mapped to x.
  %     y_column  The column mapped to y.
  %     color_column  The discrete color grouping column.
  %     layer_count  Number of geom layers added so far.
  %     layer_kinds  The geom of each layer in drawing order ("point", "line", "smooth", ...).
  %     theme_name  The selected theme's name.
  %     facet_column  The facet_wrap() column.
  %     title_text  The title set by labs().
  %     x_label  The x-axis label set by labs().
  %     y_label  The y-axis label set by labs().
  %
  %   C++: class datamunge::GGPlot
  methods
    function obj = GGPlot (varargin)
      % GGPlot  Construct a C++ datamunge::GGPlot.
      %
      %   obj = datamunge.GGPlot (data, x_column)
      %   obj = datamunge.GGPlot (data, x_column, y_column)
      %   obj = datamunge.GGPlot (data, x_column, y_column, color_column)
      %   obj = datamunge.GGPlot (data, x_column, y_column, color_column, fill_column)
      %   obj = datamunge.GGPlot (data, x_column, y_column, color_column, fill_column, group_column)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     x_column  char row
      %     y_column  char row, default ''
      %     color_column  char row, default ''
      %     fill_column  char row, default ''
      %     group_column  char row, default ''
      %
      %   C++: datamunge::GGPlot::GGPlot(const DataFrame & data, const std::string & x_column, const std::string & y_column = "", const std::string & color_column = "", const std::string & fill_column = "", const std::string & group_column = "")
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2150, false, varargin{:});
    end
    function r = geom_point (obj, varargin)
      % geom_point  C++ datamunge::GGPlot::geom_point.
      %
      %   r = obj.geom_point ()
      %   r = obj.geom_point (color)
      %   r = obj.geom_point (color, size)
      %
      %   Inputs:
      %     color  datamunge.RGB, default the C++ default
      %     size  double, default 3.0
      %
      %   Output:
      %     r  datamunge.GGPlot
      %
      %   C++: GGPlot & datamunge::GGPlot::geom_point(datamunge::graphics::RGB color = ..., double size = 3.0)
      r = __datamunge__(2119, obj, varargin{:});
    end
    function r = geom_line (obj, varargin)
      % geom_line  C++ datamunge::GGPlot::geom_line.
      %
      %   r = obj.geom_line ()
      %   r = obj.geom_line (color)
      %   r = obj.geom_line (color, width)
      %
      %   Inputs:
      %     color  datamunge.RGB, default the C++ default
      %     width  double, default 1.5
      %
      %   Output:
      %     r  datamunge.GGPlot
      %
      %   C++: GGPlot & datamunge::GGPlot::geom_line(datamunge::graphics::RGB color = ..., double width = 1.5)
      r = __datamunge__(2120, obj, varargin{:});
    end
    function r = geom_bar (obj, varargin)
      % geom_bar  C++ datamunge::GGPlot::geom_bar.
      %
      %   r = obj.geom_bar ()
      %   r = obj.geom_bar (color)
      %
      %   Inputs:
      %     color  datamunge.RGB, default the C++ default
      %
      %   Output:
      %     r  datamunge.GGPlot
      %
      %   C++: GGPlot & datamunge::GGPlot::geom_bar(datamunge::graphics::RGB color = ...)
      r = __datamunge__(2121, obj, varargin{:});
    end
    function r = geom_col (obj, varargin)
      % geom_col  C++ datamunge::GGPlot::geom_col.
      %
      %   r = obj.geom_col ()
      %   r = obj.geom_col (color)
      %
      %   Inputs:
      %     color  datamunge.RGB, default the C++ default
      %
      %   Output:
      %     r  datamunge.GGPlot
      %
      %   C++: GGPlot & datamunge::GGPlot::geom_col(datamunge::graphics::RGB color = ...)
      r = __datamunge__(2122, obj, varargin{:});
    end
    function r = geom_histogram (obj, varargin)
      % geom_histogram  C++ datamunge::GGPlot::geom_histogram.
      %
      %   r = obj.geom_histogram ()
      %   r = obj.geom_histogram (bins)
      %   r = obj.geom_histogram (bins, color)
      %
      %   Inputs:
      %     bins  integer, default 30
      %     color  datamunge.RGB, default the C++ default
      %
      %   Output:
      %     r  datamunge.GGPlot
      %
      %   C++: GGPlot & datamunge::GGPlot::geom_histogram(std::size_t bins = 30, datamunge::graphics::RGB color = ...)
      r = __datamunge__(2123, obj, varargin{:});
    end
    function r = geom_boxplot (obj, varargin)
      % geom_boxplot  C++ datamunge::GGPlot::geom_boxplot.
      %
      %   r = obj.geom_boxplot ()
      %   r = obj.geom_boxplot (color)
      %
      %   Inputs:
      %     color  datamunge.RGB, default the C++ default
      %
      %   Output:
      %     r  datamunge.GGPlot
      %
      %   C++: GGPlot & datamunge::GGPlot::geom_boxplot(datamunge::graphics::RGB color = ...)
      r = __datamunge__(2124, obj, varargin{:});
    end
    function r = geom_smooth (obj, varargin)
      % geom_smooth  C++ datamunge::GGPlot::geom_smooth.
      %
      %   r = obj.geom_smooth ()
      %   r = obj.geom_smooth (color)
      %
      %   Inputs:
      %     color  datamunge.RGB, default the C++ default
      %
      %   Output:
      %     r  datamunge.GGPlot
      %
      %   C++: GGPlot & datamunge::GGPlot::geom_smooth(datamunge::graphics::RGB color = ...)
      r = __datamunge__(2125, obj, varargin{:});
    end
    function r = geom_area (obj, varargin)
      % geom_area  C++ datamunge::GGPlot::geom_area.
      %
      %   r = obj.geom_area ()
      %   r = obj.geom_area (color)
      %
      %   Inputs:
      %     color  datamunge.RGB, default the C++ default
      %
      %   Output:
      %     r  datamunge.GGPlot
      %
      %   C++: GGPlot & datamunge::GGPlot::geom_area(datamunge::graphics::RGB color = ...)
      r = __datamunge__(2126, obj, varargin{:});
    end
    function r = geom_ribbon (obj, varargin)
      % geom_ribbon  C++ datamunge::GGPlot::geom_ribbon.
      %
      %   r = obj.geom_ribbon (ymin_column, ymax_column)
      %   r = obj.geom_ribbon (ymin_column, ymax_column, color)
      %
      %   Inputs:
      %     ymin_column  char row
      %     ymax_column  char row
      %     color  datamunge.RGB, default the C++ default
      %
      %   Output:
      %     r  datamunge.GGPlot
      %
      %   C++: GGPlot & datamunge::GGPlot::geom_ribbon(const std::string & ymin_column, const std::string & ymax_column, datamunge::graphics::RGB color = ...)
      r = __datamunge__(2127, obj, varargin{:});
    end
    function r = geom_density (obj, varargin)
      % geom_density  C++ datamunge::GGPlot::geom_density.
      %
      %   r = obj.geom_density ()
      %   r = obj.geom_density (color)
      %
      %   Inputs:
      %     color  datamunge.RGB, default the C++ default
      %
      %   Output:
      %     r  datamunge.GGPlot
      %
      %   C++: GGPlot & datamunge::GGPlot::geom_density(datamunge::graphics::RGB color = ...)
      r = __datamunge__(2128, obj, varargin{:});
    end
    function r = facet_wrap (obj, varargin)
      % facet_wrap  C++ datamunge::GGPlot::facet_wrap.
      %
      %   r = obj.facet_wrap (column)
      %   r = obj.facet_wrap (column, ncol)
      %
      %   Inputs:
      %     column  char row
      %     ncol  integer, default 0
      %
      %   Output:
      %     r  datamunge.GGPlot
      %
      %   C++: GGPlot & datamunge::GGPlot::facet_wrap(const std::string & column, std::size_t ncol = 0)
      r = __datamunge__(2129, obj, varargin{:});
    end
    function r = theme_minimal (obj)
      % theme_minimal  C++ datamunge::GGPlot::theme_minimal.
      %
      %   r = obj.theme_minimal ()
      %
      %   Output:
      %     r  datamunge.GGPlot
      %
      %   C++: GGPlot & datamunge::GGPlot::theme_minimal()
      r = __datamunge__(2130, obj);
    end
    function r = theme_bw (obj)
      % theme_bw  C++ datamunge::GGPlot::theme_bw.
      %
      %   r = obj.theme_bw ()
      %
      %   Output:
      %     r  datamunge.GGPlot
      %
      %   C++: GGPlot & datamunge::GGPlot::theme_bw()
      r = __datamunge__(2131, obj);
    end
    function r = theme_classic (obj)
      % theme_classic  C++ datamunge::GGPlot::theme_classic.
      %
      %   r = obj.theme_classic ()
      %
      %   Output:
      %     r  datamunge.GGPlot
      %
      %   C++: GGPlot & datamunge::GGPlot::theme_classic()
      r = __datamunge__(2132, obj);
    end
    function r = scale_color_manual (obj, values)
      % scale_color_manual  C++ datamunge::GGPlot::scale_color_manual.
      %
      %   r = obj.scale_color_manual (values)
      %
      %   Inputs:
      %     values  cell array of datamunge.RGB
      %
      %   Output:
      %     r  datamunge.GGPlot
      %
      %   C++: GGPlot & datamunge::GGPlot::scale_color_manual(const std::vector<datamunge::graphics::RGB> & values)
      r = __datamunge__(2133, obj, values);
    end
    function r = labs (obj, varargin)
      % labs  C++ datamunge::GGPlot::labs.
      %
      %   r = obj.labs ()
      %   r = obj.labs (title)
      %   r = obj.labs (title, x)
      %   r = obj.labs (title, x, y)
      %
      %   Inputs:
      %     title  char row, default ''
      %     x  char row, default ''
      %     y  char row, default ''
      %
      %   Output:
      %     r  datamunge.GGPlot
      %
      %   C++: GGPlot & datamunge::GGPlot::labs(const std::string & title = "", const std::string & x = "", const std::string & y = "")
      r = __datamunge__(2134, obj, varargin{:});
    end
    function r = to_svg (obj)
      % to_svg  Renders the plot as a standalone SVG document and returns it as a string.
      %
      %   r = obj.to_svg ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::GGPlot::to_svg() const
      r = __datamunge__(2135, obj);
    end
    function save (obj, path)
      % save  C++ datamunge::GGPlot::save.
      %
      %   obj.save (path)
      %
      %   Inputs:
      %     path  char row
      %
      %   C++: void datamunge::GGPlot::save(const std::string & path) const
      __datamunge__(2136, obj, path);
    end
    function save_svg (obj, path)
      % save_svg  C++ datamunge::GGPlot::save_svg.
      %
      %   obj.save_svg (path)
      %
      %   Inputs:
      %     path  char row
      %
      %   C++: void datamunge::GGPlot::save_svg(const std::string & path) const
      __datamunge__(2137, obj, path);
    end
    function show (obj, varargin)
      % show  C++ datamunge::GGPlot::show.
      %
      %   obj.show ()
      %   obj.show (title_hint)
      %
      %   Inputs:
      %     title_hint  char row, default ''
      %
      %   C++: void datamunge::GGPlot::show(const std::string & title_hint = "") const
      __datamunge__(2138, obj, varargin{:});
    end
    function r = rows (obj)
      % rows  Number of data rows being plotted.
      %
      %   r = obj.rows ()
      %
      %   Output:
      %     r  integer: The row count of the frame passed to the constructor.
      %
      %   C++: int datamunge::GGPlot::rows() const
      r = __datamunge__(2139, obj);
    end
    function r = x_column (obj)
      % x_column  The column mapped to x.
      %
      %   r = obj.x_column ()
      %
      %   Output:
      %     r  char row: The x column name.
      %
      %   C++: std::string datamunge::GGPlot::x_column() const
      r = __datamunge__(2140, obj);
    end
    function r = y_column (obj)
      % y_column  The column mapped to y.
      %
      %   r = obj.y_column ()
      %
      %   Output:
      %     r  char row: The y column name; empty when unmapped.
      %
      %   C++: std::string datamunge::GGPlot::y_column() const
      r = __datamunge__(2141, obj);
    end
    function r = color_column (obj)
      % color_column  The discrete color grouping column.
      %
      %   r = obj.color_column ()
      %
      %   Output:
      %     r  char row: The color column name; empty when unmapped.
      %
      %   C++: std::string datamunge::GGPlot::color_column() const
      r = __datamunge__(2142, obj);
    end
    function r = layer_count (obj)
      % layer_count  Number of geom layers added so far.
      %
      %   r = obj.layer_count ()
      %
      %   Output:
      %     r  integer: The layer count.
      %
      %   C++: int datamunge::GGPlot::layer_count() const
      r = __datamunge__(2143, obj);
    end
    function r = layer_kinds (obj)
      % layer_kinds  The geom of each layer in drawing order ("point", "line", "smooth", ...).
      %
      %   r = obj.layer_kinds ()
      %
      %   Output:
      %     r  cell array of char: One name per layer.
      %
      %   C++: std::vector<std::string> datamunge::GGPlot::layer_kinds() const
      r = __datamunge__(2144, obj);
    end
    function r = theme_name (obj)
      % theme_name  The selected theme's name.
      %
      %   r = obj.theme_name ()
      %
      %   Output:
      %     r  char row: "gray" (the default), "minimal", "bw" or "classic".
      %
      %   C++: std::string datamunge::GGPlot::theme_name() const
      r = __datamunge__(2145, obj);
    end
    function r = facet_column (obj)
      % facet_column  The facet_wrap() column.
      %
      %   r = obj.facet_column ()
      %
      %   Output:
      %     r  char row: The column name; empty when not faceted.
      %
      %   C++: std::string datamunge::GGPlot::facet_column() const
      r = __datamunge__(2146, obj);
    end
    function r = title_text (obj)
      % title_text  The title set by labs().
      %
      %   r = obj.title_text ()
      %
      %   Output:
      %     r  char row: The title; empty when unset.
      %
      %   C++: std::string datamunge::GGPlot::title_text() const
      r = __datamunge__(2147, obj);
    end
    function r = x_label (obj)
      % x_label  The x-axis label set by labs().
      %
      %   r = obj.x_label ()
      %
      %   Output:
      %     r  char row: The label; empty when unset.
      %
      %   C++: std::string datamunge::GGPlot::x_label() const
      r = __datamunge__(2148, obj);
    end
    function r = y_label (obj)
      % y_label  The y-axis label set by labs().
      %
      %   r = obj.y_label ()
      %
      %   Output:
      %     r  char row: The label; empty when unset.
      %
      %   C++: std::string datamunge::GGPlot::y_label() const
      r = __datamunge__(2149, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::GGPlot. Do not edit.
