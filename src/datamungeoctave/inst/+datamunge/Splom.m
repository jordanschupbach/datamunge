classdef Splom < datamunge.NativeObject
  % Splom  A scatterplot matrix (R's `pairs()`, GGally's `ggpairs()`) over the columns of a data frame, with chainable setters; the SWIG-friendly facade for graphics::Splom.
  %
  %   obj = datamunge.Splom (data)
  %   obj = datamunge.Splom (data, columns)
  %
  %   Panel (row i, column j) puts variable j on x and variable i on y. The panel type follows
  %   the two variables' kinds (numeric columns are quantitative, string columns categorical):
  %   - diagonal: kernel density curves, one per group_by() group (numeric), or a stacked bar of
  %     level counts (string), titled with the variable name;
  %   - numeric vs numeric above the diagonal: the points with a local linear smooth of the mean and
  %     a 95% prediction band, one per group; point colors follow set_col() or group_by();
  %   - numeric vs numeric below the diagonal: bivariate kernel density contours, one set per group;
  %   - string vs numeric: one violin per level with the median and interquartile range marked;
  %   - string vs string: a mosaic shaded by chi-squared Pearson residuals, with its p-value.
  %
  %   Values are copied at construction, so the frame need not outlive the matrix. Rows with a
  %   missing value in either variable are dropped from that panel only. Setters return
  %   `Splom&` for chaining; nothing is drawn until to_svg(), save(), or show(). The default
  %   figure is 960 by 960 pixels with blue `{37, 99, 235}` points of radius 2.5.
  %
  %   Example:
  %     const auto iris = dm::datasets::iris();
  %     const auto species = dm::factor(dm::c(iris.get_column<std::string>("Species").values()));
  %     const auto palette = dm::c(dm::RGB{255, 255, 80}, dm::RGB{100, 255, 255},
  %                                dm::RGB{220, 100, 255});
  %     auto pairs = dm::Splom(iris).set_title("Iris Pairs").set_col(species, palette);
  %     pairs.show();  // or pairs.save("iris_pairs.svg") without a display
  %
  %   Methods:
  %     set_col  Sets one color for every scatter point; same as color().
  %     set_title  Same as title(), matching the `set_*` style of dm::Plot.
  %     title  Sets the heading above the grid; "" removes it.
  %     size_  Sets the figure size in pixels (default 960 by 960); panels share it equally.
  %     color  Uses one color for all points and clears any grouping.
  %     group_by  Splits every numeric panel by the groups of string column `column` (GGally's `ggpairs(aes(colour = group))`); an empty palette uses the default level palette.
  %     point_colors  Sets one color per data row for the points (recycled); clears group_by().
  %     level_palette  Sets the fills cycled over categorical levels in violins and stacked bars.
  %     marker_size  Sets the point radius in pixels (default 2.5).
  %     diagonal_names  Shows only the variable names on the diagonal, like R's `pairs()`.
  %     diagonal_density  Shows kernel densities, one per group (numeric), or a stacked bar (string) on the diagonal (the default).
  %     diagonal_histogram  Shows a histogram (numeric) or stacked bar (string) on the diagonal.
  %     span  Sets the fraction of each group's points used in each local fit of the smooths above the diagonal (R's `loess(span =)`, default 0.75).
  %     prediction_level  Sets the coverage of the prediction bands above the diagonal (default 0.95).
  %     bins  Sets the histogram bin count used by diagonal_histogram() (default 10).
  %     show_grid  Toggles the light background grid of the scatter and violin panels (default on).
  %     variables  Variable names in panel order.
  %     groups  Sorted group labels set by group_by(), in palette order; empty without grouping.
  %     title_text  The figure title set by set_title() or title().
  %     width  Figure width in pixels.
  %     height  Figure height in pixels.
  %     panel  Builds the panel with variable `col` on x and variable `row` on y.
  %     to_svg  Renders the matrix as a standalone SVG document and returns it as a string.
  %     save  Writes the matrix to `path`, choosing the format from the extension.
  %     save_svg  Writes to_svg() to `path` regardless of its extension.
  %     show  Opens the matrix in the interactive X11 viewer and blocks until the window closes.
  %
  %   C++: class datamunge::Splom
  %
  %   See also: datamunge.splom(), datamunge.Plot, datamunge.Splom
  methods
    function obj = Splom (varargin)
      % Splom  Construct a C++ datamunge::Splom.
      %
      %   obj = datamunge.Splom (data)
      %   obj = datamunge.Splom (data, columns)
      %
      %   Builds the matrix from a facade DataFrame (the overload SWIG bindings use).
      %
      %   Inputs:
      %     data  datamunge.DataFrame: Source frame; its values are copied.
      %     columns  cell array of char, default the C++ default: Columns to include in order; empty selects every numeric column plus every string column with at most graphics::Splom::max_default_levels distinct values.
      %
      %   Errors:
      %     datamunge:invalid_argument  For a missing column, or fewer than two variables. (C++ std::invalid_argument)
      %
      %   C++: datamunge::Splom::Splom(const DataFrame & data, const std::vector<std::string> & columns = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2177, false, varargin{:});
    end
    function r = set_col (obj, color)
      % set_col  Sets one color for every scatter point; same as color().
      %
      %   r = obj.set_col (color)
      %
      %   Example:
      %     auto pairs = dm::Splom(dm::datasets::iris()).set_col(dm::RGBA{220, 38, 38, 0.5});
      %
      %   Inputs:
      %     color  datamunge.RGB: Point color, alpha included; also used for bars without grouping.
      %
      %   Output:
      %     r  datamunge.Splom: This matrix, for chaining.
      %
      %   C++: Splom & datamunge::Splom::set_col(datamunge::graphics::RGB color)
      r = __datamunge__(2151, obj, color);
    end
    function r = set_title (obj, value_)
      % set_title  Same as title(), matching the `set_*` style of dm::Plot.
      %
      %   r = obj.set_title (value_)
      %
      %   Example:
      %     auto pairs = dm::Splom(dm::datasets::iris()).set_title("Iris Pairs");
      %
      %   Inputs:
      %     value_  char row: Figure heading; empty removes the heading band.
      %
      %   Output:
      %     r  datamunge.Splom: This matrix, for chaining.
      %
      %   C++: Splom & datamunge::Splom::set_title(const std::string & value)
      r = __datamunge__(2152, obj, value_);
    end
    function r = title (obj, value_)
      % title  Sets the heading above the grid; "" removes it.
      %
      %   r = obj.title (value_)
      %
      %   A nonempty title reserves a band above the panels (none by default). show() also uses
      %   it as the window title when no hint is passed.
      %
      %   Inputs:
      %     value_  char row: Figure heading.
      %
      %   Output:
      %     r  datamunge.Splom: This matrix, for chaining.
      %
      %   C++: Splom & datamunge::Splom::title(const std::string & value)
      %
      %   See also: datamunge.set_title()
      r = __datamunge__(2153, obj, value_);
    end
    function r = size_ (obj, width, height)
      % size_  Sets the figure size in pixels (default 960 by 960); panels share it equally.
      %
      %   r = obj.size_ (width, height)
      %
      %   Example:
      %     auto pairs = dm::Splom(dm::datasets::iris()).size(600, 600);
      %     pairs.save("small_pairs.png");
      %
      %   Inputs:
      %     width  integer: Figure width in pixels.
      %     height  integer: Figure height in pixels.
      %
      %   Output:
      %     r  datamunge.Splom: This matrix, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If either dimension is zero. (C++ std::invalid_argument)
      %
      %   C++: Splom & datamunge::Splom::size(std::size_t width, std::size_t height)
      r = __datamunge__(2154, obj, width, height);
    end
    function r = color (obj, value_)
      % color  Uses one color for all points and clears any grouping.
      %
      %   r = obj.color (value_)
      %
      %   Clears group_by(), point_colors(), and the factor form of set_col().
      %
      %   Inputs:
      %     value_  datamunge.RGB: Point color (default `{37, 99, 235}`).
      %
      %   Output:
      %     r  datamunge.Splom: This matrix, for chaining.
      %
      %   C++: Splom & datamunge::Splom::color(datamunge::graphics::RGB value)
      %
      %   See also: datamunge.RGB)
      r = __datamunge__(2155, obj, value_);
    end
    function r = group_by (obj, varargin)
      % group_by  Splits every numeric panel by the groups of string column `column` (GGally's `ggpairs(aes(colour = group))`); an empty palette uses the default level palette.
      %
      %   r = obj.group_by (column)
      %   r = obj.group_by (column, palette)
      %
      %   The column need not be a panel variable but must be a string column of the source
      %   frame. Groups are its sorted distinct non-missing values, and group k takes
      %   `palette[k % palette.size()]`. Each group gets its own points, diagonal density, smooth
      %   and prediction band, and density contours; rows with a missing group are left out of
      %   those panels. When the column is also a panel variable its violins, bars, and mosaic axis
      %   share the colors. Pass "" to clear the grouping. Clears point_colors().
      %
      %   Example:
      %     auto pairs = dm::Splom(dm::datasets::iris()).group_by("Species");
      %     std::print("{}\n", pairs.groups());  // ["setosa", "versicolor", "virginica"]
      %
      %   Inputs:
      %     column  char row: String column to group by, or "" to clear.
      %     palette  cell array of datamunge.RGB, default the C++ default: Group colors in sorted-group order; empty uses the default.
      %
      %   Output:
      %     r  datamunge.Splom: This matrix, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If `column` is not a string column of the source frame. (C++ std::invalid_argument)
      %
      %   C++: Splom & datamunge::Splom::group_by(const std::string & column, const std::vector<datamunge::graphics::RGB> & palette = {})
      %
      %   See also: datamunge.groups(), datamunge.set_col()
      r = __datamunge__(2156, obj, varargin{:});
    end
    function r = point_colors (obj, colors)
      % point_colors  Sets one color per data row for the points (recycled); clears group_by().
      %
      %   r = obj.point_colors (colors)
      %
      %   Same as set_col(const Vector<RGB>&) for callers holding a std::vector; an empty vector
      %   restores the single color().
      %
      %   Example:
      %     auto pairs = dm::Splom(dm::datasets::iris())
      %                      .point_colors({dm::RGB{0, 0, 0}, dm::RGB{255, 0, 0}});  // alternate rows
      %
      %   Inputs:
      %     colors  cell array of datamunge.RGB: Per-row colors, recycled when shorter than the frame.
      %
      %   Output:
      %     r  datamunge.Splom: This matrix, for chaining.
      %
      %   C++: Splom & datamunge::Splom::point_colors(const std::vector<datamunge::graphics::RGB> & colors)
      r = __datamunge__(2157, obj, colors);
    end
    function r = level_palette (obj, colors)
      % level_palette  Sets the fills cycled over categorical levels in violins and stacked bars.
      %
      %   r = obj.level_palette (colors)
      %
      %   An empty vector restores the default palette. group_by() and the factor form of
      %   set_col() override it for their own column. Scatter points are unaffected.
      %
      %   Inputs:
      %     colors  cell array of datamunge.RGB: Fill colors in sorted-level order, recycled.
      %
      %   Output:
      %     r  datamunge.Splom: This matrix, for chaining.
      %
      %   C++: Splom & datamunge::Splom::level_palette(const std::vector<datamunge::graphics::RGB> & colors)
      r = __datamunge__(2158, obj, colors);
    end
    function r = marker_size (obj, value_)
      % marker_size  Sets the point radius in pixels (default 2.5).
      %
      %   r = obj.marker_size (value_)
      %
      %   Inputs:
      %     value_  double: Positive marker radius.
      %
      %   Output:
      %     r  datamunge.Splom: This matrix, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If `value` is not positive. (C++ std::invalid_argument)
      %
      %   C++: Splom & datamunge::Splom::marker_size(double value)
      r = __datamunge__(2159, obj, value_);
    end
    function r = diagonal_names (obj)
      % diagonal_names  Shows only the variable names on the diagonal, like R's `pairs()`.
      %
      %   r = obj.diagonal_names ()
      %
      %   Output:
      %     r  datamunge.Splom: This matrix, for chaining.
      %
      %   C++: Splom & datamunge::Splom::diagonal_names()
      %
      %   See also: datamunge.diagonal_density(), datamunge.diagonal_histogram()
      r = __datamunge__(2160, obj);
    end
    function r = diagonal_density (obj)
      % diagonal_density  Shows kernel densities, one per group (numeric), or a stacked bar (string) on the diagonal (the default).
      %
      %   r = obj.diagonal_density ()
      %
      %   Output:
      %     r  datamunge.Splom: This matrix, for chaining.
      %
      %   C++: Splom & datamunge::Splom::diagonal_density()
      %
      %   See also: datamunge.diagonal_histogram(), datamunge.diagonal_names()
      r = __datamunge__(2161, obj);
    end
    function r = diagonal_histogram (obj)
      % diagonal_histogram  Shows a histogram (numeric) or stacked bar (string) on the diagonal.
      %
      %   r = obj.diagonal_histogram ()
      %
      %   Output:
      %     r  datamunge.Splom: This matrix, for chaining.
      %
      %   C++: Splom & datamunge::Splom::diagonal_histogram()
      %
      %   See also: datamunge.diagonal_density(), datamunge.bins()
      r = __datamunge__(2162, obj);
    end
    function r = span (obj, value_)
      % span  Sets the fraction of each group's points used in each local fit of the smooths above the diagonal (R's `loess(span =)`, default 0.75).
      %
      %   r = obj.span (value_)
      %
      %   Example:
      %     auto pairs = dm::Splom(dm::datasets::iris()).group_by("Species").span(0.5).prediction_level(0.9);
      %
      %   Inputs:
      %     value_  double: Span in (0, 1]; smaller values follow the data more closely.
      %
      %   Output:
      %     r  datamunge.Splom: This matrix, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If `value` is not in (0, 1]. (C++ std::invalid_argument)
      %
      %   C++: Splom & datamunge::Splom::span(double value)
      r = __datamunge__(2163, obj, value_);
    end
    function r = prediction_level (obj, value_)
      % prediction_level  Sets the coverage of the prediction bands above the diagonal (default 0.95).
      %
      %   r = obj.prediction_level (value_)
      %
      %   Inputs:
      %     value_  double: Coverage in (0, 1).
      %
      %   Output:
      %     r  datamunge.Splom: This matrix, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If `value` is not in (0, 1). (C++ std::invalid_argument)
      %
      %   C++: Splom & datamunge::Splom::prediction_level(double value)
      %
      %   See also: datamunge.span()
      r = __datamunge__(2164, obj, value_);
    end
    function r = bins (obj, value_)
      % bins  Sets the histogram bin count used by diagonal_histogram() (default 10).
      %
      %   r = obj.bins (value_)
      %
      %   Inputs:
      %     value_  integer: Positive number of equal-width bins.
      %
      %   Output:
      %     r  datamunge.Splom: This matrix, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If `value` is zero. (C++ std::invalid_argument)
      %
      %   C++: Splom & datamunge::Splom::bins(std::size_t value)
      r = __datamunge__(2165, obj, value_);
    end
    function r = show_grid (obj, varargin)
      % show_grid  Toggles the light background grid of the scatter and violin panels (default on).
      %
      %   r = obj.show_grid ()
      %   r = obj.show_grid (enabled)
      %
      %   Example:
      %     auto pairs = dm::Splom(dm::datasets::iris(), {"Sepal.Length", "Sepal.Width", "Species"})
      %                      .marker_size(1.5)
      %                      .bins(15)
      %                      .show_grid(false);
      %
      %   Inputs:
      %     enabled  logical, default true: True draws the grid.
      %
      %   Output:
      %     r  datamunge.Splom: This matrix, for chaining.
      %
      %   C++: Splom & datamunge::Splom::show_grid(bool enabled = true)
      r = __datamunge__(2166, obj, varargin{:});
    end
    function r = variables (obj)
      % variables  Variable names in panel order.
      %
      %   r = obj.variables ()
      %
      %   Output:
      %     r  cell array of char: A copy of the selected column names.
      %
      %   C++: std::vector<std::string> datamunge::Splom::variables() const
      r = __datamunge__(2167, obj);
    end
    function r = groups (obj)
      % groups  Sorted group labels set by group_by(), in palette order; empty without grouping.
      %
      %   r = obj.groups ()
      %
      %   Output:
      %     r  cell array of char: A copy of the group labels; also empty after the factor form of set_col().
      %
      %   C++: std::vector<std::string> datamunge::Splom::groups() const
      r = __datamunge__(2168, obj);
    end
    function r = title_text (obj)
      % title_text  The figure title set by set_title() or title().
      %
      %   r = obj.title_text ()
      %
      %   Output:
      %     r  char row: The title; empty when unset.
      %
      %   C++: std::string datamunge::Splom::title_text() const
      r = __datamunge__(2169, obj);
    end
    function r = width (obj)
      % width  Figure width in pixels.
      %
      %   r = obj.width ()
      %
      %   Output:
      %     r  integer: The width (960 by default).
      %
      %   C++: int datamunge::Splom::width() const
      r = __datamunge__(2170, obj);
    end
    function r = height (obj)
      % height  Figure height in pixels.
      %
      %   r = obj.height ()
      %
      %   Output:
      %     r  integer: The height (960 by default).
      %
      %   C++: int datamunge::Splom::height() const
      r = __datamunge__(2171, obj);
    end
    function r = panel (obj, row, col)
      % panel  Builds the panel with variable `col` on x and variable `row` on y.
      %
      %   r = obj.panel (row, col)
      %
      %   The panel is a standalone dm::Plot sized to one grid cell, useful for inspecting or
      %   saving a single pair. Indices are zero-based positions in variables().
      %
      %   Example:
      %     const auto pairs = dm::Splom(dm::datasets::iris(), {"Sepal.Length", "Sepal.Width"});
      %     pairs.panel(1, 0).save("sepal_scatter.svg");  // Sepal.Length on x, Sepal.Width on y
      %
      %   Inputs:
      %     row  integer: Zero-based index of the y variable.
      %     col  integer: Zero-based index of the x variable; equal to `row` gives the diagonal panel.
      %
      %   Output:
      %     r  datamunge.RPlot: The rendered-on-demand panel plot.
      %
      %   Errors:
      %     datamunge:out_of_range  If either index is not below variables().size(). (C++ std::out_of_range)
      %     datamunge:invalid_argument  If the panel has no complete rows to draw. (C++ std::invalid_argument)
      %
      %   C++: datamunge::graphics::RPlot datamunge::Splom::panel(std::size_t row, std::size_t col) const
      r = __datamunge__(2172, obj, row, col);
    end
    function r = to_svg (obj)
      % to_svg  Renders the matrix as a standalone SVG document and returns it as a string.
      %
      %   r = obj.to_svg ()
      %
      %   Example:
      %     const auto svg = dm::Splom(dm::datasets::iris()).set_title("Iris Pairs").to_svg();
      %     std::print("{}\n", svg.contains("Iris Pairs"));  // true
      %
      %   Output:
      %     r  char row: The complete SVG text of the full grid, including the title band.
      %
      %   C++: std::string datamunge::Splom::to_svg() const
      r = __datamunge__(2173, obj);
    end
    function save (obj, path)
      % save  Writes the matrix to `path`, choosing the format from the extension.
      %
      %   obj.save (path)
      %
      %   `.svg` or no extension writes to_svg(); `.png` and `.pdf` write the tiled raster the
      %   viewer shows. Needs no display.
      %
      %   Example:
      %     const auto pairs = dm::Splom(dm::datasets::iris()).group_by("Species");
      %     pairs.save("iris_pairs.svg");
      %     pairs.save("iris_pairs.png");
      %
      %   Inputs:
      %     path  char row: Destination file; an existing file is overwritten.
      %
      %   Errors:
      %     datamunge:invalid_argument  For any other extension. (C++ std::invalid_argument)
      %     datamunge:native  If the file cannot be written. (C++ std::runtime_error)
      %
      %   C++: void datamunge::Splom::save(const std::string & path) const
      __datamunge__(2174, obj, path);
    end
    function save_svg (obj, path)
      % save_svg  Writes to_svg() to `path` regardless of its extension.
      %
      %   obj.save_svg (path)
      %
      %   Inputs:
      %     path  char row: Destination file; an existing file is overwritten.
      %
      %   Errors:
      %     datamunge:native  If the file cannot be opened for writing. (C++ std::runtime_error)
      %
      %   C++: void datamunge::Splom::save_svg(const std::string & path) const
      __datamunge__(2175, obj, path);
    end
    function show (obj, varargin)
      % show  Opens the matrix in the interactive X11 viewer and blocks until the window closes.
      %
      %   obj.show ()
      %   obj.show (title_hint)
      %
      %   Zoom and pan act on every panel at once; q or Escape closes the window. Requires an
      %   X11-enabled build and a reachable display; use save() in headless code.
      %
      %   Example:
      %     dm::Splom(dm::datasets::iris()).set_title("Iris Pairs").group_by("Species").show();
      %
      %   Inputs:
      %     title_hint  char row, default '': Window title; empty uses the figure title, or "datamunge splom".
      %
      %   Errors:
      %     datamunge:native  If X11 is unavailable or the display cannot be opened. (C++ std::runtime_error)
      %
      %   C++: void datamunge::Splom::show(const std::string & title_hint = "") const
      __datamunge__(2176, obj, varargin{:});
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::Splom. Do not edit.
