classdef Plot < datamunge.NativeObject
  % Plot  Renderer-facing state shared by every 2-D plot: canvas, labels, axes, and series.
  %
  %   obj = datamunge.Plot ()
  %
  %   User code works with the derived RPlot (spelled `dm::Plot`), whose setters return `RPlot&`;
  %   this base holds the data, the read-only accessors, and the output functions to_svg(),
  %   save(), and show(). The base-class setters below return `Plot&` and are what RPlot's
  %   same-named setters forward to; call the RPlot forms in user code so chains keep the RPlot
  %   type. A `const Plot&` is also what RLayout::add() accepts and RLayout::panel() returns.
  %
  %   Every output function renders from the stored state, so it throws std::runtime_error for a
  %   plot with no data series (reference lines and legend rows alone are not enough) or a canvas
  %   too small for its margins. Plots are copyable values with no shared state; distinct plots may
  %   be used from different threads, but one plot must not be mutated while it is rendered.
  %
  %   Example:
  %     const dm::graphics::Plot& base = dm::Plot(dm::c(1.0, 2.0), dm::c(1.0, 4.0)).set_title("Base");
  %     std::print("{} {}x{}\n", base.title_text(), base.width(), base.height());  // Base 960x640
  %
  %   Methods:
  %     size_  Base-class form of RPlot::size(): sets the canvas size in pixels (default 960x640).
  %     title  Base-class form of RPlot::title(): sets the main heading; empty removes it.
  %     subtitle  Sets an optional subtitle displayed beneath the title; owns the supplied text.
  %     x_label  Base-class form of RPlot::x_label(): sets the horizontal-axis label.
  %     y_label  Base-class form of RPlot::y_label(): sets the vertical-axis label.
  %     background  Base-class form of RPlot::background(): sets the canvas fill color.
  %     axis_color  Base-class form of RPlot::axis_color(): sets the axis-line and text color.
  %     grid_color  Base-class form of RPlot::grid_color(): sets the grid-line color.
  %     show_grid  Base-class form of RPlot::show_grid(): turns the background grid on or off.
  %     x_limits  Base-class form of RPlot::x_limits(): fixes the visible x range.
  %     y_limits  Base-class form of RPlot::y_limits(): fixes the visible y range.
  %     hide_axes  Base-class form of RPlot::hide_axes(): hides or restores axes, ticks, grid, and axis labels.
  %     x_tick_labels  Base-class form of RPlot::x_tick_labels(): labels the x axis at 0, 1, 2, ...
  %     x_ticks  Replaces the numeric x ticks with `labels` drawn at data positions `positions`.
  %     y_tick_labels  Replaces the numeric y ticks with `labels` drawn at y = 0, 1, 2, ...
  %     y_ticks  Replaces the numeric y ticks with `labels` drawn at data positions `positions`.
  %     margins  Overrides the automatic outer margins (in pixels) around the plotting area, e.g. for small multi-panel tiles. The defaults leave room for a title, axis labels, and ticks; tight margins suit panels whose title and axis labels are empty.
  %     tick_font_size  Sets the tick-label font size in pixels (default 12); the raster viewer draws ticks at half-size glyphs when this is below 12.
  %     pretty_ticks  Places numeric ticks at round values (steps of 1, 2 or 5 times a power of ten, like R's `pretty()`) instead of six evenly spaced ticks across the range (default on).
  %     width  Returns the canvas width in pixels (default 960).
  %     height  Returns the canvas height in pixels (default 640).
  %     title_text  Borrows the main heading ("datamunge plot" until set) for the lifetime of this plot.
  %     subtitle_text  Borrows the subtitle for the lifetime of this plot.
  %     x_label_text  Borrows the horizontal-axis label ("x" until set) for the lifetime of this plot.
  %     y_label_text  Borrows the vertical-axis label ("y" until set) for the lifetime of this plot.
  %     series  Borrows the drawn layers (points, lines, bars, ...) in the order they were added.
  %     background_color  Returns the canvas fill color set by background().
  %     axes_color  Returns the axis and text color set by axis_color().
  %     major_grid_color  Returns the grid-line color set by grid_color().
  %     grid_visible  Reports whether the background grid is drawn (see show_grid()).
  %     has_x_limits  Reports whether x_limits() fixed the x range.
  %     has_y_limits  Reports whether y_limits() fixed the y range.
  %     x_min  Returns the fixed left x limit; meaningful only when has_x_limits() is true.
  %     x_max  Returns the fixed right x limit; meaningful only when has_x_limits() is true.
  %     y_min  Returns the fixed bottom y limit; meaningful only when has_y_limits() is true.
  %     y_max  Returns the fixed top y limit; meaningful only when has_y_limits() is true.
  %     axes_hidden  Reports whether hide_axes() suppressed axes, ticks, grid, and axis labels.
  %     x_tick_label_list  Borrows the custom x tick texts; empty means numeric ticks.
  %     x_tick_position_list  Data positions of x_tick_label_list(); empty means 0, 1, 2, ...
  %     y_tick_label_list  Borrows the custom y tick texts; empty means numeric ticks.
  %     y_tick_position_list  Data positions of y_tick_label_list(); empty means 0, 1, 2, ...
  %     reference_lines  Borrows the reference lines added by abline(), abline_h(), abline_v(), and qqline().
  %     legend_entries  Borrows the manual legend rows added by legend(); empty means an automatic legend.
  %     has_custom_margins  True once margins() has been called; the renderer then uses the explicit margins.
  %     margin_left  Returns the left margin in pixels; used only when has_custom_margins() is true.
  %     margin_right  Returns the right margin in pixels; used only when has_custom_margins() is true.
  %     margin_top  Returns the top margin in pixels; used only when has_custom_margins() is true.
  %     margin_bottom  Returns the bottom margin in pixels; used only when has_custom_margins() is true.
  %     tick_font_px  Returns the tick-label font size set by tick_font_size().
  %     pretty_ticks_enabled  Reports whether pretty_ticks() is on.
  %     to_svg  Renders the plot as a standalone SVG document and returns it as a string.
  %     save  Writes the plot to `path`, choosing the format from the file extension (R's `svg()`/`png()`/`pdf()` devices followed by `dev.off()`, in one call).
  %     save_svg  Writes to_svg() to `path` regardless of its extension.
  %     to_png  Renders the plot as PNG image bytes (the same raster save() writes for `.png`).
  %     to_pdf  Renders the plot as a one-page PDF holding the raster image.
  %     save_png  Writes to_png() to `path` regardless of its extension.
  %     save_pdf  Writes to_pdf() to `path` regardless of its extension.
  %     view  Same as show(); kept as an alias for older code.
  %     show  Opens the plot in an interactive X11 window and blocks until it is closed (like printing a plot in an interactive R session).
  %
  %   C++: class datamunge::graphics::Plot
  %
  %   See also: datamunge.RPlot, datamunge.RLayout
  methods
    function obj = Plot (varargin)
      % Plot  Construct a C++ datamunge::graphics::Plot.
      %
      %   obj = datamunge.Plot ()
      %
      %   C++: datamunge::graphics::Plot::Plot()
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1018, false, varargin{:});
    end
    function r = size_ (obj, width, height)
      % size_  Base-class form of RPlot::size(): sets the canvas size in pixels (default 960x640).
      %
      %   r = obj.size_ (width, height)
      %
      %   Inputs:
      %     width  integer: Canvas width in pixels.
      %     height  integer: Canvas height in pixels.
      %
      %   Output:
      %     r  datamunge.Plot: This plot, for chaining.
      %
      %   C++: Plot & datamunge::graphics::Plot::size(std::size_t width, std::size_t height)
      %
      %   See also: datamunge.size()
      r = __datamunge__(959, obj, width, height);
    end
    function r = title (obj, value_)
      % title  Base-class form of RPlot::title(): sets the main heading; empty removes it.
      %
      %   r = obj.title (value_)
      %
      %   Inputs:
      %     value_  char row: Heading text (moved into the plot).
      %
      %   Output:
      %     r  datamunge.Plot: This plot, for chaining.
      %
      %   C++: Plot & datamunge::graphics::Plot::title(std::string value)
      %
      %   See also: datamunge.set_title()
      r = __datamunge__(960, obj, value_);
    end
    function r = subtitle (obj, value_)
      % subtitle  Sets an optional subtitle displayed beneath the title; owns the supplied text.
      %
      %   r = obj.subtitle (value_)
      %
      %   Inputs:
      %     value_  char row: Subtitle text; empty removes it.
      %
      %   Output:
      %     r  datamunge.Plot: This plot, for chaining.
      %
      %   C++: Plot & datamunge::graphics::Plot::subtitle(std::string value)
      %
      %   See also: datamunge.set_subtitle()
      r = __datamunge__(961, obj, value_);
    end
    function r = x_label (obj, value_)
      % x_label  Base-class form of RPlot::x_label(): sets the horizontal-axis label.
      %
      %   r = obj.x_label (value_)
      %
      %   Inputs:
      %     value_  char row: Label text (moved into the plot).
      %
      %   Output:
      %     r  datamunge.Plot: This plot, for chaining.
      %
      %   C++: Plot & datamunge::graphics::Plot::x_label(std::string value)
      %
      %   See also: datamunge.set_xlab()
      r = __datamunge__(962, obj, value_);
    end
    function r = y_label (obj, value_)
      % y_label  Base-class form of RPlot::y_label(): sets the vertical-axis label.
      %
      %   r = obj.y_label (value_)
      %
      %   Inputs:
      %     value_  char row: Label text (moved into the plot).
      %
      %   Output:
      %     r  datamunge.Plot: This plot, for chaining.
      %
      %   C++: Plot & datamunge::graphics::Plot::y_label(std::string value)
      %
      %   See also: datamunge.set_ylab()
      r = __datamunge__(963, obj, value_);
    end
    function r = background (obj, color)
      % background  Base-class form of RPlot::background(): sets the canvas fill color.
      %
      %   r = obj.background (color)
      %
      %   Inputs:
      %     color  datamunge.RGB: Fill for the whole canvas (default white).
      %
      %   Output:
      %     r  datamunge.Plot: This plot, for chaining.
      %
      %   C++: Plot & datamunge::graphics::Plot::background(RGBA color)
      %
      %   See also: datamunge.background()
      r = __datamunge__(964, obj, color);
    end
    function r = axis_color (obj, color)
      % axis_color  Base-class form of RPlot::axis_color(): sets the axis-line and text color.
      %
      %   r = obj.axis_color (color)
      %
      %   Inputs:
      %     color  datamunge.RGB: Color for axis lines, ticks, axis labels, title, and subtitle.
      %
      %   Output:
      %     r  datamunge.Plot: This plot, for chaining.
      %
      %   C++: Plot & datamunge::graphics::Plot::axis_color(RGBA color)
      %
      %   See also: datamunge.axis_color()
      r = __datamunge__(965, obj, color);
    end
    function r = grid_color (obj, color)
      % grid_color  Base-class form of RPlot::grid_color(): sets the grid-line color.
      %
      %   r = obj.grid_color (color)
      %
      %   Inputs:
      %     color  datamunge.RGB: Grid-line color (default light gray `{229, 231, 235}`).
      %
      %   Output:
      %     r  datamunge.Plot: This plot, for chaining.
      %
      %   C++: Plot & datamunge::graphics::Plot::grid_color(RGBA color)
      %
      %   See also: datamunge.grid_color()
      r = __datamunge__(966, obj, color);
    end
    function r = show_grid (obj, varargin)
      % show_grid  Base-class form of RPlot::show_grid(): turns the background grid on or off.
      %
      %   r = obj.show_grid ()
      %   r = obj.show_grid (enabled)
      %
      %   Inputs:
      %     enabled  logical, default true: True (the default) draws the grid.
      %
      %   Output:
      %     r  datamunge.Plot: This plot, for chaining.
      %
      %   C++: Plot & datamunge::graphics::Plot::show_grid(bool enabled = true)
      %
      %   See also: datamunge.show_grid()
      r = __datamunge__(967, obj, varargin{:});
    end
    function r = x_limits (obj, min_x, max_x)
      % x_limits  Base-class form of RPlot::x_limits(): fixes the visible x range.
      %
      %   r = obj.x_limits (min_x, max_x)
      %
      %   Inputs:
      %     min_x  double: Left edge of the plotting area, in data units.
      %     max_x  double: Right edge; must be greater than `min_x`.
      %
      %   Output:
      %     r  datamunge.Plot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  Unless `max_x > min_x` (NaN fails); the plot is unchanged. (C++ std::invalid_argument)
      %
      %   C++: Plot & datamunge::graphics::Plot::x_limits(double min_x, double max_x)
      %
      %   See also: datamunge.x_limits()
      r = __datamunge__(968, obj, min_x, max_x);
    end
    function r = y_limits (obj, min_y, max_y)
      % y_limits  Base-class form of RPlot::y_limits(): fixes the visible y range.
      %
      %   r = obj.y_limits (min_y, max_y)
      %
      %   Inputs:
      %     min_y  double: Bottom edge of the plotting area, in data units.
      %     max_y  double: Top edge; must be greater than `min_y`.
      %
      %   Output:
      %     r  datamunge.Plot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  Unless `max_y > min_y` (NaN fails); the plot is unchanged. (C++ std::invalid_argument)
      %
      %   C++: Plot & datamunge::graphics::Plot::y_limits(double min_y, double max_y)
      %
      %   See also: datamunge.y_limits()
      r = __datamunge__(969, obj, min_y, max_y);
    end
    function r = hide_axes (obj, varargin)
      % hide_axes  Base-class form of RPlot::hide_axes(): hides or restores axes, ticks, grid, and axis labels.
      %
      %   r = obj.hide_axes ()
      %   r = obj.hide_axes (enabled)
      %
      %   Inputs:
      %     enabled  logical, default true: True (the default) hides them.
      %
      %   Output:
      %     r  datamunge.Plot: This plot, for chaining.
      %
      %   C++: Plot & datamunge::graphics::Plot::hide_axes(bool enabled = true)
      %
      %   See also: datamunge.hide_axes()
      r = __datamunge__(970, obj, varargin{:});
    end
    function r = x_tick_labels (obj, labels)
      % x_tick_labels  Base-class form of RPlot::x_tick_labels(): labels the x axis at 0, 1, 2, ...
      %
      %   r = obj.x_tick_labels (labels)
      %
      %   Inputs:
      %     labels  cell array of char: Tick texts; empty restores the numeric ticks.
      %
      %   Output:
      %     r  datamunge.Plot: This plot, for chaining.
      %
      %   C++: Plot & datamunge::graphics::Plot::x_tick_labels(std::vector<std::string> labels)
      %
      %   See also: datamunge.x_tick_labels()
      r = __datamunge__(971, obj, labels);
    end
    function r = x_ticks (obj, positions, labels)
      % x_ticks  Replaces the numeric x ticks with `labels` drawn at data positions `positions`.
      %
      %   r = obj.x_ticks (positions, labels)
      %
      %   Inputs:
      %     positions  double vector: x data coordinates of the ticks.
      %     labels  cell array of char: Tick texts, one per position.
      %
      %   Output:
      %     r  datamunge.Plot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If the two vectors differ in length. (C++ std::invalid_argument)
      %
      %   C++: Plot & datamunge::graphics::Plot::x_ticks(std::vector<double> positions, std::vector<std::string> labels)
      %
      %   See also: datamunge.x_ticks()
      r = __datamunge__(972, obj, positions, labels);
    end
    function r = y_tick_labels (obj, labels)
      % y_tick_labels  Replaces the numeric y ticks with `labels` drawn at y = 0, 1, 2, ...
      %
      %   r = obj.y_tick_labels (labels)
      %
      %   Inputs:
      %     labels  cell array of char: Tick texts; empty restores the numeric ticks.
      %
      %   Output:
      %     r  datamunge.Plot: This plot, for chaining.
      %
      %   C++: Plot & datamunge::graphics::Plot::y_tick_labels(std::vector<std::string> labels)
      %
      %   See also: datamunge.y_tick_labels()
      r = __datamunge__(973, obj, labels);
    end
    function r = y_ticks (obj, positions, labels)
      % y_ticks  Replaces the numeric y ticks with `labels` drawn at data positions `positions`.
      %
      %   r = obj.y_ticks (positions, labels)
      %
      %   Inputs:
      %     positions  double vector: y data coordinates of the ticks.
      %     labels  cell array of char: Tick texts, one per position.
      %
      %   Output:
      %     r  datamunge.Plot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If the two vectors differ in length. (C++ std::invalid_argument)
      %
      %   C++: Plot & datamunge::graphics::Plot::y_ticks(std::vector<double> positions, std::vector<std::string> labels)
      %
      %   See also: datamunge.y_ticks()
      r = __datamunge__(974, obj, positions, labels);
    end
    function r = margins (obj, left, right, top, bottom)
      % margins  Overrides the automatic outer margins (in pixels) around the plotting area, e.g. for small multi-panel tiles. The defaults leave room for a title, axis labels, and ticks; tight margins suit panels whose title and axis labels are empty.
      %
      %   r = obj.margins (left, right, top, bottom)
      %
      %   Inputs:
      %     left  double: Left margin in pixels (default 90).
      %     right  double: Right margin in pixels (default 24).
      %     top  double: Top margin in pixels (default 72, plus 24 when a subtitle is set).
      %     bottom  double: Bottom margin in pixels (default 84).
      %
      %   Output:
      %     r  datamunge.Plot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If any margin is negative or non-finite. (C++ std::invalid_argument)
      %
      %   C++: Plot & datamunge::graphics::Plot::margins(double left, double right, double top, double bottom)
      %
      %   See also: datamunge.margins()
      r = __datamunge__(975, obj, left, right, top, bottom);
    end
    function r = tick_font_size (obj, pixels)
      % tick_font_size  Sets the tick-label font size in pixels (default 12); the raster viewer draws ticks at half-size glyphs when this is below 12.
      %
      %   r = obj.tick_font_size (pixels)
      %
      %   Inputs:
      %     pixels  double: Font size in pixels.
      %
      %   Output:
      %     r  datamunge.Plot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If `pixels` is not positive and finite. (C++ std::invalid_argument)
      %
      %   C++: Plot & datamunge::graphics::Plot::tick_font_size(double pixels)
      %
      %   See also: datamunge.tick_font_size()
      r = __datamunge__(976, obj, pixels);
    end
    function r = pretty_ticks (obj, varargin)
      % pretty_ticks  Places numeric ticks at round values (steps of 1, 2 or 5 times a power of ten, like R's `pretty()`) instead of six evenly spaced ticks across the range (default on).
      %
      %   r = obj.pretty_ticks ()
      %   r = obj.pretty_ticks (enabled)
      %
      %   Inputs:
      %     enabled  logical, default true: True turns round ticks on; false restores six even ticks.
      %
      %   Output:
      %     r  datamunge.Plot: This plot, for chaining.
      %
      %   C++: Plot & datamunge::graphics::Plot::pretty_ticks(bool enabled = true)
      %
      %   See also: datamunge.pretty_ticks()
      r = __datamunge__(977, obj, varargin{:});
    end
    function r = width (obj)
      % width  Returns the canvas width in pixels (default 960).
      %
      %   r = obj.width ()
      %
      %   Output:
      %     r  integer: Width set by size().
      %
      %   C++: std::size_t datamunge::graphics::Plot::width() const
      r = __datamunge__(978, obj);
    end
    function r = height (obj)
      % height  Returns the canvas height in pixels (default 640).
      %
      %   r = obj.height ()
      %
      %   Output:
      %     r  integer: Height set by size().
      %
      %   C++: std::size_t datamunge::graphics::Plot::height() const
      r = __datamunge__(979, obj);
    end
    function r = title_text (obj)
      % title_text  Borrows the main heading ("datamunge plot" until set) for the lifetime of this plot.
      %
      %   r = obj.title_text ()
      %
      %   Output:
      %     r  char row: The current title; empty when the heading was cleared.
      %
      %   C++: const std::string & datamunge::graphics::Plot::title_text() const
      r = __datamunge__(980, obj);
    end
    function r = subtitle_text (obj)
      % subtitle_text  Borrows the subtitle for the lifetime of this plot.
      %
      %   r = obj.subtitle_text ()
      %
      %   Output:
      %     r  char row: The current subtitle; empty (the default) when none is drawn.
      %
      %   C++: const std::string & datamunge::graphics::Plot::subtitle_text() const
      r = __datamunge__(981, obj);
    end
    function r = x_label_text (obj)
      % x_label_text  Borrows the horizontal-axis label ("x" until set) for the lifetime of this plot.
      %
      %   r = obj.x_label_text ()
      %
      %   Output:
      %     r  char row: The current x-axis label.
      %
      %   C++: const std::string & datamunge::graphics::Plot::x_label_text() const
      r = __datamunge__(982, obj);
    end
    function r = y_label_text (obj)
      % y_label_text  Borrows the vertical-axis label ("y" until set) for the lifetime of this plot.
      %
      %   r = obj.y_label_text ()
      %
      %   Output:
      %     r  char row: The current y-axis label.
      %
      %   C++: const std::string & datamunge::graphics::Plot::y_label_text() const
      r = __datamunge__(983, obj);
    end
    function r = series (obj)
      % series  Borrows the drawn layers (points, lines, bars, ...) in the order they were added.
      %
      %   r = obj.series ()
      %
      %   Output:
      %     r  cell array of datamunge.DataSeries: The series list; references are invalidated by adding another layer.
      %
      %   C++: const std::vector<DataSeries> & datamunge::graphics::Plot::series() const
      r = __datamunge__(984, obj);
    end
    function r = background_color (obj)
      % background_color  Returns the canvas fill color set by background().
      %
      %   r = obj.background_color ()
      %
      %   Output:
      %     r  datamunge.RGB: The background color (default white `{255, 255, 255}`).
      %
      %   C++: RGBA datamunge::graphics::Plot::background_color() const
      r = __datamunge__(985, obj);
    end
    function r = axes_color (obj)
      % axes_color  Returns the axis and text color set by axis_color().
      %
      %   r = obj.axes_color ()
      %
      %   Output:
      %     r  datamunge.RGB: The axis color (default gray `{55, 65, 81}`).
      %
      %   C++: RGBA datamunge::graphics::Plot::axes_color() const
      r = __datamunge__(986, obj);
    end
    function r = major_grid_color (obj)
      % major_grid_color  Returns the grid-line color set by grid_color().
      %
      %   r = obj.major_grid_color ()
      %
      %   Output:
      %     r  datamunge.RGB: The grid color (default light gray `{229, 231, 235}`).
      %
      %   C++: RGBA datamunge::graphics::Plot::major_grid_color() const
      r = __datamunge__(987, obj);
    end
    function r = grid_visible (obj)
      % grid_visible  Reports whether the background grid is drawn (see show_grid()).
      %
      %   r = obj.grid_visible ()
      %
      %   Output:
      %     r  logical: True (the default) when the grid is drawn.
      %
      %   C++: bool datamunge::graphics::Plot::grid_visible() const
      r = __datamunge__(988, obj);
    end
    function r = has_x_limits (obj)
      % has_x_limits  Reports whether x_limits() fixed the x range.
      %
      %   r = obj.has_x_limits ()
      %
      %   Output:
      %     r  logical: False (the default) while the x range is fitted to the data.
      %
      %   C++: bool datamunge::graphics::Plot::has_x_limits() const
      r = __datamunge__(989, obj);
    end
    function r = has_y_limits (obj)
      % has_y_limits  Reports whether y_limits() fixed the y range.
      %
      %   r = obj.has_y_limits ()
      %
      %   Output:
      %     r  logical: False (the default) while the y range is fitted to the data.
      %
      %   C++: bool datamunge::graphics::Plot::has_y_limits() const
      r = __datamunge__(990, obj);
    end
    function r = x_min (obj)
      % x_min  Returns the fixed left x limit; meaningful only when has_x_limits() is true.
      %
      %   r = obj.x_min ()
      %
      %   Output:
      %     r  double: The stored lower x limit (0 until x_limits() is called).
      %
      %   C++: double datamunge::graphics::Plot::x_min() const
      r = __datamunge__(991, obj);
    end
    function r = x_max (obj)
      % x_max  Returns the fixed right x limit; meaningful only when has_x_limits() is true.
      %
      %   r = obj.x_max ()
      %
      %   Output:
      %     r  double: The stored upper x limit (1 until x_limits() is called).
      %
      %   C++: double datamunge::graphics::Plot::x_max() const
      r = __datamunge__(992, obj);
    end
    function r = y_min (obj)
      % y_min  Returns the fixed bottom y limit; meaningful only when has_y_limits() is true.
      %
      %   r = obj.y_min ()
      %
      %   Output:
      %     r  double: The stored lower y limit (0 until y_limits() is called).
      %
      %   C++: double datamunge::graphics::Plot::y_min() const
      r = __datamunge__(993, obj);
    end
    function r = y_max (obj)
      % y_max  Returns the fixed top y limit; meaningful only when has_y_limits() is true.
      %
      %   r = obj.y_max ()
      %
      %   Output:
      %     r  double: The stored upper y limit (1 until y_limits() is called).
      %
      %   C++: double datamunge::graphics::Plot::y_max() const
      r = __datamunge__(994, obj);
    end
    function r = axes_hidden (obj)
      % axes_hidden  Reports whether hide_axes() suppressed axes, ticks, grid, and axis labels.
      %
      %   r = obj.axes_hidden ()
      %
      %   Output:
      %     r  logical: False (the default) when axes are drawn.
      %
      %   C++: bool datamunge::graphics::Plot::axes_hidden() const
      r = __datamunge__(995, obj);
    end
    function r = x_tick_label_list (obj)
      % x_tick_label_list  Borrows the custom x tick texts; empty means numeric ticks.
      %
      %   r = obj.x_tick_label_list ()
      %
      %   Output:
      %     r  cell array of char: Labels set by x_tick_labels() or x_ticks().
      %
      %   C++: const std::vector<std::string> & datamunge::graphics::Plot::x_tick_label_list() const
      r = __datamunge__(996, obj);
    end
    function r = x_tick_position_list (obj)
      % x_tick_position_list  Data positions of x_tick_label_list(); empty means 0, 1, 2, ...
      %
      %   r = obj.x_tick_position_list ()
      %
      %   Output:
      %     r  double vector: Positions set by x_ticks(); cleared by x_tick_labels().
      %
      %   C++: const std::vector<double> & datamunge::graphics::Plot::x_tick_position_list() const
      r = __datamunge__(997, obj);
    end
    function r = y_tick_label_list (obj)
      % y_tick_label_list  Borrows the custom y tick texts; empty means numeric ticks.
      %
      %   r = obj.y_tick_label_list ()
      %
      %   Output:
      %     r  cell array of char: Labels set by y_tick_labels() or y_ticks().
      %
      %   C++: const std::vector<std::string> & datamunge::graphics::Plot::y_tick_label_list() const
      r = __datamunge__(998, obj);
    end
    function r = y_tick_position_list (obj)
      % y_tick_position_list  Data positions of y_tick_label_list(); empty means 0, 1, 2, ...
      %
      %   r = obj.y_tick_position_list ()
      %
      %   Output:
      %     r  double vector: Positions set by y_ticks(); cleared by y_tick_labels().
      %
      %   C++: const std::vector<double> & datamunge::graphics::Plot::y_tick_position_list() const
      r = __datamunge__(999, obj);
    end
    function r = reference_lines (obj)
      % reference_lines  Borrows the reference lines added by abline(), abline_h(), abline_v(), and qqline().
      %
      %   r = obj.reference_lines ()
      %
      %   Output:
      %     r  cell array of datamunge.ABLine: Lines in insertion order; references are invalidated by adding another line.
      %
      %   C++: const std::vector<ABLine> & datamunge::graphics::Plot::reference_lines() const
      r = __datamunge__(1000, obj);
    end
    function r = legend_entries (obj)
      % legend_entries  Borrows the manual legend rows added by legend(); empty means an automatic legend.
      %
      %   r = obj.legend_entries ()
      %
      %   Output:
      %     r  cell array of datamunge.LegendEntry: Manual entries in insertion order.
      %
      %   C++: const std::vector<LegendEntry> & datamunge::graphics::Plot::legend_entries() const
      r = __datamunge__(1001, obj);
    end
    function r = has_custom_margins (obj)
      % has_custom_margins  True once margins() has been called; the renderer then uses the explicit margins.
      %
      %   r = obj.has_custom_margins ()
      %
      %   Output:
      %     r  logical: False (the default) while the automatic margins apply.
      %
      %   C++: bool datamunge::graphics::Plot::has_custom_margins() const
      r = __datamunge__(1002, obj);
    end
    function r = margin_left (obj)
      % margin_left  Returns the left margin in pixels; used only when has_custom_margins() is true.
      %
      %   r = obj.margin_left ()
      %
      %   Output:
      %     r  double: The stored left margin (default 90).
      %
      %   C++: double datamunge::graphics::Plot::margin_left() const
      r = __datamunge__(1003, obj);
    end
    function r = margin_right (obj)
      % margin_right  Returns the right margin in pixels; used only when has_custom_margins() is true.
      %
      %   r = obj.margin_right ()
      %
      %   Output:
      %     r  double: The stored right margin (default 24).
      %
      %   C++: double datamunge::graphics::Plot::margin_right() const
      r = __datamunge__(1004, obj);
    end
    function r = margin_top (obj)
      % margin_top  Returns the top margin in pixels; used only when has_custom_margins() is true.
      %
      %   r = obj.margin_top ()
      %
      %   Output:
      %     r  double: The stored top margin (default 72).
      %
      %   C++: double datamunge::graphics::Plot::margin_top() const
      r = __datamunge__(1005, obj);
    end
    function r = margin_bottom (obj)
      % margin_bottom  Returns the bottom margin in pixels; used only when has_custom_margins() is true.
      %
      %   r = obj.margin_bottom ()
      %
      %   Output:
      %     r  double: The stored bottom margin (default 84).
      %
      %   C++: double datamunge::graphics::Plot::margin_bottom() const
      r = __datamunge__(1006, obj);
    end
    function r = tick_font_px (obj)
      % tick_font_px  Returns the tick-label font size set by tick_font_size().
      %
      %   r = obj.tick_font_px ()
      %
      %   Output:
      %     r  double: Font size in pixels (default 12).
      %
      %   C++: double datamunge::graphics::Plot::tick_font_px() const
      r = __datamunge__(1007, obj);
    end
    function r = pretty_ticks_enabled (obj)
      % pretty_ticks_enabled  Reports whether pretty_ticks() is on.
      %
      %   r = obj.pretty_ticks_enabled ()
      %
      %   Output:
      %     r  logical: False (the default) for six evenly spaced numeric ticks.
      %
      %   C++: bool datamunge::graphics::Plot::pretty_ticks_enabled() const
      r = __datamunge__(1008, obj);
    end
    function r = to_svg (obj)
      % to_svg  Renders the plot as a standalone SVG document and returns it as a string.
      %
      %   r = obj.to_svg ()
      %
      %   The document starts with an XML declaration and draws text as embedded Computer Modern
      %   glyph outlines, so it displays identically in any SVG viewer, with or without web fonts. The plot is not modified, and no file or display is used.
      %
      %   Example:
      %     const auto svg = dm::Plot(dm::c(1.0, 2.0), dm::c(2.0, 1.0)).set_title("Tiny").to_svg();
      %     std::print("{}\n", svg.starts_with("<?xml"));  // true
      %     std::print("{}\n", svg.contains("Tiny"));      // true
      %
      %   Output:
      %     r  char row: The complete SVG text.
      %
      %   Errors:
      %     datamunge:native  If the plot has no data series, or the canvas is smaller than its margins. (C++ std::runtime_error)
      %
      %   C++: std::string datamunge::graphics::Plot::to_svg() const
      %
      %   See also: datamunge.save(), datamunge.show()
      r = __datamunge__(1009, obj);
    end
    function save (obj, path)
      % save  Writes the plot to `path`, choosing the format from the file extension (R's `svg()`/`png()`/`pdf()` devices followed by `dev.off()`, in one call).
      %
      %   obj.save (path)
      %
      %   `.svg` or no extension writes the same text as to_svg(); `.png` writes the raster image
      %   the X11 viewer displays; `.pdf` wraps that raster in a single-page PDF. The extension is
      %   case-sensitive. An existing file is overwritten. Needs no display.
      %
      %   Example:
      %     auto plt = dm::Plot(dm::c(1.0, 2.0, 3.0), dm::c(1.0, 4.0, 9.0)).set_title("Squares");
      %     plt.save("squares.svg");  // vector
      %     plt.save("squares.png");  // raster
      %     plt.save("squares.pdf");  // raster in a one-page PDF
      %
      %   Inputs:
      %     path  char row: Destination file; relative paths resolve against the working directory.
      %
      %   Errors:
      %     datamunge:invalid_argument  For any other extension (e.g. `.jpg`). (C++ std::invalid_argument)
      %     datamunge:native  If the file cannot be opened for writing, the plot has no data series, or the canvas is smaller than its margins. (C++ std::runtime_error)
      %
      %   C++: void datamunge::graphics::Plot::save(const std::string & path) const
      %
      %   See also: datamunge.save_svg(), datamunge.to_svg(), datamunge.show()
      __datamunge__(1010, obj, path);
    end
    function save_svg (obj, path)
      % save_svg  Writes to_svg() to `path` regardless of its extension.
      %
      %   obj.save_svg (path)
      %
      %   Example:
      %     dm::Plot(dm::c(1.0, 2.0), dm::c(1.0, 2.0)).save_svg("diagonal.txt");  // SVG text
      %
      %   Inputs:
      %     path  char row: Destination file; an existing file is overwritten.
      %
      %   Errors:
      %     datamunge:native  If the file cannot be opened for writing, or rendering fails as described for to_svg(). (C++ std::runtime_error)
      %
      %   C++: void datamunge::graphics::Plot::save_svg(const std::string & path) const
      %
      %   See also: datamunge.save()
      __datamunge__(1011, obj, path);
    end
    function r = to_png (obj)
      % to_png  Renders the plot as PNG image bytes (the same raster save() writes for `.png`).
      %
      %   r = obj.to_png ()
      %
      %   Example:
      %     const auto png = dm::Plot(dm::c(1.0, 2.0), dm::c(2.0, 1.0)).to_png();
      %     std::print("{}\n", png[1] == 'P');  // true: PNG signature
      %
      %   Output:
      %     r  integer vector: The complete PNG file contents.
      %
      %   Errors:
      %     datamunge:native  If rendering fails as described for to_svg(). (C++ std::runtime_error)
      %
      %   C++: std::vector<std::uint8_t> datamunge::graphics::Plot::to_png() const
      %
      %   See also: datamunge.save_png(), datamunge.to_pdf(), datamunge.to_svg()
      r = __datamunge__(1012, obj);
    end
    function r = to_pdf (obj)
      % to_pdf  Renders the plot as a one-page PDF holding the raster image.
      %
      %   r = obj.to_pdf ()
      %
      %   Output:
      %     r  integer vector: The complete PDF file contents.
      %
      %   Errors:
      %     datamunge:native  If rendering fails as described for to_svg(). (C++ std::runtime_error)
      %
      %   C++: std::vector<std::uint8_t> datamunge::graphics::Plot::to_pdf() const
      %
      %   See also: datamunge.save_pdf(), datamunge.to_png()
      r = __datamunge__(1013, obj);
    end
    function save_png (obj, path)
      % save_png  Writes to_png() to `path` regardless of its extension.
      %
      %   obj.save_png (path)
      %
      %   Inputs:
      %     path  char row: Destination file; an existing file is overwritten.
      %
      %   Errors:
      %     datamunge:native  If the file cannot be written or rendering fails. (C++ std::runtime_error)
      %
      %   C++: void datamunge::graphics::Plot::save_png(const std::string & path) const
      %
      %   See also: datamunge.save(), datamunge.to_png()
      __datamunge__(1014, obj, path);
    end
    function save_pdf (obj, path)
      % save_pdf  Writes to_pdf() to `path` regardless of its extension.
      %
      %   obj.save_pdf (path)
      %
      %   Inputs:
      %     path  char row: Destination file; an existing file is overwritten.
      %
      %   Errors:
      %     datamunge:native  If the file cannot be written or rendering fails. (C++ std::runtime_error)
      %
      %   C++: void datamunge::graphics::Plot::save_pdf(const std::string & path) const
      %
      %   See also: datamunge.save(), datamunge.to_pdf()
      __datamunge__(1015, obj, path);
    end
    function view (obj, varargin)
      % view  Same as show(); kept as an alias for older code.
      %
      %   obj.view ()
      %   obj.view (title_hint)
      %
      %   Inputs:
      %     title_hint  char row, default '': Window title; empty uses the plot's title.
      %
      %   Errors:
      %     datamunge:native  As for show(). (C++ std::runtime_error)
      %
      %   C++: void datamunge::graphics::Plot::view(const std::string & title_hint = "") const
      %
      %   See also: datamunge.show()
      __datamunge__(1016, obj, varargin{:});
    end
    function show (obj, varargin)
      % show  Opens the plot in an interactive X11 window and blocks until it is closed (like printing a plot in an interactive R session).
      %
      %   obj.show ()
      %   obj.show (title_hint)
      %
      %   The window has toolbar and keyboard zoom/pan (it re-renders from data bounds) and closes
      %   on q or Escape; the function returns only then. The window title is `title_hint`, or
      %   the plot's title (title_text()) when the hint is empty. Requires a build with X11 support and a reachable `DISPLAY`; use save()
      %   in headless code such as tests and CI.
      %
      %   Example:
      %     const auto x = dm::runif(0, 1, 1000);
      %     const auto y = dm::sin(20 / (x + 0.25)) + dm::rnorm(0, 0.2, 1000);
      %     dm::Plot(x, y).set_title("Doppler").set_col("blue").show("Doppler signal");
      %
      %   Inputs:
      %     title_hint  char row, default '': Window title; empty (the default) uses the plot's title.
      %
      %   Errors:
      %     datamunge:native  If X11 support was not compiled in, the display cannot be opened, its visual is unsupported, or the plot has no data series. (C++ std::runtime_error)
      %
      %   C++: void datamunge::graphics::Plot::show(const std::string & title_hint = "") const
      %
      %   See also: datamunge.save(), datamunge.to_svg()
      __datamunge__(1017, obj, varargin{:});
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::graphics::Plot. Do not edit.
