classdef RPlot < datamunge.Plot
  % RPlot  A single R-base-graphics-style plot, spelled `dm::Plot` in user code.
  %
  %   obj = datamunge.RPlot ()
  %   obj = datamunge.RPlot (x, y)
  %
  %   Covers R's high-level plotting calls (`plot()`, `hist()`, `barplot()`, `boxplot()`, `pie()`,
  %   `curve()`, `qqnorm()`, ...) as constructors and static factories, plus the "add to the
  %   current plot" verbs R exposes as separate functions (`points()`, `lines()`, `abline()`,
  %   `legend()`, `text()`, ...) as chainable member functions. A plot is an ordinary value: it
  %   owns copies of its data, copies cheaply enough to pass around, and draws nothing until it
  %   is rendered with to_svg(), save(), or show().
  %
  %   Two composition styles mix freely. Setters such as set_title() and verbs such as points()
  %   mutate the plot and return `RPlot&` for chaining; the pipe `plot | dm::points(...)` (see
  %   compose.hpp) appends a reusable layer to a copy. A fresh plot has the title
  %   "datamunge plot", axis labels "x" and "y", a 960x640 canvas, and a light grid.
  %
  %   Example:
  %     const auto x = dm::seql(-2.0, 2.0, 9);
  %     auto plt = dm::Plot(x, x * x)
  %                    .set_title("Parabola")
  %                    .set_xlab("x")
  %                    .set_ylab("x squared");
  %     plt.lines(x, x * x, dm::RGB{220, 38, 38}).abline_h(0.0);
  %     plt.save("parabola.svg");
  %
  %   Inherits from datamunge.Plot.
  %
  %   Methods:
  %     size_  Sets the canvas size in pixels (default 960 by 640), like the `width`/`height` of R's `png()` or `svg()` device.
  %     title  Sets the main heading, like R's `main =`; an empty string removes it.
  %     subtitle  Sets the subtitle and returns this plot for chaining RPlot verbs; same as set_subtitle().
  %     x_label  Sets the horizontal-axis label; same as xlab() and set_xlab().
  %     y_label  Sets the vertical-axis label; same as ylab() and set_ylab().
  %     xlab  R-style alias for x_label(), like R's `xlab =`.
  %     ylab  R-style alias for y_label(), like R's `ylab =`.
  %     set_title  Sets the main heading drawn above the plotting area (R's `main =`).
  %     set_subtitle  Sets the smaller second heading drawn beneath the title (unlike R's `sub =`, which sits below the x axis).
  %     set_xlab  Sets the horizontal-axis label (R's `xlab =`).
  %     set_ylab  Sets the vertical-axis label (R's `ylab =`).
  %     set_col  Recolors every data series already on the plot (R's `col =` applied after the fact).
  %     background  Sets the fill color of the whole canvas, like R's `par(bg =)`.
  %     axis_color  Sets the color of the axis lines and of all plot-level text: tick labels, axis labels, title, and subtitle (R's `col.axis`, `col.lab`, and `col.main` in one).
  %     grid_color  Sets the color of the background grid lines (default light gray `{229, 231, 235}`).
  %     show_grid  Turns the light background grid on or off (on by default), like adding or omitting R's `grid()`.
  %     x_limits  Fixes the visible x range, like R's `xlim = c(min_x, max_x)`.
  %     y_limits  Fixes the visible y range, like R's `ylim = c(min_y, max_y)`.
  %     xlim  Short spelling of x_limits(), matching R's xlim plot argument. Fixes the horizontal plotting range without automatic padding. Data outside the range follows x_limits() rendering behavior; this does not filter layers.
  %     ylim  Short spelling of y_limits(), matching R's ylim plot argument. Fixes the vertical plotting range without automatic padding; layers are retained.
  %     hide_axes  Hides the axis lines, tick labels, grid, and axis labels, like R's `axes = FALSE, ann = FALSE` (the title and subtitle are still drawn).
  %     x_tick_labels  Replaces the numeric x ticks with `labels` at x = 0, 1, 2, ..., like R's `axis(1, at = 0:(n - 1), labels = labels)`.
  %     x_ticks  Replaces the numeric x ticks with `labels` drawn at data positions `positions`, like R's `axis(1, at = positions, labels = labels)`.
  %     y_tick_labels  Replaces the numeric y ticks with `labels` at y = 0, 1, 2, ..., like R's `axis(2, at = 0:(n - 1), labels = labels)`.
  %     y_ticks  Replaces the numeric y ticks with `labels` drawn at data positions `positions`, like R's `axis(2, at = positions, labels = labels)`.
  %     margins  Overrides the automatic outer margins around the plotting area, in pixels, like R's `par(mai =)` (which uses inches and orders bottom, left, top, right).
  %     tick_font_size  Sets the tick-label font size in pixels (default 12), like R's `cex.axis`.
  %     pretty_ticks  Places numeric ticks at round values (default on) instead of six evenly spaced ticks across the range, in the spirit of R's `pretty()`.
  %     points  Adds a scatter series to this plot, like R's `points(x, y, col = color)`.
  %     points_colored  Appends scatter points with a recycled per-point RGB(A) palette. Coordinates must have equal nonzero lengths; colors must not be empty. A partial palette cycle is allowed without a warning. Colors retain their own alpha. The series is unlabeled, so it never appears in the automatic legend.
  %     line  Adds a line joining the points in input order, like R's `lines(x, y, col = color, lwd = stroke_width)`; same as lines().
  %     lines  Adds a line joining the points in input order, like R's `lines(x, y, col = color, lwd = stroke_width)`.
  %     bars  Adds vertical bars centered at arbitrary x positions, the low-level form behind barplot() and hist() (R's `rect()` from 0 to each height).
  %     box  Adds one box-and-whisker from precomputed statistics at an arbitrary x position, the low-level form behind boxplot() (R's `bxp()`).
  %     abline  Adds the reference line y = intercept + slope * x across the plotting area, like R's `abline(a = intercept, b = slope)`.
  %     abline_h  Adds a horizontal reference line at y = `y_value`, like R's `abline(h = y_value)`.
  %     abline_v  Adds a vertical reference line at x = `x_value`, like R's `abline(v = x_value)`.
  %     qqline  Adds the reference line of a normal Q-Q plot, like R's `qqline(y)`: the line through the first and third quartiles of `data` and of the standard normal.
  %     legend  Adds manual legend rows, like R's `legend(legend = labels, col = colors)`.
  %     text  Draws `label` centered on the data point (`x`, `y`), like R's `text(x, y, labels = label)`.
  %     polygon  Draws the closed polygon through the vertices (x[i], y[i]), like R's `polygon(x, y, col = color)`.
  %     image  Draws a grid of coloured rectangles, one per value of `z`, like R's `image(x, y, z)`.
  %     segments  Draws one straight segment from (x0[i], y0[i]) to (x1[i], y1[i]) for each i, like R's `segments(x0, y0, x1, y1)`.
  %
  %   Static methods:
  %     create  Returns a fresh empty plot; same as the default constructor `dm::Plot()`.
  %     plot  Builds a scatter, line, or line-plus-points plot of `y` against `x`, like R's `plot(x, y, type = "p" | "l" | "b")`.
  %     hist  Builds a frequency histogram of `data` with `bins` equal-width bins, like R's `hist(x, breaks = bins)`.
  %     barplot  Builds a bar chart with one bar per height, like R's `barplot(heights, names.arg = names)`.
  %     boxplot  Builds side-by-side box-and-whisker plots, one per group, like R's `boxplot(list(...), names = names)`.
  %     pie  Builds a pie chart with one wedge per value, like R's `pie(x, labels = names, col = colors, clockwise = TRUE)`.
  %     curve  Plots a function over [`from`, `to`] as a line, like R's `curve(expr, from, to, n)`.
  %     qqnorm  Builds a normal Q-Q plot: the sorted `data` against standard-normal quantiles, like R's `qqnorm(y)`.
  %
  %   C++: class datamunge::graphics::RPlot
  %
  %   See also: datamunge.points(), datamunge.legend(), datamunge.Histogram, datamunge.Splom, datamunge.RLayout
  methods
    function obj = RPlot (varargin)
      % RPlot  Construct a C++ datamunge::graphics::RPlot.
      %
      %   obj = datamunge.RPlot ()
      %   obj = datamunge.RPlot (x, y)
      %
      %   Example:
      %     auto plt = dm::Plot();
      %     std::print("{} | {} | {}\n", plt.title_text(), plt.x_label_text(), plt.y_label_text());
      %     // datamunge plot | x | y
      %
      %   Constructs an empty plot with default labels and styling, ready for layers.
      %
      %   C++: datamunge::graphics::RPlot::RPlot()
      %   C++: datamunge::graphics::RPlot::RPlot(std::vector<double> x, std::vector<double> y)
      obj@datamunge.Plot(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1070, false, varargin{:});
    end
    function r = size_ (obj, width, height)
      % size_  Sets the canvas size in pixels (default 960 by 640), like the `width`/`height` of R's `png()` or `svg()` device.
      %
      %   r = obj.size_ (width, height)
      %
      %   Not validated here: rendering throws std::runtime_error when the canvas leaves no room
      %   inside the margins (by default the canvas must exceed 114 by 156 pixels).
      %
      %   Example:
      %     auto plt = dm::Plot(dm::c(1.0, 2.0), dm::c(1.0, 2.0)).size(400, 300);
      %     std::print("{}x{}\n", plt.width(), plt.height());  // 400x300
      %
      %   Inputs:
      %     width  integer: Canvas width in pixels.
      %     height  integer: Canvas height in pixels.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::size(std::size_t width, std::size_t height)
      %
      %   See also: datamunge.margins()
      r = __datamunge__(1020, obj, width, height);
    end
    function r = title (obj, value_)
      % title  Sets the main heading, like R's `main =`; an empty string removes it.
      %
      %   r = obj.title (value_)
      %
      %   Inputs:
      %     value_  char row: Heading text (copied).
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::title(std::string value)
      %
      %   See also: datamunge.set_title()
      r = __datamunge__(1021, obj, value_);
    end
    function r = subtitle (obj, value_)
      % subtitle  Sets the subtitle and returns this plot for chaining RPlot verbs; same as set_subtitle().
      %
      %   r = obj.subtitle (value_)
      %
      %   Inputs:
      %     value_  char row: Subtitle text drawn under the title; empty removes it.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::subtitle(std::string value)
      %
      %   See also: datamunge.set_subtitle()
      r = __datamunge__(1022, obj, value_);
    end
    function r = x_label (obj, value_)
      % x_label  Sets the horizontal-axis label; same as xlab() and set_xlab().
      %
      %   r = obj.x_label (value_)
      %
      %   Inputs:
      %     value_  char row: Label text (copied).
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::x_label(std::string value)
      %
      %   See also: datamunge.set_xlab()
      r = __datamunge__(1023, obj, value_);
    end
    function r = y_label (obj, value_)
      % y_label  Sets the vertical-axis label; same as ylab() and set_ylab().
      %
      %   r = obj.y_label (value_)
      %
      %   Inputs:
      %     value_  char row: Label text (copied).
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::y_label(std::string value)
      %
      %   See also: datamunge.set_ylab()
      r = __datamunge__(1024, obj, value_);
    end
    function r = xlab (obj, value_)
      % xlab  R-style alias for x_label(), like R's `xlab =`.
      %
      %   r = obj.xlab (value_)
      %
      %   Inputs:
      %     value_  char row: Label text (copied).
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::xlab(std::string value)
      %
      %   See also: datamunge.set_xlab()
      r = __datamunge__(1025, obj, value_);
    end
    function r = ylab (obj, value_)
      % ylab  R-style alias for y_label(), like R's `ylab =`.
      %
      %   r = obj.ylab (value_)
      %
      %   Inputs:
      %     value_  char row: Label text (copied).
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::ylab(std::string value)
      %
      %   See also: datamunge.set_ylab()
      r = __datamunge__(1026, obj, value_);
    end
    function r = set_title (obj, value_)
      % set_title  Sets the main heading drawn above the plotting area (R's `main =`).
      %
      %   r = obj.set_title (value_)
      %
      %   The plot owns a copy of the text. An empty string clears the heading; a fresh plot
      %   shows "datamunge plot" until this is called. Same as title().
      %
      %   Example:
      %     auto plt = dm::Plot(dm::c(1.0, 2.0, 3.0), dm::c(2.0, 4.0, 8.0))
      %                    .set_title("Growth")
      %                    .set_subtitle("Doubling every step")
      %                    .set_xlab("Step")
      %                    .set_ylab("Count");
      %     std::print("{} / {}\n", plt.title_text(), plt.subtitle_text());
      %     // Growth / Doubling every step
      %
      %   Inputs:
      %     value_  char row: Heading text.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::set_title(std::string value)
      %
      %   See also: datamunge.set_subtitle(), datamunge.set_xlab(), datamunge.set_ylab(), datamunge.title()
      r = __datamunge__(1027, obj, value_);
    end
    function r = set_subtitle (obj, value_)
      % set_subtitle  Sets the smaller second heading drawn beneath the title (unlike R's `sub =`, which sits below the x axis).
      %
      %   r = obj.set_subtitle (value_)
      %
      %   The plot owns a copy of the text. Empty (the default) draws no subtitle and reserves no
      %   extra top margin. Same as subtitle().
      %
      %   Example:
      %     auto plt = dm::Plot().set_title("Iris Dataset").set_subtitle("Sepal Width vs Length");
      %     plt.set_subtitle("");  // remove it again
      %
      %   Inputs:
      %     value_  char row: Subtitle text.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::set_subtitle(std::string value)
      %
      %   See also: datamunge.set_title(), datamunge.subtitle()
      r = __datamunge__(1028, obj, value_);
    end
    function r = set_xlab (obj, value_)
      % set_xlab  Sets the horizontal-axis label (R's `xlab =`).
      %
      %   r = obj.set_xlab (value_)
      %
      %   The plot owns a copy of the text; the default is "x" and an empty string hides the
      %   label. Same as xlab() and x_label().
      %
      %   Example:
      %     auto plt = dm::Plot(dm::c(1.0, 2.0), dm::c(3.0, 4.0)).set_xlab("Sepal Width");
      %     std::print("{}\n", plt.x_label_text());  // Sepal Width
      %
      %   Inputs:
      %     value_  char row: Axis label text.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::set_xlab(std::string value)
      %
      %   See also: datamunge.set_ylab(), datamunge.xlab()
      r = __datamunge__(1029, obj, value_);
    end
    function r = set_ylab (obj, value_)
      % set_ylab  Sets the vertical-axis label (R's `ylab =`).
      %
      %   r = obj.set_ylab (value_)
      %
      %   The plot owns a copy of the text; the default is "y" and an empty string hides the
      %   label. Same as ylab() and y_label().
      %
      %   Example:
      %     auto plt = dm::Plot(dm::c(1.0, 2.0), dm::c(3.0, 4.0)).set_ylab("Sepal Length");
      %     std::print("{}\n", plt.y_label_text());  // Sepal Length
      %
      %   Inputs:
      %     value_  char row: Axis label text.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::set_ylab(std::string value)
      %
      %   See also: datamunge.set_xlab(), datamunge.ylab()
      r = __datamunge__(1030, obj, value_);
    end
    function r = set_col (obj, varargin)
      % set_col  Recolors every data series already on the plot (R's `col =` applied after the fact).
      %
      %   r = obj.set_col (color)
      %
      %   Every existing series (points, lines, bars, ...) takes `color`, and per-point palettes
      %   from dm::points(...).set_col() or points_colored() are cleared, so the whole plot becomes
      %   one color. Only layers present now are affected: points added later keep their own
      %   colors. Reference lines (abline) and manual legend entries keep their explicit colors.
      %
      %   Example:
      %     auto plt = dm::Plot(dm::c(1.0, 2.0, 3.0), dm::c(3.0, 1.0, 2.0))
      %                    .set_col(dm::RGB{220, 100, 255});
      %     std::print("{}\n", plt.series()[0].color.r);  // 220
      %
      %   Overloads are chosen at run time from the argument types (C++):
      %   C++: RPlot & datamunge::graphics::RPlot::set_col(RGBA color)
      %   C++: RPlot & datamunge::graphics::RPlot::set_col(std::string color)
      %   C++: RPlot & datamunge::graphics::RPlot::set_col(const char * color)
      %
      %   See also: datamunge.string), datamunge.points()
      r = __datamunge__(1031, obj, varargin{:});
    end
    function r = background (obj, color)
      % background  Sets the fill color of the whole canvas, like R's `par(bg =)`.
      %
      %   r = obj.background (color)
      %
      %   The default is white `{255, 255, 255}`. The color fills the margins as well as the
      %   plotting area. Its alpha is ignored: the background is always drawn opaque.
      %
      %   Example:
      %     auto plt = dm::Plot(dm::c(1.0, 2.0), dm::c(1.0, 2.0))
      %                    .background(dm::RGB{17, 24, 39})
      %                    .axis_color(dm::RGB{229, 231, 235})
      %                    .grid_color(dm::RGB{55, 65, 81});  // a dark theme
      %     std::print("{} {}\n", plt.background_color().r, plt.axes_color().r);  // 17 229
      %
      %   References:
      %   - R `par()` (`bg`): https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/par.html
      %
      %   Inputs:
      %     color  datamunge.RGB: Canvas fill color.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::background(RGBA color)
      %
      %   See also: datamunge.axis_color(), datamunge.grid_color()
      r = __datamunge__(1032, obj, color);
    end
    function r = axis_color (obj, color)
      % axis_color  Sets the color of the axis lines and of all plot-level text: tick labels, axis labels, title, and subtitle (R's `col.axis`, `col.lab`, and `col.main` in one).
      %
      %   r = obj.axis_color (color)
      %
      %   The default is slate gray `{55, 65, 81}`. Legend text follows it too; text() labels and
      %   data layers keep their own colors.
      %
      %   Example:
      %   See background(), which builds a dark theme with all three color setters.
      %
      %   References:
      %   - R `par()` (`col.axis`, `col.lab`, `col.main`): https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/par.html
      %
      %   Inputs:
      %     color  datamunge.RGB: Axis and text color.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::axis_color(RGBA color)
      %
      %   See also: datamunge.background(), datamunge.grid_color()
      r = __datamunge__(1033, obj, color);
    end
    function r = grid_color (obj, color)
      % grid_color  Sets the color of the background grid lines (default light gray `{229, 231, 235}`).
      %
      %   r = obj.grid_color (color)
      %
      %   Has no visible effect while the grid is off (show_grid(false)) or the axes are hidden.
      %
      %   Example:
      %   See background(), which builds a dark theme with all three color setters.
      %
      %   Inputs:
      %     color  datamunge.RGB: Grid-line color.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::grid_color(RGBA color)
      %
      %   See also: datamunge.show_grid(), datamunge.background()
      r = __datamunge__(1034, obj, color);
    end
    function r = show_grid (obj, varargin)
      % show_grid  Turns the light background grid on or off (on by default), like adding or omitting R's `grid()`.
      %
      %   r = obj.show_grid ()
      %   r = obj.show_grid (enabled)
      %
      %   One vertical line per x tick and one horizontal line per y tick, so the grid always lines
      %   up with the tick labels (round values, evenly spaced ticks, or custom tick positions).
      %   Hidden axes hide the grid too.
      %
      %   Example:
      %     auto plt = dm::Plot(dm::c(1.0, 2.0), dm::c(1.0, 2.0)).show_grid(false);
      %     std::print("{}\n", plt.grid_visible());  // false
      %
      %   References:
      %   - R `grid()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/grid.html
      %
      %   Inputs:
      %     enabled  logical, default true: True (the default) draws the grid; false omits it.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::show_grid(bool enabled = true)
      %
      %   See also: datamunge.grid_color(), datamunge.hide_axes()
      r = __datamunge__(1035, obj, varargin{:});
    end
    function r = x_limits (obj, min_x, max_x)
      % x_limits  Fixes the visible x range, like R's `xlim = c(min_x, max_x)`.
      %
      %   r = obj.x_limits (min_x, max_x)
      %
      %   Without limits the x range is fitted to every layer (and vertical reference line), plus
      %   5% padding on each side. With limits, the plotting area spans exactly [`min_x`, `max_x`]
      %   with no padding (R pads `xlim` by 4% unless `xaxs = "i"`). Data outside the range is not
      %   clipped and may be drawn over the margins.
      %
      %   Example:
      %     auto plt = dm::Plot(dm::c(1.0, 2.0, 3.0), dm::c(1.0, 4.0, 9.0))
      %                    .x_limits(0.0, 4.0)
      %                    .y_limits(0.0, 10.0);
      %     std::print("{} {} {}\n", plt.has_x_limits(), plt.x_max(), plt.y_max());  // true 4 10
      %
      %   References:
      %   - R `plot.default()` (`xlim`, `ylim`): https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/plot.default.html
      %
      %   Inputs:
      %     min_x  double: Left edge of the plotting area, in data units.
      %     max_x  double: Right edge; must be greater than `min_x`.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  Unless `max_x > min_x` (NaN fails); the plot is unchanged. (C++ std::invalid_argument)
      %
      %   C++: RPlot & datamunge::graphics::RPlot::x_limits(double min_x, double max_x)
      %
      %   See also: datamunge.y_limits()
      r = __datamunge__(1036, obj, min_x, max_x);
    end
    function r = y_limits (obj, min_y, max_y)
      % y_limits  Fixes the visible y range, like R's `ylim = c(min_y, max_y)`.
      %
      %   r = obj.y_limits (min_y, max_y)
      %
      %   Without limits the y range is fitted to every layer, plus 10% padding at each end (bar
      %   layers also include y = 0). With limits, the plotting area spans exactly
      %   [`min_y`, `max_y`]. Data outside the range is not clipped.
      %
      %   Example:
      %   See x_limits(), which sets both ranges.
      %
      %   References:
      %   - R `plot.default()` (`xlim`, `ylim`): https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/plot.default.html
      %
      %   Inputs:
      %     min_y  double: Bottom edge of the plotting area, in data units.
      %     max_y  double: Top edge; must be greater than `min_y`.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  Unless `max_y > min_y` (NaN fails); the plot is unchanged. (C++ std::invalid_argument)
      %
      %   C++: RPlot & datamunge::graphics::RPlot::y_limits(double min_y, double max_y)
      %
      %   See also: datamunge.x_limits()
      r = __datamunge__(1037, obj, min_y, max_y);
    end
    function r = xlim (obj, min_x, max_x)
      % xlim  Short spelling of x_limits(), matching R's xlim plot argument. Fixes the horizontal plotting range without automatic padding. Data outside the range follows x_limits() rendering behavior; this does not filter layers.
      %
      %   r = obj.xlim (min_x, max_x)
      %
      %   Example:
      %     auto plt = dm::plot().points(dm::c(0, 1), dm::c(1, 2)).xlim(0, 1).ylim(-4, 4);
      %     std::print("{} {}\n", plt.x_max(), plt.y_min()); // 1 -4
      %
      %   Inputs:
      %     min_x  double: Left edge in data units.
      %     max_x  double: Right edge; must be greater than min_x.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  Unless max_x > min_x (NaN fails); the plot is unchanged. (C++ std::invalid_argument)
      %
      %   C++: RPlot & datamunge::graphics::RPlot::xlim(double min_x, double max_x)
      %
      %   See also: datamunge.x_limits(), datamunge.ylim()
      r = __datamunge__(1038, obj, min_x, max_x);
    end
    function r = ylim (obj, min_y, max_y)
      % ylim  Short spelling of y_limits(), matching R's ylim plot argument. Fixes the vertical plotting range without automatic padding; layers are retained.
      %
      %   r = obj.ylim (min_y, max_y)
      %
      %   Example:
      %   See xlim(), whose example uses both aliases.
      %
      %   Inputs:
      %     min_y  double: Bottom edge in data units.
      %     max_y  double: Top edge; must be greater than min_y.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  Unless max_y > min_y (NaN fails); the plot is unchanged. (C++ std::invalid_argument)
      %
      %   C++: RPlot & datamunge::graphics::RPlot::ylim(double min_y, double max_y)
      %
      %   See also: datamunge.y_limits(), datamunge.xlim()
      r = __datamunge__(1039, obj, min_y, max_y);
    end
    function r = hide_axes (obj, varargin)
      % hide_axes  Hides the axis lines, tick labels, grid, and axis labels, like R's `axes = FALSE, ann = FALSE` (the title and subtitle are still drawn).
      %
      %   r = obj.hide_axes ()
      %   r = obj.hide_axes (enabled)
      %
      %   pie() turns this on. Margins are unchanged, so call margins() as well to let the data use
      %   the freed space.
      %
      %   Example:
      %     auto plt = dm::Plot(dm::c(1.0, 2.0), dm::c(1.0, 2.0)).hide_axes();
      %     std::print("{}\n", plt.axes_hidden());  // true
      %
      %   References:
      %   - R `plot.default()` (`axes`, `ann`): https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/plot.default.html
      %
      %   Inputs:
      %     enabled  logical, default true: True (the default) hides them; false shows them again.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::hide_axes(bool enabled = true)
      %
      %   See also: datamunge.show_grid(), datamunge.margins()
      r = __datamunge__(1040, obj, varargin{:});
    end
    function r = x_tick_labels (obj, labels)
      % x_tick_labels  Replaces the numeric x ticks with `labels` at x = 0, 1, 2, ..., like R's `axis(1, at = 0:(n - 1), labels = labels)`.
      %
      %   r = obj.x_tick_labels (labels)
      %
      %   Matches the bar positions of barplot() and boxplot(). Clears any positions set by
      %   x_ticks(); an empty vector restores the numeric ticks.
      %
      %   Example:
      %     auto plt = dm::Plot(dm::c(0.0, 1.0, 2.0), dm::c(5.0, 3.0, 4.0))
      %                    .x_tick_labels({"low", "mid", "high"});
      %     std::print("{}\n", plt.x_tick_label_list()[2]);  // high
      %
      %   References:
      %   - R `axis()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/axis.html
      %
      %   Inputs:
      %     labels  cell array of char: Tick texts, in order.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::x_tick_labels(std::vector<std::string> labels)
      %
      %   See also: datamunge.x_ticks(), datamunge.y_tick_labels()
      r = __datamunge__(1041, obj, labels);
    end
    function r = x_ticks (obj, positions, labels)
      % x_ticks  Replaces the numeric x ticks with `labels` drawn at data positions `positions`, like R's `axis(1, at = positions, labels = labels)`.
      %
      %   r = obj.x_ticks (positions, labels)
      %
      %   Positions need not be sorted or inside the visible range. Two empty vectors restore the
      %   numeric ticks.
      %
      %   Example:
      %     const auto x = dm::seql(0.0, 6.283, 50);
      %     auto plt = dm::Plot(x, dm::sin(x))
      %                    .x_ticks({0.0, 3.1416, 6.2832}, {"0", "pi", "2 pi"})
      %                    .y_ticks({-1.0, 0.0, 1.0}, {"min", "0", "max"});
      %     std::print("{}\n", plt.x_tick_position_list()[1]);  // 3.1416
      %
      %   References:
      %   - R `axis()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/axis.html
      %
      %   Inputs:
      %     positions  double vector: x data coordinates of the ticks.
      %     labels  cell array of char: Tick texts, one per position.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If the two vectors differ in length; the plot is unchanged. (C++ std::invalid_argument)
      %
      %   C++: RPlot & datamunge::graphics::RPlot::x_ticks(std::vector<double> positions, std::vector<std::string> labels)
      %
      %   See also: datamunge.x_tick_labels(), datamunge.y_ticks()
      r = __datamunge__(1042, obj, positions, labels);
    end
    function r = y_tick_labels (obj, labels)
      % y_tick_labels  Replaces the numeric y ticks with `labels` at y = 0, 1, 2, ..., like R's `axis(2, at = 0:(n - 1), labels = labels)`.
      %
      %   r = obj.y_tick_labels (labels)
      %
      %   Clears any positions set by y_ticks(); an empty vector restores the numeric ticks.
      %
      %   Example:
      %     auto plt = dm::Plot(dm::c(3.0, 5.0), dm::c(0.0, 1.0)).y_tick_labels({"no", "yes"});
      %     std::print("{}\n", plt.y_tick_label_list().size());  // 2
      %
      %   References:
      %   - R `axis()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/axis.html
      %
      %   Inputs:
      %     labels  cell array of char: Tick texts, in order.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::y_tick_labels(std::vector<std::string> labels)
      %
      %   See also: datamunge.y_ticks(), datamunge.x_tick_labels()
      r = __datamunge__(1043, obj, labels);
    end
    function r = y_ticks (obj, positions, labels)
      % y_ticks  Replaces the numeric y ticks with `labels` drawn at data positions `positions`, like R's `axis(2, at = positions, labels = labels)`.
      %
      %   r = obj.y_ticks (positions, labels)
      %
      %   Two empty vectors restore the numeric ticks.
      %
      %   Example:
      %   See x_ticks(), which sets both axes.
      %
      %   References:
      %   - R `axis()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/axis.html
      %
      %   Inputs:
      %     positions  double vector: y data coordinates of the ticks.
      %     labels  cell array of char: Tick texts, one per position.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If the two vectors differ in length; the plot is unchanged. (C++ std::invalid_argument)
      %
      %   C++: RPlot & datamunge::graphics::RPlot::y_ticks(std::vector<double> positions, std::vector<std::string> labels)
      %
      %   See also: datamunge.y_tick_labels(), datamunge.x_ticks()
      r = __datamunge__(1044, obj, positions, labels);
    end
    function r = margins (obj, left, right, top, bottom)
      % margins  Overrides the automatic outer margins around the plotting area, in pixels, like R's `par(mai =)` (which uses inches and orders bottom, left, top, right).
      %
      %   r = obj.margins (left, right, top, bottom)
      %
      %   The automatic margins are left 90, right 24, top 72 (96 when a subtitle is set), and
      %   bottom 84; they leave room for the title, axis labels, and ticks. Explicit margins are used
      %   as given, with no extra room for a subtitle. Tight margins suit small multi-panel tiles
      %   whose title and axis labels are empty. Rendering throws std::runtime_error if the margins
      %   leave no plotting area.
      %
      %   Example:
      %     auto tile = dm::Plot(dm::c(1.0, 2.0), dm::c(2.0, 1.0))
      %                     .set_title("").set_xlab("").set_ylab("")
      %                     .margins(30, 10, 10, 20)
      %                     .tick_font_size(9);
      %     std::print("{} {}\n", tile.has_custom_margins(), tile.margin_left());  // true 30
      %
      %   References:
      %   - R `par()` (`mai`, `mar`): https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/par.html
      %
      %   Inputs:
      %     left  double: Left margin in pixels.
      %     right  double: Right margin in pixels.
      %     top  double: Top margin in pixels.
      %     bottom  double: Bottom margin in pixels.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If any margin is negative or non-finite; the plot is unchanged. (C++ std::invalid_argument)
      %
      %   C++: RPlot & datamunge::graphics::RPlot::margins(double left, double right, double top, double bottom)
      %
      %   See also: datamunge.size(), datamunge.tick_font_size(), datamunge.RLayout
      r = __datamunge__(1045, obj, left, right, top, bottom);
    end
    function r = tick_font_size (obj, pixels)
      % tick_font_size  Sets the tick-label font size in pixels (default 12), like R's `cex.axis`.
      %
      %   r = obj.tick_font_size (pixels)
      %
      %   In SVG output the size is used as given. The raster outputs (PNG, PDF, show()) draw tick
      %   labels with half-size glyphs when the size is below 12.
      %
      %   Example:
      %   See margins(), which shrinks the ticks for a small tile.
      %
      %   References:
      %   - R `par()` (`cex.axis`): https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/par.html
      %
      %   Inputs:
      %     pixels  double: Font size in pixels.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If `pixels` is not positive and finite; the plot is unchanged. (C++ std::invalid_argument)
      %
      %   C++: RPlot & datamunge::graphics::RPlot::tick_font_size(double pixels)
      %
      %   See also: datamunge.margins()
      r = __datamunge__(1046, obj, pixels);
    end
    function r = pretty_ticks (obj, varargin)
      % pretty_ticks  Places numeric ticks at round values (default on) instead of six evenly spaced ticks across the range, in the spirit of R's `pretty()`.
      %
      %   r = obj.pretty_ticks ()
      %   r = obj.pretty_ticks (enabled)
      %
      %   The step is 1, 2, or 5 times a power of ten, chosen as the "nice number" nearest a fifth
      %   of the visible range (Heckbert's rule), and ticks are its multiples inside the range.
      %   That gives about four to seven ticks, close to R's `pretty()`: a padded range of -0.365
      %   to 7.665 gets ticks at 0, 2, 4 and 6. Custom tick labels take precedence. The background
      %   grid follows the ticks.
      %
      %   Example:
      %     auto plt = dm::Plot(dm::c(0.0, 100.0), dm::c(0.0, 1.0)).pretty_ticks();
      %     std::print("{}\n", plt.to_svg().contains(">40</text>"));  // true: ticks 0, 20, ..., 100
      %
      %   References:
      %   - R `pretty()`: https://stat.ethz.ch/R-manual/R-devel/library/base/html/pretty.html
      %   - P. S. Heckbert (1990). "Nice Numbers for Graph Labels". In Graphics Gems, 61-63.
      %     Academic Press. doi:10.1016/B978-0-08-050753-8.50020-8
      %
      %   Inputs:
      %     enabled  logical, default true: True turns round ticks on; false restores six even ticks.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::pretty_ticks(bool enabled = true)
      %
      %   See also: datamunge.x_ticks(), datamunge.show_grid()
      r = __datamunge__(1047, obj, varargin{:});
    end
    function r = points (obj, varargin)
      % points  Adds a scatter series to this plot, like R's `points(x, y, col = color)`.
      %
      %   r = obj.points (x, y)
      %   r = obj.points (x, y, label)
      %   r = obj.points (x, y, label, color)
      %   r = obj.points (x, y, label, color, marker_size)
      %   r = obj.points (x, y, color)
      %   r = obj.points (x, y, color, marker_size)
      %
      %   The coordinates are copied (or moved from rvalues) into a new series drawn above the
      %   existing ones; axis ranges grow to include it. A nonempty `label` adds a row to the
      %   automatic legend (shown only while no manual legend() entries exist).
      %
      %   Example:
      %     const auto x = dm::seql(0.0, 6.0, 13);
      %     auto plt = dm::Plot().set_title("sin and cos");
      %     plt.points(x, dm::sin(x), "sin", dm::RGB{37, 99, 235})
      %        .points(x, dm::cos(x), "cos", dm::RGB{220, 38, 38}, 6.0);
      %     plt.save("trig.svg");  // automatic legend lists "sin" and "cos"
      %
      %   References:
      %   - R `points()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/points.html
      %
      %   Errors:
      %     datamunge:invalid_argument  If the coordinates are empty or have different lengths. (C++ std::invalid_argument)
      %
      %   Overloads are chosen at run time from the argument types (C++):
      %   C++: RPlot & datamunge::graphics::RPlot::points(std::vector<double> x, std::vector<double> y, std::string label = "", RGBA color = ..., double marker_size = 4.0)
      %   C++: RPlot & datamunge::graphics::RPlot::points(std::vector<double> x, std::vector<double> y, RGBA color, double marker_size = 4.0)
      %   C++: RPlot & datamunge::graphics::RPlot::points(std::vector<double> x, std::vector<double> y, const char * label)
      %
      %   See also: datamunge.points_colored(), datamunge.points(), datamunge.set_col()
      r = __datamunge__(1055, obj, varargin{:});
    end
    function r = points_colored (obj, varargin)
      % points_colored  Appends scatter points with a recycled per-point RGB(A) palette. Coordinates must have equal nonzero lengths; colors must not be empty. A partial palette cycle is allowed without a warning. Colors retain their own alpha. The series is unlabeled, so it never appears in the automatic legend.
      %
      %   r = obj.points_colored (x, y, colors)
      %   r = obj.points_colored (x, y, colors, marker_size)
      %
      %   Example:
      %     const auto palette = dm::c(dm::RGB{255, 0, 0}, dm::RGB{0, 0, 255});
      %     auto plt = dm::Plot().points_colored(dm::c(1.0, 2.0, 3.0), dm::c(3.0, 2.0, 1.0), palette);
      %     std::print("{}\n", plt.series()[0].point_colors.size());  // 2 (red, blue, red)
      %
      %   References:
      %   - R `points()` (vector `col`): https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/points.html
      %
      %   Inputs:
      %     x  double vector: Horizontal coordinates.
      %     y  double vector: Vertical coordinates, paired with `x`.
      %     colors  cell array of datamunge.RGB: Palette; point i uses `colors[i % colors.size()]`.
      %     marker_size  double, default 4.0: Marker radius in pixels (default 4).
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If the coordinate lengths or palette are invalid. (C++ std::invalid_argument)
      %
      %   C++: RPlot & datamunge::graphics::RPlot::points_colored(std::vector<double> x, std::vector<double> y, std::vector<RGBA> colors, double marker_size = 4.0)
      %
      %   See also: datamunge.set_col(), datamunge.the pipe-friendly form of this call.
      r = __datamunge__(1056, obj, varargin{:});
    end
    function r = line (obj, varargin)
      % line  Adds a line joining the points in input order, like R's `lines(x, y, col = color, lwd = stroke_width)`; same as lines().
      %
      %   r = obj.line (x, y)
      %   r = obj.line (x, y, label)
      %   r = obj.line (x, y, label, color)
      %   r = obj.line (x, y, label, color, stroke_width)
      %
      %   The points are not sorted, so unsorted x draws a zigzag. The coordinates are copied (or
      %   moved from rvalues) into a new Line series drawn above existing layers; axis ranges grow
      %   to include it. A nonempty `label` adds a row to the automatic legend.
      %
      %   Example:
      %   See lines().
      %
      %   References:
      %   - R `lines()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/lines.html
      %
      %   Inputs:
      %     x  double vector: Horizontal coordinates, one per vertex.
      %     y  double vector: Vertical coordinates, paired with `x` by position.
      %     label  char row, default '': Legend text; empty (the default) keeps it out of the legend.
      %     color  datamunge.RGB, default the C++ default: Line color (default red `{220, 38, 38}`).
      %     stroke_width  double, default 2.0: Line width in pixels (default 2).
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If the coordinates are empty or have different lengths. (C++ std::invalid_argument)
      %
      %   C++: RPlot & datamunge::graphics::RPlot::line(std::vector<double> x, std::vector<double> y, std::string label = "", RGBA color = ..., double stroke_width = 2.0)
      %
      %   See also: datamunge.lines(), datamunge.plot()
      r = __datamunge__(1057, obj, varargin{:});
    end
    function r = lines (obj, varargin)
      % lines  Adds a line joining the points in input order, like R's `lines(x, y, col = color, lwd = stroke_width)`.
      %
      %   r = obj.lines (x, y)
      %   r = obj.lines (x, y, label)
      %   r = obj.lines (x, y, label, color)
      %   r = obj.lines (x, y, label, color, stroke_width)
      %   r = obj.lines (x, y, color)
      %   r = obj.lines (x, y, color, stroke_width)
      %
      %   Same as line(). The points are not sorted, so unsorted x draws a zigzag. The coordinates
      %   are copied (or moved from rvalues) into a new Line series drawn above existing layers;
      %   axis ranges grow to include it. A nonempty `label` adds a row to the automatic legend.
      %
      %   Example:
      %     const auto x = dm::seql(0.0, 6.0, 25);
      %     auto plt = dm::Plot(x, dm::sin(x) + dm::rnorm(0, 0.1, 25));
      %     plt.lines(x, dm::sin(x), "sin(x)", dm::RGB{220, 38, 38}, 3.0);
      %     std::print("{}\n", plt.series()[1].stroke_width);  // 3
      %
      %   References:
      %   - R `lines()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/lines.html
      %
      %   Errors:
      %     datamunge:invalid_argument  If the coordinates are empty or have different lengths. (C++ std::invalid_argument)
      %
      %   Overloads are chosen at run time from the argument types (C++):
      %   C++: RPlot & datamunge::graphics::RPlot::lines(std::vector<double> x, std::vector<double> y, std::string label = "", RGBA color = ..., double stroke_width = 2.0)
      %   C++: RPlot & datamunge::graphics::RPlot::lines(std::vector<double> x, std::vector<double> y, RGBA color, double stroke_width = 2.0)
      %   C++: RPlot & datamunge::graphics::RPlot::lines(std::vector<double> x, std::vector<double> y, const char * label)
      %
      %   See also: datamunge.line(), datamunge.vector<double>, datamunge.vector<double>, datamunge.RGB, datamunge.double), datamunge.plot()
      r = __datamunge__(1058, obj, varargin{:});
    end
    function r = bars (obj, varargin)
      % bars  Adds vertical bars centered at arbitrary x positions, the low-level form behind barplot() and hist() (R's `rect()` from 0 to each height).
      %
      %   r = obj.bars (x, y)
      %   r = obj.bars (x, y, label)
      %   r = obj.bars (x, y, label, color)
      %   r = obj.bars (x, y, label, color, bar_width)
      %
      %   Bar i spans x[i] +/- bar_width / 2 and rises from 0 to y[i] (negative heights point
      %   down); the y range always includes 0. Bars are drawn 85% opaque in SVG output. A nonempty
      %   `label` adds a row to the automatic legend.
      %
      %   Example:
      %     auto plt = dm::Plot().bars(dm::c(1.0, 2.5, 4.0), dm::c(3.0, -1.0, 2.0), "net",
      %                                dm::RGB{22, 163, 74}, 1.0);
      %     std::print("{}\n", plt.series()[0].bar_width);  // 1
      %
      %   References:
      %   - R `rect()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/rect.html
      %
      %   Inputs:
      %     x  double vector: Bar centers, in data units.
      %     y  double vector: Bar heights, paired with `x` by position.
      %     label  char row, default '': Legend text; empty (the default) keeps it out of the legend.
      %     color  datamunge.RGB, default the C++ default: Bar color (default green `{22, 163, 74}`).
      %     bar_width  double, default 0.8: Bar width in data units (default 0.8).
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If the coordinates are empty or have different lengths. (C++ std::invalid_argument)
      %
      %   C++: RPlot & datamunge::graphics::RPlot::bars(std::vector<double> x, std::vector<double> y, std::string label = "", RGBA color = ..., double bar_width = 0.80000000000000004)
      %
      %   See also: datamunge.barplot(), datamunge.hist()
      r = __datamunge__(1059, obj, varargin{:});
    end
    function r = box (obj, varargin)
      % box  Adds one box-and-whisker from precomputed statistics at an arbitrary x position, the low-level form behind boxplot() (R's `bxp()`).
      %
      %   r = obj.box (position, whisker_lo, q1, median, q3, whisker_hi)
      %   r = obj.box (position, whisker_lo, q1, median, q3, whisker_hi, outliers)
      %   r = obj.box (position, whisker_lo, q1, median, q3, whisker_hi, outliers, color)
      %   r = obj.box (position, whisker_lo, q1, median, q3, whisker_hi, outliers, color, width)
      %
      %   Draws a box from `q1` to `q3` with a thicker line at `median`, whiskers out to
      %   `whisker_lo` and `whisker_hi` with end caps, and one dot per outlier. The values are
      %   not checked for order or finiteness. In SVG the box is filled at 35% opacity.
      %
      %   Example:
      %     auto plt = dm::Plot().box(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, {8.0}).box(1.0, 2.0, 3.0, 3.5, 4.0, 6.0);
      %     std::print("{} {}\n", plt.series().size(), plt.series()[0].y.size());  // 2 6
      %
      %   References:
      %   - J. W. Tukey (1977). Exploratory Data Analysis. Addison-Wesley.
      %   - R `bxp()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/bxp.html
      %
      %   Inputs:
      %     position  double: Box center, in x data units.
      %     whisker_lo  double: Lower whisker end.
      %     q1  double: Lower box edge (first quartile).
      %     median  double: Median line.
      %     q3  double: Upper box edge (third quartile).
      %     whisker_hi  double: Upper whisker end.
      %     outliers  double vector, default the C++ default: Values drawn as individual points (default none).
      %     color  datamunge.RGB, default the C++ default: Box, whisker, and outlier color (default light blue `{96, 165, 250}`).
      %     width  double, default 0.6: Box width in data units (default 0.6).
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::box(double position, double whisker_lo, double q1, double median, double q3, double whisker_hi, std::vector<double> outliers = {}, RGBA color = ..., double width = 0.59999999999999998)
      %
      %   See also: datamunge.boxplot()
      r = __datamunge__(1060, obj, varargin{:});
    end
    function r = abline (obj, varargin)
      % abline  Adds the reference line y = intercept + slope * x across the plotting area, like R's `abline(a = intercept, b = slope)`.
      %
      %   r = obj.abline (intercept, slope)
      %   r = obj.abline (intercept, slope, color)
      %   r = obj.abline (intercept, slope, color, stroke_width)
      %
      %   The line runs between the left and right edges of the visible x range and is drawn
      %   beneath every data layer. It does not change the axis ranges, and it is not clipped to the
      %   plotting area, so a steep line can extend into the margins. Stored as an ABLine.
      %
      %   Example:
      %     const auto x = dm::c(1.0, 2.0, 3.0, 4.0);
      %     auto plt = dm::Plot(x, dm::c(2.1, 3.9, 6.2, 7.8))
      %                    .abline(0.0, 2.0, dm::RGB{220, 38, 38})  // y = 2x
      %                    .abline_h(5.0)
      %                    .abline_v(2.5, dm::RGB{37, 99, 235}, 2.0);
      %     std::print("{}\n", plt.reference_lines().size());  // 3
      %     plt.save("ablines.svg");
      %
      %   References:
      %   - R `abline()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/abline.html
      %
      %   Inputs:
      %     intercept  double: y value at x = 0.
      %     slope  double: Change in y per unit x.
      %     color  datamunge.RGB, default the C++ default: Line color (default opaque black).
      %     stroke_width  double, default 1.0: Line width in pixels (default 1).
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   Warning: Only to_svg() and `.svg` saves draw reference lines. The raster outputs (`.png`, `.pdf`, and show()) currently omit them.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::abline(double intercept, double slope, RGBA color = ..., double stroke_width = 1.0)
      %
      %   See also: datamunge.abline_h(), datamunge.abline_v(), datamunge.qqline(), datamunge.ABLine
      r = __datamunge__(1061, obj, varargin{:});
    end
    function r = abline_h (obj, varargin)
      % abline_h  Adds a horizontal reference line at y = `y_value`, like R's `abline(h = y_value)`.
      %
      %   r = obj.abline_h (y_value)
      %   r = obj.abline_h (y_value, color)
      %   r = obj.abline_h (y_value, color, stroke_width)
      %
      %   Same as `abline(y_value, 0.0, color, stroke_width)`: drawn beneath the data and not
      %   included in the axis ranges, so a value outside the visible y range is off the plot.
      %
      %   Example:
      %   See abline().
      %
      %   References:
      %   - R `abline()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/abline.html
      %
      %   Inputs:
      %     y_value  double: Height of the line, in data units.
      %     color  datamunge.RGB, default the C++ default: Line color (default opaque black).
      %     stroke_width  double, default 1.0: Line width in pixels (default 1).
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   Warning: Only to_svg() and `.svg` saves draw reference lines. The raster outputs (`.png`, `.pdf`, and show()) currently omit them.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::abline_h(double y_value, RGBA color = ..., double stroke_width = 1.0)
      %
      %   See also: datamunge.abline(), datamunge.abline_v()
      r = __datamunge__(1062, obj, varargin{:});
    end
    function r = abline_v (obj, varargin)
      % abline_v  Adds a vertical reference line at x = `x_value`, like R's `abline(v = x_value)`.
      %
      %   r = obj.abline_v (x_value)
      %   r = obj.abline_v (x_value, color)
      %   r = obj.abline_v (x_value, color, stroke_width)
      %
      %   Spans the full height of the plotting area beneath the data. Unlike horizontal and
      %   sloped lines, it widens the automatic x range to include `x_value`.
      %
      %   Example:
      %   See abline().
      %
      %   References:
      %   - R `abline()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/abline.html
      %
      %   Inputs:
      %     x_value  double: Horizontal position of the line, in data units.
      %     color  datamunge.RGB, default the C++ default: Line color (default opaque black).
      %     stroke_width  double, default 1.0: Line width in pixels (default 1).
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   Warning: Only to_svg() and `.svg` saves draw reference lines. The raster outputs (`.png`, `.pdf`, and show()) currently omit them.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::abline_v(double x_value, RGBA color = ..., double stroke_width = 1.0)
      %
      %   See also: datamunge.abline(), datamunge.abline_h()
      r = __datamunge__(1063, obj, varargin{:});
    end
    function r = qqline (obj, varargin)
      % qqline  Adds the reference line of a normal Q-Q plot, like R's `qqline(y)`: the line through the first and third quartiles of `data` and of the standard normal.
      %
      %   r = obj.qqline (data)
      %   r = obj.qqline (data, color)
      %   r = obj.qqline (data, color, stroke_width)
      %
      %   Pass the same data given to qqnorm(). The sample quartiles are R's default "type 7"
      %   quantiles, as in R's `qqline()` defaults (`probs = c(0.25, 0.75)`, `qtype = 7`). The
      %   slope is the sample IQR divided by the normal IQR (about 1.349), so for normal data it
      %   estimates the standard deviation and the intercept estimates the mean. Added through
      %   abline().
      %
      %   Example:
      %     const std::vector<double> sample{3.0, 1.0, 2.0, 5.0, 4.0};
      %     auto plt = dm::Plot::qqnorm(sample).qqline(sample);
      %     const auto& line = plt.reference_lines()[0];
      %     std::print("{:.4f} {:.4f}\n", line.value, line.slope);  // 3.0000 1.4826
      %
      %   References:
      %   - R `qqline()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/qqnorm.html
      %   - R. J. Hyndman, Y. Fan (1996). "Sample Quantiles in Statistical Packages". The American
      %     Statistician 50(4), 361-365. doi:10.1080/00031305.1996.10473566
      %
      %   Inputs:
      %     data  double vector: The sample; at least two values. Taken by value, so the caller's data is unchanged.
      %     color  datamunge.RGB, default the C++ default: Line color (default red `{220, 38, 38}`).
      %     stroke_width  double, default 1.5: Line width in pixels (default 1.5).
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If `data` has fewer than two values. (C++ std::invalid_argument)
      %
      %   Warning: Only to_svg() and `.svg` saves draw reference lines. The raster outputs (`.png`, `.pdf`, and show()) currently omit them.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::qqline(std::vector<double> data, RGBA color = ..., double stroke_width = 1.5)
      %
      %   See also: datamunge.qqnorm(), datamunge.abline()
      r = __datamunge__(1064, obj, varargin{:});
    end
    function r = legend (obj, labels, colors)
      % legend  Adds manual legend rows, like R's `legend(legend = labels, col = colors)`.
      %
      %   r = obj.legend (labels, colors)
      %
      %   Each label is paired with the color at the same position and drawn as a swatch row in
      %   the legend box. Once any manual row exists, it replaces the automatic legend built from
      %   labeled series. Calling again appends more rows. Unlike dm::legend(), colors are not
      %   recycled here: the vectors must match in length. Empty vectors add nothing.
      %
      %   Example:
      %     const auto up = dm::RGB{37, 99, 235};
      %     const auto down = dm::RGB{220, 38, 38};
      %     auto plt = dm::Plot();
      %     plt.points(dm::c(1.0, 2.0), dm::c(1.0, 2.0), up)
      %        .points(dm::c(1.0, 2.0), dm::c(2.0, 1.0), down)
      %        .legend({"rising", "falling"}, {up, down});
      %     std::print("{}\n", plt.legend_entries().size());  // 2
      %
      %   References:
      %   - R `legend()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/legend.html
      %
      %   Inputs:
      %     labels  cell array of char: Row texts, in display order.
      %     colors  cell array of datamunge.RGB: Swatch colors, one per label.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If `labels` and `colors` differ in length. (C++ std::invalid_argument)
      %
      %   C++: RPlot & datamunge::graphics::RPlot::legend(std::vector<std::string> labels, std::vector<RGBA> colors)
      %
      %   See also: datamunge.legend() for the recycling, datamunge.pipe-friendly layer.
      r = __datamunge__(1065, obj, labels, colors);
    end
    function r = text (obj, varargin)
      % text  Draws `label` centered on the data point (`x`, `y`), like R's `text(x, y, labels = label)`.
      %
      %   r = obj.text (x, y, label)
      %   r = obj.text (x, y, label, color)
      %   r = obj.text (x, y, label, color, font_size)
      %
      %   The text is centered both horizontally and vertically on the point, like R's default
      %   `adj`. The point counts toward the automatic axis ranges, but the text's extent does not,
      %   so long labels near the edge can overflow. Text layers never appear in the legend, and
      %   set_col() recolors them along with the data.
      %
      %   Example:
      %     auto plt = dm::Plot(dm::c(1.0, 2.0, 3.0), dm::c(1.0, 4.0, 2.0))
      %                    .text(2.0, 4.4, "peak", dm::RGB{220, 38, 38}, 16.0);
      %     std::print("{} {}\n", plt.series()[1].label, plt.series()[1].marker_size);  // peak 16
      %
      %   References:
      %   - R `text()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/text.html
      %
      %   Inputs:
      %     x  double: Horizontal position, in data units.
      %     y  double: Vertical position, in data units.
      %     label  char row: The text to draw (XML-escaped in SVG output).
      %     color  datamunge.RGB, default the C++ default: Text color (default near-black `{17, 24, 39}`).
      %     font_size  double, default 14.0: Font size in pixels (default 14).
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   C++: RPlot & datamunge::graphics::RPlot::text(double x, double y, std::string label, RGBA color = ..., double font_size = 14.0)
      %
      %   See also: datamunge.pie(), datamunge.legend()
      r = __datamunge__(1066, obj, varargin{:});
    end
    function r = polygon (obj, varargin)
      % polygon  Draws the closed polygon through the vertices (x[i], y[i]), like R's `polygon(x, y, col = color)`.
      %
      %   r = obj.polygon (x, y)
      %   r = obj.polygon (x, y, color)
      %   r = obj.polygon (x, y, color, filled)
      %
      %   The last vertex joins the first automatically. Filled polygons (the default) are painted
      %   with `color` (plus a 1-pixel outline in SVG; the raster fill uses the even-odd rule);
      %   unfilled ones draw a 2-pixel outline only (R's default is unfilled). Translucent colors work well
      %   for shaded regions such as confidence bands. Vertices count toward the axis ranges.
      %
      %   Example:
      %     auto plt = dm::Plot()
      %                    .polygon({0.0, 1.0, 0.5}, {0.0, 0.0, 1.0}, dm::RGBA{37, 99, 235, 0.4})
      %                    .polygon({0.2, 0.8, 0.8, 0.2}, {0.1, 0.1, 0.4, 0.4}, dm::RGB{0, 0, 0}, false);
      %     std::print("{}\n", plt.series()[1].filled);  // false
      %
      %   References:
      %   - R `polygon()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/polygon.html
      %
      %   Inputs:
      %     x  double vector: Vertex x coordinates, in order around the polygon.
      %     y  double vector: Vertex y coordinates, paired with `x`.
      %     color  datamunge.RGB, default the C++ default: Fill (and outline) color (default blue `{37, 99, 235}`).
      %     filled  logical, default true: True (the default) fills; false draws the outline only.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If the coordinates are empty or have different lengths. (C++ std::invalid_argument)
      %
      %   C++: RPlot & datamunge::graphics::RPlot::polygon(std::vector<double> x, std::vector<double> y, RGBA color = ..., bool filled = true)
      %
      %   See also: datamunge.pie(), datamunge.segments()
      r = __datamunge__(1067, obj, varargin{:});
    end
    function r = image (obj, varargin)
      % image  Draws a grid of coloured rectangles, one per value of `z`, like R's `image(x, y, z)`.
      %
      %   r = obj.image (x, y, z)
      %   r = obj.image (x, y, z, palette)
      %   r = obj.image (x, y, z, palette, z_min)
      %   r = obj.image (x, y, z, palette, z_min, z_max)
      %
      %   Cell (i, j) is centred on (x[i], y[j]) and reaches halfway to its neighbours (a half step
      %   beyond the outermost centres). Its colour is the palette entry for z[i][j] on a linear
      %   scale from `z_min` to `z_max`; NaN cells are left empty. Each cell is added as a filled
      %   polygon, so the grid should have at most tens of thousands of cells.
      %
      %   Example:
      %     auto plt = dm::graphics::RPlot::create().image({0.0, 1.0}, {0.0, 1.0, 2.0}, {{1, 2, 3}, {4, 5, 6}});
      %     std::print("{}\n", plt.series().size());  // 6
      %
      %   References:
      %   - R `image()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/image.html
      %
      %   Inputs:
      %     x  double vector: Cell centres along x, increasing (at least 2).
      %     y  double vector: Cell centres along y, increasing (at least 2).
      %     z  double matrix (rows): Values, z[i][j] at (x[i], y[j]); z.size() == x.size() and each row has y.size() entries.
      %     palette  cell array of datamunge.RGB, default the C++ default: Colours from low to high (default terrain_colors(12)).
      %     z_min  double, default the C++ default: Value mapped to the first colour; NaN (the default) uses the smallest z.
      %     z_max  double, default the C++ default: Value mapped to the last colour; NaN (the default) uses the largest z.
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If the sizes do not match, x or y is not increasing, or the palette is empty. (C++ std::invalid_argument)
      %
      %   C++: RPlot & datamunge::graphics::RPlot::image(const std::vector<double> & x, const std::vector<double> & y, const std::vector<std::vector<double>> & z, std::vector<RGBA> palette = {}, double z_min = ..., double z_max = ...)
      %
      %   See also: datamunge.terrain_colors(), datamunge.polygon()
      r = __datamunge__(1068, obj, varargin{:});
    end
    function r = segments (obj, varargin)
      % segments  Draws one straight segment from (x0[i], y0[i]) to (x1[i], y1[i]) for each i, like R's `segments(x0, y0, x1, y1)`.
      %
      %   r = obj.segments (x0, y0, x1, y1)
      %   r = obj.segments (x0, y0, x1, y1, color)
      %   r = obj.segments (x0, y0, x1, y1, color, stroke_width)
      %
      %   All segments share `color` and `stroke_width` and form one Segment layer; their end
      %   points count toward the axis ranges. Useful for error bars, residual sticks, and
      %   dendrograms.
      %
      %   Example:
      %     const auto x = dm::c(1.0, 2.0, 3.0);
      %     const auto y = dm::c(2.0, 3.5, 3.0);
      %     auto plt = dm::Plot(x, y).segments(x, y - 0.5, x, y + 0.5);  // +/- 0.5 error bars
      %     std::print("{}\n", plt.series()[1].x.size());  // 6 (start, end, start, end, ...)
      %
      %   References:
      %   - R `segments()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/segments.html
      %
      %   Inputs:
      %     x0  double vector: Start x coordinates.
      %     y0  double vector: Start y coordinates.
      %     x1  double vector: End x coordinates.
      %     y1  double vector: End y coordinates.
      %     color  datamunge.RGB, default the C++ default: Segment color (default opaque black).
      %     stroke_width  double, default 1.0: Segment width in pixels (default 1).
      %
      %   Output:
      %     r  datamunge.RPlot: This plot, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If the four vectors differ in length or are empty. (C++ std::invalid_argument)
      %
      %   C++: RPlot & datamunge::graphics::RPlot::segments(std::vector<double> x0, std::vector<double> y0, std::vector<double> x1, std::vector<double> y1, RGBA color = ..., double stroke_width = 1.0)
      %
      %   See also: datamunge.abline(), datamunge.polygon()
      r = __datamunge__(1069, obj, varargin{:});
    end
  end
  methods (Static)
    function r = create ()
      % create  Returns a fresh empty plot; same as the default constructor `dm::Plot()`.
      %
      %   r = datamunge.RPlot.create ()
      %
      %   Kept for language bindings, where a named factory reads better than a constructor.
      %
      %   Example:
      %     auto plt = dm::Plot::create();
      %     std::print("{}\n", plt.series().size());  // 0
      %
      %   Output:
      %     r  datamunge.RPlot: An empty plot with default labels and styling.
      %
      %   C++: RPlot datamunge::graphics::RPlot::create()
      %
      %   See also: datamunge.RPlot()
      r = __datamunge__(1019);
    end
    function r = plot (varargin)
      % plot  Builds a scatter, line, or line-plus-points plot of `y` against `x`, like R's `plot(x, y, type = "p" | "l" | "b")`.
      %
      %   r = datamunge.RPlot.plot (x, y)
      %   r = datamunge.RPlot.plot (x, y, type)
      %   r = datamunge.RPlot.plot (x, y, type, label)
      %   r = datamunge.RPlot.plot (x, y, type, label, color)
      %
      %   `"p"` (the default) adds one scatter series with 4-pixel markers; `"l"` adds one 2-pixel
      %   line joining the points in input order; `"b"` adds the line and then unlabeled points on
      %   top (R's `"b"` leaves gaps in the line around each point; this draws it through them). The
      %   title and axis labels keep their defaults. `RPlot(x, y)` is `plot(x, y)`.
      %
      %   Example:
      %     const auto x = dm::seql(0.0, 4.0, 5);
      %     auto plt = dm::Plot::plot(x, x * x, "b", "x squared");
      %     std::print("{} series\n", plt.series().size());  // 2 series
      %
      %   References:
      %   - R `plot.default()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/plot.default.html
      %
      %   Inputs:
      %     x  double vector: Horizontal coordinates, one per point.
      %     y  double vector: Vertical coordinates, paired with `x` by position.
      %     type  char row, default 'p': `"p"` (default), `"l"`, or `"b"`; case-sensitive.
      %     label  char row, default '': Legend text; empty (the default) keeps the plot out of the automatic legend. With `"b"` only the line carries it.
      %     color  datamunge.RGB, default the C++ default: Color of the points and line (default blue `{37, 99, 235}`).
      %
      %   Output:
      %     r  datamunge.RPlot: A new plot holding the series.
      %
      %   Errors:
      %     datamunge:invalid_argument  If the coordinates are empty or differ in length, or `type` is not one of the three codes. (C++ std::invalid_argument)
      %
      %   C++: RPlot datamunge::graphics::RPlot::plot(std::vector<double> x, std::vector<double> y, std::string type = "p", std::string label = "", RGBA color = ...)
      %
      %   See also: datamunge.vector<double>, datamunge.vector<double>), datamunge.points(), datamunge.lines()
      r = __datamunge__(1048, varargin{:});
    end
    function r = hist (varargin)
      % hist  Builds a frequency histogram of `data` with `bins` equal-width bins, like R's `hist(x, breaks = bins)`.
      %
      %   r = datamunge.RPlot.hist (data)
      %   r = datamunge.RPlot.hist (data, bins)
      %   r = datamunge.RPlot.hist (data, bins, label)
      %   r = datamunge.RPlot.hist (data, bins, label, color)
      %
      %   The bins split [min, max] of the data into equal widths (R instead treats `breaks` as a
      %   suggestion and uses `pretty()` boundaries). Bins include their left edge and exclude their
      %   right edge, except that the maximum goes in the final bin; R's default (`right = TRUE`) is
      %   the opposite, so a value on an inner boundary is counted one bin higher here. Constant
      %   data uses the range [value, value + 1]. Each bin is one bar of height equal to its count,
      %   centered on the bin with width equal to the bin width. The title is "Histogram", the x
      %   label "x", and the y label "Frequency". The data is consumed; the plot keeps only counts.
      %
      %   Example:
      %     auto plt = dm::Plot::hist({1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0}, 3);
      %     const auto& bars = plt.series()[0];  // bins [1,2) [2,3) [3,4]
      %     std::print("{} {} {}\n", bars.y[0], bars.y[1], bars.y[2]);  // 1 2 4
      %
      %   References:
      %   - R `hist()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/hist.html
      %
      %   Inputs:
      %     data  double vector: Observations; all must be finite.
      %     bins  integer, default 10: Number of bins (default 10).
      %     label  char row, default '': Legend text; empty (the default) shows no legend.
      %     color  datamunge.RGB, default the C++ default: Bar color (default light blue `{96, 165, 250}`).
      %
      %   Output:
      %     r  datamunge.RPlot: A new plot holding one Bar series.
      %
      %   Errors:
      %     datamunge:invalid_argument  For empty/non-finite data, zero bins, or a range/bin width that cannot be represented as a finite positive double. (C++ std::invalid_argument)
      %
      %   C++: RPlot datamunge::graphics::RPlot::hist(std::vector<double> data, std::size_t bins = 10, std::string label = "", RGBA color = ...)
      %
      %   See also: datamunge.Histogram, datamunge.bars()
      r = __datamunge__(1049, varargin{:});
    end
    function r = barplot (varargin)
      % barplot  Builds a bar chart with one bar per height, like R's `barplot(heights, names.arg = names)`.
      %
      %   r = datamunge.RPlot.barplot (heights)
      %   r = datamunge.RPlot.barplot (heights, names)
      %   r = datamunge.RPlot.barplot (heights, names, label)
      %   r = datamunge.RPlot.barplot (heights, names, label, color)
      %
      %   Bar i is centered at x = i (0, 1, 2, ...) and is 0.8 units wide (R puts default bars at
      %   0.7, 1.9, 3.1, ...). Bars rise from 0, so negative heights point down, and the y range
      %   always includes 0. Nonempty `names` become x tick labels under the bars. The title and
      %   axis labels keep their defaults.
      %
      %   Example:
      %     auto plt = dm::Plot::barplot({3.0, 5.0, 2.0}, {"a", "b", "c"}).set_title("Counts");
      %     std::print("{} {}\n", plt.series()[0].x[2], plt.x_tick_label_list()[2]);  // 2 c
      %
      %   References:
      %   - R `barplot()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/barplot.html
      %
      %   Inputs:
      %     heights  double vector: Bar heights, in display order.
      %     names  cell array of char, default the C++ default: Tick labels, one per bar; empty (the default) keeps numeric ticks.
      %     label  char row, default '': Legend text; empty (the default) shows no legend.
      %     color  datamunge.RGB, default the C++ default: Bar color (default green `{22, 163, 74}`).
      %
      %   Output:
      %     r  datamunge.RPlot: A new plot holding one Bar series.
      %
      %   Errors:
      %     datamunge:invalid_argument  If `heights` is empty or `names` is nonempty with a different length. (C++ std::invalid_argument)
      %
      %   C++: RPlot datamunge::graphics::RPlot::barplot(std::vector<double> heights, std::vector<std::string> names = {}, std::string label = "", RGBA color = ...)
      %
      %   See also: datamunge.bars(), datamunge.x_tick_labels()
      r = __datamunge__(1050, varargin{:});
    end
    function r = boxplot (varargin)
      % boxplot  Builds side-by-side box-and-whisker plots, one per group, like R's `boxplot(list(...), names = names)`.
      %
      %   r = datamunge.RPlot.boxplot (groups)
      %   r = datamunge.RPlot.boxplot (groups, names)
      %   r = datamunge.RPlot.boxplot (groups, names, color)
      %   r = datamunge.RPlot.boxplot (groups, names, colors)
      %
      %   Group g is drawn at x = g (0, 1, 2, ...) as a 0.6-unit-wide box from the first to the
      %   third quartile with a line at the median. Quartiles are R's default "type 7" sample
      %   quantiles. Whiskers reach the most extreme values within 1.5 IQR of the box (Tukey's
      %   rule), and values beyond are drawn as outlier points. R's `boxplot()` instead uses
      %   Tukey's hinges (`fivenum()`), so for some sample sizes the box edges and the fences differ
      %   slightly from R: for 1, ..., 9, 40 this gives quartiles 3.25 and 7.75 where R draws
      %   hinges at 3 and 8. All boxes share `color`; the title and axis labels keep their defaults.
      %
      %   Example:
      %     auto plt = dm::Plot::boxplot({{1, 2, 3, 4, 5, 6, 7, 8, 9, 40}, {2, 4, 6}}, {"a", "b"});
      %     const auto& y = plt.series()[0].y;  // whisker_lo, q1, median, q3, whisker_hi, outliers...
      %     std::print("{} {} {} {} {} | {}\n", y[0], y[1], y[2], y[3], y[4], y[5]);
      %     // 1 3.25 5.5 7.75 9 | 40
      %     plt.save("boxes.svg");
      %
      %   References:
      %   - J. W. Tukey (1977). Exploratory Data Analysis. Addison-Wesley. (Box plot and the 1.5 IQR fences.)
      %   - R. McGill, J. W. Tukey, W. A. Larsen (1978). "Variations of Box Plots". The American
      %     Statistician 32(1), 12-16. doi:10.1080/00031305.1978.10479236
      %   - R. J. Hyndman, Y. Fan (1996). "Sample Quantiles in Statistical Packages". The American
      %     Statistician 50(4), 361-365. doi:10.1080/00031305.1996.10473566 (quantile type 7.)
      %   - R `boxplot()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/boxplot.html
      %   - R `boxplot.stats()`: https://stat.ethz.ch/R-manual/R-devel/library/grDevices/html/boxplot.stats.html
      %
      %   Errors:
      %     datamunge:invalid_argument  If `groups` or any group is empty, or `names` is nonempty with a different length. (C++ std::invalid_argument)
      %
      %   Overloads are chosen at run time from the argument types (C++):
      %   C++: RPlot datamunge::graphics::RPlot::boxplot(std::vector<std::vector<double>> groups, std::vector<std::string> names = {}, RGBA color = ...)
      %   C++: RPlot datamunge::graphics::RPlot::boxplot(std::vector<std::vector<double>> groups, std::vector<std::string> names, std::vector<RGBA> colors)
      %
      %   See also: datamunge.box(), datamunge.x_tick_labels()
      r = __datamunge__(1051, varargin{:});
    end
    function r = pie (varargin)
      % pie  Builds a pie chart with one wedge per value, like R's `pie(x, labels = names, col = colors, clockwise = TRUE)`.
      %
      %   r = datamunge.RPlot.pie (values)
      %   r = datamunge.RPlot.pie (values, names)
      %   r = datamunge.RPlot.pie (values, names, colors)
      %
      %   Wedge i spans the fraction values[i] / sum(values) of the circle. Wedges start at 12
      %   o'clock and run clockwise (R's default starts at 3 o'clock and runs counter-clockwise).
      %   Each wedge is a filled unit-radius polygon with 24 arc segments. Names, when given, are
      %   drawn at radius 1.15 beside the middle of each wedge's arc and also label the wedges, so the
      %   automatic legend lists them. Colors are recycled; empty uses an 8-color built-in palette
      %   starting blue, red, green. The axes and grid are hidden and both ranges are fixed at
      %   [-1.4, 1.4]. The pie is drawn in data units, so it is round only when the plotting area is
      %   square, e.g. `.size(598, 640)` with the default margins; the default 960x640 canvas draws an
      %   ellipse.
      %
      %   Example:
      %     auto plt = dm::Plot::pie({3.0, 2.0, 1.0}, {"rent", "food", "fun"}).size(598, 640);
      %     std::print("{}\n", plt.series().size());  // 6
      %     plt.save("budget.svg");
      %
      %   References:
      %   - R `pie()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/pie.html
      %
      %   Inputs:
      %     values  double vector: Wedge sizes; nonnegative, with a positive sum.
      %     names  cell array of char, default the C++ default: Wedge labels; empty (the default) draws none.
      %     colors  cell array of datamunge.RGB, default the C++ default: Wedge colors, recycled; empty (the default) uses the built-in palette.
      %
      %   Output:
      %     r  datamunge.RPlot: A new plot holding one Polygon series per wedge plus one Text series per name.
      %
      %   Errors:
      %     datamunge:invalid_argument  If `values` is empty, has a negative entry, or sums to 0, or `names` is nonempty with a different length. (C++ std::invalid_argument)
      %
      %   C++: RPlot datamunge::graphics::RPlot::pie(std::vector<double> values, std::vector<std::string> names = {}, std::vector<RGBA> colors = {})
      %
      %   See also: datamunge.polygon(), datamunge.text()
      r = __datamunge__(1052, varargin{:});
    end
    function r = curve (varargin)
      % curve  Plots a function over [`from`, `to`] as a line, like R's `curve(expr, from, to, n)`.
      %
      %   r = datamunge.RPlot.curve (f, from, to)
      %   r = datamunge.RPlot.curve (f, from, to, n)
      %   r = datamunge.RPlot.curve (f, from, to, n, label)
      %   r = datamunge.RPlot.curve (f, from, to, n, label, color)
      %
      %   Evaluates `f.call(x)` at `n` evenly spaced points including both ends, in increasing
      %   order, and joins them with a 2-pixel line. `f` is used only during this call and is not
      %   retained. Non-finite results are stored as-is. Exceptions thrown by `f.call()` propagate.
      %
      %   Example:
      %     struct Square : dm::Callback {
      %       double call(double x) override { return x * x; }
      %     };
      %     Square square;
      %     auto plt = dm::Plot::curve(square, -1.0, 1.0, 5);
      %     std::print("{}\n", plt.series()[0].y);  // [1, 0.25, 0, 0.25, 1]
      %
      %   References:
      %   - R `curve()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/curve.html
      %
      %   Inputs:
      %     f  datamunge.Callback: The function to draw: a dm::Callback subclass overriding `call(double)`.
      %     from  double: Left end of the range.
      %     to  double: Right end; must be greater than `from`.
      %     n  integer, default 101: Number of sample points, at least 2 (default 101, as in R).
      %     label  char row, default '': Legend text; empty (the default) shows no legend.
      %     color  datamunge.RGB, default the C++ default: Line color (default red `{220, 38, 38}`).
      %
      %   Output:
      %     r  datamunge.RPlot: A new plot holding one Line series.
      %
      %   Errors:
      %     datamunge:invalid_argument  If `n` is below 2 or `to` is not greater than `from`. (C++ std::invalid_argument)
      %
      %   C++: RPlot datamunge::graphics::RPlot::curve(datamunge::Callback & f, double from, double to, std::size_t n = 101, std::string label = "", RGBA color = ...)
      %
      %   See also: datamunge.lines()
      r = __datamunge__(1053, varargin{:});
    end
    function r = qqnorm (varargin)
      % qqnorm  Builds a normal Q-Q plot: the sorted `data` against standard-normal quantiles, like R's `qqnorm(y)`.
      %
      %   r = datamunge.RPlot.qqnorm (data)
      %   r = datamunge.RPlot.qqnorm (data, label)
      %   r = datamunge.RPlot.qqnorm (data, label, color)
      %
      %   The i-th smallest value (i = 1..n) is plotted against the normal quantile of the plotting
      %   position (i - 0.5) / n. R's `ppoints()` uses that rule only for n > 10 and
      %   (i - 3/8) / (n + 1/4) for n <= 10, so the x coordinates differ from R for small samples.
      %   Points that fall on a straight line indicate normally distributed data; add the reference
      %   line with qqline(). The title is "Normal Q-Q", the axis labels "Theoretical Quantiles"
      %   and "Sample Quantiles".
      %
      %   Example:
      %     const std::vector<double> sample{3.0, 1.0, 2.0, 5.0, 4.0};
      %     auto plt = dm::Plot::qqnorm(sample).qqline(sample);
      %     std::print("{:.4f}\n", plt.series()[0].x[0]);  // -1.2816
      %
      %   References:
      %   - M. B. Wilk, R. Gnanadesikan (1968). "Probability plotting methods for the analysis of
      %     data". Biometrika 55(1), 1-17. doi:10.1093/biomet/55.1.1
      %   - R `qqnorm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/qqnorm.html
      %   - R `ppoints()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/ppoints.html
      %
      %   Inputs:
      %     data  double vector: The sample; at least two values. Consumed (sorted in place).
      %     label  char row, default 'sample': Legend text (default "sample", so the automatic legend shows one row; pass "" to hide it).
      %     color  datamunge.RGB, default the C++ default: Point color (default blue `{37, 99, 235}`).
      %
      %   Output:
      %     r  datamunge.RPlot: A new plot holding one Scatter series.
      %
      %   Errors:
      %     datamunge:invalid_argument  If `data` has fewer than two values. (C++ std::invalid_argument)
      %
      %   C++: RPlot datamunge::graphics::RPlot::qqnorm(std::vector<double> data, std::string label = "sample", RGBA color = ...)
      %
      %   See also: datamunge.qqline()
      r = __datamunge__(1054, varargin{:});
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::graphics::RPlot. Do not edit.
