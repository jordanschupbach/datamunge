classdef RLayout < datamunge.NativeObject
  % RLayout  Tiles independently built plots into one multi-panel figure, like R's `par(mfrow = c(rows, cols))` followed by one plot per panel.
  %
  %   obj = datamunge.RLayout ()
  %
  %   Create it with create(rows, cols), then add() up to rows * cols panels; they fill the grid
  %   row by row. Each panel is copied (and its SVG rendered) at add() time, so later changes to
  %   the original plot do not show and panels need not outlive the layout. Each cell is
  %   `width / cols` by `(height - band) / rows` pixels, where band is title_band_px when a
  %   figure title is set and 0 otherwise; every panel is scaled from its own canvas size into its
  %   cell. The figure background is white. Unused cells stay empty.
  %
  %   Example:
  %     const auto x = dm::seql(0.0, 6.0, 30);
  %     auto fig = dm::graphics::RLayout::create(1, 2)
  %                    .size(1200, 500)
  %                    .title("Two views")
  %                    .add(dm::Plot(x, dm::sin(x)).set_title("sin"))
  %                    .add(dm::Plot(x, dm::cos(x)).set_title("cos"));
  %     std::print("{} panels in {}x{}\n", fig.panel_count(), fig.rows(), fig.cols());
  %     // 2 panels in 1x2
  %     fig.save("two_views.svg");
  %
  %   References:
  %   - R `par()` (`mfrow`): https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/par.html
  %
  %   Methods:
  %     add  Copies `panel` into the next free cell, filling rows left to right, top to bottom.
  %     size_  Sets the size of the whole figure in pixels (default 960 by 640).
  %     title  Sets an optional figure-level heading drawn in a band above the panels, like R's `mtext(main, outer = TRUE)`; an empty string (the default) reserves no band.
  %     rows  Returns the number of panel rows given to create().
  %     cols  Returns the number of panel columns given to create().
  %     width  Returns the figure width in pixels.
  %     height  Returns the figure height in pixels.
  %     panel_count  Returns how many panels have been added.
  %     title_text  Borrows the figure-level heading for the lifetime of this layout.
  %     panel  Borrows the copy of the `index-th` added panel.
  %     to_svg  Renders the multi-panel figure as a standalone SVG document and returns it as a string.
  %     save  Saves the figure, choosing the format from the extension: `.svg` (or none) writes SVG, `.png` / `.pdf` write the same tiled raster the X11 viewer shows.
  %     save_svg  Writes to_svg() to `path` regardless of its extension.
  %     to_png  Renders the tiled figure as PNG image bytes (the raster save() writes for `.png`).
  %     to_pdf  Renders the tiled figure as a one-page PDF holding the raster image.
  %     save_png  Writes to_png() to `path` regardless of its extension.
  %     save_pdf  Writes to_pdf() to `path` regardless of its extension.
  %     fit_panels  Resizes every stored panel's canvas to its cell, so text keeps its pixel size instead of being scaled with the panel.
  %     show  Opens the tiled figure in the interactive X11 viewer; the toolbar zoom/pan buttons act on every panel's data bounds at once, keyboard zoom/pan scales the whole image.
  %     view  Same as show(); kept as an alias for older code.
  %
  %   Static methods:
  %     create  Creates an empty layout with `rows` by `cols` cells, like R's `par(mfrow = c(rows, cols))`.
  %
  %   Constants:
  %     title_band_px  Height in pixels of the heading band reserved above the panels when a title is set.
  %     min_panel_px  Smallest canvas side fit_panels() gives a panel; smaller cells get a larger canvas with the cell's aspect ratio, scaled down when drawn, so the margins still fit.
  %
  %   C++: class datamunge::graphics::RLayout
  %
  %   See also: datamunge.RPlot, datamunge.margins(), datamunge.Splom
  properties (Constant)
    title_band_px = __datamunge__(1092);  % Height in pixels of the heading band reserved above the panels when a title is set.
    min_panel_px = __datamunge__(1093);  % Smallest canvas side fit_panels() gives a panel; smaller cells get a larger canvas with the cell's aspect ratio, scaled down when drawn, so the margins still fit.
  end
  methods
    function obj = RLayout (varargin)
      % RLayout  Construct a C++ datamunge::graphics::RLayout.
      %
      %   obj = datamunge.RLayout ()
      %
      %   C++: datamunge::graphics::RLayout::RLayout()
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1094, false, varargin{:});
    end
    function r = add (obj, panel)
      % add  Copies `panel` into the next free cell, filling rows left to right, top to bottom.
      %
      %   r = obj.add (panel)
      %
      %   The panel's SVG is rendered immediately and stored with the copy, so rendering errors
      %   surface here and later edits to the original plot have no effect on the layout.
      %
      %   Example:
      %     auto fig = dm::graphics::RLayout::create(1, 1).add(dm::Plot(dm::c(1.0, 2.0), dm::c(2.0, 1.0)));
      %     try {
      %       fig.add(dm::Plot(dm::c(1.0), dm::c(1.0)));
      %     } catch (const std::invalid_argument& e) {
      %       std::print("{}\n", e.what());  // RLayout::add: layout is already full
      %     }
      %
      %   Inputs:
      %     panel  datamunge.Plot: The plot to add; any RPlot (or other Plot) converts.
      %
      %   Output:
      %     r  datamunge.RLayout: This layout, for chaining.
      %
      %   Errors:
      %     datamunge:invalid_argument  If all rows * cols cells are already filled. (C++ std::invalid_argument)
      %     datamunge:native  If `panel` has no data series or its canvas is smaller than its margins. (C++ std::runtime_error)
      %
      %   C++: RLayout & datamunge::graphics::RLayout::add(const Plot & panel)
      %
      %   See also: datamunge.panel(), datamunge.panel_count()
      r = __datamunge__(1072, obj, panel);
    end
    function r = size_ (obj, width, height)
      % size_  Sets the size of the whole figure in pixels (default 960 by 640).
      %
      %   r = obj.size_ (width, height)
      %
      %   Panels keep their own canvas sizes and are scaled into their cells, so text in a panel
      %   shrinks or grows with it; give panels a size close to their cell for readable labels.
      %
      %   Example:
      %   See RLayout.
      %
      %   Inputs:
      %     width  integer: Figure width in pixels.
      %     height  integer: Figure height in pixels, including the title band.
      %
      %   Output:
      %     r  datamunge.RLayout: This layout, for chaining.
      %
      %   C++: RLayout & datamunge::graphics::RLayout::size(std::size_t width, std::size_t height)
      %
      %   See also: datamunge.title()
      r = __datamunge__(1073, obj, width, height);
    end
    function r = title (obj, value_)
      % title  Sets an optional figure-level heading drawn in a band above the panels, like R's `mtext(main, outer = TRUE)`; an empty string (the default) reserves no band.
      %
      %   r = obj.title (value_)
      %
      %   Example:
      %   See RLayout.
      %
      %   References:
      %   - R `mtext()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/mtext.html
      %
      %   Inputs:
      %     value_  char row: Heading text; the band is title_band_px pixels tall.
      %
      %   Output:
      %     r  datamunge.RLayout: This layout, for chaining.
      %
      %   C++: RLayout & datamunge::graphics::RLayout::title(std::string value)
      %
      %   See also: datamunge.title_text()
      r = __datamunge__(1074, obj, value_);
    end
    function r = rows (obj)
      % rows  Returns the number of panel rows given to create().
      %
      %   r = obj.rows ()
      %
      %   Output:
      %     r  integer: Row count (at least 1).
      %
      %   C++: std::size_t datamunge::graphics::RLayout::rows() const
      r = __datamunge__(1075, obj);
    end
    function r = cols (obj)
      % cols  Returns the number of panel columns given to create().
      %
      %   r = obj.cols ()
      %
      %   Output:
      %     r  integer: Column count (at least 1).
      %
      %   C++: std::size_t datamunge::graphics::RLayout::cols() const
      r = __datamunge__(1076, obj);
    end
    function r = width (obj)
      % width  Returns the figure width in pixels.
      %
      %   r = obj.width ()
      %
      %   Output:
      %     r  integer: Width set by size() (default 960).
      %
      %   C++: std::size_t datamunge::graphics::RLayout::width() const
      r = __datamunge__(1077, obj);
    end
    function r = height (obj)
      % height  Returns the figure height in pixels.
      %
      %   r = obj.height ()
      %
      %   Output:
      %     r  integer: Height set by size() (default 640).
      %
      %   C++: std::size_t datamunge::graphics::RLayout::height() const
      r = __datamunge__(1078, obj);
    end
    function r = panel_count (obj)
      % panel_count  Returns how many panels have been added.
      %
      %   r = obj.panel_count ()
      %
      %   Output:
      %     r  integer: Panel count, at most rows() * cols().
      %
      %   C++: std::size_t datamunge::graphics::RLayout::panel_count() const
      r = __datamunge__(1079, obj);
    end
    function r = title_text (obj)
      % title_text  Borrows the figure-level heading for the lifetime of this layout.
      %
      %   r = obj.title_text ()
      %
      %   Output:
      %     r  char row: The title; empty (the default) when none is drawn.
      %
      %   C++: const std::string & datamunge::graphics::RLayout::title_text() const
      r = __datamunge__(1080, obj);
    end
    function r = panel (obj, index)
      % panel  Borrows the copy of the `index-th` added panel.
      %
      %   r = obj.panel (index)
      %
      %   Example:
      %     auto fig = dm::graphics::RLayout::create(1, 2).add(dm::Plot(dm::c(1.0), dm::c(1.0)).set_title("A"));
      %     std::print("{}\n", fig.panel(0).title_text());  // A
      %
      %   Inputs:
      %     index  integer: Zero-based panel index, in the order panels were added.
      %
      %   Output:
      %     r  datamunge.Plot: The stored copy, valid until the next add() or the layout's destruction.
      %
      %   Errors:
      %     datamunge:out_of_range  If `index` is not below panel_count(). (C++ std::out_of_range)
      %
      %   C++: const Plot & datamunge::graphics::RLayout::panel(std::size_t index) const
      r = __datamunge__(1081, obj, index);
    end
    function r = to_svg (obj)
      % to_svg  Renders the multi-panel figure as a standalone SVG document and returns it as a string.
      %
      %   r = obj.to_svg ()
      %
      %   Each panel becomes a nested `<svg>` scaled into its cell. A layout with no panels renders
      %   as an empty white figure. The layout is not modified.
      %
      %   Example:
      %     auto fig = dm::graphics::RLayout::create(2, 1).add(dm::Plot(dm::c(1.0, 2.0), dm::c(1.0, 2.0)));
      %     std::print("{}\n", fig.to_svg().starts_with("<?xml"));  // true
      %
      %   Output:
      %     r  char row: The complete SVG text.
      %
      %   C++: std::string datamunge::graphics::RLayout::to_svg() const
      %
      %   See also: datamunge.save()
      r = __datamunge__(1082, obj);
    end
    function save (obj, path)
      % save  Saves the figure, choosing the format from the extension: `.svg` (or none) writes SVG, `.png` / `.pdf` write the same tiled raster the X11 viewer shows.
      %
      %   obj.save (path)
      %
      %   The extension is case-sensitive and an existing file is overwritten. Raster panels are
      %   re-rendered from the stored copies and scaled into their cells.
      %
      %   Example:
      %   See RLayout.
      %
      %   Inputs:
      %     path  char row: Destination file.
      %
      %   Errors:
      %     datamunge:invalid_argument  For any other extension. (C++ std::invalid_argument)
      %     datamunge:native  If the file cannot be written. (C++ std::runtime_error)
      %
      %   C++: void datamunge::graphics::RLayout::save(const std::string & path) const
      %
      %   See also: datamunge.save_svg(), datamunge.to_svg(), datamunge.show()
      __datamunge__(1083, obj, path);
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
      %   C++: void datamunge::graphics::RLayout::save_svg(const std::string & path) const
      %
      %   See also: datamunge.save()
      __datamunge__(1084, obj, path);
    end
    function r = to_png (obj)
      % to_png  Renders the tiled figure as PNG image bytes (the raster save() writes for `.png`).
      %
      %   r = obj.to_png ()
      %
      %   Output:
      %     r  integer vector: The complete PNG file contents.
      %
      %   Errors:
      %     datamunge:native  If a panel cannot be rendered. (C++ std::runtime_error)
      %
      %   C++: std::vector<std::uint8_t> datamunge::graphics::RLayout::to_png() const
      %
      %   See also: datamunge.save_png(), datamunge.to_svg()
      r = __datamunge__(1085, obj);
    end
    function r = to_pdf (obj)
      % to_pdf  Renders the tiled figure as a one-page PDF holding the raster image.
      %
      %   r = obj.to_pdf ()
      %
      %   Output:
      %     r  integer vector: The complete PDF file contents.
      %
      %   Errors:
      %     datamunge:native  If a panel cannot be rendered. (C++ std::runtime_error)
      %
      %   C++: std::vector<std::uint8_t> datamunge::graphics::RLayout::to_pdf() const
      %
      %   See also: datamunge.save_pdf(), datamunge.to_png()
      r = __datamunge__(1086, obj);
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
      %     datamunge:native  If the file cannot be written or a panel cannot be rendered. (C++ std::runtime_error)
      %
      %   C++: void datamunge::graphics::RLayout::save_png(const std::string & path) const
      __datamunge__(1087, obj, path);
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
      %     datamunge:native  If the file cannot be written or a panel cannot be rendered. (C++ std::runtime_error)
      %
      %   C++: void datamunge::graphics::RLayout::save_pdf(const std::string & path) const
      __datamunge__(1088, obj, path);
    end
    function r = fit_panels (obj)
      % fit_panels  Resizes every stored panel's canvas to its cell, so text keeps its pixel size instead of being scaled with the panel.
      %
      %   r = obj.fit_panels ()
      %
      %   Call it after size() and title() (the title band changes the cell height). Panels added
      %   later keep their own size until fit_panels() is called again. A cell whose shorter side is
      %   under min_panel_px gets a canvas scaled up to that side (same aspect ratio), so its text is
      %   drawn proportionally smaller instead of the margins overflowing the panel.
      %
      %   Example:
      %     auto fig = dm::graphics::RLayout::create(1, 2)
      %                    .add(dm::Plot(dm::c(1.0, 2.0), dm::c(2.0, 1.0)))
      %                    .add(dm::Plot(dm::c(1.0, 2.0), dm::c(1.0, 2.0)))
      %                    .size(1000, 400)
      %                    .fit_panels();
      %     std::print("{}x{}\n", fig.panel(0).width(), fig.panel(0).height());  // 500x400
      %
      %   Output:
      %     r  datamunge.RLayout: This layout, for chaining.
      %
      %   C++: RLayout & datamunge::graphics::RLayout::fit_panels()
      %
      %   See also: datamunge.size()
      r = __datamunge__(1089, obj);
    end
    function show (obj, varargin)
      % show  Opens the tiled figure in the interactive X11 viewer; the toolbar zoom/pan buttons act on every panel's data bounds at once, keyboard zoom/pan scales the whole image.
      %
      %   obj.show ()
      %   obj.show (title_hint)
      %
      %   Blocks until the window is closed (q or Escape). Use save() in headless code.
      %
      %   Example:
      %     const auto x = dm::seql(0.0, 6.0, 30);
      %     dm::graphics::RLayout::create(1, 2)
      %         .add(dm::Plot(x, dm::sin(x)))
      %         .add(dm::Plot(x, dm::cos(x)))
      %         .show("Trig");
      %
      %   Inputs:
      %     title_hint  char row, default '': Window title; empty (the default) uses title_text().
      %
      %   Errors:
      %     datamunge:native  If X11 support was not enabled at build time or no display is available. (C++ std::runtime_error)
      %     datamunge:invalid_argument  If no panel was added. (C++ std::invalid_argument)
      %
      %   C++: void datamunge::graphics::RLayout::show(const std::string & title_hint = "") const
      %
      %   See also: datamunge.save(), datamunge.show()
      __datamunge__(1090, obj, varargin{:});
    end
    function view (obj, varargin)
      % view  Same as show(); kept as an alias for older code.
      %
      %   obj.view ()
      %   obj.view (title_hint)
      %
      %   Inputs:
      %     title_hint  char row, default '': Window title; empty (the default) uses title_text().
      %
      %   Errors:
      %     datamunge:native  As for show(). (C++ std::runtime_error)
      %     datamunge:invalid_argument  If no panel was added. (C++ std::invalid_argument)
      %
      %   C++: void datamunge::graphics::RLayout::view(const std::string & title_hint = "") const
      %
      %   See also: datamunge.show()
      __datamunge__(1091, obj, varargin{:});
    end
  end
  methods (Static)
    function r = create (rows, cols)
      % create  Creates an empty layout with `rows` by `cols` cells, like R's `par(mfrow = c(rows, cols))`.
      %
      %   r = datamunge.RLayout.create (rows, cols)
      %
      %   The figure starts at 960 by 640 pixels with no title.
      %
      %   Example:
      %   See RLayout.
      %
      %   References:
      %   - R `par()` (`mfrow`): https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/par.html
      %
      %   Inputs:
      %     rows  integer: Number of panel rows; must be positive.
      %     cols  integer: Number of panel columns; must be positive.
      %
      %   Output:
      %     r  datamunge.RLayout: The empty layout.
      %
      %   Errors:
      %     datamunge:invalid_argument  If `rows` or `cols` is 0. (C++ std::invalid_argument)
      %
      %   C++: RLayout datamunge::graphics::RLayout::create(std::size_t rows, std::size_t cols)
      %
      %   See also: datamunge.add()
      r = __datamunge__(1071, rows, cols);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::graphics::RLayout. Do not edit.
