1;

% Demonstrates GGPlot -- the ggplot2-style grammar-of-graphics library.
datamunge;

function c = rgb(r, g, b)
  datamunge;
  c = RGB();
  RGB_r_set(c, r);
  RGB_g_set(c, g);
  RGB_b_set(c, b);
endfunction

iris = DataFrame_iris();

% ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
scatter = GGPlot(iris, "Sepal.Length", "Sepal.Width", "Species");
GGPlot_geom_point(scatter);
GGPlot_labs(scatter, "Iris Sepal Dimensions", "Sepal Length", "Sepal Width");
GGPlot_theme_minimal(scatter);
GGPlot_save_svg(scatter, "octave_ggplot_point.svg");

% + geom_smooth(): an lm() fit line, reusing stats::LM internally.
smooth = GGPlot(iris, "Sepal.Length", "Petal.Length");
GGPlot_geom_point(smooth, rgb(156, 163, 175), 2.5);
GGPlot_geom_smooth(smooth);
GGPlot_labs(smooth, "Petal Length vs Sepal Length With a Linear Fit", "Sepal Length", "Petal Length");
GGPlot_save_svg(smooth, "octave_ggplot_smooth.svg");

% geom_bar(): counts a discrete column (stat = "count").
bar = GGPlot(iris, "Species");
GGPlot_geom_bar(bar);
GGPlot_labs(bar, "Observations per Species", "Species", "Count");
GGPlot_theme_bw(bar);
GGPlot_save_svg(bar, "octave_ggplot_bar.svg");

% geom_boxplot(): grouped by a discrete x column.
box = GGPlot(iris, "Species", "Petal.Width");
GGPlot_geom_boxplot(box);
GGPlot_labs(box, "Petal Width by Species", "Species", "Petal Width");
GGPlot_save_svg(box, "octave_ggplot_boxplot.svg");

% geom_histogram() + geom_density(): distribution of a single numeric column.
hist = GGPlot(iris, "Sepal.Length");
GGPlot_geom_histogram(hist, 20);
GGPlot_labs(hist, "Distribution of Sepal Length", "Sepal Length", "Count");
GGPlot_save_svg(hist, "octave_ggplot_histogram.svg");

density = GGPlot(iris, "Sepal.Length");
GGPlot_geom_density(density);
GGPlot_labs(density, "Density of Sepal Length", "Sepal Length", "Density");
GGPlot_save_svg(density, "octave_ggplot_density.svg");

% facet_wrap(): one panel per Species, composed via RLayout under the hood.
faceted = GGPlot(iris, "Petal.Length", "Petal.Width");
GGPlot_geom_point(faceted);
GGPlot_facet_wrap(faceted, "Species");
GGPlot_labs(faceted, "Petal Dimensions", "Petal Length", "Petal Width");
GGPlot_save_svg(faceted, "octave_ggplot_facet.svg");

% theme_classic(): another built-in theme. (scale_color_manual is omitted here -- the
% std::vector<RGB> overload it needs has no %template in the Octave binding's SWIG interface,
% so RGBVector is not exposed.)
classic = GGPlot(iris, "Sepal.Length", "Sepal.Width", "Species");
GGPlot_geom_point(classic);
GGPlot_theme_classic(classic);
GGPlot_save_svg(classic, "octave_ggplot_classic.svg");

printf("Wrote 7 SVGs to octave_ggplot_*.svg\n");
