package require Datamunge 0.0.1

# Demonstrates datamunge::GGPlot -- the ggplot2-style grammar-of-graphics library.

proc rgb {r g b} {
  set c [datamunge::new_RGB]
  datamunge::RGB_r_set $c $r
  datamunge::RGB_g_set $c $g
  datamunge::RGB_b_set $c $b
  return $c
}

set iris [datamunge::DataFrame_iris]

# ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
set scatter [datamunge::new_GGPlot $iris "Sepal.Length" "Sepal.Width" "Species"]
datamunge::GGPlot_geom_point $scatter
datamunge::GGPlot_labs $scatter "Iris Sepal Dimensions" "Sepal Length" "Sepal Width"
datamunge::GGPlot_theme_minimal $scatter
datamunge::GGPlot_save_svg $scatter "tcl_ggplot_point.svg"

# + geom_smooth(): an lm() fit line, reusing stats::LM internally.
set smooth [datamunge::new_GGPlot $iris "Sepal.Length" "Petal.Length"]
datamunge::GGPlot_geom_point $smooth [rgb 156 163 175] 2.5
datamunge::GGPlot_geom_smooth $smooth
datamunge::GGPlot_labs $smooth "Petal Length vs Sepal Length With a Linear Fit" "Sepal Length" "Petal Length"
datamunge::GGPlot_save_svg $smooth "tcl_ggplot_smooth.svg"

# geom_bar(): counts a discrete column (stat = "count").
set bar [datamunge::new_GGPlot $iris "Species"]
datamunge::GGPlot_geom_bar $bar
datamunge::GGPlot_labs $bar "Observations per Species" "Species" "Count"
datamunge::GGPlot_theme_bw $bar
datamunge::GGPlot_save_svg $bar "tcl_ggplot_bar.svg"

# geom_boxplot(): grouped by a discrete x column.
set box [datamunge::new_GGPlot $iris "Species" "Petal.Width"]
datamunge::GGPlot_geom_boxplot $box
datamunge::GGPlot_labs $box "Petal Width by Species" "Species" "Petal Width"
datamunge::GGPlot_save_svg $box "tcl_ggplot_boxplot.svg"

# geom_histogram() + geom_density(): distribution of a single numeric column.
set hist [datamunge::new_GGPlot $iris "Sepal.Length"]
datamunge::GGPlot_geom_histogram $hist 20
datamunge::GGPlot_labs $hist "Distribution of Sepal Length" "Sepal Length" "Count"
datamunge::GGPlot_save_svg $hist "tcl_ggplot_histogram.svg"

set density [datamunge::new_GGPlot $iris "Sepal.Length"]
datamunge::GGPlot_geom_density $density
datamunge::GGPlot_labs $density "Density of Sepal Length" "Sepal Length" "Density"
datamunge::GGPlot_save_svg $density "tcl_ggplot_density.svg"

# facet_wrap(): one panel per Species, composed via RLayout under the hood.
set faceted [datamunge::new_GGPlot $iris "Petal.Length" "Petal.Width"]
datamunge::GGPlot_geom_point $faceted
datamunge::GGPlot_facet_wrap $faceted "Species"
datamunge::GGPlot_labs $faceted "Petal Dimensions" "Petal Length" "Petal Width"
datamunge::GGPlot_save_svg $faceted "tcl_ggplot_facet.svg"

# theme_classic(): another built-in theme. (scale_color_manual is omitted here -- the
# std::vector<RGB> overload it needs (RGBVector) is not one of the binding's %template'd
# containers, so it is not part of the Tcl binding's exposed surface, matching the Lua template.)
set classic [datamunge::new_GGPlot $iris "Sepal.Length" "Sepal.Width" "Species"]
datamunge::GGPlot_geom_point $classic
datamunge::GGPlot_theme_classic $classic
datamunge::GGPlot_save_svg $classic "tcl_ggplot_classic.svg"

puts "Wrote 7 SVGs to tcl_ggplot_*.svg"
