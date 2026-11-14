# CMD: just TARGET=ggplot_ex run-tcl
# Demonstrates GGPlot -- the ggplot2-style grammar-of-graphics library.
package require datamunge

set iris [datamunge::Data::iris]

# Each GGPlot method returns the plot, so a ggplot2 "+ layer" chain becomes nested calls;
# chain applies a sequence of {method args...} steps to one plot.
proc chain {plot args} {
    foreach step $args {
        set plot [$plot {*}$step]
    }
    return $plot
}

# ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
chain [datamunge::GGPlot new $iris Sepal.Length Sepal.Width Species] \
    geom_point \
    {labs "Iris Sepal Dimensions" "Sepal Length" "Sepal Width"} \
    theme_minimal \
    {save_svg tcl_ggplot_point.svg}

# + geom_smooth(): an lm() fit line, reusing stats::LM internally.
chain [datamunge::GGPlot new $iris Sepal.Length Petal.Length] \
    [list geom_point [datamunge::RGB new 156 163 175] 2.5] \
    geom_smooth \
    {labs "Petal Length vs Sepal Length With a Linear Fit" "Sepal Length" "Petal Length"} \
    {save_svg tcl_ggplot_smooth.svg}

# geom_bar(): counts a discrete column (stat = "count").
chain [datamunge::GGPlot new $iris Species] \
    geom_bar \
    {labs "Observations per Species" Species Count} \
    theme_bw \
    {save_svg tcl_ggplot_bar.svg}

# geom_boxplot(): grouped by a discrete x column.
chain [datamunge::GGPlot new $iris Species Petal.Width] \
    geom_boxplot \
    {labs "Petal Width by Species" Species "Petal Width"} \
    {save_svg tcl_ggplot_boxplot.svg}

# geom_histogram() + geom_density(): distribution of a single numeric column.
chain [datamunge::GGPlot new $iris Sepal.Length] \
    {geom_histogram 20} \
    {labs "Distribution of Sepal Length" "Sepal Length" Count} \
    {save_svg tcl_ggplot_histogram.svg}

chain [datamunge::GGPlot new $iris Sepal.Length] \
    geom_density \
    {labs "Density of Sepal Length" "Sepal Length" Density} \
    {save_svg tcl_ggplot_density.svg}

# facet_wrap(): one panel per Species, composed via RLayout under the hood.
chain [datamunge::GGPlot new $iris Petal.Length Petal.Width] \
    geom_point \
    {facet_wrap Species} \
    {labs "Petal Dimensions" "Petal Length" "Petal Width"} \
    {save_svg tcl_ggplot_facet.svg}

# theme_classic(): another built-in theme.
chain [datamunge::GGPlot new $iris Sepal.Length Sepal.Width Species] \
    geom_point \
    theme_classic \
    {save_svg tcl_ggplot_classic.svg}

puts "Wrote 7 SVGs to tcl_ggplot_*.svg"
