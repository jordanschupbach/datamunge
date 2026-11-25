⍝ CMD: just TARGET=ggplot_ex run-apl
⍝ Demonstrates GGPlot -- the ggplot2-style grammar-of-graphics library.
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
⍝ mkdir -p build/debug/examples (⎕FIO[20] is mkdir; an existing directory just returns an error code).
Z←⎕FIO[20]¨'build' 'build/debug' 'build/debug/examples'
⍝ The layer/label/theme methods return the plot's own handle, so P←P ... keeps one handle per plot
⍝ (dm∆Delete it once, at the end).
IRIS←dm∆data∆iris

⍝ ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
SCATTER←dm∆GGPlot∆new IRIS 'Sepal.Length' 'Sepal.Width' 'Species'
SCATTER←SCATTER dm∆GGPlot∆geom_point ⍬
SCATTER←SCATTER dm∆GGPlot∆labs 'Iris Sepal Dimensions' 'Sepal Length' 'Sepal Width'
SCATTER←dm∆GGPlot∆theme_minimal SCATTER
SCATTER dm∆GGPlot∆save_svg 'build/debug/examples/apl_ggplot_point.svg'

⍝ + geom_smooth(): an lm() fit line per group, reusing stats::LM internally.
GRAY←dm∆RGB∆new 156 163 175
SMOOTH←dm∆GGPlot∆new IRIS 'Sepal.Length' 'Petal.Length'
SMOOTH←SMOOTH dm∆GGPlot∆geom_point GRAY 2.5
SMOOTH←SMOOTH dm∆GGPlot∆geom_smooth ⍬
SMOOTH←SMOOTH dm∆GGPlot∆labs 'Petal Length vs Sepal Length With a Linear Fit' 'Sepal Length' 'Petal Length'
SMOOTH dm∆GGPlot∆save_svg 'build/debug/examples/apl_ggplot_smooth.svg'

⍝ geom_bar(): counts a discrete column (stat = "count").
BAR←dm∆GGPlot∆new IRIS 'Species'
BAR←BAR dm∆GGPlot∆geom_bar ⍬
BAR←BAR dm∆GGPlot∆labs 'Observations per Species' 'Species' 'Count'
BAR←dm∆GGPlot∆theme_bw BAR
BAR dm∆GGPlot∆save_svg 'build/debug/examples/apl_ggplot_bar.svg'

⍝ geom_boxplot(): grouped by a discrete x column.
BOX←dm∆GGPlot∆new IRIS 'Species' 'Petal.Width'
BOX←BOX dm∆GGPlot∆geom_boxplot ⍬
BOX←BOX dm∆GGPlot∆labs 'Petal Width by Species' 'Species' 'Petal Width'
BOX dm∆GGPlot∆save_svg 'build/debug/examples/apl_ggplot_boxplot.svg'

⍝ geom_histogram() + geom_density(): distribution of a single numeric column.
HIST←dm∆GGPlot∆new IRIS 'Sepal.Length'
HIST←HIST dm∆GGPlot∆geom_histogram 20
HIST←HIST dm∆GGPlot∆labs 'Distribution of Sepal Length' 'Sepal Length' 'Count'
HIST dm∆GGPlot∆save_svg 'build/debug/examples/apl_ggplot_histogram.svg'

DENSITY←dm∆GGPlot∆new IRIS 'Sepal.Length'
DENSITY←DENSITY dm∆GGPlot∆geom_density ⍬
DENSITY←DENSITY dm∆GGPlot∆labs 'Density of Sepal Length' 'Sepal Length' 'Density'
DENSITY dm∆GGPlot∆save_svg 'build/debug/examples/apl_ggplot_density.svg'

⍝ facet_wrap(): one panel per Species, composed via RLayout under the hood.
FACETED←dm∆GGPlot∆new IRIS 'Petal.Length' 'Petal.Width'
FACETED←FACETED dm∆GGPlot∆geom_point ⍬
FACETED←FACETED dm∆GGPlot∆facet_wrap 'Species'
FACETED←FACETED dm∆GGPlot∆labs 'Petal Dimensions' 'Petal Length' 'Petal Width'
FACETED dm∆GGPlot∆save_svg 'build/debug/examples/apl_ggplot_facet.svg'

⍝ scale_color_manual(): override the default discrete palette; theme_classic(): another built-in theme.
COLORS←(dm∆RGB∆new 16 185 129) (dm∆RGB∆new 245 158 11) (dm∆RGB∆new 99 102 241)
CUSTOM←dm∆GGPlot∆new IRIS 'Sepal.Length' 'Sepal.Width' 'Species'
CUSTOM←CUSTOM dm∆GGPlot∆geom_point ⍬
CUSTOM←CUSTOM dm∆GGPlot∆scale_color_manual ⊂COLORS
CUSTOM←dm∆GGPlot∆theme_classic CUSTOM
CUSTOM dm∆GGPlot∆save_svg 'build/debug/examples/apl_ggplot_custom_colors.svg'

PUT 'Wrote 8 SVGs to build/debug/examples/apl_ggplot_*.svg'
dm∆Delete IRIS GRAY SCATTER SMOOTH BAR BOX HIST DENSITY FACETED CUSTOM
dm∆Delete COLORS
