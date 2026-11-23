-- CMD: just TARGET=ggplot_ex run-haskell
-- Demonstrates GGPlot -- the ggplot2-style grammar-of-graphics library.
module Main (main) where

import Datamunge

main :: IO ()
main = do
  iris <- dataIris

  -- ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
  newGgPlotWithColorColumn iris "Sepal.Length" "Sepal.Width" "Species"
    >>= ggPlotGeomPoint
    >>= (\p -> ggPlotLabsWithY p "Iris Sepal Dimensions" "Sepal Length" "Sepal Width")
    >>= ggPlotThemeMinimal
    >>= (`ggPlotSaveSvg` "haskell_ggplot_point.svg")

  -- + geom_smooth(): an lm() fit line, reusing stats::LM internally.
  gray <- newRgbRedGreenBlueOpacity 156 163 175
  newGgPlotWithYColumn iris "Sepal.Length" "Petal.Length"
    >>= (\p -> ggPlotGeomPointWithSize p gray 2.5)
    >>= ggPlotGeomSmooth
    >>= (\p -> ggPlotLabsWithY p "Petal Length vs Sepal Length With a Linear Fit" "Sepal Length" "Petal Length")
    >>= (`ggPlotSaveSvg` "haskell_ggplot_smooth.svg")

  -- geom_bar(): counts a discrete column (stat = "count").
  newGgPlot iris "Species"
    >>= ggPlotGeomBar
    >>= (\p -> ggPlotLabsWithY p "Observations per Species" "Species" "Count")
    >>= ggPlotThemeBw
    >>= (`ggPlotSaveSvg` "haskell_ggplot_bar.svg")

  -- geom_boxplot(): grouped by a discrete x column.
  newGgPlotWithYColumn iris "Species" "Petal.Width"
    >>= ggPlotGeomBoxplot
    >>= (\p -> ggPlotLabsWithY p "Petal Width by Species" "Species" "Petal Width")
    >>= (`ggPlotSaveSvg` "haskell_ggplot_boxplot.svg")

  -- geom_histogram() + geom_density(): distribution of a single numeric column.
  newGgPlot iris "Sepal.Length"
    >>= (`ggPlotGeomHistogramWithBins` 20)
    >>= (\p -> ggPlotLabsWithY p "Distribution of Sepal Length" "Sepal Length" "Count")
    >>= (`ggPlotSaveSvg` "haskell_ggplot_histogram.svg")

  newGgPlot iris "Sepal.Length"
    >>= ggPlotGeomDensity
    >>= (\p -> ggPlotLabsWithY p "Density of Sepal Length" "Sepal Length" "Density")
    >>= (`ggPlotSaveSvg` "haskell_ggplot_density.svg")

  -- facet_wrap(): one panel per Species, composed via RLayout under the hood.
  newGgPlotWithYColumn iris "Petal.Length" "Petal.Width"
    >>= ggPlotGeomPoint
    >>= (`ggPlotFacetWrap` "Species")
    >>= (\p -> ggPlotLabsWithY p "Petal Dimensions" "Petal Length" "Petal Width")
    >>= (`ggPlotSaveSvg` "haskell_ggplot_facet.svg")

  -- theme_classic(): another built-in theme.
  newGgPlotWithColorColumn iris "Sepal.Length" "Sepal.Width" "Species"
    >>= ggPlotGeomPoint
    >>= ggPlotThemeClassic
    >>= (`ggPlotSaveSvg` "haskell_ggplot_classic.svg")

  putStrLn "Wrote 7 SVGs to haskell_ggplot_*.svg"
