-- CMD: just TARGET=rplot_ex run-haskell
-- R base-graphics-style plotting with RPlot / RLayout: plot, hist, barplot, boxplot, pie, qqnorm, layouts.
module Main (main) where

import Datamunge

main :: IO ()
main = do
  -- plot(x, y, type = "p") then abline() layered on afterward.
  scatter <- rPlotPlotWithLabel [1.0, 2.0, 3.0, 4.0, 5.0] [2.1, 3.9, 6.2, 7.8, 10.1] "p" "observed"
  red <- newRgbRedGreenBlueOpacity 220 38 38
  _ <- rPlotAblineWithStrokeWidth scatter 0.0 2.0 red 1.5
  _ <- rPlotTitle scatter "plot() + abline()" >>= (`rPlotXLabel` "x") >>= (`rPlotYLabel` "y")
  plotSaveSvg scatter "haskell_rplot_scatter.svg"

  -- hist(): equal-width binning over the data range.
  histogram <- rPlotHistWithLabel [1.0, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7] 6 "counts"
  plotSaveSvg histogram "haskell_rplot_hist.svg"

  -- barplot(): categorical positions with x tick labels.
  rPlotBarplotWithNames [23.0, 41.0, 12.0] ["A", "B", "C"] >>= (`rPlotTitle` "barplot()") >>= (`plotSaveSvg` "haskell_rplot_barplot.svg")

  -- boxplot(): Tukey five-number summary per group.
  rPlotBoxplotWithNames [[2.0, 4, 4, 4, 5, 5, 7, 9], [1.0, 2, 2, 2, 3, 3, 3, 3, 4, 20]] ["low variance", "has outlier"]
    >>= (`rPlotTitle` "boxplot()") >>= (`plotSaveSvg` "haskell_rplot_boxplot.svg")

  -- pie(): wedge areas proportional to value, axes hidden automatically.
  rPlotPieWithNames [35.0, 25.0, 20.0, 20.0] ["Q1", "Q2", "Q3", "Q4"] >>= (`rPlotTitle` "pie()") >>= (`plotSaveSvg` "haskell_rplot_pie.svg")

  -- curve() is omitted to stay in step with the Lua example (Lua cannot supply the callback).

  -- qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
  let residuals = [-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3]
  rPlotQqnorm residuals >>= (`rPlotQqline` residuals) >>= (`plotSaveSvg` "haskell_rplot_qqnorm.svg")

  -- par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
  rLayoutCreate 1 2 >>= (`rLayoutAdd` scatter) >>= (`rLayoutAdd` histogram) >>= (`rLayoutSaveSvg` "haskell_rplot_layout.svg")

  putStrLn "Wrote 7 SVGs to haskell_rplot_*.svg"
