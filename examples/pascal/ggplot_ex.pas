// CMD: just TARGET=ggplot_ex run-pascal
// Demonstrates GGPlot -- the ggplot2-style grammar-of-graphics library.
program ggplot_ex;

{$mode objfpc}{$H+}

uses
  SysUtils, datamunge;

// Saves a plot as SVG and frees it. The layer/label/theme methods return the plot itself.
procedure SaveAndFree(APlot: TGgPlot; const APath: String);
begin
  try
    APlot.SaveSvg(APath);
  finally
    APlot.Free;
  end;
end;

var
  Iris: TDataFrame;
  Gray: TRgb;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;

  Iris := Data.Iris;
  Gray := TRgb.Create(156, 163, 175);
  try
    // ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
    SaveAndFree(TGgPlot.Create(Iris, 'Sepal.Length', 'Sepal.Width', 'Species')
      .GeomPoint
      .Labs('Iris Sepal Dimensions', 'Sepal Length', 'Sepal Width')
      .ThemeMinimal,
      'pascal_ggplot_point.svg');

    // + geom_smooth(): an lm() fit line, reusing stats::LM internally.
    SaveAndFree(TGgPlot.Create(Iris, 'Sepal.Length', 'Petal.Length')
      .GeomPoint(Gray, 2.5)
      .GeomSmooth
      .Labs('Petal Length vs Sepal Length With a Linear Fit', 'Sepal Length', 'Petal Length'),
      'pascal_ggplot_smooth.svg');

    // geom_bar(): counts a discrete column (stat = "count").
    SaveAndFree(TGgPlot.Create(Iris, 'Species')
      .GeomBar
      .Labs('Observations per Species', 'Species', 'Count')
      .ThemeBw,
      'pascal_ggplot_bar.svg');

    // geom_boxplot(): grouped by a discrete x column.
    SaveAndFree(TGgPlot.Create(Iris, 'Species', 'Petal.Width')
      .GeomBoxplot
      .Labs('Petal Width by Species', 'Species', 'Petal Width'),
      'pascal_ggplot_boxplot.svg');

    // geom_histogram() + geom_density(): distribution of a single numeric column.
    SaveAndFree(TGgPlot.Create(Iris, 'Sepal.Length')
      .GeomHistogram(20)
      .Labs('Distribution of Sepal Length', 'Sepal Length', 'Count'),
      'pascal_ggplot_histogram.svg');

    SaveAndFree(TGgPlot.Create(Iris, 'Sepal.Length')
      .GeomDensity
      .Labs('Density of Sepal Length', 'Sepal Length', 'Density'),
      'pascal_ggplot_density.svg');

    // facet_wrap(): one panel per Species, composed via RLayout under the hood.
    SaveAndFree(TGgPlot.Create(Iris, 'Petal.Length', 'Petal.Width')
      .GeomPoint
      .FacetWrap('Species')
      .Labs('Petal Dimensions', 'Petal Length', 'Petal Width'),
      'pascal_ggplot_facet.svg');

    // theme_classic(): another built-in theme.
    SaveAndFree(TGgPlot.Create(Iris, 'Sepal.Length', 'Sepal.Width', 'Species')
      .GeomPoint
      .ThemeClassic,
      'pascal_ggplot_classic.svg');
  finally
    Gray.Free;
    Iris.Free;
  end;

  WriteLn('Wrote 7 SVGs to pascal_ggplot_*.svg');
end.
