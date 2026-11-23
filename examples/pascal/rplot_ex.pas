// CMD: just TARGET=rplot_ex run-pascal
// R base-graphics-style plotting with RPlot / RLayout: plot, hist, barplot, boxplot, pie, qqnorm, layouts.
program rplot_ex;

{$mode objfpc}{$H+}

uses
  SysUtils, datamunge;

// Saves a plot as SVG and frees it. The fluent methods (Title2, Qqline, ...) return the plot itself.
procedure SaveAndFree(APlot: TRPlot; const APath: String);
begin
  try
    APlot.SaveSvg(APath);
  finally
    APlot.Free;
  end;
end;

const
  Residuals: array[0..9] of Double = (-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3);
var
  Scatter, Histogram, Plot: TRPlot;
  Red: TRgb;
  Layout: TRLayout;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;

  // plot(x, y, type = "p") then abline() layered on afterward.
  Scatter := TRPlot.Plot([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 3.9, 6.2, 7.8, 10.1], 'p', 'observed');
  // hist(): equal-width binning over the data range.
  Histogram := TRPlot.Hist([1.0, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7], 6, 'counts');
  Red := TRgb.Create(220, 38, 38);
  try
    Scatter.Abline(0.0, 2.0, Red, 1.5);
    // RPlot's title/xlab/ylab are Title2/XLabel2/YLabel2 in Pascal (Plot's Title returns a TPlot).
    Scatter.Title2('plot() + abline()').XLabel2('x').YLabel2('y');
    Scatter.SaveSvg('pascal_rplot_scatter.svg');

    Histogram.SaveSvg('pascal_rplot_hist.svg');

    // barplot(): categorical positions with x tick labels.
    Plot := TRPlot.Barplot([23.0, 41.0, 12.0], ['A', 'B', 'C']);
    SaveAndFree(Plot.Title2('barplot()'), 'pascal_rplot_barplot.svg');

    // boxplot(): Tukey five-number summary per group.
    Plot := TRPlot.Boxplot([[2.0, 4, 4, 4, 5, 5, 7, 9], [1.0, 2, 2, 2, 3, 3, 3, 3, 4, 20]],
                           ['low variance', 'has outlier']);
    SaveAndFree(Plot.Title2('boxplot()'), 'pascal_rplot_boxplot.svg');

    // pie(): wedge areas proportional to value, axes hidden automatically.
    Plot := TRPlot.Pie([35.0, 25.0, 20.0, 20.0], ['Q1', 'Q2', 'Q3', 'Q4']);
    SaveAndFree(Plot.Title2('pie()'), 'pascal_rplot_pie.svg');

    // curve() is omitted to stay in step with the Lua example (Lua cannot supply the callback).

    // qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
    Plot := TRPlot.Qqnorm(Residuals);
    SaveAndFree(Plot.Qqline(Residuals), 'pascal_rplot_qqnorm.svg');

    // par(mfrow = c(1, 2))-style multi-panel composition via RLayout (RLayout::create is Create2).
    Layout := TRLayout.Create2(1, 2);
    try
      Layout.Add(Scatter).Add(Histogram).SaveSvg('pascal_rplot_layout.svg');
    finally
      Layout.Free;
    end;
  finally
    Red.Free;
    Histogram.Free;
    Scatter.Free;
  end;

  WriteLn('Wrote 7 SVGs to pascal_rplot_*.svg');
end.
