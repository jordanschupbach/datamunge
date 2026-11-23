--  CMD: just TARGET=rplot_ex run-ada
--  R base-graphics-style plotting with RPlot / RLayout: plot, hist, barplot, boxplot, pie, qqnorm, layouts.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;

procedure Rplot_Ex is
   --  Saves a plot as SVG and frees it.
   procedure Save_And_Free (P : in out R_Plot; Path : String) is
   begin
      P.Save_Svg (Path);
      Free (P);
   end Save_And_Free;

   --  Discards what a fluent method returns: the plot itself, which belongs to the plot.
   procedure Ignore (Plot : R_Plot) is null;

   Residuals : constant Long_Float_Vector := [-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3];
   Scatter, Histogram, P : R_Plot;
   Red : Rgb;
   Layout : R_Layout;
begin
   --  plot(x, y, type = "p") then abline() layered on afterward. The fluent methods return the
   --  plot itself (an object that belongs to it), which is not freed separately.
   Scatter := R_Plot_Plot ([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 3.9, 6.2, 7.8, 10.1], "p", "observed");
   Red := New_Rgb (220, 38, 38);
   Ignore (Scatter.Abline (0.0, 2.0, Red, 1.5));
   Ignore (Scatter.Title_2 ("plot() + abline()").X_Label_2 ("x").Y_Label_2 ("y"));
   Scatter.Save_Svg ("ada_rplot_scatter.svg");

   --  hist(): equal-width binning over the data range.
   Histogram := R_Plot_Hist ([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 7.0], 6, "counts");
   Histogram.Save_Svg ("ada_rplot_hist.svg");

   --  barplot(): categorical positions with x tick labels.
   P := R_Plot_Barplot ([23.0, 41.0, 12.0], ["A", "B", "C"]);
   Ignore (P.Title_2 ("barplot()"));
   Save_And_Free (P, "ada_rplot_barplot.svg");

   --  boxplot(): Tukey five-number summary per group.
   P := R_Plot_Boxplot ([[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], [1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 20.0]],
                        ["low variance", "has outlier"]);
   Ignore (P.Title_2 ("boxplot()"));
   Save_And_Free (P, "ada_rplot_boxplot.svg");

   --  pie(): wedge areas proportional to value, axes hidden automatically.
   P := R_Plot_Pie ([35.0, 25.0, 20.0, 20.0], ["Q1", "Q2", "Q3", "Q4"]);
   Ignore (P.Title_2 ("pie()"));
   Save_And_Free (P, "ada_rplot_pie.svg");

   --  curve() is omitted to stay in step with the Lua example (Lua cannot supply the callback).

   --  qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
   P := R_Plot_Qqnorm (Residuals);
   Ignore (P.Qqline (Residuals));
   Save_And_Free (P, "ada_rplot_qqnorm.svg");

   --  par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
   Layout := R_Layout_Create (1, 2);
   Layout.Add (Scatter).Add (Histogram).Save_Svg ("ada_rplot_layout.svg");
   Free (Layout);

   Free (Red);
   Free (Histogram);
   Free (Scatter);

   Put_Line ("Wrote 7 SVGs to ada_rplot_*.svg");
end Rplot_Ex;
