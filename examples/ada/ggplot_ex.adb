--  CMD: just TARGET=ggplot_ex run-ada
--  Demonstrates GGPlot -- the ggplot2-style grammar-of-graphics library.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;

procedure Ggplot_Ex is
   Iris : Data_Frame := Data.Iris;
   Gray : Rgb := New_Rgb (156, 163, 175);
   P    : Gg_Plot;
begin
   --  ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
   --  The layer/label/theme functions return the plot itself (owned by P, freed with it).
   P := New_Gg_Plot (Iris, "Sepal.Length", "Sepal.Width", "Species");
   P.Geom_Point
    .Labs ("Iris Sepal Dimensions", "Sepal Length", "Sepal Width")
    .Theme_Minimal
    .Save_Svg ("ada_ggplot_point.svg");
   Free (P);

   --  + geom_smooth(): an lm() fit line, reusing stats::LM internally.
   P := New_Gg_Plot (Iris, "Sepal.Length", "Petal.Length");
   P.Geom_Point (Gray, 2.5)
    .Geom_Smooth
    .Labs ("Petal Length vs Sepal Length With a Linear Fit", "Sepal Length", "Petal Length")
    .Save_Svg ("ada_ggplot_smooth.svg");
   Free (P);

   --  geom_bar(): counts a discrete column (stat = "count").
   P := New_Gg_Plot (Iris, "Species");
   P.Geom_Bar
    .Labs ("Observations per Species", "Species", "Count")
    .Theme_Bw
    .Save_Svg ("ada_ggplot_bar.svg");
   Free (P);

   --  geom_boxplot(): grouped by a discrete x column.
   P := New_Gg_Plot (Iris, "Species", "Petal.Width");
   P.Geom_Boxplot
    .Labs ("Petal Width by Species", "Species", "Petal Width")
    .Save_Svg ("ada_ggplot_boxplot.svg");
   Free (P);

   --  geom_histogram() + geom_density(): distribution of a single numeric column.
   P := New_Gg_Plot (Iris, "Sepal.Length");
   P.Geom_Histogram (20)
    .Labs ("Distribution of Sepal Length", "Sepal Length", "Count")
    .Save_Svg ("ada_ggplot_histogram.svg");
   Free (P);

   P := New_Gg_Plot (Iris, "Sepal.Length");
   P.Geom_Density
    .Labs ("Density of Sepal Length", "Sepal Length", "Density")
    .Save_Svg ("ada_ggplot_density.svg");
   Free (P);

   --  facet_wrap(): one panel per Species, composed via RLayout under the hood.
   P := New_Gg_Plot (Iris, "Petal.Length", "Petal.Width");
   P.Geom_Point
    .Facet_Wrap ("Species")
    .Labs ("Petal Dimensions", "Petal Length", "Petal Width")
    .Save_Svg ("ada_ggplot_facet.svg");
   Free (P);

   --  theme_classic(): another built-in theme.
   P := New_Gg_Plot (Iris, "Sepal.Length", "Sepal.Width", "Species");
   P.Geom_Point
    .Theme_Classic
    .Save_Svg ("ada_ggplot_classic.svg");
   Free (P);

   Put_Line ("Wrote 7 SVGs to ada_ggplot_*.svg");

   Free (Gray);
   Free (Iris);
end Ggplot_Ex;
