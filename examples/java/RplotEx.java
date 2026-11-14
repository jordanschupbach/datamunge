// CMD: just TARGET=rplot_ex run-java
// R base-graphics-style plotting with RPlot / RLayout: plot, hist, barplot, boxplot, pie, qqnorm, layouts.
import datamunge.RGB;
import datamunge.RLayout;
import datamunge.RPlot;
import java.util.List;

public class RplotEx {
  public static void main(String[] args) {
    // plot(x, y, type = "p") then abline() layered on afterward.
    var scatter =
        RPlot.plot(
            new double[] {1.0, 2.0, 3.0, 4.0, 5.0},
            new double[] {2.1, 3.9, 6.2, 7.8, 10.1},
            "p",
            "observed");
    scatter.abline(0.0, 2.0, new RGB(220, 38, 38), 1.5);
    scatter.title("plot() + abline()");
    scatter.xLabel("x");
    scatter.yLabel("y");
    scatter.saveSvg("java_rplot_scatter.svg");

    // hist(): equal-width binning over the data range.
    var histogram =
        RPlot.hist(new double[] {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7}, 6, "counts");
    histogram.saveSvg("java_rplot_hist.svg");

    // barplot(): categorical positions with x tick labels.
    RPlot.barplot(new double[] {23.0, 41.0, 12.0}, List.of("A", "B", "C"))
        .title("barplot()")
        .saveSvg("java_rplot_barplot.svg");

    // boxplot(): Tukey five-number summary per group.
    List<double[]> groups =
        List.of(
            new double[] {2.0, 4, 4, 4, 5, 5, 7, 9},
            new double[] {1.0, 2, 2, 2, 3, 3, 3, 3, 4, 20});
    RPlot.boxplot(groups, List.of("low variance", "has outlier"))
        .title("boxplot()")
        .saveSvg("java_rplot_boxplot.svg");

    // pie(): wedge areas proportional to value, axes hidden automatically.
    RPlot.pie(new double[] {35.0, 25.0, 20.0, 20.0}, List.of("Q1", "Q2", "Q3", "Q4"))
        .title("pie()")
        .saveSvg("java_rplot_pie.svg");

    // curve() is omitted to stay in step with the Lua example (Lua cannot supply the callback).

    // qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
    double[] residuals = {-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3};
    RPlot.qqnorm(residuals).qqline(residuals).saveSvg("java_rplot_qqnorm.svg");

    // par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
    RLayout.create(1, 2).add(scatter).add(histogram).saveSvg("java_rplot_layout.svg");

    System.out.println("Wrote 7 SVGs to java_rplot_*.svg");
  }
}
