module app;

// Demonstrates datamunge.RPlot / datamunge.RLayout -- the R-base-graphics-style plotting library.
import std.stdio : writeln;
import std.math : sin, PI;
import datamunge;

DVector dv(double[] t) {
  auto v = new DVector();
  foreach (x; t) v.push_back(x);
  return v;
}

SVector sv(string[] t) {
  auto v = new SVector();
  foreach (x; t) v.push_back(x);
  return v;
}

DVectorVector dvv(double[][] rows) {
  auto m = new DVectorVector();
  foreach (r; rows) m.push_back(dv(r));
  return m;
}

RGB rgb(int r, int g, int b) {
  auto c = new RGB();
  c.r = r;
  c.g = g;
  c.b = b;
  return c;
}

// curve() samples a Callback (a SWIG director). Unlike the Lua/Ruby ports, the D binding
// DOES generate director code for Callback, so we can subclass it live here.
class SineWave : Callback {
  this() { super(); }
  override double call(double x) { return sin(x); }
}

void main() {
  // plot(x, y, type = "p") then abline() layered on afterward.
  auto scatter = RPlot.plot(dv([1.0, 2.0, 3.0, 4.0, 5.0]), dv([2.1, 3.9, 6.2, 7.8, 10.1]), "p", "observed");
  scatter.abline(0.0, 2.0, rgb(220, 38, 38), 1.5);
  scatter.title("plot() + abline()").x_label("x").y_label("y");
  scatter.save_svg("d_rplot_scatter.svg");

  // hist(): equal-width binning over the data range.
  auto hist = RPlot.hist(dv([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7]), 6, "counts");
  hist.save_svg("d_rplot_hist.svg");

  // barplot(): categorical positions with x tick labels.
  auto bars = RPlot.barplot(dv([23.0, 41.0, 12.0]), sv(["A", "B", "C"]));
  bars.title("barplot()");
  bars.save_svg("d_rplot_barplot.svg");

  // boxplot(): Tukey five-number summary per group.
  auto groups = dvv([[2, 4, 4, 4, 5, 5, 7, 9], [1, 2, 2, 2, 3, 3, 3, 3, 4, 20]]);
  auto box = RPlot.boxplot(groups, sv(["low variance", "has outlier"]));
  box.title("boxplot()");
  box.save_svg("d_rplot_boxplot.svg");

  // pie(): wedge areas proportional to value, axes hidden automatically.
  auto pie = RPlot.pie(dv([35.0, 25.0, 20.0, 20.0]), sv(["Q1", "Q2", "Q3", "Q4"]));
  pie.title("pie()");
  pie.save_svg("d_rplot_pie.svg");

  // curve(): samples a Callback (a D-subclassed director) over a range.
  auto sine = new SineWave();
  auto curve = RPlot.curve(sine, 0.0, 2.0 * PI, 200, "sin(x)");
  curve.title("curve()");
  curve.save_svg("d_rplot_curve.svg");

  // qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
  double[] residuals = [-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3];
  auto qq = RPlot.qqnorm(dv(residuals));
  qq.qqline(dv(residuals));
  qq.save_svg("d_rplot_qqnorm.svg");

  // par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
  auto layout = RLayout.create(1, 2);
  layout.add(scatter);
  layout.add(hist);
  layout.save_svg("d_rplot_layout.svg");

  writeln("Wrote 8 SVGs to d_rplot_*.svg");
}
