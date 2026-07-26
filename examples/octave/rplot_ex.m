1;

% Demonstrates RPlot / RLayout -- the R-base-graphics-style plotting library.
datamunge;

function v = dv(t)
  datamunge;
  v = DVector();
  for i = 1:numel(t)
    DVector_push_back(v, t(i));
  end
endfunction

function v = sv(t)
  datamunge;
  v = SVector();
  for i = 1:numel(t)
    SVector_push_back(v, t{i});
  end
endfunction

function m = dvv(rows)
  datamunge;
  m = DVectorVector();
  for i = 1:numel(rows)
    DVectorVector_push_back(m, dv(rows{i}));
  end
endfunction

function c = rgb(r, g, b)
  datamunge;
  c = RGB();
  RGB_r_set(c, r);
  RGB_g_set(c, g);
  RGB_b_set(c, b);
endfunction

function y = sine_call(self, x)
  y = sin(x);
endfunction

% plot(x, y, type = "p") then abline() layered on afterward.
scatter = RPlot_plot(dv([1.0, 2.0, 3.0, 4.0, 5.0]), dv([2.1, 3.9, 6.2, 7.8, 10.1]), "p", "observed");
RPlot_abline(scatter, 0.0, 2.0, rgb(220, 38, 38), 1.5);
Plot_title(scatter, "plot() + abline()");
Plot_x_label(scatter, "x");
Plot_y_label(scatter, "y");
Plot_save_svg(scatter, "octave_rplot_scatter.svg");

% hist(): equal-width binning over the data range.
hist = RPlot_hist(dv([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7]), 6, "counts");
Plot_save_svg(hist, "octave_rplot_hist.svg");

% barplot(): categorical positions with x tick labels.
bars = RPlot_barplot(dv([23.0, 41.0, 12.0]), sv({"A", "B", "C"}));
Plot_title(bars, "barplot()");
Plot_save_svg(bars, "octave_rplot_barplot.svg");

% boxplot(): Tukey five-number summary per group.
groups = dvv({[2, 4, 4, 4, 5, 5, 7, 9], [1, 2, 2, 2, 3, 3, 3, 3, 4, 20]});
box = RPlot_boxplot(groups, sv({"low variance", "has outlier"}));
Plot_title(box, "boxplot()");
Plot_save_svg(box, "octave_rplot_boxplot.svg");

% pie(): wedge areas proportional to value, axes hidden automatically.
pie = RPlot_pie(dv([35.0, 25.0, 20.0, 20.0]), sv({"Q1", "Q2", "Q3", "Q4"}));
Plot_title(pie, "pie()");
Plot_save_svg(pie, "octave_rplot_pie.svg");

% curve(): samples a Callback (a SWIG director) over a range. Octave's SWIG binding supports
% directors, and Callback::call() returns a plain double, so this works here (unlike the Lua
% binding, which has no director codegen at all).
sine = Callback();
sine.call = @sine_call;
curve = RPlot_curve(sine, 0.0, 2.0 * pi, 200, "sin(x)");
Plot_title(curve, "curve()");
Plot_save_svg(curve, "octave_rplot_curve.svg");

% qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
residuals = [-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3];
qq = RPlot_qqnorm(dv(residuals));
RPlot_qqline(qq, dv(residuals));
Plot_save_svg(qq, "octave_rplot_qqnorm.svg");

% par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
layout = RLayout_create(1, 2);
RLayout_add(layout, scatter);
RLayout_add(layout, hist);
RLayout_save_svg(layout, "octave_rplot_layout.svg");

printf("Wrote 8 SVGs to octave_rplot_*.svg\n");
