% CMD: just TARGET=rplot_ex run-prolog
% R base-graphics-style plotting with RPlot / RLayout: plot, hist, barplot, boxplot, pie, qqnorm, layouts.
:- use_module(library(datamunge)).
:- initialization(main, main).

main :-
    % plot(x, y, type = "p") then abline() layered on afterward.
    r_plot_plot([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 3.9, 6.2, 7.8, 10.1], "p", "observed", Scatter),
    rgb_new(220, 38, 38, Red),
    r_plot_abline(Scatter, 0.0, 2.0, Red, 1.5, _),
    r_plot_title(Scatter, "plot() + abline()", S1), r_plot_x_label(S1, "x", S2), r_plot_y_label(S2, "y", _),
    plot_save_svg(Scatter, "prolog_rplot_scatter.svg"),

    % hist(): equal-width binning over the data range.
    r_plot_hist([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 7.0], 6, "counts", Histogram),
    plot_save_svg(Histogram, "prolog_rplot_hist.svg"),

    % barplot(): categorical positions with x tick labels.
    r_plot_barplot([23.0, 41.0, 12.0], ["A", "B", "C"], Bar0),
    r_plot_title(Bar0, "barplot()", Bar),
    plot_save_svg(Bar, "prolog_rplot_barplot.svg"),

    % boxplot(): Tukey five-number summary per group.
    r_plot_boxplot([[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], [1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 20.0]],
                   ["low variance", "has outlier"], Box0),
    r_plot_title(Box0, "boxplot()", Box),
    plot_save_svg(Box, "prolog_rplot_boxplot.svg"),

    % pie(): wedge areas proportional to value, axes hidden automatically.
    r_plot_pie([35.0, 25.0, 20.0, 20.0], ["Q1", "Q2", "Q3", "Q4"], Pie0),
    r_plot_title(Pie0, "pie()", Pie),
    plot_save_svg(Pie, "prolog_rplot_pie.svg"),

    % curve() is omitted to stay in step with the Lua example (Lua cannot supply the callback).

    % qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
    Residuals = [-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3],
    r_plot_qqnorm(Residuals, Qq0),
    r_plot_qqline(Qq0, Residuals, Qq),
    plot_save_svg(Qq, "prolog_rplot_qqnorm.svg"),

    % par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
    r_layout_create(1, 2, Layout0),
    r_layout_add(Layout0, Scatter, Layout1),
    r_layout_add(Layout1, Histogram, Layout),
    r_layout_save_svg(Layout, "prolog_rplot_layout.svg"),

    format("Wrote 7 SVGs to prolog_rplot_*.svg~n").
