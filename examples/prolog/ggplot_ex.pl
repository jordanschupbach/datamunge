% CMD: just TARGET=ggplot_ex run-prolog
% Demonstrates GGPlot -- the ggplot2-style grammar-of-graphics library.
:- use_module(library(datamunge)).
:- initialization(main, main).

main :-
    data_iris(Iris),

    % ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
    gg_plot_new(Iris, "Sepal.Length", "Sepal.Width", "Species", P1a),
    gg_plot_geom_point(P1a, P1b),
    gg_plot_labs(P1b, "Iris Sepal Dimensions", "Sepal Length", "Sepal Width", P1c),
    gg_plot_theme_minimal(P1c, P1),
    gg_plot_save_svg(P1, "prolog_ggplot_point.svg"),

    % + geom_smooth(): an lm() fit line, reusing stats::LM internally.
    gg_plot_new(Iris, "Sepal.Length", "Petal.Length", P2a),
    rgb_new(156, 163, 175, Grey),
    gg_plot_geom_point(P2a, Grey, 2.5, P2b),
    gg_plot_geom_smooth(P2b, P2c),
    gg_plot_labs(P2c, "Petal Length vs Sepal Length With a Linear Fit", "Sepal Length", "Petal Length", P2),
    gg_plot_save_svg(P2, "prolog_ggplot_smooth.svg"),

    % geom_bar(): counts a discrete column (stat = "count").
    gg_plot_new(Iris, "Species", P3a),
    gg_plot_geom_bar(P3a, P3b),
    gg_plot_labs(P3b, "Observations per Species", "Species", "Count", P3c),
    gg_plot_theme_bw(P3c, P3),
    gg_plot_save_svg(P3, "prolog_ggplot_bar.svg"),

    % geom_boxplot(): grouped by a discrete x column.
    gg_plot_new(Iris, "Species", "Petal.Width", P4a),
    gg_plot_geom_boxplot(P4a, P4b),
    gg_plot_labs(P4b, "Petal Width by Species", "Species", "Petal Width", P4),
    gg_plot_save_svg(P4, "prolog_ggplot_boxplot.svg"),

    % geom_histogram() + geom_density(): distribution of a single numeric column.
    gg_plot_new(Iris, "Sepal.Length", P5a),
    gg_plot_geom_histogram(P5a, 20, P5b),
    gg_plot_labs(P5b, "Distribution of Sepal Length", "Sepal Length", "Count", P5),
    gg_plot_save_svg(P5, "prolog_ggplot_histogram.svg"),

    gg_plot_new(Iris, "Sepal.Length", P6a),
    gg_plot_geom_density(P6a, P6b),
    gg_plot_labs(P6b, "Density of Sepal Length", "Sepal Length", "Density", P6),
    gg_plot_save_svg(P6, "prolog_ggplot_density.svg"),

    % facet_wrap(): one panel per Species, composed via RLayout under the hood.
    gg_plot_new(Iris, "Petal.Length", "Petal.Width", P7a),
    gg_plot_geom_point(P7a, P7b),
    gg_plot_facet_wrap(P7b, "Species", P7c),
    gg_plot_labs(P7c, "Petal Dimensions", "Petal Length", "Petal Width", P7),
    gg_plot_save_svg(P7, "prolog_ggplot_facet.svg"),

    % theme_classic(): another built-in theme.
    gg_plot_new(Iris, "Sepal.Length", "Sepal.Width", "Species", P8a),
    gg_plot_geom_point(P8a, P8b),
    gg_plot_theme_classic(P8b, P8),
    gg_plot_save_svg(P8, "prolog_ggplot_classic.svg"),

    format("Wrote 7 SVGs to prolog_ggplot_*.svg~n").
