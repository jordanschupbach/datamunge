// CMD: just TARGET=ggplot_ex run-rust
// Demonstrates GGPlot -- the ggplot2-style grammar-of-graphics library.
use datamunge::*;

fn main() -> datamunge::Result<()> {
    let iris = data::iris()?.expect("iris");

    // ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
    GGPlot::new_with_color_column(&iris, "Sepal.Length", "Sepal.Width", "Species")?
        .geom_point()?
        .labs_with_y("Iris Sepal Dimensions", "Sepal Length", "Sepal Width")?
        .theme_minimal()?
        .save_svg("rust_ggplot_point.svg")?;

    // + geom_smooth(): an lm() fit line, reusing stats::LM internally.
    GGPlot::new_with_y_column(&iris, "Sepal.Length", "Petal.Length")?
        .geom_point_with_size(&RGB::new_red_green_blue(156, 163, 175)?, 2.5)?
        .geom_smooth()?
        .labs_with_y("Petal Length vs Sepal Length With a Linear Fit", "Sepal Length", "Petal Length")?
        .save_svg("rust_ggplot_smooth.svg")?;

    // geom_bar(): counts a discrete column (stat = "count").
    GGPlot::new(&iris, "Species")?
        .geom_bar()?
        .labs_with_y("Observations per Species", "Species", "Count")?
        .theme_bw()?
        .save_svg("rust_ggplot_bar.svg")?;

    // geom_boxplot(): grouped by a discrete x column.
    GGPlot::new_with_y_column(&iris, "Species", "Petal.Width")?
        .geom_boxplot()?
        .labs_with_y("Petal Width by Species", "Species", "Petal Width")?
        .save_svg("rust_ggplot_boxplot.svg")?;

    // geom_histogram() + geom_density(): distribution of a single numeric column.
    GGPlot::new(&iris, "Sepal.Length")?
        .geom_histogram_with_bins(20)?
        .labs_with_y("Distribution of Sepal Length", "Sepal Length", "Count")?
        .save_svg("rust_ggplot_histogram.svg")?;

    GGPlot::new(&iris, "Sepal.Length")?
        .geom_density()?
        .labs_with_y("Density of Sepal Length", "Sepal Length", "Density")?
        .save_svg("rust_ggplot_density.svg")?;

    // facet_wrap(): one panel per Species, composed via RLayout under the hood.
    GGPlot::new_with_y_column(&iris, "Petal.Length", "Petal.Width")?
        .geom_point()?
        .facet_wrap("Species")?
        .labs_with_y("Petal Dimensions", "Petal Length", "Petal Width")?
        .save_svg("rust_ggplot_facet.svg")?;

    // theme_classic(): another built-in theme.
    GGPlot::new_with_color_column(&iris, "Sepal.Length", "Sepal.Width", "Species")?
        .geom_point()?
        .theme_classic()?
        .save_svg("rust_ggplot_classic.svg")?;

    println!("Wrote 7 SVGs to rust_ggplot_*.svg");
    Ok(())
}
