// CMD: just TARGET=rplot_ex run-rust
// R base-graphics-style plotting with RPlot / RLayout: plot, hist, barplot, boxplot, pie, qqnorm, layouts.
use datamunge::*;

fn main() -> datamunge::Result<()> {
    // plot(x, y, type = "p") then abline() layered on afterward.
    let scatter = RPlot::plot_with_label(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.1, 3.9, 6.2, 7.8, 10.1], "p", "observed")?;
    scatter.abline_with_stroke_width(0.0, 2.0, &RGB::new_red_green_blue(220, 38, 38)?, 1.5)?;
    scatter.title("plot() + abline()")?.x_label("x")?.y_label("y")?;
    scatter.save_svg("rust_rplot_scatter.svg")?;

    // hist(): equal-width binning over the data range.
    let data = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 7.0];
    let histogram = RPlot::hist_with_label(&data, 6, "counts")?;
    histogram.save_svg("rust_rplot_hist.svg")?;

    // barplot(): categorical positions with x tick labels.
    RPlot::barplot_with_names(&[23.0, 41.0, 12.0], &["A", "B", "C"])?.title("barplot()")?.save_svg("rust_rplot_barplot.svg")?;

    // boxplot(): Tukey five-number summary per group.
    let groups = [
        vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0],
        vec![1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 20.0],
    ];
    RPlot::boxplot_with_names(&groups, &["low variance", "has outlier"])?
        .title("boxplot()")?
        .save_svg("rust_rplot_boxplot.svg")?;

    // pie(): wedge areas proportional to value, axes hidden automatically.
    RPlot::pie_with_names(&[35.0, 25.0, 20.0, 20.0], &["Q1", "Q2", "Q3", "Q4"])?.title("pie()")?.save_svg("rust_rplot_pie.svg")?;

    // curve() is omitted to stay in step with the Lua example (Lua cannot supply the callback).

    // qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
    let residuals = [-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3];
    RPlot::qqnorm(&residuals)?.qqline(&residuals)?.save_svg("rust_rplot_qqnorm.svg")?;

    // par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
    RLayout::create(1, 2)?.add(&scatter)?.add(&histogram)?.save_svg("rust_rplot_layout.svg")?;

    println!("Wrote 7 SVGs to rust_rplot_*.svg");
    Ok(())
}
