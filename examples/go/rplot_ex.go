package main

import (
	"datamunge"
	"fmt"
)

func dvector(values []float64) datamunge.DVector {
	out := datamunge.NewDVector(int64(len(values)))
	for i, v := range values {
		out.Set(i, v)
	}
	return out
}

func svector(values []string) datamunge.SVector {
	out := datamunge.NewSVector(int64(len(values)))
	for i, v := range values {
		out.Set(i, v)
	}
	return out
}

func dvectorvector(rows [][]float64) datamunge.DVectorVector {
	out := datamunge.NewDVectorVector(int64(len(rows)))
	for i, row := range rows {
		out.Set(i, dvector(row))
	}
	return out
}

func rgb(r, g, b int) datamunge.RGB {
	c := datamunge.NewRGB()
	c.SetR(r)
	c.SetG(g)
	c.SetB(b)
	return c
}

func main() {
	// plot(x, y, type = "p") then abline() layered on afterward.
	scatter := datamunge.RPlotPlot(dvector([]float64{1.0, 2.0, 3.0, 4.0, 5.0}), dvector([]float64{2.1, 3.9, 6.2, 7.8, 10.1}), "p", "observed")
	scatter.Abline(0.0, 2.0, rgb(220, 38, 38), 1.5)
	scatter.Title("plot() + abline()").X_label("x").Y_label("y")
	scatter.Save_svg("go_rplot_scatter.svg")

	// hist(): equal-width binning over the data range.
	hist := datamunge.RPlotHist(dvector([]float64{1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7}), int64(6), "counts")
	hist.Save_svg("go_rplot_hist.svg")

	// barplot(): categorical positions with x tick labels.
	bars := datamunge.RPlotBarplot(dvector([]float64{23.0, 41.0, 12.0}), svector([]string{"A", "B", "C"}))
	bars.Title("barplot()")
	bars.Save_svg("go_rplot_barplot.svg")

	// boxplot(): Tukey five-number summary per group.
	groups := dvectorvector([][]float64{{2, 4, 4, 4, 5, 5, 7, 9}, {1, 2, 2, 2, 3, 3, 3, 3, 4, 20}})
	box := datamunge.RPlotBoxplot(groups, svector([]string{"low variance", "has outlier"}))
	box.Title("boxplot()")
	box.Save_svg("go_rplot_boxplot.svg")

	// pie(): wedge areas proportional to value, axes hidden automatically.
	pie := datamunge.RPlotPie(dvector([]float64{35.0, 25.0, 20.0, 20.0}), svector([]string{"Q1", "Q2", "Q3", "Q4"}))
	pie.Title("pie()")
	pie.Save_svg("go_rplot_pie.svg")

	// curve() is omitted here: it samples a Callback (a SWIG director), matching the Lua
	// template. Every other RPlot method below is director-free.

	// qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
	residuals := []float64{-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3}
	qq := datamunge.RPlotQqnorm(dvector(residuals))
	qq.Qqline(dvector(residuals))
	qq.Save_svg("go_rplot_qqnorm.svg")

	// par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
	layout := datamunge.RLayoutCreate(int64(1), int64(2))
	layout.Add(scatter)
	layout.Add(hist)
	layout.Save_svg("go_rplot_layout.svg")

	fmt.Println("Wrote 7 SVGs to go_rplot_*.svg")
}
