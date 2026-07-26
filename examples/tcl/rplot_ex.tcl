package require Datamunge 0.0.1

# Demonstrates datamunge::RPlot / datamunge::RLayout -- the R-base-graphics-style plotting library.

proc rgb {r g b} {
  set c [datamunge::new_RGB]
  datamunge::RGB_r_set $c $r
  datamunge::RGB_g_set $c $g
  datamunge::RGB_b_set $c $b
  return $c
}

# Nested vector<vector<double>> (DVectorVector) parameters do NOT auto-convert from plain Tcl
# lists (only scalar-element vector<T> does) -- build a real DVectorVector of real DVector rows.
proc dvec {values} {
  set v [datamunge::new_DVector]
  foreach x $values { datamunge::DVector_push_back $v $x }
  return $v
}
proc dvv {rows} {
  set m [datamunge::new_DVectorVector]
  foreach r $rows { datamunge::DVectorVector_push_back $m [dvec $r] }
  return $m
}

# plot(x, y, type = "p") then abline() layered on afterward.
set scatter [datamunge::RPlot_plot {1.0 2.0 3.0 4.0 5.0} {2.1 3.9 6.2 7.8 10.1} "p" "observed"]
datamunge::RPlot_abline $scatter 0.0 2.0 [rgb 220 38 38] 1.5
datamunge::Plot_title $scatter "plot() + abline()"
datamunge::Plot_x_label $scatter "x"
datamunge::Plot_y_label $scatter "y"
datamunge::Plot_save_svg $scatter "tcl_rplot_scatter.svg"

# hist(): equal-width binning over the data range.
set hist [datamunge::RPlot_hist {1 2 2 3 3 3 4 4 4 4 5 5 5 6 6 7} 6 "counts"]
datamunge::Plot_save_svg $hist "tcl_rplot_hist.svg"

# barplot(): categorical positions with x tick labels.
set bars [datamunge::RPlot_barplot {23.0 41.0 12.0} {A B C}]
datamunge::Plot_title $bars "barplot()"
datamunge::Plot_save_svg $bars "tcl_rplot_barplot.svg"

# boxplot(): Tukey five-number summary per group.
set groups [dvv {{2 4 4 4 5 5 7 9} {1 2 2 2 3 3 3 3 4 20}}]
set box [datamunge::RPlot_boxplot $groups {"low variance" "has outlier"}]
datamunge::Plot_title $box "boxplot()"
datamunge::Plot_save_svg $box "tcl_rplot_boxplot.svg"

# pie(): wedge areas proportional to value, axes hidden automatically.
set pie [datamunge::RPlot_pie {35.0 25.0 20.0 20.0} {Q1 Q2 Q3 Q4}]
datamunge::Plot_title $pie "pie()"
datamunge::Plot_save_svg $pie "tcl_rplot_pie.svg"

# curve() is omitted here: it samples a Callback (a SWIG director), which stock SWIG's Tcl backend
# cannot subclass. Every other RPlot method below is director-free.

# qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
set residuals {-2.1 -1.3 -0.8 -0.4 -0.1 0.2 0.5 0.9 1.4 2.3}
set qq [datamunge::RPlot_qqnorm $residuals]
datamunge::RPlot_qqline $qq $residuals
datamunge::Plot_save_svg $qq "tcl_rplot_qqnorm.svg"

# par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
set layout [datamunge::RLayout_create 1 2]
datamunge::RLayout_add $layout $scatter
datamunge::RLayout_add $layout $hist
datamunge::RLayout_save_svg $layout "tcl_rplot_layout.svg"

puts "Wrote 7 SVGs to tcl_rplot_*.svg"
