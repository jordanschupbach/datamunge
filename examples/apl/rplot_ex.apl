⍝ CMD: just TARGET=rplot_ex run-apl
⍝ Demonstrates RPlot / RLayout -- the R-base-graphics-style plotting library.
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
⍝ mkdir -p build/debug/examples (⎕FIO[20] is mkdir; an existing directory just returns an error code).
Z←⎕FIO[20]¨'build' 'build/debug' 'build/debug/examples'
⍝ RPlot's fluent methods (abline, title, x_label, ...) return the plot's own handle; RPlot is a Plot, so
⍝ dm∆Plot∆save_svg saves it.

⍝ plot(x, y, type = "p") then abline() layered on afterward.
RED←dm∆RGB∆new 220 38 38
SCATTER←dm∆RPlot∆plot (1 2 3 4 5) (2.1 3.9 6.2 7.8 10.1) 'p' 'observed'
SCATTER←SCATTER dm∆RPlot∆abline 0 2 RED 1.5
SCATTER←SCATTER dm∆RPlot∆title 'plot() + abline()'
SCATTER←SCATTER dm∆RPlot∆x_label 'x'
SCATTER←SCATTER dm∆RPlot∆y_label 'y'
SCATTER dm∆Plot∆save_svg 'build/debug/examples/apl_rplot_scatter.svg'

⍝ hist(): equal-width binning over the data range.
HIST←dm∆RPlot∆hist (1 2 2 3 3 3 4 4 4 4 5 5 5 6 6 7) 6 'counts'
HIST dm∆Plot∆save_svg 'build/debug/examples/apl_rplot_hist.svg'

⍝ barplot(): categorical positions with x_tick_labels.
BARS←dm∆RPlot∆barplot (23 41 12) (,¨'ABC')
BARS←BARS dm∆RPlot∆title 'barplot()'
BARS dm∆Plot∆save_svg 'build/debug/examples/apl_rplot_barplot.svg'

⍝ boxplot(): Tukey five-number summary per group.
BOX←dm∆RPlot∆boxplot ((2 4 4 4 5 5 7 9) (1 2 2 2 3 3 3 3 4 20)) ('low variance' 'has outlier')
BOX←BOX dm∆RPlot∆title 'boxplot()'
BOX dm∆Plot∆save_svg 'build/debug/examples/apl_rplot_boxplot.svg'

⍝ pie(): wedge areas proportional to value, axes hidden automatically.
PIE←dm∆RPlot∆pie (35 25 20 20) ('Q1' 'Q2' 'Q3' 'Q4')
PIE←PIE dm∆RPlot∆title 'pie()'
PIE dm∆Plot∆save_svg 'build/debug/examples/apl_rplot_pie.svg'

⍝ curve(): samples a Callback -- an APL implementation of its virtual call() (Sine∆call) -- over a range.
∇Z←H Sine∆call X
 Z←1○X
∇
SINE←'Sine' dm∆Callback∆subclass ⍬
CURVE←dm∆RPlot∆curve SINE 0 (○2) 200 'sin(x)'
CURVE←CURVE dm∆RPlot∆title 'curve()'
CURVE dm∆Plot∆save_svg 'build/debug/examples/apl_rplot_curve.svg'

⍝ qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
⍝ (A lone vector argument is enclosed: the functions take the vector of their arguments.)
RESIDUALS←¯2.1 ¯1.3 ¯0.8 ¯0.4 ¯0.1 0.2 0.5 0.9 1.4 2.3
QQ←dm∆RPlot∆qqnorm ⊂RESIDUALS
QQ←QQ dm∆RPlot∆qqline ⊂RESIDUALS
QQ dm∆Plot∆save_svg 'build/debug/examples/apl_rplot_qqnorm.svg'

⍝ par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
LAYOUT←dm∆RLayout∆create 1 2
LAYOUT←LAYOUT dm∆RLayout∆add SCATTER
LAYOUT←LAYOUT dm∆RLayout∆add HIST
LAYOUT dm∆RLayout∆save_svg 'build/debug/examples/apl_rplot_layout.svg'

PUT 'Wrote 8 SVGs to build/debug/examples/apl_rplot_*.svg'
dm∆Delete RED SCATTER HIST BARS BOX PIE SINE CURVE QQ LAYOUT
