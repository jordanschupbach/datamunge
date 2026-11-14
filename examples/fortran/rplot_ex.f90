! CMD: just TARGET=rplot_ex run-fortran

! R base-graphics-style plotting with RPlot / RLayout: plot, hist, barplot, boxplot, pie, qqnorm, layouts.
program rplot_ex
use datamunge
implicit none
real(c_double), parameter :: residuals(10) = [-2.1d0, -1.3d0, -0.8d0, -0.4d0, -0.1d0, 0.2d0, 0.5d0, 0.9d0, &
    1.4d0, 2.3d0]
type(RPlot) :: scatter, histogram, bar, boxes, wedges, qq
type(Plot) :: titled
type(RLayout) :: layout
type(vector_real) :: groups(2)

! Fortran cannot call a type-bound procedure on a function result: one variable per step. The
! chained methods return the same C++ object, so `scatter` sees the title and labels.

! plot(x, y, type = "p") then abline() layered on afterward.
scatter = rplot_plot([1d0, 2d0, 3d0, 4d0, 5d0], [2.1d0, 3.9d0, 6.2d0, 7.8d0, 10.1d0], 'p', 'observed')
scatter = scatter%abline(0d0, 2d0, RGB(220, 38, 38), 1.5d0)
titled = scatter%title('plot() + abline()')
titled = titled%x_label('x')
titled = titled%y_label('y')
call scatter%save_svg('fortran_rplot_scatter.svg')

! hist(): equal-width binning over the data range.
histogram = rplot_hist([1d0, 2d0, 2d0, 3d0, 3d0, 3d0, 4d0, 4d0, 4d0, 4d0, 5d0, 5d0, 5d0, 6d0, 6d0, 7d0], 6, 'counts')
call histogram%save_svg('fortran_rplot_hist.svg')

! barplot(): categorical positions with x tick labels.
bar = rplot_barplot([23d0, 41d0, 12d0], ['A', 'B', 'C'])
titled = bar%title('barplot()')
call titled%save_svg('fortran_rplot_barplot.svg')

! boxplot(): Tukey five-number summary per group.
groups(1)%v = [2d0, 4d0, 4d0, 4d0, 5d0, 5d0, 7d0, 9d0]
groups(2)%v = [1d0, 2d0, 2d0, 2d0, 3d0, 3d0, 3d0, 3d0, 4d0, 20d0]
boxes = rplot_boxplot(groups, [character(len=12) :: 'low variance', 'has outlier'])
titled = boxes%title('boxplot()')
call titled%save_svg('fortran_rplot_boxplot.svg')

! pie(): wedge areas proportional to value, axes hidden automatically.
wedges = rplot_pie([35d0, 25d0, 20d0, 20d0], ['Q1', 'Q2', 'Q3', 'Q4'])
titled = wedges%title('pie()')
call titled%save_svg('fortran_rplot_pie.svg')

! curve() is omitted to stay in step with the Lua example (Lua cannot supply the callback).

! qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
qq = rplot_qqnorm(residuals)
qq = qq%qqline(residuals)
call qq%save_svg('fortran_rplot_qqnorm.svg')

! par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
layout = rlayout_create(1, 2)
layout = layout%add(scatter)
layout = layout%add(histogram)
call layout%save_svg('fortran_rplot_layout.svg')

print '(a)', 'Wrote 7 SVGs to fortran_rplot_*.svg'
end program
