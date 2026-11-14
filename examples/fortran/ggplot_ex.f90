! CMD: just TARGET=ggplot_ex run-fortran

! Demonstrates GGPlot -- the ggplot2-style grammar-of-graphics library.
program ggplot_ex
use datamunge
use datamunge_data, only: iris
implicit none
type(DataFrame) :: frame
type(GGPlot) :: gp

frame = iris()

! Each verb returns the same plot, so `gp = gp%verb(...)` chains one step at a time.

! ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
gp = GGPlot(frame, 'Sepal.Length', 'Sepal.Width', 'Species')
gp = gp%geom_point()
gp = gp%labs('Iris Sepal Dimensions', 'Sepal Length', 'Sepal Width')
gp = gp%theme_minimal()
call gp%save_svg('fortran_ggplot_point.svg')

! + geom_smooth(): an lm() fit line, reusing stats::LM internally.
gp = GGPlot(frame, 'Sepal.Length', 'Petal.Length')
gp = gp%geom_point(RGB(156, 163, 175), 2.5d0)
gp = gp%geom_smooth()
gp = gp%labs('Petal Length vs Sepal Length With a Linear Fit', 'Sepal Length', 'Petal Length')
call gp%save_svg('fortran_ggplot_smooth.svg')

! geom_bar(): counts a discrete column (stat = "count").
gp = GGPlot(frame, 'Species')
gp = gp%geom_bar()
gp = gp%labs('Observations per Species', 'Species', 'Count')
gp = gp%theme_bw()
call gp%save_svg('fortran_ggplot_bar.svg')

! geom_boxplot(): grouped by a discrete x column.
gp = GGPlot(frame, 'Species', 'Petal.Width')
gp = gp%geom_boxplot()
gp = gp%labs('Petal Width by Species', 'Species', 'Petal Width')
call gp%save_svg('fortran_ggplot_boxplot.svg')

! geom_histogram() + geom_density(): distribution of a single numeric column.
gp = GGPlot(frame, 'Sepal.Length')
gp = gp%geom_histogram(20)
gp = gp%labs('Distribution of Sepal Length', 'Sepal Length', 'Count')
call gp%save_svg('fortran_ggplot_histogram.svg')

gp = GGPlot(frame, 'Sepal.Length')
gp = gp%geom_density()
gp = gp%labs('Density of Sepal Length', 'Sepal Length', 'Density')
call gp%save_svg('fortran_ggplot_density.svg')

! facet_wrap(): one panel per Species, composed via RLayout under the hood.
gp = GGPlot(frame, 'Petal.Length', 'Petal.Width')
gp = gp%geom_point()
gp = gp%facet_wrap('Species')
gp = gp%labs('Petal Dimensions', 'Petal Length', 'Petal Width')
call gp%save_svg('fortran_ggplot_facet.svg')

! theme_classic(): another built-in theme.
gp = GGPlot(frame, 'Sepal.Length', 'Sepal.Width', 'Species')
gp = gp%geom_point()
gp = gp%theme_classic()
call gp%save_svg('fortran_ggplot_classic.svg')

print '(a)', 'Wrote 7 SVGs to fortran_ggplot_*.svg'
end program
