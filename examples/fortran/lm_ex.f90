! CMD: just TARGET=lm_ex run-fortran

! Linear model: mpg ~ hp + wt + transmission, its ANOVA table and predictions.
program lm_ex
use datamunge
implicit none
type(DataFrame) :: cars, newcars, anova_table, predictions
type(LM) :: model

cars = DataFrame()
call cars%add_numeric_column('hp', [110d0, 110d0, 93d0, 110d0, 175d0, 105d0, 245d0, 62d0, 95d0, 123d0])
call cars%add_numeric_column('wt', [2.62d0, 2.875d0, 2.32d0, 3.215d0, 3.44d0, 3.46d0, 3.57d0, 3.19d0, 3.15d0, 3.44d0])
call cars%add_string_column('transmission', [character(len=9) :: 'manual', 'manual', 'manual', 'automatic', &
    'automatic', 'automatic', 'automatic', 'automatic', 'automatic', 'automatic'])
call cars%add_numeric_column('mpg', [21.0d0, 21.0d0, 22.8d0, 21.4d0, 18.7d0, 18.1d0, 14.3d0, 24.4d0, 22.8d0, 19.2d0])

print '(a)', 'Fitting: mpg ~ hp + wt + transmission' // new_line('a')
model = LM(cars, 'mpg ~ hp + wt + transmission')
call model%print_summary()

anova_table = model%anova()
print '(a)', new_line('a') // 'Sequential ANOVA:' // new_line('a') // anova_table%to_string()

newcars = DataFrame()
call newcars%add_numeric_column('hp', [150d0, 90d0])
call newcars%add_numeric_column('wt', [3.0d0, 2.5d0])
call newcars%add_string_column('transmission', [character(len=9) :: 'manual', 'automatic'])
predictions = model%predict_frame(newcars, 'confidence')
print '(a)', new_line('a') // 'Predictions with 95% confidence intervals:' // new_line('a') // predictions%to_string()

call model%save_diagnostic_plots('lm_ex_diagnostics')
print '(a)', new_line('a') // 'Saved diagnostic plots as lm_ex_diagnostics_*.svg'
end program
