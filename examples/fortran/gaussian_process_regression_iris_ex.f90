! CMD: just TARGET=gaussian_process_regression_iris_ex run-fortran

! Gaussian process regression of Petal.Length on Petal.Width (iris), with confidence intervals.
program gaussian_process_regression_iris_ex
use datamunge
use datamunge_data
implicit none
character(len=*), parameter :: formula = 'Petal.Length ~ Petal.Width'
type(DataFrame) :: frame, query, detail
type(GaussianProcessRegression) :: model
type(RPlot) :: fit_plot, profile_plot

frame = iris()
print '(a, i0, a, i0, a)', 'iris: ', frame%nrows(), ' rows x ', frame%ncols(), ' cols'
print '(a)', 'formula: ' // formula // new_line('a')

! Both the length scale and the noise ratio are auto-selected by maximizing the exact log
! marginal likelihood.
model = GaussianProcessRegression(frame, formula)
call model%print_summary()

fit_plot = model%plot_fit(frame)
call fit_plot%save('gpr_iris_fit.svg')
profile_plot = model%plot_length_scale_profile()
call profile_plot%save('gpr_iris_length_scale_profile.svg')
print '(a)', new_line('a') // 'Saved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg'

! Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
! interval at every point -- including far outside the training data, where it should widen
! substantially as the model's uncertainty grows.
query = DataFrame()
call query%add_numeric_column('Petal.Width', [0.2d0, 1.3d0, 2.5d0, 10.0d0])
detail = model%predict_frame(query, 'confidence')
print '(a)', new_line('a') // 'Predictions with 95% confidence intervals:'
print '(a)', detail%to_string()
print '(a)', '(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its' &
    // new_line('a') // ' interval is than the in-range predictions.)'
end program
