! CMD: just TARGET=elastic_net_iris_ex run-fortran

! Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures.
program elastic_net_iris_ex
use datamunge
use datamunge_data, only: iris
implicit none
character(len=*), parameter :: formula = &
    'Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width'
type(DataFrame) :: frame
type(Ridge) :: ridge_model
type(Lasso) :: lasso_model
type(ElasticNet) :: elastic

frame = iris()
print '(a, i0, a, i0, a)', 'iris: ', frame%nrows(), ' rows x ', frame%ncols(), ' cols'
print '(a)', 'formula: ' // formula // new_line('a')

print '(a)', '=================== Ridge ==================='
ridge_model = Ridge(frame, formula)
call ridge_model%print_summary()

print '(a)', new_line('a') // '=================== Lasso ==================='
lasso_model = Lasso(frame, formula)
call lasso_model%print_summary()

print '(a)', new_line('a') // '================= Elastic Net ================='
! ElasticNet(data, formula, alpha=0.5, lambda, n_lambda, cv_folds, standardize, seed)
elastic = ElasticNet(frame, formula, 0.5d0)
call elastic%print_summary()

print '(a)', new_line('a') // 'Saved figures showing how each model''s coefficients respond to the ' // &
    'regularization strength, and the cross-validation curve used to pick it:'

! The three model types share no base class, so each pair of figures is saved by one helper call.
call save_figures(ridge_model%plot_coefficient_path(), ridge_model%plot_cv_curve(), 'ridge', 'ridge:      ')
call save_figures(lasso_model%plot_coefficient_path(), lasso_model%plot_cv_curve(), 'lasso', 'lasso:      ')
call save_figures(elastic%plot_coefficient_path(), elastic%plot_cv_curve(), 'elasticnet', 'elastic net:')

contains

subroutine save_figures(path_plot, cv_plot, name, label)
    type(RPlot), intent(in) :: path_plot, cv_plot
    character(len=*), intent(in) :: name, label
    call path_plot%save('elastic_net_' // name // '_path.svg')
    call cv_plot%save('elastic_net_' // name // '_cv.svg')
    print '(a)', '  ' // label // ' elastic_net_' // name // '_path.svg, elastic_net_' // name // '_cv.svg'
end subroutine
end program
