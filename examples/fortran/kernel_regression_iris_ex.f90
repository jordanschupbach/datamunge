! CMD: just TARGET=kernel_regression_iris_ex run-fortran

! Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones.
program kernel_regression_iris_ex
use datamunge
use datamunge_data, only: iris
implicit none
character(len=*), parameter :: formula = 'Petal.Length ~ Petal.Width'
type(DataFrame) :: frame
type(KernelRegression) :: model, small, large
type(RPlot) :: fig

frame = iris()
print '(a, i0, a, i0, a)', 'iris: ', frame%nrows(), ' rows x ', frame%ncols(), ' cols'
print '(a)', 'formula: ' // formula // new_line('a')

model = KernelRegression(frame, formula)  ! bandwidth auto-selected via leave-one-out CV
call model%print_summary()

fig = model%plot_fit(frame)
call fig%save('kernel_regression_iris_fit.svg')
fig = model%plot_cv_curve()
call fig%save('kernel_regression_iris_cv.svg')
print '(a)', new_line('a') // 'Saved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg'

! A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
! bias-variance story. KernelRegression(data, formula, kernel, bandwidth, n_bandwidth, standardize)
small = KernelRegression(frame, formula, 'gaussian', 0.05d0)
print '(a)', new_line('a') // 'bandwidth=0.05 (too small): LOO R-squared=' // num(small%r_squared()) // &
    '  LOO RMSE=' // num(small%rmse())
fig = small%plot_fit(frame)
call fig%save('kernel_regression_iris_fit_small_bandwidth.svg')

large = KernelRegression(frame, formula, 'gaussian', 5d0)
print '(a)', 'bandwidth=5.0 (too large):  LOO R-squared=' // num(large%r_squared()) // &
    '  LOO RMSE=' // num(large%rmse())
fig = large%plot_fit(frame)
call fig%save('kernel_regression_iris_fit_large_bandwidth.svg')

print '(a)', 'bandwidth=' // num(model%bandwidth()) // ' (CV-selected): LOO R-squared=' // &
    num(model%r_squared()) // '  LOO RMSE=' // num(model%rmse())
print '(a)', new_line('a') // &
    'Saved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg'

contains

! Shortest round-trip decimal text for x (like C#'s/Julia's default double printing).
function num(x) result(s)
    real(c_double), intent(in) :: x
    character(len=:), allocatable :: s
    character(len=48) :: buf, fmt
    real(c_double) :: y
    integer :: d, e, k
    if (x == aint(x) .and. abs(x) < 1d15) then
        write(buf, '(i0)') int(x, c_int64_t)
        s = trim(buf)
        return
    end if
    do d = 2, 17
        write(fmt, '(a, i0, a, i0, a)') '(es', d + 10, '.', d - 1, 'e3)'
        write(buf, fmt) x
        read(buf, *) y
        if (y == x) exit
    end do
    read(buf(index(buf, 'E') + 1:), *) e
    if (e < -5 .or. e >= 15) then
        s = trim(adjustl(buf))
        write(buf, '(a, i0)') s(:index(s, 'E') - 1) // 'e', e
        s = trim(buf)
        return
    end if
    write(fmt, '(a, i0, a)') '(f40.', max(d - 1 - e, 1), ')'
    write(buf, fmt) x
    s = trim(adjustl(buf))
    k = len(s)
    do while (s(k:k) == '0' .and. s(k - 1:k - 1) /= '.')
        k = k - 1
    end do
    s = s(1:k)
end function
end program
