! CMD: just TARGET=glm_iris_ex run-fortran

! Generalized linear models on iris: logistic regression (binomial) and Poisson regression.
program glm_iris_ex
use datamunge
use datamunge_data, only: iris
implicit none
type(DataFrame) :: frame, sub, count_data
type(GLM) :: logit, poisson_model
real(c_double), allocatable :: is_virginica(:), petal_length(:), petal_width(:), fitted(:)
character(len=:), allocatable :: species
integer :: i, n, m, correct

frame = iris()
n = frame%nrows()

! Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
! the other two on these predictors, which sends logistic regression's coefficients toward
! +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
! binary split for a demo.
allocate(is_virginica(n), petal_length(n), petal_width(n))
m = 0
do i = 0, n - 1
    species = frame%string_at('Species', i)
    if (species /= 'versicolor' .and. species /= 'virginica') cycle
    m = m + 1
    is_virginica(m) = merge(1d0, 0d0, species == 'virginica')
    petal_length(m) = frame%numeric_at('Petal.Length', i)
    petal_width(m) = frame%numeric_at('Petal.Width', i)
end do
is_virginica = is_virginica(1:m)
petal_length = petal_length(1:m)
petal_width = petal_width(1:m)

sub = DataFrame()
call sub%add_numeric_column('Petal.Length', petal_length)
call sub%add_numeric_column('Petal.Width', petal_width)
call sub%add_numeric_column('is_virginica', is_virginica)

print '(a)', '=================== Logistic regression (binomial, logit link) ==================='
logit = GLM(sub, 'is_virginica ~ Petal.Length + Petal.Width', 'binomial')
call logit%print_summary()

fitted = logit%fitted_values()
correct = count((fitted >= 0.5d0) .eqv. (is_virginica >= 0.5d0))
print '(/, a, a, a)', 'Resubstitution accuracy at 0.5 threshold: ', num(100d0 * correct / m), '%'

call logit%save_diagnostic_plots('glm_logistic_iris')
print '(/, a)', 'Saved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg'

! Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
! classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
block
    integer, parameter :: grid_n = 100
    type(DataFrame) :: grid, curve_frame
    real(c_double) :: width_mean, pl_min, pl_max, grid_values(grid_n)
    integer :: j

    width_mean = sum(petal_width) / size(petal_width)
    pl_min = minval(petal_length) - 0.3d0
    pl_max = maxval(petal_length) + 0.3d0
    do j = 0, grid_n - 1
        grid_values(j + 1) = pl_min + (pl_max - pl_min) * j / (grid_n - 1)
    end do
    grid = DataFrame()
    call grid%add_numeric_column('Petal.Length', grid_values)
    call grid%add_numeric_column('Petal.Width', spread(width_mean, 1, grid_n))
    curve_frame = logit%predict_frame(grid, 'confidence')
    print '(/, a)', 'Predicted-probability curve (first 5 rows):'
    print '(a)', curve_frame%to_string(5)
end block

! Poisson regression, for contrast: same IRLS engine, different family/link.
print '(/, a)', '=================== Poisson regression (log link) ==================='
block
    real(c_double), allocatable :: sepal_width(:), all_petal_length(:), counts(:)

    allocate(sepal_width(n), all_petal_length(n), counts(n))
    do i = 0, n - 1
        sepal_width(i + 1) = frame%numeric_at('Sepal.Width', i)
        all_petal_length(i + 1) = frame%numeric_at('Petal.Length', i)
        counts(i + 1) = floor(frame%numeric_at('Sepal.Length', i) + 0.5d0)
    end do
    count_data = DataFrame()
    call count_data%add_numeric_column('Sepal.Width', sepal_width)
    call count_data%add_numeric_column('Petal.Length', all_petal_length)
    call count_data%add_numeric_column('count', counts)
    poisson_model = GLM(count_data, 'count ~ Sepal.Width + Petal.Length', 'poisson')
    call poisson_model%print_summary()
end block
contains

function num(x) result(s)
    ! Shortest round-trip decimal form of x (like Julia/C# default double printing).
    real(c_double), intent(in) :: x
    character(len=:), allocatable :: s
    character(len=40) :: buf, fmt
    real(c_double) :: y
    integer :: k, e
    if (x == 0d0) then
        s = '0.0'
        return
    end if
    do k = 1, 17
        write(fmt, '(a, i0, a)') '(es30.', k, 'e3)'
        write(buf, fmt) x
        read(buf, *) y
        if (y == x) exit
    end do
    read(buf(index(buf, 'E') + 1:), *) e
    if (e >= -5 .and. e < 16) then
        write(fmt, '(a, i0, a)') '(f40.', max(1, k - e), ')'
        write(buf, fmt) x
    end if
    s = trim(adjustl(buf))
    if (index(s, 'E') > 0) then
        write(buf, '(a, i0)') s(:index(s, 'E') - 1) // 'e', e
        s = trim(buf)
    end if
    if (scan(s, 'Ee') == 0) then
        do while (s(len(s):len(s)) == '0' .and. s(len(s) - 1:len(s) - 1) /= '.')
            s = s(:len(s) - 1)
        end do
    end if
end function
end program
