! CMD: just TARGET=xgboost_iris_ex run-fortran

! XGBoost classifier on iris: summary, confusion matrix, misclassified rows, L2-regularization comparison.
program xgboost_iris_ex
use datamunge
use datamunge_data, only: iris
implicit none
type(DataFrame) :: frame, confusion
type(XGBoostClassifier) :: model, heavy
type(RPlot) :: fig
type(rosetta_string), allocatable :: predictions(:)
real(c_double), allocatable :: deviance(:), heavy_deviance(:)
character(len=:), allocatable :: actual
integer :: i, n, misclassified

frame = iris()
print '(a, i0, a, i0, a)', 'iris: ', frame%nrows(), ' rows x ', frame%ncols(), ' cols' // new_line('a')

model = XGBoostClassifier(frame, 'Species ~ Petal.Length + Petal.Width')
call model%print_summary()

confusion = model%confusion_matrix()
print '(a)', new_line('a') // 'Confusion matrix (rows = actual, cols = predicted):' // new_line('a') // &
    confusion%to_string()

print '(a)', new_line('a') // 'Misclassified rows:'
predictions = model%predict(frame)
n = frame%nrows()
misclassified = 0
do i = 0, n - 1
    actual = frame%string_at('Species', i)
    if (predictions(i + 1)%s == actual) cycle
    misclassified = misclassified + 1
    print '(a)', '  row ' // itoa(i) // ': Petal.Length=' // num(frame%numeric_at('Petal.Length', i)) // &
        ' Petal.Width=' // num(frame%numeric_at('Petal.Width', i)) // '  actual=' // actual // &
        '  predicted=' // predictions(i + 1)%s
end do
print '(a)', itoa(misclassified) // ' of ' // itoa(n) // ' misclassified (' // &
    num(100d0 * misclassified / n) // '%)'

fig = model%plot_training_deviance()
call fig%save('xgboost_iris_training_deviance.svg')
fig = model%plot_decision_regions('Petal.Length', 'Petal.Width')
call fig%save('xgboost_iris_decision_regions.svg')
print '(a)', new_line('a') // 'Saved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg'

! XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
! grows the same deep trees but keeps every leaf weight small, producing a much smoother
! decision boundary than the lightly-regularized default.
! XGBoostClassifier(data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma,
!                   min_child_weight, min_samples_leaf, subsample, colsample_bytree, seed)
heavy = XGBoostClassifier(frame, 'Species ~ Petal.Length + Petal.Width', 100, 0.3d0, 6, 50d0)
deviance = model%training_deviance()
heavy_deviance = heavy%training_deviance()
print '(a)', new_line('a') // 'lambda=1 (default):   training accuracy=' // num(100 * model%training_accuracy()) // &
    '%  deviance=' // num(deviance(size(deviance)))
print '(a)', 'lambda=50 (heavy L2): training accuracy=' // num(100 * heavy%training_accuracy()) // &
    '%  deviance=' // num(heavy_deviance(size(heavy_deviance)))
fig = heavy%plot_decision_regions('Petal.Length', 'Petal.Width')
call fig%save('xgboost_iris_decision_regions_heavy_lambda.svg')
print '(a)', 'Saved xgboost_iris_decision_regions_heavy_lambda.svg'

contains

function itoa(i) result(s)
    integer, intent(in) :: i
    character(len=:), allocatable :: s
    character(len=24) :: buf
    write(buf, '(i0)') i
    s = trim(buf)
end function

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
