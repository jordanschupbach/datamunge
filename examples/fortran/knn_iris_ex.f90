! CMD: just TARGET=knn_iris_ex run-fortran

! k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for several k.
program knn_iris_ex
use datamunge
use datamunge_data, only: iris
implicit none
type(DataFrame) :: frame, confusion
type(KNNClassifier) :: model, knn
type(RPlot) :: fig
type(rosetta_string), allocatable :: fitted(:)
character(len=:), allocatable :: actual, fname
character(len=2) :: kpad
integer :: i, n, misclassified, j
integer, parameter :: ks(2) = [1, 25]

frame = iris()
print '(a, i0, a, i0, a)', 'iris: ', frame%nrows(), ' rows x ', frame%ncols(), ' cols' // new_line('a')

model = KNNClassifier(frame, 'Species ~ Petal.Length + Petal.Width')
call model%print_summary()

print '(a)', new_line('a') // 'Confusion matrix (leave-one-out; rows = actual, cols = predicted):'
confusion = model%confusion_matrix()
print '(a)', confusion%to_string()

print '(a)', new_line('a') // 'Leave-one-out misclassified rows:'
fitted = model%predict(frame)
n = frame%nrows()
misclassified = 0
do i = 0, n - 1
    actual = frame%string_at('Species', i)
    if (fitted(i + 1)%s == actual) cycle
    misclassified = misclassified + 1
    print '(a)', '  row ' // itoa(i) // ': Petal.Length=' // num(frame%numeric_at('Petal.Length', i)) // &
        ' Petal.Width=' // num(frame%numeric_at('Petal.Width', i)) // '  actual=' // actual // &
        '  predicted=' // fitted(i + 1)%s
end do
print '(a)', itoa(misclassified) // ' of ' // itoa(n) // ' misclassified (' // &
    num(100d0 * misclassified / n) // '%)'

fig = model%plot_decision_regions('Petal.Length', 'Petal.Width')
call fig%save('knn_iris_decision_regions_k5.svg')
print '(a)', new_line('a') // 'Saved knn_iris_decision_regions_k5.svg'

! k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
! every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
! underfit boundary). KNNClassifier(data, formula, k, metric, weighted, standardize)
do j = 1, size(ks)
    knn = KNNClassifier(frame, 'Species ~ Petal.Length + Petal.Width', ks(j))
    fname = 'knn_iris_decision_regions_k' // itoa(ks(j)) // '.svg'
    kpad = itoa(ks(j))
    print '(a)', new_line('a') // 'k=' // kpad // ' leave-one-out accuracy: ' // &
        num(knn%training_accuracy() * 100) // '%'
    fig = knn%plot_decision_regions('Petal.Length', 'Petal.Width')
    call fig%save(fname)
    print '(a)', 'Saved ' // fname
end do

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
