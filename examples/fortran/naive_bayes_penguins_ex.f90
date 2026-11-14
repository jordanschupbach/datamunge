! CMD: just TARGET=naive_bayes_penguins_ex run-fortran

! Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots.
program naive_bayes_penguins_ex
use datamunge
use datamunge_data, only: penguins
implicit none
character(len=14), parameter :: columns(5) = [character(len=14) :: 'species', 'bill_length_mm', 'bill_depth_mm', &
    'island', 'sex']
type(DataFrame) :: frame, confusion
type(NaiveBayesClassifier) :: model, bill_only
type(RPlot) :: fig
type(rosetta_string), allocatable :: predictions(:)
character(len=:), allocatable :: actual
integer :: i, j, n, misclassified
logical :: incomplete

! Note: NaiveBayesClassifier drops incomplete rows for its own formula columns internally
! when fitting, matching every other formula-based model in this library.
frame = penguins()
print '(a, i0, a, i0, a)', 'penguins: ', frame%nrows(), ' rows x ', frame%ncols(), ' cols' // new_line('a')

! A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
! predictors in one formula -- each modeled independently given the class, per the naive
! Bayes assumption.
model = NaiveBayesClassifier(frame, 'species ~ bill_length_mm + bill_depth_mm + island + sex')
call model%print_summary()

confusion = model%confusion_matrix()
print '(a)', new_line('a') // 'Confusion matrix (rows = actual, cols = predicted):' // new_line('a') // &
    confusion%to_string()

print '(a)', new_line('a') // 'Misclassified rows:'
predictions = model%predict(frame)
n = frame%nrows()
misclassified = 0
do i = 0, n - 1
    incomplete = .false.
    do j = 1, size(columns)
        incomplete = incomplete .or. frame%is_null(trim(columns(j)), i)
    end do
    if (incomplete) cycle
    actual = frame%string_at('species', i)
    if (predictions(i + 1)%s == actual) cycle
    misclassified = misclassified + 1
    print '(a)', '  row ' // itoa(i) // ': bill_length=' // num(frame%numeric_at('bill_length_mm', i)) // &
        ' bill_depth=' // num(frame%numeric_at('bill_depth_mm', i)) // ' island=' // &
        frame%string_at('island', i) // ' sex=' // frame%string_at('sex', i) // '  actual=' // actual // &
        '  predicted=' // predictions(i + 1)%s
end do
print '(a)', itoa(misclassified) // ' misclassified (of ' // itoa(n) // ' rows, some incomplete)'

! PlotDecisionRegions requires exactly two NUMERIC predictors, so build a separate
! two-predictor model (bill measurements alone) just for visualization.
bill_only = NaiveBayesClassifier(frame, 'species ~ bill_length_mm + bill_depth_mm')
print '(a)', new_line('a') // 'bill-measurements-only model training accuracy: ' // &
    num(100 * bill_only%training_accuracy()) // '%'
fig = bill_only%plot_classification(frame, 'bill_length_mm', 'bill_depth_mm')
call fig%save('naive_bayes_penguins_classification.svg')
fig = bill_only%plot_decision_regions('bill_length_mm', 'bill_depth_mm')
call fig%save('naive_bayes_penguins_decision_regions.svg')
print '(a)', 'Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg'

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
