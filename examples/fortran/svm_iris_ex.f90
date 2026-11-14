! CMD: just TARGET=svm_iris_ex run-fortran

! Support vector machine on iris: RBF vs. linear kernels, confusion matrix and predictions.
program svm_iris_ex
use datamunge
use datamunge_data
implicit none
character(len=*), parameter :: formula = 'Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width'
type(DataFrame) :: frame, newdata, confusion, predictions
type(SVM) :: rbf_model, linear_model

frame = iris()
print '(a, i0, a, i0, a)', 'iris: ', frame%nrows(), ' rows x ', frame%ncols(), ' cols' // new_line('a')

print '(a)', '=== RBF kernel (default) ==='
rbf_model = SVM(frame, formula)
call rbf_model%print_summary()
confusion = rbf_model%confusion_matrix()
print '(a)', new_line('a') // 'Confusion matrix (rows = actual, cols = predicted):' // new_line('a') // confusion%to_string()

print '(a)', new_line('a') // '=== Linear kernel, for comparison ==='
linear_model = SVM(frame, formula, 'linear')
print '(a)', 'Training accuracy: ' // num(linear_model%training_accuracy() * 100) // '%'
print '(a, i0)', 'Support vectors: ', linear_model%num_support_vectors()

newdata = DataFrame()
call newdata%add_numeric_column('Sepal.Length', [5.1d0, 6.0d0, 6.5d0, 6.2d0])
call newdata%add_numeric_column('Sepal.Width', [3.5d0, 2.7d0, 3.0d0, 2.8d0])
call newdata%add_numeric_column('Petal.Length', [1.4d0, 4.5d0, 5.5d0, 4.8d0])
call newdata%add_numeric_column('Petal.Width', [0.2d0, 1.5d0, 2.0d0, 1.8d0])
predictions = rbf_model%predict_frame(newdata)
print '(a)', new_line('a') // 'RBF predictions for new flowers (votes out of 3 one-vs-one pairs):' // new_line('a') &
    // predictions%to_string()
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
