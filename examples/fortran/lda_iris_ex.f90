! CMD: just TARGET=lda_iris_ex run-fortran

! Linear discriminant analysis on iris: summary, confusion matrix, predictions and discriminant plot.
program lda_iris_ex
use datamunge
use datamunge_data
implicit none
type(DataFrame) :: frame, newdata, confusion, predictions
type(LDA) :: model

frame = iris()
print '(a, i0, a, i0, a)', 'iris: ', frame%nrows(), ' rows x ', frame%ncols(), ' cols' // new_line('a')

model = LDA(frame, 'Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width')
call model%print_summary()

confusion = model%confusion_matrix()
print '(a)', new_line('a') // 'Confusion matrix (rows = actual, cols = predicted):' // new_line('a') // confusion%to_string()

newdata = DataFrame()
call newdata%add_numeric_column('Sepal.Length', [5.1d0, 6.0d0, 6.5d0, 6.2d0])
call newdata%add_numeric_column('Sepal.Width', [3.5d0, 2.7d0, 3.0d0, 2.8d0])
call newdata%add_numeric_column('Petal.Length', [1.4d0, 4.5d0, 5.5d0, 4.8d0])
call newdata%add_numeric_column('Petal.Width', [0.2d0, 1.5d0, 2.0d0, 1.8d0])
predictions = model%predict_frame(newdata)
print '(a)', new_line('a') // 'Predictions for new flowers:' // new_line('a') // predictions%to_string()

call model%save_discriminant_plot('lda_iris_discriminants.svg')
print '(a)', new_line('a') // 'Saved discriminant plot as lda_iris_discriminants.svg'
end program
