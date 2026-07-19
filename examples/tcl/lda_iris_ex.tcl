package require Datamunge 0.0.1

set iris [datamunge::DataFrame_iris]
puts "iris: [datamunge::DataFrame_nrows $iris] rows x [datamunge::DataFrame_ncols $iris] cols\n"

set model [datamunge::new_LDA $iris "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width"]
datamunge::LDA_print_summary $model

puts "\nConfusion matrix (rows = actual, cols = predicted):"
puts "[datamunge::DataFrame_to_string [datamunge::LDA_confusion_matrix $model]]"

set newdata [datamunge::DataFrame_empty]
datamunge::DataFrame_add_numeric_column $newdata "Sepal.Length" {5.1 6.0 6.5 6.2}
datamunge::DataFrame_add_numeric_column $newdata "Sepal.Width" {3.5 2.7 3.0 2.8}
datamunge::DataFrame_add_numeric_column $newdata "Petal.Length" {1.4 4.5 5.5 4.8}
datamunge::DataFrame_add_numeric_column $newdata "Petal.Width" {0.2 1.5 2.0 1.8}

puts "\nPredictions for new flowers:"
puts "[datamunge::DataFrame_to_string [datamunge::LDA_predict_frame $model $newdata]]"

datamunge::LDA_save_discriminant_plot $model "lda_iris_discriminants.svg"
puts "\nSaved discriminant plot as lda_iris_discriminants.svg"
