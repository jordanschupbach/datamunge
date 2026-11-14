# CMD: just TARGET=naive_bayes_penguins_ex run-tcl
# Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots.
package require datamunge

# Note: NaiveBayesClassifier drops incomplete rows for its own formula columns internally
# when fitting, matching every other formula-based model in this library.
set penguins [datamunge::Data::penguins]
puts "penguins: [$penguins nrows] rows x [$penguins ncols] cols\n"

# A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
# predictors in one formula -- each modeled independently given the class, per the naive
# Bayes assumption.
set model [datamunge::NaiveBayesClassifier new $penguins \
    "species ~ bill_length_mm + bill_depth_mm + island + sex"]
$model print_summary

puts "\nConfusion matrix (rows = actual, cols = predicted):\n[[$model confusion_matrix] to_string]"

puts "\nMisclassified rows:"
set predictions [$model predict $penguins]
set misclassified 0
set columns {species bill_length_mm bill_depth_mm island sex}
for {set i 0} {$i < [$penguins nrows]} {incr i} {
    set incomplete 0
    foreach column $columns {
        if {[$penguins is_null $column $i]} {
            set incomplete 1
            break
        }
    }
    if {$incomplete} {
        continue
    }
    set actual [$penguins string_at species $i]
    set predicted [lindex $predictions $i]
    if {$predicted eq $actual} {
        continue
    }
    incr misclassified
    set row "  row $i: bill_length=[$penguins numeric_at bill_length_mm $i]"
    append row " bill_depth=[$penguins numeric_at bill_depth_mm $i]"
    append row " island=[$penguins string_at island $i] sex=[$penguins string_at sex $i]"
    puts "$row  actual=$actual  predicted=$predicted"
}
puts "$misclassified misclassified (of [$penguins nrows] rows, some incomplete)"

# plot_decision_regions requires exactly two NUMERIC predictors, so build a separate
# two-predictor model (bill measurements alone) just for visualization.
set bill_only [datamunge::NaiveBayesClassifier new $penguins "species ~ bill_length_mm + bill_depth_mm"]
puts "\nbill-measurements-only model training accuracy: [expr {100 * [$bill_only training_accuracy]}]%"
[$bill_only plot_classification $penguins bill_length_mm bill_depth_mm] save naive_bayes_penguins_classification.svg
[$bill_only plot_decision_regions bill_length_mm bill_depth_mm] save naive_bayes_penguins_decision_regions.svg
puts "Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg"
