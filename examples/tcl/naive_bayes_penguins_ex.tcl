package require Datamunge 0.0.1

set penguins [datamunge::DataFrame_penguins]
puts "penguins: [datamunge::DataFrame_nrows $penguins] rows x [datamunge::DataFrame_ncols $penguins] cols\n"

set model [datamunge::new_NaiveBayesClassifier $penguins "species ~ bill_length_mm + bill_depth_mm + island + sex"]
datamunge::NaiveBayesClassifier_print_summary $model

puts "\nConfusion matrix (rows = actual, cols = predicted):"
puts "[datamunge::DataFrame_to_string [datamunge::NaiveBayesClassifier_confusion_matrix $model]]"

puts "\nMisclassified rows:"
set predictions [datamunge::NaiveBayesClassifier_predict $model $penguins]
set misclassified 0
set n [datamunge::DataFrame_nrows $penguins]
for {set i 0} {$i < $n} {incr i} {
  if {![datamunge::DataFrame_is_null $penguins "species" $i] &&
      ![datamunge::DataFrame_is_null $penguins "bill_length_mm" $i] &&
      ![datamunge::DataFrame_is_null $penguins "bill_depth_mm" $i] &&
      ![datamunge::DataFrame_is_null $penguins "island" $i] &&
      ![datamunge::DataFrame_is_null $penguins "sex" $i]} {
    set actual [datamunge::DataFrame_string_at $penguins "species" $i]
    set pred [lindex $predictions $i]
    if {$pred ne $actual} {
      incr misclassified
      set bl [datamunge::DataFrame_numeric_at $penguins "bill_length_mm" $i]
      set bd [datamunge::DataFrame_numeric_at $penguins "bill_depth_mm" $i]
      set island [datamunge::DataFrame_string_at $penguins "island" $i]
      set sex [datamunge::DataFrame_string_at $penguins "sex" $i]
      puts "  row $i: bill_length=$bl bill_depth=$bd island=$island sex=$sex  actual=$actual  predicted=$pred"
    }
  }
}
puts "$misclassified misclassified (of $n rows, some incomplete)"

set bill_only [datamunge::new_NaiveBayesClassifier $penguins "species ~ bill_length_mm + bill_depth_mm"]
puts "\nbill-measurements-only model training accuracy: [expr {[datamunge::NaiveBayesClassifier_training_accuracy $bill_only] * 100.0}]%"
datamunge::Plot_save [datamunge::NaiveBayesClassifier_plot_classification $bill_only $penguins "bill_length_mm" "bill_depth_mm"] "naive_bayes_penguins_classification.svg"
datamunge::Plot_save [datamunge::NaiveBayesClassifier_plot_decision_regions $bill_only "bill_length_mm" "bill_depth_mm"] "naive_bayes_penguins_decision_regions.svg"
puts "Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg"
