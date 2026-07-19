use strict;
use warnings;

use Datamunge;

# Note: unlike the C++ example, this doesn't call drop_nulls() first (that method isn't
# exposed on the SWIG-bound DataFrame facade) -- NaiveBayesClassifier drops incomplete rows
# for its own formula columns internally when fitting, matching every other formula-based
# model in this library.
my $penguins = Datamunge::DataFrame::penguins();
print "penguins: " . $penguins->nrows() . " rows x " . $penguins->ncols() . " cols\n\n";

# A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
# predictors in one formula -- each modeled independently given the class, per the naive
# Bayes assumption.
my $model = Datamunge::NaiveBayesClassifier->new($penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex");
$model->print_summary();

print "\nConfusion matrix (rows = actual, cols = predicted):\n";
print $model->confusion_matrix()->to_string(), "\n";

print "\nMisclassified rows:\n";
my $predictions = $model->predict($penguins);
my $misclassified = 0;
for (my $i = 0; $i < $penguins->nrows(); $i++) {
  next if $penguins->is_null("species", $i) || $penguins->is_null("bill_length_mm", $i) ||
          $penguins->is_null("bill_depth_mm", $i) || $penguins->is_null("island", $i) || $penguins->is_null("sex", $i);
  my $actual = $penguins->string_at("species", $i);
  next if $predictions->[$i] eq $actual;
  $misclassified++;
  print "  row $i: bill_length=" . $penguins->numeric_at("bill_length_mm", $i) .
        " bill_depth=" . $penguins->numeric_at("bill_depth_mm", $i) .
        " island=" . $penguins->string_at("island", $i) .
        " sex=" . $penguins->string_at("sex", $i) .
        "  actual=$actual  predicted=" . $predictions->[$i] . "\n";
}
print "$misclassified misclassified (of " . $penguins->nrows() . " rows, some incomplete)\n";

# plot_decision_regions requires exactly two NUMERIC predictors, so build a separate
# two-predictor model (bill measurements alone) just for visualization.
my $bill_only = Datamunge::NaiveBayesClassifier->new($penguins, "species ~ bill_length_mm + bill_depth_mm");
print "\nbill-measurements-only model training accuracy: " . ($bill_only->training_accuracy() * 100.0) . "%\n";
$bill_only->plot_classification($penguins, "bill_length_mm", "bill_depth_mm")->save(
  "naive_bayes_penguins_classification.svg");
$bill_only->plot_decision_regions("bill_length_mm", "bill_depth_mm")->save(
  "naive_bayes_penguins_decision_regions.svg");
print "Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg\n";
