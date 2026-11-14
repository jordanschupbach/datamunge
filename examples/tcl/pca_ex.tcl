# CMD: just TARGET=pca_ex run-tcl
# PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.
package require datamunge

set iris [datamunge::Data::iris]
set feature_columns {Sepal.Length Sepal.Width Petal.Length Petal.Width}

# species labels, for the grouped (colored-by-species) scores plot
set species {}
for {set i 0} {$i < [$iris nrows]} {incr i} {
    lappend species [$iris string_at Species $i]
}

puts "=================== PCA on iris (scaled) ==================="
set pca [datamunge::PCA new $iris $feature_columns true true]
$pca print_summary

puts "\nPC1 loadings (which original features drive it):"
foreach name [$pca feature_names] loading [$pca component_loadings 0] {
    puts [format "  %s: %.4f" $name $loading]
}

puts "\nTraining scores as a DataFrame:\n[[$pca scores_frame] to_string]"

[$pca plot_scores_grouped $species] save_svg pca_iris_scores.svg
puts "\nScores scatter (colored by species) saved as pca_iris_scores.svg"

[$pca plot_scree] save_svg pca_iris_scree.svg
puts "Scree plot saved as pca_iris_scree.svg"

puts "\n=================== PCA on iris (unscaled) ==================="
set unscaled [datamunge::PCA new $iris $feature_columns true false]
puts [format "PC1 explains %.2f%% of variance (vs. %.2f%% scaled) -- Sepal.Length's larger raw variance\
    dominates the unscaled covariance matrix." \
    [expr {100 * [lindex [$unscaled explained_variance_ratio] 0]}] \
    [expr {100 * [lindex [$pca explained_variance_ratio] 0]}]]
