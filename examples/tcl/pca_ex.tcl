package require Datamunge 0.0.1

# Tcl's overload dispatch can't distinguish a vector<string> argument from a std::string one (both
# are just Tcl_Obj* strings at the C API level), so PCA's overloaded constructor always resolves to
# the encoded-string form. Route around it exactly like clustering_ex.tcl: join the column names
# with a unit-separator (\x1f, decimal 31); the facade's split_encoded_strings() decodes that.
proc encode_cols {names} {
  return [join $names [format %c 31]]
}

set iris [datamunge::DataFrame_iris]
set features [encode_cols {Sepal.Length Sepal.Width Petal.Length Petal.Width}]

# species labels as a plain list, for the grouped (colored-by-species) scores plot
set n [datamunge::DataFrame_nrows $iris]
set species {}
for {set i 0} {$i < $n} {incr i} {
  lappend species [datamunge::DataFrame_string_at $iris "Species" $i]
}

puts "=================== PCA on iris (scaled) ==================="
set pca [datamunge::new_PCA $iris $features 1 1]
datamunge::PCA_print_summary $pca

puts "\nPC1 loadings (which original features drive it):"
# The binding's specialize_std_vector out-typemaps convert every std::vector return into a plain
# Tcl list, so iterate these with foreach/lindex (not SVector_get/DVector_get, which are for
# explicitly constructed new_SVector/new_DVector objects).
set names [datamunge::PCA_feature_names $pca]
set loadings [datamunge::PCA_component_loadings $pca 0]
foreach nm $names ld $loadings {
  puts [format "  %s: %.4f" $nm $ld]
}

puts "\nTraining scores as a DataFrame:"
puts [datamunge::DataFrame_to_string [datamunge::PCA_scores_frame $pca]]

set scatter [datamunge::PCA_plot_scores_grouped $pca $species]
datamunge::Plot_save_svg $scatter "pca_iris_scores.svg"
puts "\nScores scatter (colored by species) saved as pca_iris_scores.svg"

set scree [datamunge::PCA_plot_scree $pca]
datamunge::Plot_save_svg $scree "pca_iris_scree.svg"
puts "Scree plot saved as pca_iris_scree.svg"

puts "\n=================== PCA on iris (unscaled) ==================="
set unscaled [datamunge::new_PCA $iris $features 1 0]
set scaled_ratio [datamunge::PCA_explained_variance_ratio $pca]
set unscaled_ratio [datamunge::PCA_explained_variance_ratio $unscaled]
puts [format "PC1 explains %.2f%% of variance (vs. %.2f%% scaled) -- Sepal.Length's larger raw variance dominates the unscaled covariance matrix." \
  [expr {[lindex $unscaled_ratio 0] * 100}] \
  [expr {[lindex $scaled_ratio 0] * 100}]]
