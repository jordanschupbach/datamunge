package main

import (
	"datamunge"
	"fmt"
)

func svector(values []string) datamunge.SVector {
	out := datamunge.NewSVector(int64(len(values)))
	for i, v := range values {
		out.Set(i, v)
	}
	return out
}

func main() {
	iris := datamunge.DataFrameIris()
	features := svector([]string{"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"})

	// species labels as an SVector, for the grouped (colored-by-species) scores plot
	n := iris.Nrows()
	speciesT := make([]string, 0, n)
	for i := int64(0); i < n; i++ {
		speciesT = append(speciesT, iris.String_at("Species", i))
	}
	species := svector(speciesT)

	fmt.Println("=================== PCA on iris (scaled) ===================")
	pca := datamunge.NewPCA(iris, features, true, true)
	pca.Print_summary()

	fmt.Println("\nPC1 loadings (which original features drive it):")
	names := pca.Feature_names()
	loadings := pca.Component_loadings(int64(0))
	for i := 0; i < int(names.Size()); i++ {
		fmt.Printf("  %s: %.4f\n", names.Get(i), loadings.Get(i))
	}

	fmt.Println("\nTraining scores as a DataFrame:")
	fmt.Println(pca.Scores_frame().To_string())

	scatter := pca.Plot_scores_grouped(species)
	scatter.Save_svg("pca_iris_scores.svg")
	fmt.Println("\nScores scatter (colored by species) saved as pca_iris_scores.svg")

	scree := pca.Plot_scree()
	scree.Save_svg("pca_iris_scree.svg")
	fmt.Println("Scree plot saved as pca_iris_scree.svg")

	fmt.Println("\n=================== PCA on iris (unscaled) ===================")
	unscaled := datamunge.NewPCA(iris, features, true, false)
	scaledRatio := pca.Explained_variance_ratio()
	unscaledRatio := unscaled.Explained_variance_ratio()
	fmt.Printf(
		"PC1 explains %.2f%% of variance (vs. %.2f%% scaled) -- Sepal.Length's larger raw variance "+
			"dominates the unscaled covariance matrix.\n",
		unscaledRatio.Get(0)*100, scaledRatio.Get(0)*100)
}
