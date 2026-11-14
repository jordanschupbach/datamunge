// CMD: just TARGET=pca_ex run-go

//go:build ignore

// PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.
package main

import (
	"datamunge"
	"datamunge/data"
	"fmt"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	iris := datamunge.Must(data.Iris())
	defer iris.Close()
	featureColumns := []string{"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"}

	// species labels, for the grouped (colored-by-species) scores plot
	var species []string
	for i := 0; i < datamunge.Must(iris.Nrows()); i++ {
		species = append(species, datamunge.Must(iris.StringAt("Species", i)))
	}

	fmt.Println("=================== PCA on iris (scaled) ===================")
	pca := datamunge.Must(datamunge.NewPCAWithScale(iris, featureColumns, true, true))
	defer pca.Close()
	check(pca.PrintSummary())

	fmt.Println("\nPC1 loadings (which original features drive it):")
	names := datamunge.Must(pca.FeatureNames())
	loadings := datamunge.Must(pca.ComponentLoadings(0))
	for i, name := range names {
		fmt.Printf("  %s: %.4f\n", name, loadings[i])
	}

	fmt.Println("\nTraining scores as a DataFrame:\n" + datamunge.Must(pca.ScoresFrame()).String())

	check(datamunge.Must(pca.PlotScoresGrouped(species)).SaveSvg("pca_iris_scores.svg"))
	fmt.Println("\nScores scatter (colored by species) saved as pca_iris_scores.svg")

	check(datamunge.Must(pca.PlotScree()).SaveSvg("pca_iris_scree.svg"))
	fmt.Println("Scree plot saved as pca_iris_scree.svg")

	fmt.Println("\n=================== PCA on iris (unscaled) ===================")
	unscaled := datamunge.Must(datamunge.NewPCAWithScale(iris, featureColumns, true, false))
	defer unscaled.Close()
	fmt.Printf("PC1 explains %.2f%% of variance (vs. %.2f%% scaled) -- Sepal.Length's larger raw"+
		" variance dominates the unscaled covariance matrix.\n",
		100*datamunge.Must(unscaled.ExplainedVarianceRatio())[0],
		100*datamunge.Must(pca.ExplainedVarianceRatio())[0])
}
