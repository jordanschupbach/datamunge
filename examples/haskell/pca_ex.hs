-- CMD: just TARGET=pca_ex run-haskell
-- PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.
module Main (main) where

import Control.Monad (forM_)
import Datamunge

main :: IO ()
main = do
  iris <- dataIris
  let featureColumns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]

  -- species labels, for the grouped (colored-by-species) scores plot
  n <- dataFrameNrows iris
  species <- mapM (dataFrameStringAt iris "Species") [0 .. n - 1]

  putStrLn "=================== PCA on iris (scaled) ==================="
  pca <- newPcaWithScale iris featureColumns True True
  pcaPrintSummary pca

  putStrLn "\nPC1 loadings (which original features drive it):"
  names <- pcaFeatureNames pca
  loadings <- pcaComponentLoadings pca 0
  forM_ (zip names loadings) $ \(name, loading) ->
    putStrLn ("  " ++ name ++ ": " ++ fixed 4 loading)

  scores <- pcaScoresFrame pca >>= dataFrameToString
  putStrLn ("\nTraining scores as a DataFrame:\n" ++ scores)

  pcaPlotScoresGrouped pca species >>= (`plotSaveSvg` "pca_iris_scores.svg")
  putStrLn "\nScores scatter (colored by species) saved as pca_iris_scores.svg"

  pcaPlotScree pca >>= (`plotSaveSvg` "pca_iris_scree.svg")
  putStrLn "Scree plot saved as pca_iris_scree.svg"

  putStrLn "\n=================== PCA on iris (unscaled) ==================="
  unscaled <- newPcaWithScale iris featureColumns True False
  unscaledRatio <- (!! 0) <$> pcaExplainedVarianceRatio unscaled
  scaledRatio <- (!! 0) <$> pcaExplainedVarianceRatio pca
  putStrLn ("PC1 explains " ++ fixed 2 (100 * unscaledRatio) ++ "% of variance (vs. " ++ fixed 2 (100 * scaledRatio)
              ++ "% scaled) -- Sepal.Length's larger raw variance dominates the unscaled covariance matrix.")

-- | C#'s ToString("F<n>"): the exact value rounded to n decimals, midpoints away from zero.
fixed :: Int -> Double -> String
fixed n x
  | isNaN x = "NaN"
  | isInfinite x = if x < 0 then "-∞" else "∞"
  | otherwise = sign ++ whole ++ (if n > 0 then '.' : frac else "")
  where
    scaled = floor (abs (toRational x) * 10 ^ n + 1 / 2) :: Integer
    sign = if x < 0 || isNegativeZero x then "-" else ""
    padded = let s = show scaled in replicate (n + 1 - length s) '0' ++ s
    (whole, frac) = splitAt (length padded - n) padded
