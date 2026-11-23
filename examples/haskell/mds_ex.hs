-- CMD: just TARGET=mds_ex run-haskell
-- Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.
module Main (main) where

import Datamunge

main :: IO ()
main = do
  iris <- dataIris
  let featureColumns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
  n <- dataFrameNrows iris
  species <- mapM (dataFrameStringAt iris "Species") [0 .. n - 1]

  putStrLn "=================== Classical MDS on iris (euclidean) ==================="
  mds <- newMdsWithMetric iris featureColumns 2 "euclidean"
  mdsPrintSummary mds

  embedding <- mdsEmbeddingFrame mds >>= dataFrameToString
  putStrLn ("\nEmbedding as a DataFrame:\n" ++ embedding)

  mdsPlotEmbeddingGrouped mds species >>= (`plotSaveSvg` "mds_iris_embedding.svg")
  putStrLn "\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg"

  putStrLn "\n=================== Classical MDS on iris (manhattan) ==================="
  manhattanMds <- newMdsWithMetric iris featureColumns 2 "manhattan"
  euclideanFit <- mdsGoodnessOfFit mds
  manhattanFit <- mdsGoodnessOfFit manhattanMds
  putStrLn ("Goodness of fit: euclidean=" ++ fixed 4 euclideanFit ++ ", manhattan=" ++ fixed 4 manhattanFit)

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
