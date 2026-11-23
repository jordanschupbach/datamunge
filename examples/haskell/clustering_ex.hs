-- CMD: just TARGET=clustering_ex run-haskell
-- Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN.
module Main (main) where

import Control.Monad (forM_, unless)
import Datamunge
import Numeric (floatToDigits)

-- | For each cluster, prints its size and its majority species (negative labels are noise).
reportPurity :: [String] -> [Int] -> Int -> IO ()
reportPurity species clusterLabels nClusters =
  forM_ [0 .. nClusters - 1] $ \c -> do
    let votes = countInOrder [sp | (label, sp) <- zip clusterLabels species, label == c]
        total = sum (map snd votes)
    unless (total == 0) $ do
      -- MaxBy keeps the first of equal maxima.
      let (bestSpecies, best) = foldl1 (\a b -> if snd b > snd a then b else a) votes
      putStrLn ("  cluster " ++ show c ++ ": " ++ show total ++ " points, majority " ++ bestSpecies
                  ++ " (" ++ show best ++ "/" ++ show total ++ ")")

-- | Each species with its count, in first-seen order (as C#'s Dictionary enumerates them).
countInOrder :: [String] -> [(String, Int)]
countInOrder = foldl' bump []
  where
    bump acc sp = case lookup sp acc of
      Nothing -> acc ++ [(sp, 1)]
      Just _ -> [(s, if s == sp then k + 1 else k) | (s, k) <- acc]

main :: IO ()
main = do
  iris <- dataIris
  n <- dataFrameNrows iris
  species <- mapM (dataFrameStringAt iris "Species") [0 .. n - 1]
  let featureColumns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]

  putStrLn "=================== K-means (k=3) on iris ==================="
  kmeans <- newKMeansWithNClusters iris featureColumns 3
  inertia <- kMeansInertia kmeans
  iterations <- kMeansIterationsUsed kmeans
  putStrLn ("Inertia: " ++ fmt inertia ++ ", iterations (best run): " ++ show iterations)
  kMeansLabels kmeans >>= \labels -> reportPurity species labels 3

  putStrLn "\n=================== Agglomerative clustering on iris ==================="
  forM_ ["ward", "complete", "average"] $ \linkage -> do
    model <- newAgglomerativeClusteringWithLinkage iris featureColumns 3 linkage
    putStrLn ("--- linkage=" ++ linkage ++ " ---")
    agglomerativeClusteringLabels model >>= \labels -> reportPurity species labels 3

  putStrLn "\n--- Re-cutting the ward dendrogram at k=2 without refitting ---"
  wardModel <- newAgglomerativeClusteringWithLinkage iris featureColumns 3 "ward"
  agglomerativeClusteringCut wardModel 2 >>= \labels -> reportPurity species labels 2

  putStrLn "\n=================== DBSCAN on iris ==================="
  dbscan <- newDbscanWithMinSamples iris featureColumns 0.6 5
  nClusters <- dbscanNClusters dbscan
  nNoise <- dbscanNNoise dbscan
  observations <- dbscanObservations dbscan
  putStrLn ("Clusters found: " ++ show nClusters ++ ", noise points: " ++ show nNoise ++ " (of " ++ show observations ++ ")")
  dbscanLabels dbscan >>= \labels -> reportPurity species labels nClusters

-- | A Double as C#'s default ToString() prints it: the shortest round-trip digits, in E notation
-- when the decimal exponent is below -4 or at least 15 (6, 0.1, 1E-05, 1E+15).
fmt :: Double -> String
fmt x
  | isNaN x = "NaN"
  | isInfinite x = if x < 0 then "-∞" else "∞"
  | x < 0 || isNegativeZero x = '-' : fmt (negate x)
  | x == 0 = "0"
  | exp10 < -4 || exp10 >= 15 =
      let (d, rest) = splitAt 1 digits
          magnitude = show (abs exp10)
       in d ++ (if null rest then "" else '.' : rest) ++ "E" ++ (if exp10 < 0 then "-" else "+")
            ++ replicate (2 - length magnitude) '0' ++ magnitude
  | e <= 0 = "0." ++ replicate (negate e) '0' ++ digits
  | e >= length digits = digits ++ replicate (e - length digits) '0'
  | otherwise = let (whole, frac) = splitAt e digits in whole ++ "." ++ frac
  where
    (ds, e) = floatToDigits 10 x
    digits = concatMap show ds
    exp10 = e - 1
