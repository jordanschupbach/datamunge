-- CMD: just TARGET=tensor_ex run-haskell
-- Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.
module Main (main) where

import Data.List (intercalate)
import Datamunge
import Numeric (floatToDigits)

shapeString :: Tensor -> IO String
shapeString tensor = do
  shape <- tensorShape tensor
  pure ("[" ++ intercalate ", " (map show shape) ++ "]")

-- | Prints a label followed by a tensor's text.
printTensor :: String -> Tensor -> IO ()
printTensor label tensor = tensorToString tensor >>= \text -> putStrLn (label ++ text)

main :: IO ()
main = do
  putStrLn "=================== Construction ==================="
  tensorZeros [2, 3] >>= printTensor "zeros([2,3]): "
  tensorEye 3 >>= printTensor "eye(3): "

  r <- tensorArangeWithStep 0.0 12.0 1.0 >>= (`tensorReshape` [3, 4])
  printTensor "arange(0,12).reshape([3,4]): " r

  putStrLn "\n=================== Shape ops ==================="
  transposedShape <- tensorTranspose r >>= shapeString
  putStrLn ("transpose -> shape " ++ transposedShape)
  tensorSlice r 1 1 3 >>= printTensor "slice(axis=1, start=1, stop=3): "

  putStrLn "\n=================== Broadcasting arithmetic ==================="
  do
    col <- tensorFromValues [3, 1] [1.0, 2.0, 3.0]
    row <- tensorFromValues [1, 4] [10.0, 20.0, 30.0, 40.0]
    tensorAdd col row >>= printTensor "(3,1) + (1,4) -> "

  putStrLn "\n=================== Reductions ==================="
  total <- tensorSum r
  mean <- tensorMean r
  putStrLn ("r.sum() = " ++ fmt total ++ ", r.mean() = " ++ fmt mean)
  tensorMeanAxis r 0 >>= printTensor "column means (axis=0): "

  putStrLn "\n=================== Linear algebra ==================="
  do
    a <- tensorFromValues [2, 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    b <- tensorFromValues [3, 2] [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    tensorMatmul a b >>= printTensor "matmul(2x3, 3x2) -> "
  do
    v1 <- tensorFromValues [3] [1.0, 2.0, 3.0]
    v2 <- tensorFromValues [3] [4.0, 5.0, 6.0]
    dot <- tensorDot v1 v2
    putStrLn ("dot([1,2,3], [4,5,6]) = " ++ fmt dot)
    tensorOuter v1 v2 >>= printTensor "outer(v1, v2) -> "

  putStrLn "\n=================== Comparisons & masks ==================="
  do
    mask <- tensorFull [3, 4] 6.0 >>= tensorGreaterEqual r
    printTensor "r >= 6 -> " mask
    count <- tensorSum mask
    putStrLn ("count(r >= 6) = " ++ fmt count)

  putStrLn "\n=================== A real dataset as a Tensor ==================="
  do
    iris <- dataIris
    n <- dataFrameNrows iris
    let columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
    flat <- sequence [dataFrameNumericAt iris column i | i <- [0 .. n - 1], column <- columns]
    x <- tensorFromValues [n, 4] flat
    xShape <- shapeString x
    putStrLn ("iris feature tensor shape: " ++ xShape)

    featureMeans <- tensorMeanAxis x 0
    centered <- tensorReshape featureMeans [1, 4] >>= tensorSubtract x
    scatter <- tensorTranspose centered >>= (`tensorMatmul` centered)
    printTensor "feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): " featureMeans
    scatterText <- tensorToStringWithMaxElements scatter 16
    putStrLn ("(X-mean)^T (X-mean) [4x4 scatter matrix]: " ++ scatterText)

    species <- mapM (dataFrameStringAt iris "Species") [0 .. n - 1]
    speciesTensor <- tensorFromStringValues [n] species
    setosaMask <- tensorFromStringValues [1] ["setosa"] >>= tensorEqual speciesTensor
    setosaCount <- tensorSum setosaMask
    putStrLn ("setosa count = " ++ fmt setosaCount ++ " (of " ++ show n ++ " rows)")

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
