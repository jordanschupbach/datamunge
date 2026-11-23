-- CMD: just TARGET=dataframe_ex run-haskell
-- DataFrame basics: nulls, drop_duplicates, fill_null, select/sort, group-by, join and summaries.
module Main (main) where

import Datamunge
import Numeric (floatToDigits)

main :: IO ()
main = do
  sales <- newDataFrame
  dataFrameAddStringColumn sales "region" ["west", "west", "east", "south", "south", "south"]
  dataFrameAddStringColumn sales "product" ["widget", "widget", "widget", "gizmo", "gizmo", "gizmo"]
  dataFrameAddNumericColumnWithValidMask sales "sales" [10.0, 10.0, 14.0, 8.0, 0.0, 11.0] [1, 1, 1, 1, 0, 1]
  dataFrameAddStringColumnWithValidMask sales "quarter" ["Q1", "Q1", "Q1", "Q2", "Q2", ""] [1, 1, 1, 1, 1, 0]

  raw <- dataFrameToString sales
  putStrLn ("raw data\n" ++ raw ++ "\n")

  cleaned <- dataFrameDropDuplicatesWithSubset sales ["region", "product", "sales", "quarter"]
  dataFrameFillNullString cleaned "quarter" "unknown"
  dataFrameFillNullNumeric cleaned "sales" 0.0
  cleanedText <- dataFrameToString cleaned
  putStrLn ("after drop_duplicates + fill_null\n" ++ cleanedText ++ "\n")

  selected <- dataFrameSelect cleaned ["region", "sales", "quarter"] >>= \frame -> dataFrameSortByWithAscending frame "sales" False
  selectedText <- dataFrameToString selected
  putStrLn ("selected + sorted\n" ++ selectedText ++ "\n")

  grouped <- dataFrameGroupBySum cleaned ["region"] ["sales"] >>= \frame -> dataFrameSortByWithAscending frame "sales" False
  groupedText <- dataFrameToString grouped
  putStrLn ("group_by_sum(region)\n" ++ groupedText ++ "\n")

  targets <- newDataFrame
  dataFrameAddStringColumn targets "region" ["west", "east", "south"]
  dataFrameAddNumericColumn targets "target" [18.0, 12.0, 25.0]
  joined <- dataFrameJoinWithJoinType grouped targets "region" "region" "left" >>= dataFrameToString
  putStrLn ("joined with targets\n" ++ joined ++ "\n")

  shape <- dataFrameShape cleaned
  putStrLn ("shape = (" ++ show (shape !! 0) ++ ", " ++ show (shape !! 1) ++ ")")
  dataFrameNumericCount cleaned "sales" >>= \count -> putStrLn ("sales count = " ++ show count)
  dataFrameNumericNullCount cleaned "sales" >>= \nulls -> putStrLn ("sales nulls = " ++ show nulls)
  dataFrameNumericSum cleaned "sales" >>= \total -> putStrLn ("sales sum = " ++ fmt total)
  dataFrameNumericMean cleaned "sales" >>= \mean -> putStrLn ("sales mean = " ++ fmt mean)

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
