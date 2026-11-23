-- CMD: just TARGET=gis_ex run-haskell
-- ShapeLayer: write a tiny synthetic shapefile (.shp + .dbf) by hand, read it back and draw it as a map.
-- Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this example
-- writes two "counties" (one plain square, one with a lake-shaped hole) using the same ESRI byte
-- layout shapeLayerRead expects, then reads them back through the public API.
module Main (main) where

import qualified Data.ByteString as B
import qualified Data.ByteString.Builder as BB
import qualified Data.ByteString.Lazy as BL
import Data.List (intercalate)
import Datamunge

type Ring = [(Double, Double)]

basePath :: FilePath
basePath = "datamunge_gis_ex_counties_haskell"

main :: IO ()
main = do
  -- Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
  let plain = [[(10, 0), (10, 10), (20, 10), (20, 0), (10, 0)]]
      withLake =
        [ [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]
        , [(3, 3), (4, 3), (4, 4), (3, 4), (3, 3)]
        ]

  writeCountiesShp (basePath ++ ".shp") [withLake, plain]
  writeCountiesDbf (basePath ++ ".dbf") ["Lakeside", "Plainview"] ["48231", "19876"]

  counties <- shapeLayerRead basePath

  size <- shapeLayerSize counties
  shapeType <- shapeLayerShapeType counties
  putStrLn ("shapes = " ++ show size ++ ", shape_type = " ++ shapeType)
  bounds <- shapeLayerBounds counties
  putStrLn ("bounds = [" ++ intercalate ", " (map fmtG bounds) ++ "]\n")

  attrs <- shapeLayerAttributes counties
  attrsText <- dataFrameToString attrs
  putStrLn ("attributes\n" ++ attrsText ++ "\n")

  mapM_ (\i -> do
           name <- dataFrameStringAt attrs "NAME" i
           parts <- shapeLayerNumParts counties i
           putStrLn (name ++ ": " ++ show parts ++ " ring(s)"))
        [0 .. size - 1]

  let svgPath = "datamunge_gis_ex_map_haskell.svg"
  shapeLayerPlot counties >>= (`plotSaveSvg` svgPath)
  putStrLn ("\nmap saved to " ++ svgPath)

-- | Roughly C's %g: integral doubles print without a fractional part.
fmtG :: Double -> String
fmtG x
  | x == fromIntegral (floor x :: Integer) = show (floor x :: Integer)
  | otherwise = fixed 6 x

-- | Bounding box (minx, miny, maxx, maxy) of every point in rings.
ringBounds :: [Ring] -> [Double]
ringBounds rings = [minimum xs, minimum ys, maximum xs, maximum ys]
  where
    (xs, ys) = unzip (concat rings)

-- | Bytes of one Polygon record's contents (without the record header).
polygonRecord :: [Ring] -> B.ByteString
polygonRecord rings =
  BL.toStrict . BB.toLazyByteString $
    BB.int32LE 5 -- shape type: Polygon
      <> foldMap BB.doubleLE (ringBounds rings)
      <> BB.int32LE (fromIntegral (length rings))
      <> BB.int32LE (fromIntegral (sum (map length rings)))
      <> foldMap (BB.int32LE . fromIntegral) (scanl (+) 0 (map length (init rings)))
      <> foldMap (\(x, y) -> BB.doubleLE x <> BB.doubleLE y) (concat rings)

writeCountiesShp :: FilePath -> [[Ring]] -> IO ()
writeCountiesShp path shapes = do
  let contents = map polygonRecord shapes
      totalWords = sum [4 + B.length c `div` 2 | c <- contents]
  BB.writeFile path $
    BB.int32BE 9994
      <> mconcat (replicate 5 (BB.int32BE 0))
      <> BB.int32BE (fromIntegral (50 + totalWords))
      <> BB.int32LE 1000
      <> BB.int32LE 5 -- Polygon
      <> foldMap BB.doubleLE (ringBounds (concat shapes))
      <> mconcat (replicate 4 (BB.doubleLE 0.0))
      <> mconcat
        [ BB.int32BE i <> BB.int32BE (fromIntegral (B.length c `div` 2)) <> BB.byteString c
        | (i, c) <- zip [1 ..] contents
        ]

ljust :: String -> Int -> Char -> String
ljust s n ch = take n (s ++ replicate n ch)

writeCountiesDbf :: FilePath -> [String] -> [String] -> IO ()
writeCountiesDbf path names populations = do
  let headerSize = 32 + 2 * 32 + 1
      recordSize = 1 + 12 + 8
      zeros n = mconcat (replicate n (BB.word8 0))
      field name fieldType len =
        BB.string7 (ljust name 11 '\0') <> BB.string7 fieldType <> zeros 4 <> BB.word8 len <> zeros 15
  BB.writeFile path $
    BB.word8 0x03 <> zeros 3
      <> BB.int32LE (fromIntegral (length names))
      <> BB.int16LE headerSize
      <> BB.int16LE recordSize
      <> zeros 20
      <> field "NAME" "C" 12
      <> field "POP" "N" 8
      <> BB.word8 0x0D
      <> mconcat [BB.string7 " " <> BB.string7 (ljust name 12 ' ') <> BB.string7 (ljust pop 8 ' ') | (name, pop) <- zip names populations]

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
