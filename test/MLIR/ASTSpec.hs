-- Copyright 2021 Google LLC
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

module MLIR.ASTSpec where

import Test.Hspec

import Text.RawString.QQ
import Data.Int
import Data.Char
import Data.Maybe
import Data.Foldable
import Foreign.Ptr
import Foreign.Storable
import Foreign.ForeignPtr
import qualified Data.ByteString as BS
import qualified Data.Vector.Storable as V
import Control.Monad.Trans.Cont
import Control.Monad.IO.Class

import MLIR.AST
import MLIR.AST.Serialize
import qualified MLIR.AST.Dialect.Arith  as Arith
import qualified MLIR.AST.Dialect.Func   as Func
import qualified MLIR.AST.Dialect.MemRef as MemRef
import qualified MLIR.AST.Dialect.Affine as Affine
import qualified MLIR.AST.Dialect.Vector as Vector
import qualified MLIR.Native                 as MLIR
import qualified MLIR.Native.Pass            as MLIR
import qualified MLIR.Native.ExecutionEngine as MLIR


newtype AlignedStorable a = Aligned a
deriving instance Num a => Num (AlignedStorable a)
deriving instance Fractional a => Fractional (AlignedStorable a)
deriving instance Show a => Show (AlignedStorable a)
deriving instance Eq a => Eq (AlignedStorable a)
deriving instance Ord a => Ord (AlignedStorable a)
instance Storable a => Storable (AlignedStorable a) where
  sizeOf        (Aligned x) = sizeOf x
  alignment     (Aligned _) = 64
  peek      ptr             = Aligned <$> peek (castPtr ptr)
  poke      ptr (Aligned x) = poke (castPtr ptr) x


trimLeadingSpaces :: BS.ByteString -> BS.ByteString
trimLeadingSpaces str = BS.intercalate "\n" strippedLines
  where
    space = fromIntegral $ ord ' '
    ls = BS.split (fromIntegral $ ord '\n') str
    indentDepth = fromJust $ asum $ (BS.findIndex (/= space)) <$> filter (/= "") ls
    indent = BS.replicate indentDepth space
    strippedLines = flip fmap ls \case "" -> ""
                                       l  -> fromJust $ BS.stripPrefix indent l


shouldShowAs :: Operation -> BS.ByteString -> Expectation
shouldShowAs op expectedWithLeadingNewline = do
  MLIR.withContext \ctx -> do
    MLIR.registerAllDialects ctx
    nativeOp <- fromAST ctx (mempty, mempty) op
    MLIR.verifyOperation nativeOp >>= (`shouldBe` True)
    let expected = trimLeadingSpaces $ BS.append (BS.tail expectedWithLeadingNewline) "\n"
    MLIR.showOperation nativeOp >>= (`shouldBe` expected)

shouldShowWithLocationAs :: Operation -> BS.ByteString -> Expectation
shouldShowWithLocationAs op expectedWithLeadingNewline = do
  MLIR.withContext \ctx -> do
    MLIR.registerAllDialects ctx
    nativeOp <- fromAST ctx (mempty, mempty) op
    MLIR.verifyOperation nativeOp >>= (`shouldBe` True)
    let expected = trimLeadingSpaces $ BS.append (BS.tail expectedWithLeadingNewline) "\n"
    MLIR.showOperationWithLocation nativeOp >>= (`shouldBe` expected)

shouldImplementMatmul :: Operation -> Expectation
shouldImplementMatmul op = evalContT $ do
  ctx <- ContT $ MLIR.withContext
  m <- liftIO $ do
    MLIR.registerAllDialects ctx
    Just m <- MLIR.moduleFromOperation =<< fromAST ctx (mempty, mempty) op
    MLIR.withPassManager ctx \pm -> do
      MLIR.addConvertMemRefToLLVMPass   pm
      MLIR.addConvertVectorToLLVMPass   pm
      MLIR.addConvertStandardToLLVMPass pm
      MLIR.addConvertReconcileUnrealizedCastsPass pm
      result <- MLIR.runPasses pm m
      result `shouldBe` MLIR.Success
    return m
  (a, _   ) <- withMemrefArg0 $ V.unsafeThaw $ V.iterateN  64 (+1.0) (1.0 :: AlignedStorable Float)
  (b, _   ) <- withMemrefArg0 $ V.unsafeThaw $ V.iterateN  64 (+2.0) (1.0 :: AlignedStorable Float)
  (c, cVec) <- withMemrefArg0 $ V.unsafeThaw $ V.replicate 64        (0.0 :: AlignedStorable Float)
  Just eng <- ContT $ MLIR.withExecutionEngine m
  name     <- ContT $ MLIR.withStringRef "matmul8x8x8"
  liftIO $ do
    Just () <- MLIR.executionEngineInvoke @() eng name [a, b, c]
    cVecFinal <- V.unsafeFreeze cVec
    cVecFinal `shouldBe` expectedOutput
  where
    -- Packs a vector into a struct representing a rank-0 memref
    withMemrefArg0 :: ContT r IO (V.MVector s a) -> ContT r IO (MLIR.SomeStorable, V.MVector s a)
    withMemrefArg0 mkVec = do
      vec@(V.MVector _ fptr) <- mkVec
      ptr <- ContT $ withForeignPtr fptr
      structPtr <- ContT $ MLIR.packStruct64
          [MLIR.SomeStorable ptr, MLIR.SomeStorable ptr, MLIR.SomeStorable (0 :: Int64)]
      return (MLIR.SomeStorable structPtr, vec)

    expectedOutput :: V.Vector (AlignedStorable Float)
    expectedOutput = V.fromList $ fmap Aligned
      [ 2724,  2796,  2868,  2940,  3012,  3084,  3156,  3228
      , 6372,  6572,  6772,  6972,  7172,  7372,  7572,  7772
      , 10020, 10348, 10676, 11004, 11332, 11660, 11988, 12316
      , 13668, 14124, 14580, 15036, 15492, 15948, 16404, 16860
      , 17316, 17900, 18484, 19068, 19652, 20236, 20820, 21404
      , 20964, 21676, 22388, 23100, 23812, 24524, 25236, 25948
      , 24612, 25452, 26292, 27132, 27972, 28812, 29652, 30492
      , 28260, 29228, 30196, 31164, 32132, 33100, 34068, 35036
      ]


emitted :: Operation -> Operation
emitted op = op { opAttributes = opAttributes op <> namedAttribute "llvm.emit_c_interface" UnitAttr }


spec :: Spec
spec = do
  describe "AST translation" $ do
    it "Can translate an empty module" $ do
      let m = ModuleOp $ Block "0" [] []
      m `shouldShowAs` [r|
        module {
        }|]

    it "Can translate a module with location" $ do
      -- TODO(jpienaar): This builds the module explicitly using the Operation
      -- interface to set the opLocation on an Operation corresponding to a
      -- Module.
      let m = Operation {
        opName = "builtin.module"
        , opLocation = FusedLocation [
            NameLocation "first" UnknownLocation
            , NameLocation "last" UnknownLocation
            ] Nothing
        , opResultTypes = Explicit []
        , opOperands = []
        , opRegions = [Region [Block "0" [] []]]
        , opSuccessors = []
        , opAttributes = NoAttrs
        }
      m `shouldShowWithLocationAs` [r|
        module {
        } loc(#loc)
        #loc = loc(fused["first", "last"])|]

    it "Can construct a matmul via vector.matrix_multiply" $ do
      let v64Ty = VectorType [64] Float32Type
      let v64refTy = MemRefType { memrefTypeShape = []
                                , memrefTypeElement = v64Ty
                                , memrefTypeLayout = Nothing
                                , memrefTypeMemorySpace = Nothing }
      let m = ModuleOp $ Block "0" [] [
                Do $ emitted $ FuncOp UnknownLocation "matmul8x8x8" (FunctionType [v64refTy, v64refTy, v64refTy] []) $ Region [
                  Block "0" [("arg0", v64refTy), ("arg1", v64refTy), ("arg2", v64refTy)]
                    [ "0" := MemRef.Load v64Ty "arg0" []
                    , "1" := MemRef.Load v64Ty "arg1" []
                    , "2" := Vector.Matmul UnknownLocation v64Ty "0" "1" 8 8 8
                    , "3" := MemRef.Load v64Ty "arg2" []
                    , "4" := Arith.AddF UnknownLocation v64Ty "3" "2"
                    , Do $ MemRef.Store "4" "arg2" []
                    , Do $ Func.Return UnknownLocation []
                    ]
                ]
              ]
      m `shouldShowAs` [r|
        module {
          func @matmul8x8x8(%arg0: memref<vector<64xf32>>, %arg1: memref<vector<64xf32>>, %arg2: memref<vector<64xf32>>) attributes {llvm.emit_c_interface} {
            %0 = memref.load %arg0[] : memref<vector<64xf32>>
            %1 = memref.load %arg1[] : memref<vector<64xf32>>
            %2 = vector.matrix_multiply %0, %1 {lhs_columns = 8 : i32, lhs_rows = 8 : i32, rhs_columns = 8 : i32} : (vector<64xf32>, vector<64xf32>) -> vector<64xf32>
            %3 = memref.load %arg2[] : memref<vector<64xf32>>
            %4 = arith.addf %3, %2 : vector<64xf32>
            memref.store %4, %arg2[] : memref<vector<64xf32>>
            return
          }
        }|]

    it "Can translate matmul via vector.contract" $ do
      let v8x8Ty = VectorType [8, 8] Float32Type
      let v8x8RefTy = MemRefType [] v8x8Ty Nothing Nothing
      let m = ModuleOp $ Block "0" [] [
                Do $ emitted $ FuncOp UnknownLocation "matmul8x8x8" (FunctionType [v8x8RefTy, v8x8RefTy, v8x8RefTy] []) $ Region [
                  Block "0" [("arg0", v8x8RefTy), ("arg1", v8x8RefTy), ("arg2", v8x8RefTy)]
                    [ "0" := MemRef.Load v8x8Ty "arg0" []
                    , "1" := MemRef.Load v8x8Ty "arg1" []
                    , "2" := MemRef.Load v8x8Ty "arg2" []
                    , "3" := Vector.Contract UnknownLocation v8x8Ty "0" "1" "2"
                                (Affine.Map 3 0 [Affine.Dimension 0, Affine.Dimension 2])
                                (Affine.Map 3 0 [Affine.Dimension 2, Affine.Dimension 1])
                                (Affine.Map 3 0 [Affine.Dimension 0, Affine.Dimension 1])
                                [Vector.Parallel, Vector.Parallel, Vector.Reduction]
                    , Do $ MemRef.Store "3" "arg2" []
                    , Do $ Func.Return UnknownLocation []
                    ]
                ]
              ]
      shouldImplementMatmul m
      m `shouldShowAs` [r|
        #map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
        #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
        #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
        module {
          func @matmul8x8x8(%arg0: memref<vector<8x8xf32>>, %arg1: memref<vector<8x8xf32>>, %arg2: memref<vector<8x8xf32>>) attributes {llvm.emit_c_interface} {
            %0 = memref.load %arg0[] : memref<vector<8x8xf32>>
            %1 = memref.load %arg1[] : memref<vector<8x8xf32>>
            %2 = memref.load %arg2[] : memref<vector<8x8xf32>>
            %3 = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %0, %1, %2 : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
            memref.store %3, %arg2[] : memref<vector<8x8xf32>>
            return
          }
        }|]


main :: IO ()
main = hspec spec
