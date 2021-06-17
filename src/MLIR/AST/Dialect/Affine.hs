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


module MLIR.AST.Dialect.Affine where

import Control.Monad.IO.Class
import Control.Monad.Trans.Cont
import qualified Language.C.Inline as C

import qualified MLIR.Native.FFI as Native
import MLIR.AST.Serialize

C.context $ C.baseCtx <> Native.mlirCtx

C.include "mlir-c/AffineExpr.h"
C.include "mlir-c/AffineMap.h"

data Expr =
    Dimension Int
  | Symbol    Int
  | Constant  Int
  | Add       Expr Expr
  | Mul       Expr Expr
  | Mod       Expr Expr
  | FloorDiv  Expr Expr
  | CeilDiv   Expr Expr
  deriving Eq

data Map = Map { mapDimensionCount :: Int
               , mapSymbolCount :: Int
               , mapExprs :: [Expr]
               }
               deriving Eq


instance FromAST Expr Native.AffineExpr where
  fromAST ctx env expr = case expr of
    Dimension idx -> do
      let natIdx = fromIntegral idx
      [C.exp| MlirAffineExpr { mlirAffineDimExprGet($(MlirContext ctx), $(intptr_t natIdx)) } |]
    Symbol    idx -> do
      let natIdx = fromIntegral idx
      [C.exp| MlirAffineExpr { mlirAffineSymbolExprGet($(MlirContext ctx), $(intptr_t natIdx)) } |]
    Constant  val -> do
      let natVal = fromIntegral val
      [C.exp| MlirAffineExpr { mlirAffineConstantExprGet($(MlirContext ctx), $(int64_t natVal)) } |]
    Add       l r -> do
      natL <- fromAST ctx env l
      natR <- fromAST ctx env r
      [C.exp| MlirAffineExpr { mlirAffineAddExprGet($(MlirAffineExpr natL), $(MlirAffineExpr natR)) } |]
    Mul       l r -> do
      natL <- fromAST ctx env l
      natR <- fromAST ctx env r
      [C.exp| MlirAffineExpr { mlirAffineMulExprGet($(MlirAffineExpr natL), $(MlirAffineExpr natR)) } |]
    Mod       l r -> do
      natL <- fromAST ctx env l
      natR <- fromAST ctx env r
      [C.exp| MlirAffineExpr { mlirAffineModExprGet($(MlirAffineExpr natL), $(MlirAffineExpr natR)) } |]
    FloorDiv  l r -> do
      natL <- fromAST ctx env l
      natR <- fromAST ctx env r
      [C.exp| MlirAffineExpr { mlirAffineFloorDivExprGet($(MlirAffineExpr natL), $(MlirAffineExpr natR)) } |]
    CeilDiv   l r -> do
      natL <- fromAST ctx env l
      natR <- fromAST ctx env r
      [C.exp| MlirAffineExpr { mlirAffineCeilDivExprGet($(MlirAffineExpr natL), $(MlirAffineExpr natR)) } |]


instance FromAST Map Native.AffineMap where
  fromAST ctx env Map{..} = evalContT $ do
    (numExprs, nativeExprs) <- packFromAST ctx env mapExprs
    let nativeDimCount      =  fromIntegral mapDimensionCount
    let nativeSymbolCount   =  fromIntegral mapSymbolCount
    liftIO $ [C.exp| MlirAffineMap {
      mlirAffineMapGet($(MlirContext ctx),
                       $(intptr_t nativeDimCount),
                       $(intptr_t nativeSymbolCount),
                       $(intptr_t numExprs), $(MlirAffineExpr* nativeExprs))
    } |]
