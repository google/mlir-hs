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

{-# OPTIONS_GHC -Wno-name-shadowing #-}

module MLIR.AST.Dialect.Vector
  ( module MLIR.AST.Dialect.Vector
  , module MLIR.AST.Dialect.Generated.Vector
  ) where

import Data.Typeable
import qualified Data.Map.Strict as M
import qualified Data.ByteString as BS
import qualified Language.C.Inline as C

import MLIR.AST.Dialect.Generated.Vector
import qualified MLIR.AST                as AST
import qualified MLIR.AST.Serialize      as AST
import qualified MLIR.AST.Dialect.Affine as Affine
import qualified MLIR.Native             as Native
import qualified MLIR.Native.FFI         as Native

data IteratorKind = Parallel | Reduction
          deriving (Eq, Show)

data Attribute = IteratorAttr IteratorKind
          deriving (Eq, Show)

castVectorAttr :: AST.Attribute -> Maybe Attribute
castVectorAttr ty = case ty of
  AST.DialectAttr dty -> cast dty
  _                   -> Nothing

showIterator :: IteratorKind -> BS.ByteString
showIterator Parallel  = "#vector.iterator_type<parallel>"
showIterator Reduction = "#vector.iterator_type<reduction>"

C.context $ C.baseCtx <> Native.mlirCtx

C.include "mlir-c/IR.h"

instance AST.FromAST Attribute Native.Attribute where
  fromAST ctx _ ty = case ty of
    IteratorAttr t -> do
      let value = showIterator t
      Native.withStringRef value \(Native.StringRef ptr len) ->
        [C.exp| MlirAttribute {
          mlirAttributeParseGet($(MlirContext ctx), (MlirStringRef){$(char* ptr), $(size_t len)})
        } |]

iterFromAttribute :: AST.Attribute -> Maybe IteratorKind
iterFromAttribute attr = case attr of
  AST.DialectAttr subAttr -> case cast subAttr of
    Just (IteratorAttr kind) -> Just kind
    _ -> Nothing
  _ -> Nothing

itersFromAttribute :: AST.Attribute -> Maybe [IteratorKind]
itersFromAttribute attr = case attr of
  AST.ArrayAttr subAttrs -> traverse iterFromAttribute subAttrs
  _                      -> Nothing

pattern IteratorAttrs :: [IteratorKind] -> AST.Attribute
pattern IteratorAttrs iterTypes <- (itersFromAttribute -> Just iterTypes)
  where IteratorAttrs iterTypes = AST.ArrayAttr $ fmap (AST.DialectAttr . IteratorAttr) iterTypes

pattern ContractAttrs :: Affine.Map -> Affine.Map -> Affine.Map -> [IteratorKind] -> AST.NamedAttributes
pattern ContractAttrs lhsMap rhsMap accMap iterKinds <-
  ((\m -> (M.lookup "indexing_maps" m, M.lookup "iterator_types" m)) ->
     (Just (AST.ArrayAttr [AST.AffineMapAttr lhsMap, AST.AffineMapAttr rhsMap, AST.AffineMapAttr accMap]),
      Just (IteratorAttrs iterKinds)))
  where ContractAttrs lhsMap rhsMap accMap iterKinds = M.fromList
          [ ("indexing_maps", AST.ArrayAttr [ AST.AffineMapAttr lhsMap
                                            , AST.AffineMapAttr rhsMap
                                            , AST.AffineMapAttr accMap])
          , ("iterator_types", IteratorAttrs iterKinds)
          ]

pattern Contract :: AST.Location -> AST.Type -> AST.Name -> AST.Name -> AST.Name
                 -> Affine.Map -> Affine.Map -> Affine.Map -> [IteratorKind]
                 -> AST.Operation
pattern Contract location resultType lhs rhs acc lhsMap rhsMap accMap iterKinds = AST.Operation
  { opName = "vector.contract"
  , opLocation = location
  , opResultTypes = AST.Explicit [resultType]
  , opOperands = [lhs, rhs, acc]
  , opRegions = []
  , opSuccessors = []
  , opAttributes = ContractAttrs lhsMap rhsMap accMap iterKinds
  }

