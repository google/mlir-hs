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

import qualified Data.Map.Strict as M
import qualified Data.ByteString as BS

import MLIR.AST
import MLIR.AST.Dialect.Generated.Vector
import qualified MLIR.AST.Dialect.Affine as Affine

data IteratorType = Parallel | Reduction

showIterator :: IteratorType -> BS.ByteString
showIterator Parallel  = "#vector.iterator_type<parallel>"
showIterator Reduction = "#vector.iterator_type<reduction>"

itersFromAttribute :: Attribute -> Maybe [IteratorType]
itersFromAttribute attr = case attr of
  ArrayAttr subAttrs -> traverse iterFromString subAttrs
  _                  -> Nothing
  where iterFromString (AsmTextAttr "#vector.iterator_type<parallel>")  = Just Parallel
        iterFromString (AsmTextAttr "#vector.iterator_type<reduction>") = Just Reduction
        iterFromString _                        = Nothing

pattern IteratorAttrs :: [IteratorType] -> Attribute
pattern IteratorAttrs iterTypes <- (itersFromAttribute -> Just iterTypes)
  where IteratorAttrs iterTypes = ArrayAttr $ fmap (AsmTextAttr . showIterator) iterTypes

pattern ContractAttrs :: Affine.Map -> Affine.Map -> Affine.Map -> [IteratorType] -> NamedAttributes
pattern ContractAttrs lhsMap rhsMap accMap iterTypes <-
  ((\m -> (M.lookup "indexing_maps" m, M.lookup "iterator_types" m)) ->
     (Just (ArrayAttr [AffineMapAttr lhsMap, AffineMapAttr rhsMap, AffineMapAttr accMap]),
      Just (IteratorAttrs iterTypes)))
  where ContractAttrs lhsMap rhsMap accMap iterTypes = M.fromList
          [ ("indexing_maps", ArrayAttr [ AffineMapAttr lhsMap
                                        , AffineMapAttr rhsMap
                                        , AffineMapAttr accMap])
          , ("iterator_types", IteratorAttrs iterTypes)
          ]


pattern Contract :: Location -> Type -> Name -> Name -> Name
                 -> Affine.Map -> Affine.Map -> Affine.Map -> [IteratorType]
                 -> Operation
pattern Contract location resultType lhs rhs acc lhsMap rhsMap accMap iterTypes = Operation
  { opName = "vector.contract"
  , opLocation = location
  , opResultTypes = Explicit [resultType]
  , opOperands = [lhs, rhs, acc]
  , opRegions = []
  , opSuccessors = []
  , opAttributes = ContractAttrs lhsMap rhsMap accMap iterTypes
  }

