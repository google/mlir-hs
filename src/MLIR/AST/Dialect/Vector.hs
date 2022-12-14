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

import qualified MLIR.AST as AST
import MLIR.AST.Dialect.Generated.Vector
import qualified MLIR.AST.Dialect.Affine as Affine

data IteratorType = Parallel | Reduction

showIterator :: IteratorType -> BS.ByteString
showIterator Parallel  = "#vector.iterator_type<parallel>"
showIterator Reduction = "#vector.iterator_type<reduction>"

itersFromAttribute :: AST.Attribute -> Maybe [IteratorType]
itersFromAttribute attr = case attr of
  AST.ArrayAttr subAttrs -> traverse iterFromString subAttrs
  _                  -> Nothing
  where iterFromString (AST.AsmTextAttr "#vector.iterator_type<parallel>")  = Just Parallel
        iterFromString (AST.AsmTextAttr "#vector.iterator_type<reduction>") = Just Reduction
        iterFromString _                        = Nothing

pattern IteratorAttrs :: [IteratorType] -> AST.Attribute
pattern IteratorAttrs iterTypes <- (itersFromAttribute -> Just iterTypes)
  where IteratorAttrs iterTypes = AST.ArrayAttr $ fmap (AST.AsmTextAttr . showIterator) iterTypes

pattern ContractAttrs :: Affine.Map -> Affine.Map -> Affine.Map -> [IteratorType] -> AST.NamedAttributes
pattern ContractAttrs lhsMap rhsMap accMap iterTypes <-
  ((\m -> (M.lookup "indexing_maps" m, M.lookup "iterator_types" m)) ->
     (Just (AST.ArrayAttr [AST.AffineMapAttr lhsMap, AST.AffineMapAttr rhsMap, AST.AffineMapAttr accMap]),
      Just (IteratorAttrs iterTypes)))
  where ContractAttrs lhsMap rhsMap accMap iterTypes = M.fromList
          [ ("indexing_maps", AST.ArrayAttr [ AST.AffineMapAttr lhsMap
                                            , AST.AffineMapAttr rhsMap
                                            , AST.AffineMapAttr accMap])
          , ("iterator_types", IteratorAttrs iterTypes)
          ]


pattern Contract :: AST.Location -> AST.Type -> AST.Name -> AST.Name -> AST.Name
                 -> Affine.Map -> Affine.Map -> Affine.Map -> [IteratorType]
                 -> AST.Operation
pattern Contract location resultType lhs rhs acc lhsMap rhsMap accMap iterTypes = AST.Operation
  { opName = "vector.contract"
  , opLocation = location
  , opResultTypes = AST.Explicit [resultType]
  , opOperands = [lhs, rhs, acc]
  , opRegions = []
  , opSuccessors = []
  , opAttributes = ContractAttrs lhsMap rhsMap accMap iterTypes
  }

