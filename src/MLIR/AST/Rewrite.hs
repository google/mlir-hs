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

module MLIR.AST.Rewrite
  ( RewriteBuilderT
  , OpRewriteM
  , OpRewrite
  , RewriteResult(..)
  , pattern ReplaceOne
  , applyClosedOpRewrite
  , applyClosedOpRewriteT
  ) where

import qualified Data.Map.Strict as M
import Control.Monad.Reader
import Control.Monad.Identity

import qualified MLIR.AST as AST
import MLIR.AST hiding (Operation)
import MLIR.AST.Builder

-- For convenience we pass in operations with operands already substituted
-- for builder-compilant values that can be used to replace the operation
-- with an arbitrary program constructed through a builder expression.
type Operation = AST.AbstractOperation Value

type ValueMapping = M.Map Name      Value
type BlockMapping = M.Map BlockName BlockName
type BlockAndValueMapping = (ValueMapping, BlockMapping)

type SubstT = ReaderT BlockAndValueMapping
type RewriteT m = SubstT (NameSupplyT m)
type RewriteBuilderT m = BlockBuilderT (RewriteT m)

data RewriteResult = Replace [Value] | Skip | Traverse

pattern ReplaceOne :: Value -> RewriteResult
pattern ReplaceOne val = Replace [val]

type OpRewriteM m = Operation -> RewriteBuilderT m RewriteResult
type OpRewrite = OpRewriteM Identity

extendValueMap :: MonadReader BlockAndValueMapping m => ValueMapping -> m a -> m a
extendValueMap upd = local \(vm, bm) -> (vm <> upd, bm)

extendBlockMap :: MonadReader BlockAndValueMapping m => BlockMapping -> m a -> m a
extendBlockMap upd = local \(vm, bm) -> (vm, bm <> upd)

applyClosedOpRewrite :: OpRewrite -> AST.Operation -> AST.Operation
applyClosedOpRewrite rule op = runIdentity $ applyClosedOpRewriteT rule op

applyClosedOpRewriteT :: MonadFix m => OpRewriteM m -> AST.Operation -> m AST.Operation
applyClosedOpRewriteT rule op = evalNameSupplyT $ applyOpRewrite rule op

applyOpRewrite :: MonadFix m => OpRewriteM m -> AST.Operation -> NameSupplyT m AST.Operation
applyOpRewrite rule op = flip runReaderT (mempty, mempty) $ do
  newRegions <- mapM (applyOpRewriteRegion rule) $ opRegions op
  return $ op { opRegions = newRegions }

applyOpRewriteRegion :: MonadFix m => OpRewriteM m -> Region -> RewriteT m Region
applyOpRewriteRegion rule (Region blocks) = do
  buildRegion $ void $ mfix \blockSubst -> extendBlockMap blockSubst $ go mempty blocks
  where
    go blockSubst bs = case bs of
      []                                 -> return blockSubst
      (block@(Block oldName _ _) : rest) -> do
        newName <- applyOpRewriteBlock rule block
        go (blockSubst <> M.singleton oldName newName) rest

applyOpRewriteBlock :: MonadFix m => OpRewriteM m -> Block -> RegionBuilderT (RewriteT m) BlockName
applyOpRewriteBlock rule Block{..} = do
  buildBlock $ do
    let (blockArgNames, blockArgTypes) = unzip blockArgs
    newBlockArgs <- mapM blockArgument blockArgTypes
    extendValueMap (M.fromList $ zip blockArgNames newBlockArgs) $ go blockBody
  where
    go bs = case bs of
      [] -> terminateBlock
      ((Bind names astOp) : rest) -> do
        op <- substOp astOp
        answer <- rule op
        newValues <- case answer of
          Replace newValues -> do
            unless (length names == length newValues) $
              error "Rewrite rule returned an incorrect number of values"
            return newValues
          Traverse          -> opRewriteTraverse op
          Skip              -> opRewriteSkip     op
        extendValueMap (M.fromList $ zip names newValues) $ go rest

    opRewriteTraverse op = do
      newRegions <- lift $ mapM (applyOpRewriteRegion rule) $ opRegions op
      emitOp $ op { opRegions = newRegions, opOperands = operands (opOperands op) }

    opRewriteSkip op = do
      -- Note that we still have to traverse the subregions in case they close-over
      -- any values that have had their names updated.
      newRegions <- lift $ mapM (applyOpRewriteRegion (const $ return Skip)) $ opRegions op
      emitOp $ op { opRegions = newRegions, opOperands = operands (opOperands op) }

substOp :: MonadReader BlockAndValueMapping m => AST.Operation -> m Operation
substOp op = do
  (valueMap, blockMap) <- ask
  let newOperands = fmap (valueMap M.!) $ opOperands op
  let newSuccessors = fmap (blockMap M.!) $ opSuccessors op
  return $ op { opOperands = newOperands, opSuccessors = newSuccessors }

-- TODO(apaszke): Multi-op-patterns. Sketch:
--
-- type OperationExpr = AbstractOperation Value
-- data Value = BlockArgument Builder.Value | OperationResult Builder.Value Int OperationExpr
--
-- instance Builder.IsValue Value where
--   getValue (BlockArgument val) = val
--   getValue (Result val _ _)    = val
--
-- pattern Result :: Int -> OperationExpr -> Value
-- pattern Result idx opExpr <- OperationResult _ idx opExpr
--
-- pattern Result0 :: OperationExpr -> Value
-- pattern Result0 opExpr <- OperationResult _ 0 opExpr
--
-- asOperationExpr :: MonadRewrite m => Operation -> m OperationExpr
-- asOperationExpr = undefined
--
-- TODO(apaszke): Multi-op-patterns removals with multi-op-patterns. Sketch:
--
-- erase :: MonadRewrite m => Operation -> m ()
-- erase = undefined
