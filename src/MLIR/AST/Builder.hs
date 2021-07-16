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

{-# LANGUAGE UndecidableInstances #-}
module MLIR.AST.Builder where

import MLIR.AST
import Data.String
import Data.Functor
import Control.Monad
import Control.Monad.State.Strict
import Control.Monad.Writer
import Control.Monad.Reader

--------------------------------------------------------------------------------
-- Value

data Value = Name :> Type

typeOf :: Value -> Type
typeOf (_ :> ty) = ty

operand :: Value -> Name
operand (n :> _) = n

operands :: [Value] -> [Name]
operands = fmap operand

--------------------------------------------------------------------------------
-- Name supply monad

newtype NameSupply = NameSupply { nextName :: Int }
newtype NameSupplyT m a = NameSupplyT (StateT NameSupply m a)
                          deriving (Functor, Applicative, Monad,
                                    MonadTrans, MonadFix,
                                    MonadReader r, MonadWriter w)

instance MonadState s m => MonadState s (NameSupplyT m) where
  get = lift get
  put = lift . put

class Monad m => MonadNameSupply m where
  freshName :: m Name

instance MonadNameSupply m => MonadNameSupply (ReaderT r m) where
  freshName = lift freshName

evalNameSupplyT :: Monad m => NameSupplyT m a -> m a
evalNameSupplyT (NameSupplyT a) = evalStateT a $ NameSupply 0

instance Monad m => MonadNameSupply (NameSupplyT m) where
  freshName = NameSupplyT $ do
    curId <- gets nextName
    modify \s -> s { nextName = nextName s + 1 }
    return $ fromString $ show curId

freshValue :: MonadNameSupply m => Type -> m Value
freshValue ty = freshName <&> (:> ty)

freshBlockArg :: MonadNameSupply m => Type -> m Value
freshBlockArg ty = (("arg" <>) <$> freshName) <&> (:> ty)

--------------------------------------------------------------------------------
-- Block builder monad

-- TODO(apaszke): Thread locations through
-- TODO(apaszke): Use a writer monad
data BlockBindings = BlockBindings
  { blockBindings :: SnocList Binding
  , blockArguments :: SnocList Value
  , blockLocation :: Location
  }

instance Semigroup BlockBindings where
  BlockBindings bs args _ <> BlockBindings bs' args' loc' =
    BlockBindings (bs <> bs') (args <> args') loc'

instance Monoid BlockBindings where
  mempty = BlockBindings mempty mempty UnknownLocation

newtype BlockBuilderT m a = BlockBuilderT (StateT BlockBindings m a)
                            deriving (Functor, Applicative, Monad,
                                      MonadTrans, MonadFix,
                                      MonadReader r, MonadWriter w)

instance MonadState s m => MonadState s (BlockBuilderT m) where
  get = lift get
  put = lift . put

class Monad m => MonadBlockDecl m where
  emitOp_ :: Operation -> m ()
class MonadBlockDecl m => MonadBlockBuilder m where
  emitOp :: Operation -> m [Value]
  blockArgument :: Type -> m Value
  setLocation :: Location -> m ()

data EndOfBlock = EndOfBlock

terminateBlock :: Monad m => m EndOfBlock
terminateBlock = return EndOfBlock

noTerminator :: Monad m => m EndOfBlock
noTerminator = return EndOfBlock

runBlockBuilder :: Monad m => BlockBuilderT m a -> m (a, ([Value], [Binding]))
runBlockBuilder (BlockBuilderT act) = do
  (result, BlockBindings binds args _) <- runStateT act mempty
  return (result, (unsnocList args, unsnocList binds))

instance Monad m => MonadBlockDecl (BlockBuilderT m) where
  emitOp_ op = BlockBuilderT $ do
    case opResultTypes op of
      Inferred    -> error "Builder doesn't support inferred result types!"
      Explicit [] -> modify \s -> s { blockBindings = blockBindings s .:. (Do op) }
      Explicit _  -> error "emitOp_ can only be used on ops that have no results"

instance MonadNameSupply m => MonadBlockBuilder (BlockBuilderT m) where
  emitOp opNoLoc = BlockBuilderT $ do
    loc <- gets blockLocation
    let op = case opLocation opNoLoc of
          UnknownLocation -> opNoLoc { opLocation = loc }
          _ -> opNoLoc
    results <- case opResultTypes op of
      Inferred     -> error "Builder doesn't support inferred result types!"
      Explicit tys -> lift $ mapM freshValue tys
    modify \s -> s { blockBindings = blockBindings s .:. (operands results ::= op) }
    return results
  blockArgument ty = BlockBuilderT $ do
    value <- lift $ freshValue ty
    modify \s -> s { blockArguments = blockArguments s .:. value }
    return value
  setLocation loc = BlockBuilderT $ modify \s -> s { blockLocation = loc }

--------------------------------------------------------------------------------
-- Region builder monad

-- TODO(apaszke): Make this a writer, assign block names only at the very end
data RegionBuilderState = RegionBuilderState
  { blocks      :: SnocList Block
  , nextBlockId :: Int
  }
newtype RegionBuilderT m a = RegionBuilderT (StateT RegionBuilderState m a)
                             deriving (Functor, Applicative, Monad,
                                       MonadTrans, MonadFix,
                                       MonadReader r, MonadWriter w)

instance MonadState s m => MonadState s (RegionBuilderT m) where
  get = lift get
  put = lift . put

type BlockName = Name

class Monad m => MonadRegionBuilder m where
  appendBlock :: BlockBuilderT m EndOfBlock -> m BlockName

endOfRegion :: Monad m => m ()
endOfRegion = return ()

buildRegion :: Monad m => RegionBuilderT m () -> m Region
buildRegion (RegionBuilderT regionBuilder) =
  Region . unsnocList . blocks <$> execStateT regionBuilder (RegionBuilderState mempty 0)

buildBlock :: Monad m => BlockBuilderT m EndOfBlock -> RegionBuilderT m BlockName
buildBlock builder = RegionBuilderT $ do
  (EndOfBlock, (args, body)) <- lift $ runBlockBuilder builder
  makeBlock args body
  where
    makeBlock args body = do
      curBlockId <- gets nextBlockId
      modify (\s -> s { nextBlockId = nextBlockId s + 1 })
      let blockName = "bb" <> (fromString $ show curBlockId)
      let block = Block blockName (fmap (\(n :> t) -> (n, t)) args) body
      modify (\s -> s { blocks = blocks s .:. block })
      return blockName

--------------------------------------------------------------------------------
-- Builtin dialect

soleBlock :: Monad m => BlockBuilderT m EndOfBlock -> m Block
soleBlock builder = do
  (EndOfBlock, (args, body)) <- runBlockBuilder builder
  return $ Block "0" (fmap (\(n :> t) -> (n, t)) args) body

buildModule :: Monad m => BlockBuilderT m () -> m Operation
buildModule build = liftM ModuleOp $ soleBlock $ build >> noTerminator

declareFunction :: MonadBlockDecl m => Name -> Type -> m ()
declareFunction name funcTy =
  emitOp_ $ FuncOp UnknownLocation name funcTy $ Region []

buildFunction :: MonadBlockDecl m
              => Name -> [Type] -> NamedAttributes
              -> RegionBuilderT (NameSupplyT m) () -> m ()
buildFunction name retTypes attrs bodyBuilder = do
  body@(Region blocks) <- evalNameSupplyT $ buildRegion bodyBuilder
  let argTypes = case blocks of
        [] -> error $ "buildFunction cannot be used for function declarations! " ++
                      "Build at least one block!"
        (Block _ args _) : _ -> fmap snd args
  let op = FuncOp UnknownLocation name (FunctionType argTypes retTypes) body
  emitOp_ $ op { opAttributes = opAttributes op <> attrs }

buildSimpleFunction :: MonadBlockDecl m
                    => Name -> [Type] -> NamedAttributes
                    -> BlockBuilderT (NameSupplyT m) EndOfBlock -> m ()
buildSimpleFunction name retTypes attrs bodyBuilder = do
  block <- evalNameSupplyT $ soleBlock bodyBuilder
  let argTypes = fmap snd $ blockArgs block
  let fTy = FunctionType argTypes retTypes
  let op = FuncOp UnknownLocation name fTy $ Region [block]
  emitOp_ $ op { opAttributes = opAttributes op <> attrs }

--------------------------------------------------------------------------------
-- Utilities

newtype SnocList a = SnocList [a]

(.:.) :: SnocList a -> a -> SnocList a
(SnocList t) .:. h = SnocList (h : t)

unsnocList :: SnocList a -> [a]
unsnocList (SnocList l) = reverse l

instance Semigroup (SnocList a) where
  SnocList l <> SnocList r = SnocList (r <> l)

instance Monoid (SnocList a) where
  mempty = SnocList []
