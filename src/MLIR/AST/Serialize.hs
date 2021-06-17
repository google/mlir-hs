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

module MLIR.AST.Serialize (
  ValueMapping,
  BlockMapping,
  ValueAndBlockMapping,
  FromAST(..),
  packFromAST, packArray, unpackArray) where

import Foreign.Ptr
import Foreign.Storable
import Foreign.Marshal.Array
import Control.Monad.IO.Class
import Control.Monad.Trans.Cont
import qualified Language.C.Inline as C
import qualified Data.ByteString as BS
import qualified Data.Map.Strict as M

import qualified MLIR.Native     as Native
import qualified MLIR.Native.FFI as Native

type Name = BS.ByteString

type ValueMapping = M.Map Name Native.Value
type BlockMapping = M.Map Name Native.Block
type ValueAndBlockMapping = (ValueMapping, BlockMapping)

class FromAST ast native | ast -> native where
  fromAST :: Native.Context -> ValueAndBlockMapping -> ast -> IO native

packFromAST :: (FromAST ast native, Storable native)
            => Native.Context -> ValueAndBlockMapping
            -> [ast] -> ContT r IO (C.CIntPtr, Ptr native)
packFromAST ctx env asts = packArray =<< liftIO (mapM (fromAST ctx env) asts)

-- TODO(apaszke): Unify this with packing utilities from ExecutionEngine?
packArray :: Storable a => [a] -> ContT r IO (C.CIntPtr, Ptr a)
packArray xs = do
  let arrSize = (length xs)
  ptr <- ContT $ allocaArray arrSize
  liftIO $ mapM_ (uncurry $ pokeElemOff ptr) $ zip [0..] xs
  return (fromIntegral arrSize, ptr)

unpackArray :: Storable a => C.CIntPtr -> Ptr a -> IO [a]
unpackArray size arrPtr = mapM (peekElemOff arrPtr) [0..fromIntegral size - 1]
