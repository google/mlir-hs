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

module MLIR.Native.ExecutionEngine where

import Foreign.Ptr
import Foreign.Storable
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Data.Int
import qualified Language.C.Inline as C

import Control.Exception (bracket)
import Control.Monad

import MLIR.Native
import MLIR.Native.FFI

C.context $ C.baseCtx <> mlirCtx

C.include "mlir-c/ExecutionEngine.h"

-- TODO(apaszke): Flesh this out based on the header

--------------------------------------------------------------------------------
-- Execution engine

-- TODO(apaszke): Make the opt level configurable
-- TODO(apaszke): Allow loading shared libraries
createExecutionEngine :: Module -> IO (Maybe ExecutionEngine)
createExecutionEngine m = nullable <$>
  [C.exp| MlirExecutionEngine { mlirExecutionEngineCreate($(MlirModule m), 3, 0, NULL, false, false) } |]

destroyExecutionEngine :: ExecutionEngine -> IO ()
destroyExecutionEngine eng =
  [C.exp| void { mlirExecutionEngineDestroy($(MlirExecutionEngine eng)) } |]

withExecutionEngine :: Module -> (Maybe ExecutionEngine -> IO a) -> IO a
withExecutionEngine m = bracket (createExecutionEngine m)
                                (\case Just e  -> destroyExecutionEngine e
                                       Nothing -> return ())


data SomeStorable = forall a. Storable a => SomeStorable a

executionEngineInvoke :: forall result. Storable result
                      => ExecutionEngine -> StringRef -> [SomeStorable] -> IO (Maybe result)
executionEngineInvoke eng (StringRef namePtr nameLen) args =
  withPackedPtr \packPtr resultPtr -> do
    result <- [C.exp| MlirLogicalResult {
      mlirExecutionEngineInvokePacked($(MlirExecutionEngine eng),
                                      (MlirStringRef){$(char* namePtr), $(size_t nameLen)},
                                      $(void** packPtr))
    } |]
    case result of
      Success -> Just <$> peek resultPtr
      Failure -> return Nothing
  where
    numArgs = length args

    -- TODO(apaszke): Are tuples exploded, or stored as pointers?
    withPackedPtr :: (Ptr (Ptr ()) -> Ptr result -> IO a) -> IO a
    withPackedPtr f =
      allocaArray (numArgs + 1) \packedPtr ->
        alloca @result \resultPtr -> do
          pokeElemOff packedPtr numArgs (castPtr resultPtr)
          withStoredArgs args packedPtr $ f packedPtr resultPtr

    withStoredArgs :: [SomeStorable] -> Ptr (Ptr ()) -> IO a -> IO a
    withStoredArgs [] _ m = m
    withStoredArgs (SomeStorable h:t) nextArgPtr m =
      alloca \argPtr -> do
        poke argPtr h
        poke nextArgPtr (castPtr argPtr)
        withStoredArgs t (advancePtr nextArgPtr 1) m

packStruct64 :: [SomeStorable] -> (Ptr () -> IO a) -> IO a
packStruct64 fields f = do
  allocaArray (length fields) \(structPtr :: Ptr Int64) -> do
    forM_ (zip [0..] fields) \(i, SomeStorable field) -> do
      unless (sizeOf field == 8) $
        error "packStruct64 expects all fields to be exactly 8 bytes in size"
      unless (alignment field <= 8) $
        error "packStruct64 expects all fields to have an alignment of at most 8 bytes"
      pokeElemOff (castPtr structPtr) i field
    f $ castPtr structPtr
