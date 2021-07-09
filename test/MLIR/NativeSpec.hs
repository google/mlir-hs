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

module MLIR.NativeSpec where

import Test.Hspec

import Text.RawString.QQ

import Data.Int
import Data.Maybe
import Data.Char (ord)
import qualified Data.ByteString as BS
import Control.Monad
import Foreign.Storable

import qualified MLIR.Native as MLIR
import qualified MLIR.Native.Pass as MLIR
import qualified MLIR.Native.ExecutionEngine as MLIR

exampleModuleStr :: BS.ByteString
exampleModuleStr = pack $ [r|module  {
  func @add(%arg0: i32) -> i32 attributes {llvm.emit_c_interface} {
    %0 = addi %arg0, %arg0 : i32
    return %0 : i32
  }
}
|]

-- XXX: Only valid for ASCII strings
pack :: String -> BS.ByteString
pack = BS.pack . fmap (fromIntegral . ord)

-- TODO(apaszke): Clean up
prepareContext :: IO MLIR.Context
prepareContext = do
  ctx <- MLIR.createContext
  MLIR.registerAllDialects ctx
  return ctx

spec :: Spec
spec = do
  describe "Basics" $ do
    it "Can create a context" $ MLIR.withContext $ const $ return ()

    it "Can load dialects" $ do
      MLIR.withContext \ctx -> do
        MLIR.registerAllDialects ctx
        numDialects <- MLIR.getNumLoadedDialects ctx
        numDialects `shouldSatisfy` (> 1)

  describe "Modules" $ beforeAll prepareContext $ do
    it "Can create an empty module" $ \ctx -> do
      loc <- MLIR.getUnknownLocation ctx
      m <- MLIR.createEmptyModule loc
      str <- MLIR.showModule m
      MLIR.destroyModule m
      str `shouldBe` "module  {\n}\n"

    it "Can parse an example module" $ \ctx -> do
      exampleModule <- liftM fromJust $
        MLIR.withStringRef exampleModuleStr $ MLIR.parseModule ctx
      exampleModuleStr' <- MLIR.showModule exampleModule
      exampleModuleStr' `shouldBe` exampleModuleStr
      MLIR.destroyModule exampleModule

    it "Fails to parse garbage" $ \ctx -> do
      maybeModule <- MLIR.withStringRef "asdf" $ MLIR.parseModule ctx
      (isNothing maybeModule) `shouldBe` True

    it "Can create an empty module with location" $ \ctx -> do
      MLIR.withStringRef "test.cc" $ \nameRef -> do
        loc <- MLIR.getFileLineColLocation ctx nameRef 21 45
        m <- MLIR.createEmptyModule loc
        str <- (MLIR.moduleAsOperation >=> MLIR.showOperationWithLocation) m
        MLIR.destroyModule m
        str `shouldBe` "module  {\n} loc(#loc)\n#loc = loc(\"test.cc\":21:45)\n"

  describe "Evaluation engine" $ beforeAll prepareContext $ do
    it "Can evaluate the example module" $ \ctx -> do
      m <- liftM fromJust $
        MLIR.withStringRef exampleModuleStr $ MLIR.parseModule ctx
      lowerToLLVM m
      result <- run @Int32 m "add" [MLIR.SomeStorable (123 :: Int32)]
      result `shouldBe` 246
      MLIR.destroyModule m
      where
        lowerToLLVM :: MLIR.Module -> IO ()
        lowerToLLVM m = do
          ctx <- MLIR.getContext m
          MLIR.withPassManager ctx \pm -> do
            MLIR.addConvertStandardToLLVMPass pm
            result <- MLIR.runPasses pm m
            when (result == MLIR.Failure) $ error "Failed to lower to LLVM!"

        run :: forall result. Storable result
            => MLIR.Module -> BS.ByteString -> [MLIR.SomeStorable] -> IO result
        run m name args = do
          MLIR.withExecutionEngine m \maybeEng -> do
            let eng = fromMaybe (error "Failed to compile the module") maybeEng
            MLIR.withStringRef name $ \nameRef -> do
              maybeValue <- MLIR.executionEngineInvoke eng nameRef args
              case maybeValue of
                Just value -> return value
                Nothing -> error "Failed to run the example program!"

main :: IO ()
main = hspec spec
