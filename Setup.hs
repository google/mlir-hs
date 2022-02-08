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

import Data.Char
import Data.List
import Data.Maybe

import System.Directory
import System.FilePath

import Distribution.ModuleName hiding (main)
import Distribution.Simple
import Distribution.Simple.Setup
import Distribution.Simple.Program
import Distribution.Types.BuildInfo.Lens
import Distribution.Types.Library.Lens
import Distribution.Types.TestSuite.Lens
import Distribution.Types.GenericPackageDescription
import Distribution.Types.CondTree

import Control.Monad
import Control.Lens.Setter
import Control.Lens.Operators ((&))

llvmVersion :: Version
llvmVersion = mkVersion [15]

llvmConfigProgram :: Program
llvmConfigProgram = (simpleProgram "llvm-config")
  { programFindVersion =
      findProgramVersion "--version" (takeWhile (\c -> isDigit c || c == '.'))
  }

getLLVMConfig :: ConfigFlags -> IO ([String] -> IO String)
getLLVMConfig confFlags = do
  (program, _, _) <- requireProgramVersion
                       verbosity
                       llvmConfigProgram
                       (withinVersion llvmVersion)
                       (configPrograms confFlags)
  return $ getProgramOutput verbosity program
  where verbosity = fromFlag $ configVerbosity confFlags

ccProgram = simpleProgram "c++"

getCC :: ConfigFlags -> IO ([String] -> IO ())
getCC confFlags = do
  (program, _) <- requireProgram verbosity ccProgram (configPrograms confFlags)
  return $ runProgram verbosity program
  where verbosity = fromFlag $ configVerbosity confFlags

isIncludeDir :: String -> Bool
isIncludeDir = ("-I" `isPrefixOf`)

isLibDir :: String -> Bool
isLibDir = ("-L" `isPrefixOf`)

data TblGenerator = OpGenerator | TestGenerator
instance Show TblGenerator where
  show OpGenerator   = "hs-op-defs"
  show TestGenerator = "hs-tests"

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

buildTblgen :: ConfigFlags -> IO (TblGenerator -> FilePath -> FilePath -> [ProgArg] -> IO ())
buildTblgen confFlags = do
  -- TODO(apaszke): Cache compilation.
  cwd <- getCurrentDirectory
  llvmConfig <- getLLVMConfig confFlags
  cxxFlags   <- words <$> llvmConfig ["--cxxflags"]
  ldFlags    <- words <$> llvmConfig ["--ldflags"]
  cppFlags   <- words <$> llvmConfig ["--cppflags"]
  includeDir <- trim  <$> llvmConfig ["--includedir"]
  cc <- getCC confFlags
  ensureDirectory $ cwd </> ".bin"
  cc $ sources ++ cxxFlags ++ ldFlags ++
        [ "-lMLIR", "-lLLVM", "-lMLIRTableGen", "-lLLVMTableGen"
        , "-o", cwd </> ".bin/mlir-hs-tblgen"]
  let tblgenProgram = ConfiguredProgram
        { programId           = "mlir-hs-tblgen"
        , programVersion      = Nothing
        , programDefaultArgs  = ("-I" <> includeDir) : cppFlags
        , programOverrideArgs = []
        , programOverrideEnv  = []
        , programProperties   = mempty
        , programLocation     = FoundOnSystem $ cwd </> ".bin/mlir-hs-tblgen"
        , programMonitorFiles = []
        }
  return $ \generator tdPath outputPath opts -> do
    putStrLn $ "Generating " <> (cwd </> outputPath)
    runProgram verbosity tblgenProgram $
        [ "--write-if-changed"
        , "--generator", show generator
        , includeDir </> tdPath
        , "-o", cwd </> outputPath
        ] ++ opts
  where
    verbosity = fromFlag $ configVerbosity confFlags
    sources =
      [ "tblgen/mlir-hs-tblgen.cc"
      , "tblgen/hs-generators.cc"
      ]

ensureDirectory :: FilePath -> IO ()
ensureDirectory path =
    mapM_ ensureDirectoryNonrec $ tail $ scanl' (++) "" $ splitPath path
  where
    ensureDirectoryNonrec dir = do
      exists <- doesDirectoryExist dir
      if exists then return () else createDirectory dir

main :: IO ()
main = defaultMainWithHooks simpleUserHooks
  { hookedPrograms = [ llvmConfigProgram, ccProgram ]
  , confHook = \(genericPackageDesc, hookedBuildInfo) confFlags -> do
      tblgen <- buildTblgen confFlags
      let dialects =
            [ ("Std"             , "mlir/Dialect/StandardOps/IR/Ops.td", [])
            , ("Arith"           , "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td", ["-strip-prefix", "Arith_"])
            , ("ControlFlow"     , "mlir/Dialect/ControlFlow/IR/ControlFlowOps.td", ["-dialect-name", "ControlFlow"])
            , ("Vector"          , "mlir/Dialect/Vector/IR/VectorOps.td", ["-strip-prefix", "Vector_"])
            , ("Shape"           , "mlir/Dialect/Shape/IR/ShapeOps.td", ["-strip-prefix", "Shape_"])
            , ("LLVM"            , "mlir/Dialect/LLVMIR/LLVMOps.td", ["-strip-prefix", "LLVM_", "-dialect-name", "LLVM"])
            , ("Linalg"          , "mlir/Dialect/Linalg/IR/LinalgOps.td", [])
            , ("LinalgStructured", "mlir/Dialect/Linalg/IR/LinalgStructuredOps.td", ["-dialect-name", "LinalgStructured"])
            , ("Tensor"          , "mlir/Dialect/Tensor/IR/TensorOps.td", ["-strip-prefix", "Tensor_"])
            , ("X86Vector"       , "mlir/Dialect/X86Vector/X86Vector.td", ["-dialect-name", "X86Vector"])
            ]
      ensureDirectory "src/MLIR/AST/Dialect/Generated"
      generatedModules <- forM dialects $ \(dialect, tdPath, opts) -> do
        tblgen OpGenerator tdPath ("src/MLIR/AST/Dialect/Generated/" <> dialect <> ".hs") opts
        return $ fromString $ "MLIR.AST.Dialect.Generated." <> dialect

      -- TODO: Do I need to do anything about the rpath?
      llvmConfig  <- getLLVMConfig confFlags
      (llvmLibDirFlags , llvmLdFlags) <- partition isLibDir     . words <$> llvmConfig ["--ldflags"]
      (llvmIncludeFlags, llvmCcFlags) <- partition isIncludeDir . words <$> llvmConfig ["--cflags"]
      let llvmIncludeDirs = (fromJust . (stripPrefix "-I")) <$> llvmIncludeFlags
      let llvmLibDirs     = (fromJust . (stripPrefix "-L")) <$> llvmLibDirFlags
      let Just condLib = condLibrary genericPackageDesc
      let newLibrary = condTreeData condLib
            & over (libBuildInfo . buildInfo . ccOptions     ) (<> llvmCcFlags     )
            & over (libBuildInfo . buildInfo . includeDirs   ) (<> llvmIncludeDirs )
            & over (libBuildInfo . buildInfo . ldOptions     ) (<> llvmLdFlags     )
            & over (libBuildInfo . buildInfo . extraLibDirs  ) (<> llvmLibDirs     )
            & over (libBuildInfo . buildInfo . otherModules  ) (<> generatedModules)
            & over (libBuildInfo . buildInfo . autogenModules) (<> generatedModules)
      let newCondLibrary = condLib { condTreeData = newLibrary }

      ensureDirectory "test/MLIR/AST/Dialect/Generated"
      generatedSpecModules <- liftM catMaybes $ forM dialects $ \(dialect, tdPath, opts) -> do
        case dialect of
          "LinalgStructured" -> return Nothing
          _ -> do
            tblgen TestGenerator tdPath ("test/MLIR/AST/Dialect/Generated/" <> dialect <> "Spec.hs") opts
            return $ Just $ fromString $ "MLIR.AST.Dialect.Generated." <> dialect <> "Spec"
      let [(testSuiteName, condTestSuite)] = condTestSuites genericPackageDesc
      let newTestSuite = condTreeData condTestSuite
            & over (testBuildInfo . otherModules) (<> generatedSpecModules)


      let newGenericPackageDesc = genericPackageDesc
            { condLibrary = Just newCondLibrary
            , condTestSuites = [(testSuiteName, condTestSuite { condTreeData = newTestSuite })]
            }
      confHook simpleUserHooks (newGenericPackageDesc, hookedBuildInfo) confFlags
  }


