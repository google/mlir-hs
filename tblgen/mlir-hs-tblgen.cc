// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

using generator_function = bool(const llvm::RecordKeeper& recordKeeper,
                                llvm::raw_ostream& os);

struct GeneratorInfo {
  const char* name;
  generator_function* generator;
};

extern generator_function emitOpTableDefs;
extern generator_function emitTestTableDefs;

static std::array<GeneratorInfo, 2> generators {{
  {"hs-op-defs", emitOpTableDefs},
  {"hs-tests", emitTestTableDefs},
}};

generator_function* generator;

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::opt<std::string> generatorOpt("generator", llvm::cl::desc("Generator to run"), cl::Required);
  cl::ParseCommandLineOptions(argc, argv);
  for (const auto& spec : generators) {
    if (generatorOpt == spec.name) {
      generator = spec.generator;
      break;
    }
  }
  if (!generator) {
    llvm::errs() << "Invalid generator type\n";
    abort();
  }

  return TableGenMain(argv[0], [](raw_ostream& os, const RecordKeeper &records) {
    return generator(records, os);
  });
}
