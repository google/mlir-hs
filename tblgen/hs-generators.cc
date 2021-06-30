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

#include <functional>
#include <regex>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatCommon.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/TableGen/Argument.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/OpClass.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Region.h"
#include "mlir/TableGen/SideEffects.h"
#include "mlir/TableGen/Trait.h"

namespace {

llvm::cl::opt<bool> ExplainMissing(
    "explain-missing",
    llvm::cl::desc("Print the reason for skipping operations from output"));

llvm::cl::opt<std::string> StripOpPrefix(
    "strip-prefix", llvm::cl::desc("Prefix to strip from def names"),
    llvm::cl::value_desc("prefix"));

llvm::cl::opt<std::string> DialectName(
    "dialect-name", llvm::cl::desc("Override the inferred dialect name"),
    llvm::cl::value_desc("dialect"));

template <class C>
llvm::iterator_range<typename C::const_iterator> make_range(const C& x) {
  return llvm::make_range(x.begin(), x.end());
}

template <class C, class FunTy,
          typename ResultTy = decltype(std::declval<FunTy>()(
              std::declval<typename C::value_type>()))>
std::vector<ResultTy> map_vector(const C& container, FunTy f) {
  std::vector<ResultTy> results;
  for (const auto& v : container) {
    results.push_back(f(v));
  }
  return results;
}

void warn(llvm::StringRef op_name, const std::string& reason) {
  if (!ExplainMissing) return;
  llvm::errs() << llvm::formatv(
      "{0} {1}\n", llvm::fmt_align(op_name, llvm::AlignStyle::Left, 40),
      reason);
}

void warn(const mlir::tblgen::Operator& op, const std::string& reason) {
  warn(op.getOperationName(), reason);
}

struct AttrHandler {
  std::string _pattern;
  std::string type;

  std::string pattern(std::string name) {
    return llvm::formatv(_pattern.c_str(), name);
  }
};
using attr_handler_map = llvm::StringMap<AttrHandler>;

const attr_handler_map& getAttrHandlers() {
  static const attr_handler_map* kAttrHandlers = new attr_handler_map{
      {"AnyAttr", {"{0}", "Attribute"}},
      {"AffineMapArrayAttr", {"PatternUtil.AffineMapArrayAttr {0}", "[Affine.Map]"}},
      {"AffineMapAttr", {"AffineMapAttr {0}", "Affine.Map"}},
      {"ArrayAttr", {"ArrayAttr {0}", "[Attribute]"}},
      {"BoolAttr", {"BoolAttr {0}", "Bool"}},
      {"DictionaryAttr", {"DictionaryAttr {0}", "(M.Map Name Attribute)"}},
      {"I32Attr", {"IntegerAttr (IntegerType Signless 32) {0}", "Int"}},
      {"I64ArrayAttr", {"PatternUtil.I64ArrayAttr {0}", "[Int]"}},
      {"IndexAttr", {"IntegerAttr IndexType {0}", "Int"}},
      {"StrAttr", {"StringAttr {0}", "BS.ByteString"}},
  };
  return *kAttrHandlers;
}

const std::string sanitizeName(llvm::StringRef name) {
  static const llvm::StringSet<>* kReservedNames = new llvm::StringSet<>{
      // TODO(apaszke): Add more keywords
      // Haskell keywords
      "in",
  };
  if (kReservedNames->contains(name)) {
    auto new_name = name.str();
    new_name.push_back('_');
    return new_name;
  } else {
    return name.str();
  }
}

std::string getDialectName(llvm::ArrayRef<llvm::Record*> op_defs) {
  mlir::tblgen::Operator any_op(op_defs.front());
  assert(
      std::all_of(op_defs.begin(), op_defs.end(), [&any_op](llvm::Record* op) {
        return mlir::tblgen::Operator(op).getDialectName() ==
               any_op.getDialectName();
      }));
  std::string dialect_name;
  if (DialectName.empty()) {
    dialect_name = any_op.getDialectName().str();
    dialect_name[0] = llvm::toUpper(dialect_name[0]);
  } else {
    dialect_name = DialectName;
  }
  return dialect_name;
}

class AttrPattern {
  AttrPattern(std::string name, std::vector<std::string> binders,
              std::vector<mlir::tblgen::NamedAttribute> attrs)
      : name(std::move(name)),
        binders(std::move(binders)),
        attrs(std::move(attrs)) {
    for (const auto& nattr : this->attrs) {
      types.push_back(getHandlerType(nattr));
    }
  }

 public:
  static llvm::Optional<AttrPattern> buildFor(mlir::tblgen::Operator& op) {
    if (op.getNumAttributes() == 0) return AttrPattern("NoAttrs", {}, {});

    const auto& attr_handlers = getAttrHandlers();
    std::vector<std::string> binders;
    std::vector<mlir::tblgen::NamedAttribute> attrs;
    for (const auto& named_attr : op.getAttributes()) {
      // Derived attributes are never materialized and don't have to be
      // specified.
      if (named_attr.attr.isDerivedAttr()) continue;

      auto handler_it = attr_handlers.find(named_attr.attr.getAttrDefName());
      if (handler_it == attr_handlers.end()) {
        if (named_attr.attr.hasDefaultValue()) {
          warn(op, llvm::formatv("unsupported attr {0} (but has default value)",
                                 named_attr.attr.getAttrDefName()));
          continue;
        }
        if (named_attr.attr.isOptional()) {
          warn(op, llvm::formatv("unsupported attr {0} (but is optional)",
                                 named_attr.attr.getAttrDefName()));
          continue;
        }
        warn(op, llvm::formatv("unsupported attr ({0})",
                               named_attr.attr.getAttrDefName()));
        return llvm::None;
      }
      std::string attr_arg_name = sanitizeName(named_attr.name.str());
      binders.push_back(attr_arg_name);
      attrs.push_back(named_attr);
    }
    if (binders.empty()) return AttrPattern("NoAttrs", {}, {});
    std::string name = "Internal" + op.getCppClassName().str() + "Attributes";
    return AttrPattern(std::move(name), std::move(binders), std::move(attrs));
  }

  using print_state = llvm::StringSet<>;

  void print(llvm::raw_ostream& os, print_state& optional_attr_defs) {
    if (name == "NoAttrs") return;
    // `M.lookup "attr_name" m` for every attribute
    std::vector<std::string> lookups;
    // Patterns from handlers, but wrapped in "Just (...)" when non-optional
    std::vector<std::string> lookup_patterns;
    // `[("attr_name", attr_pattern)]` for every non-optional attribute
    std::vector<std::string> singleton_pairs;
    for (size_t i = 0; i < attrs.size(); ++i) {
      const mlir::tblgen::NamedAttribute& nattr = attrs[i];
      AttrHandler handler = getHandler(nattr, os, optional_attr_defs);
      lookups.push_back(llvm::formatv("M.lookup \"{0}\" m", nattr.name));
      std::string pattern = handler.pattern(binders[i]);
      if (nattr.attr.isOptional()) {
        lookup_patterns.push_back(pattern);
        singleton_pairs.push_back(llvm::formatv(
            "(Data.Maybe.maybeToList $ (\"{0}\",) <$> {1})", nattr.name, pattern));
      } else {
        lookup_patterns.push_back(llvm::formatv("Just ({0})", pattern));
        singleton_pairs.push_back(
            llvm::formatv("[(\"{0}\", {1})]", nattr.name, pattern));
      }
    }
    const char* kAttributePattern = R"(
pattern {0} :: {1:$[ -> ]} -> NamedAttributes
pattern {0} {2:$[ ]} <- ((\m -> ({3:$[, ]})) -> ({4:$[, ]}))
  where {0} {2:$[ ]} = M.fromList $ {5:$[ ++ ]}
)";
    os << llvm::formatv(kAttributePattern,
                        name,                          // 0
                        make_range(types),             // 1
                        make_range(binders),           // 2
                        make_range(lookups),           // 3
                        make_range(lookup_patterns),   // 4
                        make_range(singleton_pairs));  // 5
  }

  std::string getHandlerType(const mlir::tblgen::NamedAttribute& nattr) {
    llvm::StringRef attr_kind = nattr.attr.getAttrDefName();
    assert(getAttrHandlers().count(attr_kind) == 1);
    AttrHandler handler = getAttrHandlers().lookup(attr_kind);
    if (!nattr.attr.isOptional()) return handler.type;
    return "Maybe (" + handler.type + ")";
  }

  AttrHandler getHandler(const mlir::tblgen::NamedAttribute& nattr,
                         llvm::raw_ostream& os,
                         print_state& optional_attr_defs) {
    llvm::StringRef attr_kind = nattr.attr.getAttrDefName();
    assert(getAttrHandlers().count(attr_kind) == 1);
    AttrHandler handler = getAttrHandlers().lookup(attr_kind);
    if (!nattr.attr.isOptional()) return handler;
    if (!optional_attr_defs.contains(attr_kind)) {
      const char* kOptionalHandler = R"(
pattern Optional{0} :: Maybe {1} -> Maybe Attribute
pattern Optional{0} x <- ((\case Just ({2}) -> Just y; Nothing -> Nothing) -> x)
  where Optional{0} x = case x of Just y -> Just ({2}); Nothing -> Nothing
)";
      os << llvm::formatv(kOptionalHandler, attr_kind, handler.type,
                          handler.pattern("y"));
      optional_attr_defs.insert(attr_kind);
    }
    return {"Optional" + attr_kind.str() + " {0}",
            "Maybe (" + handler.type + ")"};
  }

  std::string name;
  std::vector<std::string> binders;
  std::vector<std::string> types;

 private:
  std::vector<mlir::tblgen::NamedAttribute> attrs;
};

llvm::Optional<std::string> buildOperation(
    const llvm::Record* def, bool is_pattern, const std::string& what_for,
    const std::string& location_expr,
    const std::vector<std::string>& type_exprs,
    const std::vector<std::string>& operand_exprs,
    const std::vector<std::string>& region_exprs,
    const AttrPattern& attr_pattern) {
  mlir::tblgen::Operator op(def);
  auto fail = [&op, &what_for](std::string reason) {
    warn(op, llvm::formatv("couldn't construct {0}: {1}", what_for, reason));
    return llvm::Optional<std::string>();
  };

  // Skip currently unsupported cases
  if (op.getNumVariadicRegions() != 0) return fail("variadic regions");
  if (op.getNumSuccessors() != 0) return fail("successors");

  // Prepare results
  std::string type_expr;
  if (op.getNumResults() == 0) {
    assert(type_exprs.size() == op.getNumResults());
    type_expr = "[]";
  } else if (op.getNumVariableLengthResults() == 0 &&
             op.getTrait("::mlir::OpTrait::SameOperandsAndResultType")) {
    assert(type_exprs.size() == 1);
    type_expr = llvm::formatv("[{0:$[, ]}]",
                              make_range(std::vector<llvm::StringRef>(
                                  op.getNumResults(), type_exprs.front())));
  } else if (op.getNumVariableLengthResults() == 0) {
    assert(type_exprs.size() == op.getNumResults());
    type_expr = llvm::formatv("[{0:$[, ]}]", make_range(type_exprs));
  } else if (!is_pattern) {
    assert(type_exprs.size() == op.getNumResults());
    std::vector<std::string> list_type_exprs;
    for (int i = 0; i < op.getNumResults(); ++i) {
      auto& result = op.getResult(i);
      if (result.isOptional()) {
        list_type_exprs.push_back("(Data.Maybe.maybeToList " + type_exprs[i] + ")");
      } else if (result.isVariadic()) {
        list_type_exprs.push_back(type_exprs[i]);
      } else {
        assert(!result.isVariableLength());
        list_type_exprs.push_back("[" + type_exprs[i] + "]");
      }
    }
    type_expr = llvm::formatv("({0:$[ ++ ]})", make_range(list_type_exprs));
  } else {
    return fail("unsupported variable length results");
  }

  // Prepare operands
  std::string operand_expr;
  assert(operand_exprs.size() == op.getNumOperands());
  if (op.getNumOperands() == 1 && op.getOperand(0).isVariadic()) {
    // Note that this expr already should represent a list
    operand_expr = operand_exprs.front();
  } else if (op.getNumVariableLengthOperands() == 0) {
    operand_expr = llvm::formatv("[{0:$[, ]}]", make_range(operand_exprs));
  } else if (!is_pattern) {
    std::vector<std::string> operand_list_exprs;
    for (int i = 0; i < op.getNumOperands(); ++i) {
      auto& operand = op.getOperand(i);
      if (operand.isOptional()) {
        operand_list_exprs.push_back("(Data.Maybe.maybeToList " + operand_exprs[i] + ")");
      } else if (operand.isVariadic()) {
        operand_list_exprs.push_back(operand_exprs[i]);
      } else {
        assert(!operand.isVariableLength());
        operand_list_exprs.push_back("[" + operand_exprs[i] + "]");
      }
    }
    operand_expr =
        llvm::formatv("({0:$[ ++ ]})", make_range(operand_list_exprs));
  } else {
    return fail("unsupported variable length operands");
  }

  std::string extra_attrs;
  if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments")) {
    std::vector<std::string> segment_sizes;
    for (int i = 0; i < op.getNumOperands(); ++i) {
      auto& operand = op.getOperand(i);
      if (operand.isOptional()) {
        segment_sizes.push_back(llvm::formatv(
            "case {0} of Just _ -> 1; Nothing -> 0", operand_exprs[i]));
      } else if (operand.isVariadic()) {
        segment_sizes.push_back("Prelude.length " + operand_exprs[i]);
      } else {
        assert(!operand.isVariableLength());
        segment_sizes.push_back("1");
      }
    }
    const char* kOperandSegmentsAttr = R"(
              <> AST.namedAttribute "operand_segment_sizes"
                   (DenseElementsAttr (VectorType [{0}] $ IntegerType Unsigned 32) $
                      DenseUInt32 $ IArray.listArray (1 :: Int, {0}) $ Prelude.fromIntegral <$> [{1:$[, ]}])
)";
    extra_attrs = llvm::formatv(kOperandSegmentsAttr,
                                segment_sizes.size(),
                                make_range(segment_sizes));
  }

  const char* kPatternExplicitType = R"(Operation
          { opName = "{0}"
          , opLocation = {1}
          , opResultTypes = Explicit {2}
          , opOperands = {3}
          , opRegions = [{4:$[ ]}]
          , opSuccessors = []
          , opAttributes = ({5}{6}{7:$[ ]}){8}
          })";
  return llvm::formatv(kPatternExplicitType,
                       op.getOperationName(),                    // 0
                       location_expr,                            // 1
                       type_expr,                                // 2
                       operand_expr,                             // 3
                       make_range(region_exprs),                 // 4
                       attr_pattern.name,                        // 5
                       attr_pattern.binders.empty() ? "" : " ",  // 6
                       make_range(attr_pattern.binders),         // 7
                       extra_attrs)                              // 8
      .str();
}

// TODO(apaszke): Make this more reliable
std::string legalizeBuilderName(std::string name) {
  for (size_t i = 0; i < name.size(); ++i) {
    if (name[i] == '.') name[i] = '_';
  }
  return name;
}

std::string stripDialect(std::string name) {
  size_t dialect_sep_loc = name.find('.');
  assert(dialect_sep_loc != std::string::npos);
  return name.substr(dialect_sep_loc + 1);
}

void emitBuilderMethod(mlir::tblgen::Operator& op,
                       const AttrPattern& attr_pattern, llvm::raw_ostream& os) {
  auto fail = [&op](std::string reason) {
    warn(op, "couldn't construct builder: " + reason);
  };

  if (op.getNumVariadicRegions() != 0) return fail("variadic regions");
  if (op.getNumSuccessors() != 0) return fail("successors");

  const char* result_type;
  std::string prologue;
  const char* continuation = "";
  if (op.getNumResults() == 0) {
    prologue = "Control.Monad.void ";
    if (op.getTrait("::mlir::OpTrait::IsTerminator")) {
      result_type = "EndOfBlock";
      continuation = "\n  AST.terminateBlock";
    } else {
      result_type = "()";
    }
  } else if (op.getNumResults() == 1) {
    result_type = "Value";
    prologue = "Control.Monad.liftM Prelude.head ";
  } else {
    result_type = "[Value]";
    prologue = "";
  }

  std::string builder_name = legalizeBuilderName(stripDialect(op.getOperationName()));

  std::vector<std::string> builder_arg_types;

  // TODO(apaszke): Use inference (op.getSameTypeAsResult)
  std::vector<std::string> type_exprs;
  std::vector<std::string> type_binders;
  if (op.getNumResults() == 0) {
    // Nothing to do.
  } else if (op.getNumVariableLengthResults() == 0 &&
             op.getTrait("::mlir::OpTrait::SameOperandsAndResultType")) {
    for (const mlir::tblgen::NamedTypeConstraint& operand : op.getOperands()) {
      if (operand.isVariableLength()) continue;
      type_exprs.push_back("(AST.typeOf " + sanitizeName(operand.name) + ")");
      break;
    }
    if (type_exprs.empty()) return fail("type inference failed");
  } else {
    int result_nr = 0;
    for (const mlir::tblgen::NamedTypeConstraint& result : op.getResults()) {
      type_binders.push_back(llvm::formatv("ty{0}", result_nr++));
      type_exprs.push_back(type_binders.back());
      if (result.isOptional()) {
        builder_arg_types.push_back("Maybe Type");
      } else if (result.isVariadic()) {
        builder_arg_types.push_back("[Type]");
      } else {
        assert(!result.isVariableLength());
        builder_arg_types.push_back("Type");
      }
    }
  }

  std::vector<std::string> operand_binders;
  std::vector<std::string> operand_name_exprs;
  operand_name_exprs.reserve(op.getNumOperands());
  for (const auto& operand : op.getOperands()) {
    std::string operand_name = sanitizeName(operand.name);
    operand_binders.push_back(operand_name);
    if (operand.isOptional()) {
      builder_arg_types.push_back("Maybe Value");
      operand_name_exprs.push_back("(AST.operand <$> " + operand_name + ")");
    } else if (operand.isVariadic()) {
      builder_arg_types.push_back("[Value]");
      operand_name_exprs.push_back("(AST.operands " + operand_name + ")");
    } else {
      assert(!operand.isVariableLength());
      builder_arg_types.push_back("Value");
      operand_name_exprs.push_back("(AST.operand " + operand_name + ")");
    }
  }

  builder_arg_types.insert(builder_arg_types.end(), attr_pattern.types.begin(),
                           attr_pattern.types.end());

  std::vector<std::string> region_builder_binders;
  std::vector<std::string> region_binders;
  if (op.getNumRegions() > 0) {
    std::string region_prologue;
    for (const mlir::tblgen::NamedRegion& region : op.getRegions()) {
      region_builder_binders.push_back(sanitizeName(region.name) + "Builder");
      region_binders.push_back(sanitizeName(region.name));
      builder_arg_types.push_back("RegionBuilderT m ()");
      region_prologue += llvm::formatv(
          "{0} <- AST.buildRegion {1}\n  ",
          region_binders.back(), region_builder_binders.back());
    }
    prologue = region_prologue + prologue;
  }

  builder_arg_types.push_back("");  // To add the arrow before m

  llvm::Optional<std::string> operation =
      buildOperation(&op.getDef(), false, "builder", "UnknownLocation",
                     type_exprs, operand_name_exprs, region_binders,
                     attr_pattern);
  if (!operation) return;

  const char* kBuilder = R"(
-- | A builder for @{10}@.
{0} :: MonadBlockBuilder m => {1:$[ -> ]}m {2}
{0} {3:$[ ]} {4:$[ ]} {5:$[ ]} {6:$[ ]} = do
  {7}(AST.emitOp ({8})){9}
)";
  os << llvm::formatv(kBuilder,
                      builder_name,                        // 0
                      make_range(builder_arg_types),       // 1
                      result_type,                         // 2
                      make_range(type_binders),            // 3
                      make_range(operand_binders),         // 4
                      make_range(attr_pattern.binders),    // 5
                      make_range(region_builder_binders),  // 6
                      prologue,                            // 7
                      *operation,                          // 8
                      continuation,                        // 9
                      op.getOperationName());              // 10
}

void emitPattern(const llvm::Record* def, const AttrPattern& attr_pattern,
                 llvm::raw_ostream& os) {
  mlir::tblgen::Operator op(def);
  auto fail = [&op](std::string reason) {
    return warn(op, llvm::formatv("couldn't construct pattern: {0}", reason));
  };

  // Skip currently unsupported cases
  if (op.getNumVariableLengthResults() != 0) return fail("variadic results");
  if (op.getNumRegions() != 0) return fail("regions");
  if (op.getNumSuccessors() != 0) return fail("successors");
  if (!def->getName().endswith("Op")) return fail("unsupported name format");
  if (!def->getName().startswith(StripOpPrefix)) return fail("missing prefix");

  // Drop the stripped prefix and "Op" from the end.
  llvm::StringRef pattern_name =
      def->getName().drop_back(2).drop_front(StripOpPrefix.length());

  std::vector<std::string> pattern_arg_types{"Location"};

  // Prepare results
  std::vector<std::string> type_binders;
  if (op.getNumResults() > 0 &&
      op.getTrait("::mlir::OpTrait::SameOperandsAndResultType")) {
    assert(op.getNumVariableLengthResults() == 0);
    pattern_arg_types.push_back("Type");
    type_binders.push_back("ty");
  } else {
    size_t result_count = 0;
    for (int i = 0; i < op.getNumResults(); ++i) {
      pattern_arg_types.push_back("Type");
      type_binders.push_back(llvm::formatv("ty{0}", result_count++));
    }
  }

  // Prepare operands
  std::vector<std::string> operand_binders;
  if (op.getNumOperands() == 1 && op.getOperand(0).isVariadic()) {
    // Single variadic arg is easy to handle
    pattern_arg_types.push_back("[operand]");
    operand_binders.push_back(sanitizeName(op.getOperand(0).name));
  } else {
    // Non-variadic case
    for (auto operand : op.getOperands()) {
      if (operand.isVariableLength())
        return fail("unsupported variable length operand");
      pattern_arg_types.push_back("operand");
      operand_binders.push_back(sanitizeName(operand.name));
    }
  }

  // Prepare attribute pattern
  pattern_arg_types.insert(pattern_arg_types.end(), attr_pattern.types.begin(),
                           attr_pattern.types.end());

  llvm::Optional<std::string> operation = buildOperation(
      def, true, "pattern", "loc",
      type_binders, operand_binders, {}, attr_pattern);
  if (!operation) return;

  const char* kPatternExplicitType = R"(
-- | A pattern for @{6}@.
pattern {0} :: {1:$[ -> ]} -> AbstractOperation operand
pattern {0} loc {2:$[ ]} {3:$[ ]} {4:$[ ]} = {5}
)";
  os << llvm::formatv(kPatternExplicitType,
                      pattern_name,                      // 0
                      make_range(pattern_arg_types),     // 1
                      make_range(type_binders),          // 2
                      make_range(operand_binders),       // 3
                      make_range(attr_pattern.binders),  // 4
                      *operation,                        // 5
                      op.getOperationName());            // 6

}

std::string formatDescription(mlir::tblgen::Operator op) {
  std::string description;
  description = "\n" + op.getDescription().str();
  size_t pos = 0;
  while (description[pos] == '\n') ++pos;
  size_t leading_spaces = 0;
  while (description[pos++] == ' ') ++leading_spaces;
  if (leading_spaces) {
    std::string leading_spaces_str;
    for (size_t i = 0; i < leading_spaces; ++i) leading_spaces_str += "[ ]";
    description = std::regex_replace(description, std::regex("\n" + leading_spaces_str), "\n");
  }
  description = std::regex_replace(description, std::regex("\\[(.*)\\]\\(.*\\)"), "$1");
  description = std::regex_replace(description, std::regex("(['\"@<$#])"), "\\$1");
  description = std::regex_replace(description, std::regex("```mlir"), "@");
  description = std::regex_replace(description, std::regex("```"), "@");
  description = std::regex_replace(description, std::regex("`"), "@");
  description = std::regex_replace(description, std::regex("\n"), "\n-- ");
  return description;
}

}  // namespace

bool emitOpTableDefs(const llvm::RecordKeeper& recordKeeper,
                     llvm::raw_ostream& os) {
  std::vector<llvm::Record*> defs = recordKeeper.getAllDerivedDefinitions("Op");

  if (defs.empty()) return true;
  // TODO(apaszke): Emit a module header to avoid leaking internal definitions.
  auto dialect_name = getDialectName(defs);
  os << "{-# OPTIONS_GHC -Wno-unused-imports #-}\n";
  os << "{-# OPTIONS_HADDOCK hide, prune, not-home #-}\n\n";
  os << "module MLIR.AST.Dialect.Generated." << dialect_name << " where\n";
  os << R"(
import Prelude (Int, Maybe(..), Bool(..), (++), (<$>), ($), (<>))
import qualified Prelude
import qualified Data.Maybe
import qualified Data.Array.IArray as IArray
import qualified Data.ByteString as BS
import qualified Data.Map.Strict as M
import qualified Control.Monad

import MLIR.AST ( Attribute(..), Type(..), AbstractOperation(..), ResultTypes(..)
                , Location(..), Signedness(..), DenseElements(..)
                , NamedAttributes, Name
                , pattern NoAttrs )
import qualified MLIR.AST as AST
import MLIR.AST.Builder (Value, EndOfBlock, MonadBlockBuilder, RegionBuilderT)
import qualified MLIR.AST.Builder as AST
import qualified MLIR.AST.PatternUtil as PatternUtil
import qualified MLIR.AST.Dialect.Affine as Affine
)";

  AttrPattern::print_state attr_pattern_state;
  for (const auto* def : defs) {
    mlir::tblgen::Operator op(*def);
    if (op.hasDescription()) {
      os << llvm::formatv("\n-- * {0}\n-- ${0}", stripDialect(op.getOperationName()));
      os << formatDescription(op);
      os << "\n";
    }
    llvm::Optional<AttrPattern> attr_pattern = AttrPattern::buildFor(op);
    if (!attr_pattern) continue;
    attr_pattern->print(os, attr_pattern_state);
    emitPattern(def, *attr_pattern, os);
    emitBuilderMethod(op, *attr_pattern, os);
  }

  return false;
}

bool emitTestTableDefs(const llvm::RecordKeeper& recordKeeper,
                       llvm::raw_ostream& os) {
  std::vector<llvm::Record*> defs = recordKeeper.getAllDerivedDefinitions("Op");
  if (defs.empty()) return true;

  auto dialect_name = getDialectName(defs);
  os << "{-# OPTIONS_GHC -Wno-unused-imports #-}\n\n";
  const char* module_header = R"(
module MLIR.AST.Dialect.Generated.{0}Spec where

import Prelude (IO, Maybe(..), ($), (<>))
import qualified Prelude
import Test.Hspec (Spec)
import qualified Test.Hspec as Hspec
import Test.QuickCheck ((===))
import qualified Test.QuickCheck as QC

import MLIR.AST (pattern NoAttrs)
import MLIR.AST.Dialect.{0}

import MLIR.Test.Generators ()

main :: IO ()
main = Hspec.hspec spec

spec :: Spec
spec = do
)";
  os << llvm::formatv(module_header, dialect_name);
  for (const auto* def : defs) {
    mlir::tblgen::Operator op(*def);
    llvm::Optional<AttrPattern> attr_pattern = AttrPattern::buildFor(op);
    if (!attr_pattern) continue;
    os << "\n  Hspec.describe \"" << op.getOperationName() << "\" $ do";
    const char* bidirectional_test_template = R"(
    Hspec.it "has a bidirectional attr pattern" $ do
      let wrapUnwrap ({1:$[, ]}) = case ({0} {1:$[ ]}) <> Prelude.mempty of
              {0} {2:$[ ]} -> Just ({2:$[, ]})
              _ -> Nothing
      QC.property $ \args -> wrapUnwrap args === Just args
)";
    os << llvm::formatv(
        bidirectional_test_template, attr_pattern->name,
        make_range(attr_pattern->binders),
        make_range(map_vector(attr_pattern->binders, [](const std::string& b) {
          return b + "_match";
        })));
    const char* pattern_extensibility_test_template = R"(
    Hspec.it "accepts additional attributes" $ do
      QC.property $ do
        ({1:$[, ]}) <- QC.arbitrary
        extraAttrs <- QC.arbitrary
        let match = case ({0} {1:$[ ]}) <> extraAttrs of
              {0} {2:$[ ]} -> Just ({2:$[, ]})
              _ -> Nothing
        Prelude.return $ match === Just ({1:$[, ]})
)";
    os << llvm::formatv(
        pattern_extensibility_test_template, attr_pattern->name,
        make_range(attr_pattern->binders),
        make_range(map_vector(attr_pattern->binders, [](const std::string& b) {
          return b + "_match";
        })));
    // TODO(apaszke): Test attr pattern matches with more attributes.
    // TODO(apaszke): Test bidirectionality of op pattern.
    // TODO(apaszke): Test op pattern matches with more attributes.
    // TODO(apaszke): Test builder output matches op pattern.
    // TODO(apaszke): Figure out how to do tests with translation.
  }
  return false;
}
