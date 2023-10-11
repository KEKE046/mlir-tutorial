#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "toy/ToyDialect.h"
#include "toy/ToyOps.h"

#include "toy/ToyDialect.cpp.inc"
#define GET_OP_CLASSES
#include "toy/Toy.cpp.inc"

#include "toy/ToyTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#define GET_TYPEDEF_CLASSES
#include "toy/ToyTypes.cpp.inc"

using namespace mlir;
using namespace toy;

void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Toy.cpp.inc"
  >();
  registerTypes();
}

void ToyDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "toy/ToyTypes.cpp.inc"
      >();
}

// mlir::LogicalResult ConstantOp::inferReturnTypes(
//   mlir::MLIRContext * context,
//   std::optional<::mlir::Location> location,
//   mlir::ValueRange operands,
//   mlir::DictionaryAttr attributes,
//   mlir::OpaqueProperties properties,
//   mlir::RegionRange regions,
//   llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes
// ) {
//   ConstantOp::Adaptor adaptor(operands, attributes, properties, regions);
//   inferredReturnTypes.push_back(adaptor.getValueAttr().getType());
//   return success();
// }

mlir::LogicalResult ConstantOp::inferReturnTypes(
  mlir::MLIRContext * context,
  std::optional<mlir::Location> location,
  Adaptor adaptor,
  llvm::SmallVectorImpl<mlir::Type> & inferedReturnType
) {
  inferedReturnType.push_back(adaptor.getValueAttr().getType());
  return mlir::success();
}

mlir::ParseResult FuncOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  auto buildFuncType = [](auto & builder, auto argTypes, auto results, auto, auto) {
    return builder.getFunctionType(argTypes, results);
  };
  return function_interface_impl::parseFunctionOp(
    parser, result, false, 
    getFunctionTypeAttrName(result.name), buildFuncType, 
    getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name)
  );
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operation.
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}
