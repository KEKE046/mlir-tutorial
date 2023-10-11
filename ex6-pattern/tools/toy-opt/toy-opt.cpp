#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// 导入 Func Dialect
#include "mlir/Dialect/Func/IR/FuncOps.h"
// 导入 MLIR 自带 Pass
#include "mlir/Transforms/Passes.h"
// 导入我们新建的 Dialect
#include "toy/ToyDialect.h"
#include "toy/ToyPasses.h"
using namespace mlir;
using namespace llvm;

int main(int argc, char ** argv) {
  DialectRegistry registry;
  // 注册 Dialect
  registry.insert<toy::ToyDialect, func::FuncDialect, arith::ArithDialect>();
  // registry.insert<LLVM::LLVMDialect>();
  // 注册两个 Pass
  registerCSEPass();
  registerCanonicalizerPass();
  toy::registerPasses();
  return asMainReturnCode(MlirOptMain(argc, argv, "toy-opt", registry));
}
