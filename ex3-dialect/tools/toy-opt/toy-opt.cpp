#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
// 导入 Func Dialect
#include "mlir/Dialect/Func/IR/FuncOps.h"
// 导入 MLIR 自带 Pass
#include "mlir/Transforms/Passes.h"
// 导入我们新建的 Dialect
#include "toy/ToyDialect.h"
using namespace mlir;
using namespace llvm;

int main(int argc, char ** argv) {
  DialectRegistry registry;
  // 注册 Dialect
  registry.insert<toy::ToyDialect, func::FuncDialect>();
  // 注册两个 Pass
  registerCSEPass();
  registerCanonicalizerPass();
  return asMainReturnCode(MlirOptMain(argc, argv, "toy-opt", registry));
}