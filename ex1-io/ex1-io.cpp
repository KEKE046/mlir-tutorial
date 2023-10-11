#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

int main(int argc, char ** argv) {
  MLIRContext ctx;
  // 首先，注册需要的 dialect
  ctx.loadDialect<func::FuncDialect, arith::ArithDialect>();
  // 读入dialect
  auto src = parseSourceFile<ModuleOp>(argv[1], &ctx);
  // 输出dialect
  src->print(llvm::outs());
  // 简单的输出，在 debug 的时候常用
  src->dump();
  return 0;
}