#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#define GEN_PASS_DEF_DCE
#include "toy/ToyPasses.h"

using namespace mlir;
using namespace llvm;
using namespace toy;

struct DCEPass : toy::impl::DCEBase<DCEPass> {
  void visitAll(llvm::DenseSet<Operation*> &visited, Operation * op) {
    if(visited.contains(op)) return;
    visited.insert(op);
    for(auto operand: op->getOperands()) 
      if(auto def = operand.getDefiningOp()) 
        visitAll(visited, def);
  }
  void runOnOperation() final {
    llvm::DenseSet<Operation*> visited;
    getOperation()->walk([&](toy::ReturnOp op) {
      visitAll(visited, op);
    });
    llvm::SmallVector<Operation*> opToRemove;
    getOperation().walk([&](Operation * op) {
      if(op == getOperation()) return;
      if(!visited.contains(op)) opToRemove.push_back(op);
    });
    for(auto v: reverse(opToRemove)) {
      v->erase();
    }
  }
};

std::unique_ptr<mlir::Pass> toy::createDCEPass() {
  return std::make_unique<DCEPass>();
}
