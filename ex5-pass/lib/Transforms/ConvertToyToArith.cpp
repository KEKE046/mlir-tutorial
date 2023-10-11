#include "llvm/Support/raw_ostream.h"
#define GEN_PASS_DEF_CONVERTTOYTOARITH
#include "toy/ToyPasses.h"

struct ConvertToyToArithPass : toy::impl::ConvertToyToArithBase<ConvertToyToArithPass> {
  using toy::impl::ConvertToyToArithBase<ConvertToyToArithPass>::ConvertToyToArithBase;
  void runOnOperation() final {
    llvm::errs() << "get name: " << name << "\n";
  }
};

std::unique_ptr<mlir::Pass> toy::createConvertToyToArithPass(ConvertToyToArithOptions options) {
  return std::make_unique<ConvertToyToArithPass>(options);
}
