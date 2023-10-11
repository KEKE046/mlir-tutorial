#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "toy/ToyDialect.h"
#include "toy/ToyOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
// #include "mlir/Conversion/LLVMCommon/TypeConverter.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#define GEN_PASS_DEF_CONVERTTOYTOARITH
#include "toy/ToyPasses.h"

#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace llvm;
using namespace toy;

struct AddOpPat: OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op, PatternRewriter & rewriter) const {
    auto inputs = to_vector(op.getInputs());
    auto result = inputs[0];
    for(size_t i = 1; i< inputs.size(); i++) {
      result = rewriter.create<arith::AddIOp>(op->getLoc(), result, inputs[i]);
    }
    rewriter.replaceOp(op, ValueRange(result));
    return success();
  }
};

struct SubOpPat: OpRewritePattern<SubOp> {
  using OpRewritePattern<SubOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SubOp op, PatternRewriter & rewriter) const {
    rewriter.replaceOpWithNewOp<arith::SubIOp>(op, op.getLhs(), op.getRhs());
    return success();
  }
};

struct ConstantOpPat: OpRewritePattern<ConstantOp> {
  using OpRewritePattern<ConstantOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ConstantOp op, PatternRewriter & rewriter) const {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValueAttr());
    return success();
  }
};

struct ConvertToyToArithPass : toy::impl::ConvertToyToArithBase<ConvertToyToArithPass> {
  using toy::impl::ConvertToyToArithBase<ConvertToyToArithPass>::ConvertToyToArithBase;
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect>();
    // registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() final {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();
    // target.addLegalDialect<LLVM::LLVMDialect>();
    // LLVMTypeConverter converter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<AddOpPat, SubOpPat, ConstantOpPat>(&getContext());
    // arith::populateArithToLLVMConversionPatterns(converter, patterns);
    if(failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> toy::createConvertToyToArithPass(ConvertToyToArithOptions options) {
  return std::make_unique<ConvertToyToArithPass>(options);
}
