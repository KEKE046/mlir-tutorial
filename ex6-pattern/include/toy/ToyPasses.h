#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "toy/ToyOps.h"
#include <memory>

namespace toy {

#define GEN_PASS_DECL
#include "toy/ToyPasses.h.inc"

std::unique_ptr<mlir::Pass> createConvertToyToArithPass(ConvertToyToArithOptions options={});
std::unique_ptr<mlir::Pass> createDCEPass();

#define GEN_PASS_REGISTRATION
#include "toy/ToyPasses.h.inc"

}