add_mlir_dialect(Toy toy)


set(LLVM_TARGET_DEFINITIONS ToyPasses.td)
mlir_tablegen(ToyPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRToyTransformsIncGen)

add_custom_target(header DEPENDS MLIRToyIncGen MLIRToyTransformsIncGen)