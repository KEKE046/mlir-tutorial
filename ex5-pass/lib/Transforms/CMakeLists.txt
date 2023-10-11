add_mlir_library(
  ToyTransforms
  ConvertToyToArith.cpp
  DCE.cpp
  DEPENDS
  MLIRToyTransformsIncGen
)