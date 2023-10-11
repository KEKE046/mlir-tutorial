#ifndef TOY_PASSES_TD
#define TOY_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertToyToArith : Pass<"convert-toy-to-arith"> {
  let summary = "Convert Toy To Arith";
  let constructor = "toy::createConvertToyToArithPass()";
  let options = [
    Option<"name", "name", "std::string", "", "help">
  ];
}

def DCE : Pass<"toy-dce", "toy::FuncOp"> {
  let summary = "dce";
  let constructor = "toy::createDCEPass()";
}
#endif
