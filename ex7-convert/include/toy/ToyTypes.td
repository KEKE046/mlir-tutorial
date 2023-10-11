#ifndef TOY_TYPES_TD
#define TOY_TYPES_TD

include "toy/ToyDialect.td"
include "mlir/IR/AttrTypeBase.td"

class ToyType<string name, list<Trait> traits=[]>: TypeDef<ToyDialect, name, traits>;

def ToyInteger: ToyType<"ToyInteger"> {
  let mnemonic = "int";
  let parameters = (ins "uint64_t":$width);
  let assemblyFormat = "`<` $width `>`";
}

#endif
