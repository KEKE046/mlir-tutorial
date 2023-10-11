#ifndef TOY_DIALECT_TD
#define TOY_DIALECT_TD

include "mlir/IR/OpBase.td"

def ToyDialect : Dialect {
  let name = "toy";
  let cppNamespace = "::toy";
  let summary = "Toy Dialect";
}

#endif
