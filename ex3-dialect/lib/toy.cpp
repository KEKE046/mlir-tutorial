#include "toy/ToyDialect.h"
#include "toy/ToyOps.h"

#include "toy/ToyDialect.cpp.inc"
#define GET_OP_CLASSES
#include "toy/Toy.cpp.inc"

using namespace mlir;
using namespace toy;

void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Toy.cpp.inc"
  >();
}
