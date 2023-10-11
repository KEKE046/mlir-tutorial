!t32 = !toy.int<32>

toy.func @add(%a: !t32, %b: !t32) -> !t32 {
  %c = toy.add %a, %b : !t32
  toy.ret %c : !t32
}

toy.func @test(%a: !t32, %b: !t32) -> !t32 {
  %c = toy.call @add(%a, %b) : (!t32, !t32) -> !t32
  %d = toy.add %a, %b : !t32
  %f = toy.add %d, %d : !t32
  toy.ret %f : !t32
}