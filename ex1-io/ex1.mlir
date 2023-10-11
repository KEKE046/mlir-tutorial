func.func @test(%a: i32, %b: i32) -> i32 {
  %c = arith.addi %a, %b : i32
  func.return %c : i32
}