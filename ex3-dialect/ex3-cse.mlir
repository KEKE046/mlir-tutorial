func.func @test(%a: i32, %b: i32) -> i32 {
  %c = "toy.add"(%a, %b): (i32, i32) -> i32
  %d = "toy.add"(%a, %b): (i32, i32) -> i32
  %e = "toy.add"(%c, %d): (i32, i32) -> i32
  %f = "toy.add"(%e, %e): (i32, i32) -> i32
  func.return %e : i32
}