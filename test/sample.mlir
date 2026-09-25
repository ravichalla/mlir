// Sample IR exercising the precision-lowering pass on a small unrolled
// dot-product accumulation -- the same pattern that shows up in the inner
// loop of a matmul tile before vectorization/codegen.
//
// Run with:
//   precision-lower-opt test/sample.mlir --precision-lowering

func.func @dot4(%a0: f32, %a1: f32, %a2: f32, %a3: f32,
                 %b0: f32, %b1: f32, %b2: f32, %b3: f32) -> f32 {
  %p0 = arith.mulf %a0, %b0 : f32
  %p1 = arith.mulf %a1, %b1 : f32
  %p2 = arith.mulf %a2, %b2 : f32
  %p3 = arith.mulf %a3, %b3 : f32

  %s0 = arith.addf %p0, %p1 : f32
  %s1 = arith.addf %p2, %p3 : f32
  %sum = arith.addf %s0, %s1 : f32

  return %sum : f32
}

// A near-cancellation case included deliberately: this is the kind of
// input the differential harness's catastrophic-cancellation check is
// meant to flag if fp16 lowering pushes the relative error past tolerance.
func.func @near_cancellation(%x: f32) -> f32 {
  %big = arith.constant 1000.0 : f32
  %almost_big = arith.constant 999.9375 : f32   // exactly representable in f16
  %sum = arith.addf %big, %x : f32
  %diff = arith.subf %sum, %almost_big : f32
  return %diff : f32
}
