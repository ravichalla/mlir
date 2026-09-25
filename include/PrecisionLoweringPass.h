//===- PrecisionLoweringPass.h - fp32 -> fp16 lowering pass -------------===//
//
// Declares a pass that rewrites arith-dialect floating point ops from f32
// to f16, inserting truncf/extf casts at value boundaries so the rest of
// the IR remains type-consistent. Intended as a stand-in for the kind of
// precision-lowering transformation used ahead of GPU tensor-core codegen.
//
//===----------------------------------------------------------------------===//

#ifndef PRECISION_LOWERING_PASS_H
#define PRECISION_LOWERING_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace precision {

/// Creates a pass that lowers eligible f32 arith ops to f16.
///
/// "Eligible" here (deliberately conservative, see .cpp for rationale)
/// means: arith.addf, arith.subf, arith.mulf operating on plain f32
/// (not vector/tensor-of-f32, to keep the first version simple) where
/// neither operand is produced by a reduction/accumulation op already
/// flagged as precision-sensitive via an `arith.fastmath` attribute
/// opt-out.
std::unique_ptr<mlir::Pass> createPrecisionLoweringPass();

/// Registers the pass with MLIR's pass registry so it can be invoked as
/// `--precision-lowering` from precision-lower-opt / mlir-opt.
void registerPrecisionLoweringPass();

} // namespace precision
} // namespace mlir

#endif // PRECISION_LOWERING_PASS_H
