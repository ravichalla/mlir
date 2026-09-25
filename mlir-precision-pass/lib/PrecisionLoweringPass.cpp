//===- PrecisionLoweringPass.cpp -----------------------------------------===//
//
// fp32 -> fp16 precision-lowering pass with conservative eligibility rules.
//
// Design notes (the part that matters for a verification-oriented review):
//
//   * We only lower ops whose *both* operands are already f32 SSA values
//     with a single use each in the op being lowered. This avoids silently
//     changing the precision of a value that's also consumed elsewhere at
//     f32 precision, which would be a correctness footgun disguised as an
//     optimization.
//
//   * Every lowered op gets f16 operands via arith::TruncFOp and produces
//     an f16 result that is immediately widened back to f32 via
//     arith::ExtFOp before being handed to any downstream consumer that
//     still expects f32. This keeps the pass *locally* sound: it changes
//     the precision of individual op evaluations without changing the
//     type signature of the surrounding function. It is intentionally NOT
//     optimal (a real pass would hoist casts and lower whole chains), but
//     that tradeoff is what makes it differential-testable in small,
//     attributable units -- each lowered op is independently checkable.
//
//   * Ops under an `arith.fastmath` attribute containing `contract` are
//     skipped: fused multiply-add chains are exactly where fp16
//     intermediate rounding causes the largest, hardest-to-bound errors,
//     so they're carved out of v1 rather than "optimized" past the point
//     where a differential test could catch a regression.
//
//===----------------------------------------------------------------------===//

#include "PrecisionLoweringPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

/// Returns true if `op` is a binary float arith op eligible for lowering
/// under the conservative rules described above.
template <typename OpTy>
static bool isEligible(OpTy op) {
  auto resultType = llvm::dyn_cast<FloatType>(op.getType());
  if (!resultType || !resultType.isF32())
    return false;

  // Skip fma/contraction-flagged ops -- see file header rationale.
  if (auto fastmath = op.getFastmathAttr()) {
    if (bitEnumContainsAny(fastmath.getValue(), arith::FastMathFlags::contract))
      return false;
  }

  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  return lhs.getType().isF32() && rhs.getType().isF32() &&
         lhs.hasOneUse() && rhs.hasOneUse();
}

/// Rewrites a single eligible op: truncf both operands to f16, clone the
/// op at f16, extf the result back to f32.
template <typename OpTy>
struct LowerBinaryFloatOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                 PatternRewriter &rewriter) const override {
    if (!isEligible(op))
      return failure();

    Location loc = op.getLoc();
    Type f16Ty = rewriter.getF16Type();

    Value lhs16 = rewriter.create<arith::TruncFOp>(loc, f16Ty, op.getLhs());
    Value rhs16 = rewriter.create<arith::TruncFOp>(loc, f16Ty, op.getRhs());

    Value result16 = rewriter.create<OpTy>(loc, f16Ty, lhs16, rhs16);

    Value result32 =
        rewriter.create<arith::ExtFOp>(loc, rewriter.getF32Type(), result16);

    rewriter.replaceOp(op, result32);
    return success();
  }
};

struct PrecisionLoweringPass
    : public PassWrapper<PrecisionLoweringPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrecisionLoweringPass)

  StringRef getArgument() const final { return "precision-lowering"; }
  StringRef getDescription() const final {
    return "Conservatively lower eligible f32 arith ops to f16 with "
           "boundary casts, for differential-testable precision "
           "experiments.";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerBinaryFloatOp<arith::AddFOp>,
                 LowerBinaryFloatOp<arith::SubFOp>,
                 LowerBinaryFloatOp<arith::MulFOp>>(&getContext());

    // Greedy application is fine here: patterns are non-overlapping in the
    // ops they match post-rewrite (the newly created f16 op doesn't match
    // isEligible again since its result type is f16, not f32), so this
    // naturally terminates in one pass over eligible ops.
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::precision::createPrecisionLoweringPass() {
  return std::make_unique<PrecisionLoweringPass>();
}

void mlir::precision::registerPrecisionLoweringPass() {
  PassRegistration<PrecisionLoweringPass>();
}
