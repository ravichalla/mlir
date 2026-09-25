//===- precision-lower-opt.cpp -------------------------------------------===//
//
// Minimal mlir-opt-style driver: registers the arith/func dialects and our
// precision-lowering pass, then hands off to MLIR's standard opt main.
// This gives us a standalone binary (rather than requiring a patched
// mlir-opt) so the pass is easy to invoke from the differential harness.
//
//===----------------------------------------------------------------------===//

#include "PrecisionLoweringPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect>();

  mlir::precision::registerPrecisionLoweringPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "precision-lower-opt\n", registry));
}
