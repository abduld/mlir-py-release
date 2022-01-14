from mlir.ir import *
from mlir.dialects import arith
from mlir.dialects import scf
from mlir.dialects import memref
from mlir.dialects import builtin 


def constructAndPrintInModule(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f() 
        module.dump()
    return f


@constructAndPrintInModule
def vectorAdd():
    N = 128
    memrefTy = MemRefType.get([N], F32Type.get())

    @builtin.FuncOp.from_py_func(memrefTy, memrefTy, memrefTy)
    def simple_loop(A, B, C):
        lb = arith.ConstantOp.create_index(0)
        ub = arith.ConstantOp.create_index(N)
        step = arith.ConstantOp.create_index(1)
        loop = scf.ForOp(lb, ub, step)
        with InsertionPoint(loop.body):
            idx = [loop.body.arguments[0]] 
            a = memref.LoadOp(A, idx)
            b = memref.LoadOp(B, idx)
            res = arith.AddFOp(a, b)
            memref.StoreOp(res, C, idx)
            # scf.YieldOp(loop.inner_iter_args)
        return C