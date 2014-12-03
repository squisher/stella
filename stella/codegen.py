#!/usr/bin/env python

import llvm
import llvm.core
import llvm.ee
import llvm.passes

import logging
import time

from . import tp
from . import utils
from . import ir


class CGEnv(object):
    module = None
    builder = None


class Program(object):
    def __init__(self, module):
        self.module = module
        self.module.translate()

        self.cge = CGEnv()
        self.cge.module = module

        self.llvm = self.makeStub()

        for _, func in self.module.namestore.all(ir.Function):
            self.blockAndCode(func)

    def blockAndCode(self, impl):
        func = impl.llvm
        # create blocks
        bb = func.append_basic_block("entry")

        for bc in impl.bytecodes:
            if bc.discard:
                impl.remove(bc)
                impl.log.debug("BLOCK skipped {0}".format(bc))
                continue

            newblock = ''
            if bc in impl.incoming_jumps:
                assert not bc.block
                bc.block = func.append_basic_block(str(bc.loc))
                bb = bc.block
                newblock = ' NEW BLOCK (' + str(bc.loc) + ')'
            else:
                bc.block = bb
            impl.log.debug("BLOCK'D {0}{1}".format(bc, newblock))

        for ext_module in self.module.getExternalModules():
            ext_module.translate(self.module.llvm)

        impl.log.debug("Printing all bytecodes:")
        impl.bytecodes.printAll(impl.log)

        impl.log.debug("Emitting code:")
        bb = None
        cge = self.cge
        for bc in impl.bytecodes:
            if bb != bc.block:
                # new basic block, use a new builder
                cge.builder = llvm.core.Builder.new(bc.block)

            bc.translate(cge)
            impl.log.debug("TRANS'D {0}".format(bc))
            # Note: the `and not' part is a basic form of dead code elimination
            #       This is used to drop unreachable "return None" which are implicitly added
            #       by Python to the end of functions.
            #       TODO is this the proper way to handle those returns? Any side effects?
            #            NEEDS REVIEW
            #       See also analysis.Function.analyze
            if isinstance(bc, utils.BlockTerminal) and \
                    bc.next and bc.next not in impl.incoming_jumps:
                impl.log.debug("TRANS stopping")
                break

    def makeStub(self):
        impl = self.module.entry
        func_tp = llvm.core.Type.function(impl.result.type.llvmType(), [])
        func = self.module.llvm.add_function(func_tp, str(impl.function)+'__stub__')
        bb = func.append_basic_block("entry")
        builder = llvm.core.Builder.new(bb)
        self.cge.builder = builder

        for name, var in self.module.namestore.all(ir.GlobalVariable):
            var.translate(self.cge)

        llvm_args = [arg.translate(self.cge) for arg in self.module.entry_args]

        call = builder.call(impl.llvm, llvm_args)

        if impl.result.type is tp.Void:
            builder.ret_void()
        else:
            builder.ret(call)
        return func

    def elapsed(self):
        if self.start is None or self.end is None:
            return None
        return self.end - self.start

    def optimize(self, opt):
        if opt is not None:
            logging.warn("Running optimizations level {0}... ".format(opt))
            self.module.llvm.verify()

            tm = llvm.ee.TargetMachine.new(opt=opt)
            pm = llvm.passes.build_pass_managers(tm, opt=opt, loop_vectorize=True, fpm=False).pm
            pm.run(self.module.llvm)

    def destruct(self):
        self.module.destruct()
        del self.module

    def __del__(self):
        logging.debug("DEL  {}: {}".format(repr(self), hasattr(self, 'module')))

    def run(self, stats):
        logging.debug("Verifying... ")
        self.module.llvm.verify()

        logging.debug("Preparing execution...")

        # m = Module.new('-lm')
        # fntp = Type.function(Type.float(), [Type.int()])
        # func = m.add_function(fntp, '__powidf2')

        import ctypes
        from llvmpy import _api
        clib = ctypes.cdll.LoadLibrary(_api.__file__)
        logging.debug(str(clib))

        # BUG: clib.__powidf2 gets turned into the following bytecode:
        # 103 LOAD_FAST                3 (clib)
        # 106 LOAD_ATTR               11 (_Program__powidf2)
        # 109 STORE_FAST               5 (f)
        # which is not correct. I have no idea where _Program is coming from,
        # I'm assuming it is some internal Python magic going wrong
        f = getattr(clib, '__powidf2')

        logging.debug(str(f))

        llvm.ee.dylib_add_symbol('__powidf2', ctypes.cast(f, ctypes.c_void_p).value)

        # ee = ExecutionEngine.new(self.module)
        eb = llvm.ee.EngineBuilder.new(self.module.llvm)

        logging.debug("Enabling mcjit...")
        eb.mcjit(True)

        ee = eb.create()

        entry = self.module.entry

        logging.info("running {0}{1}".format(entry,
                                             list(zip(entry.type_.arg_types,
                                                      self.module.entry_args))))

        # Now let's compile and run!

        time_start = time.time()
        retval = ee.run_function(self.llvm, [])
        stats['elapsed'] = time.time() - time_start

        for arg in self.module.entry_args:
            arg.copy2Python(self.cge)  # may be a no-op if not necessary
            arg.destruct()  # may be a no-op if not necessary

        logging.debug("Returning...")
        self.destruct()

        return tp.llvm_to_py(entry.result.type, retval)

    def getLlvmIR(self):
        ret = self.module.getLlvmIR()

        for arg in self.module.entry_args:
            arg.destruct()  # may be a no-op if not necessary

        logging.debug("Returning...")
        self.destruct()

        return ret
