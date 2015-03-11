#!/usr/bin/env python

import llvmlite.ir as ll
import llvmlite.binding as llvm

import logging
import time

from . import tp
from . import utils
from . import ir
from . import exc


class CGEnv(object):
    module = None
    builder = None


class Program(object):
    def __init__(self, module):
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()

        self.module = module
        self.module.translate()

        self.cge = CGEnv()
        self.cge.module = module

        self.llvm = self.makeStub()

        for _, func in self.module.namestore.all(ir.Function):
            self.blockAndCode(func)

        self.target_machine = llvm.Target.from_default_triple().create_target_machine()

        logging.debug("Verifying... ")
        self._llmod = None

    def llmod(self):
        if not self._llmod:
            self._llmod = llvm.parse_assembly(str(self.module.llvm))
        return self._llmod

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
            try:
                if bb != bc.block:
                    # new basic block, use a new builder
                    cge.builder = ll.IRBuilder(bc.block)

                bc.translate(cge)
                impl.log.debug("TRANS'D {0}".format(bc.locStr()))
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
            except exc.StellaException as e:
                e.addDebug(bc.debuginfo)
                raise

    def makeStub(self):
        impl = self.module.entry
        func_tp = ll.FunctionType(impl.result.type.llvmType(), [])
        func = ll.Function(self.module.llvm, func_tp, name=str(impl.function)+'__stub__')
        bb = func.append_basic_block("entry")
        builder = ll.IRBuilder(bb)
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

            # TODO was build_pass_managers(tm, opt=opt, loop_vectorize=True, fpm=False)
            pmb = llvm.create_pass_manager_builder()
            pmb.opt_level = opt
            pm = llvm.create_module_pass_manager()
            pmb.populate(pm)
            pm.run(self.llmod())

    def destruct(self):
        self.module.destruct()
        del self.module

    def __del__(self):
        logging.debug("DEL  {}: {}".format(repr(self), hasattr(self, 'module')))

    def run(self, stats):
        logging.debug("Preparing execution...")

        import ctypes
        import llvmlite
        import os

        _lib_dir = os.path.dirname(llvm.ffi.__file__)
        clib = ctypes.CDLL(os.path.join(_lib_dir, llvmlite.utils.get_library_name()))
        # Direct access as below mangles the name
        # f = clib.__powidf2
        f = getattr(clib, '__powidf2')
        llvm.add_symbol('__powidf2', ctypes.cast(f, ctypes.c_void_p).value)

        with llvm.create_mcjit_compiler(self.llmod(), self.target_machine) as ee:
            ee.finalize_object()

            entry = self.module.entry

            logging.info("running {0}{1}".format(entry,
                                                 list(zip(entry.type_.arg_types,
                                                          self.module.entry_args))))

            entry_ptr = ee.get_pointer_to_global(self.llmod().get_function(self.llvm.name))
            cfunc = ctypes.CFUNCTYPE(entry.result.type.Ctype())(entry_ptr)

            time_start = time.time()
            retval = cfunc()
            stats['elapsed'] = time.time() - time_start

        for arg in self.module.entry_args:
            arg.ctype2Python(self.cge)  # may be a no-op if not necessary
            arg.destruct()  # may be a no-op if not necessary

        ret_type = self.module.entry.result.type
        if isinstance(ret_type, tp.TupleType):
            retval = ret_type.unpack(retval)

        logging.debug("Returning...")
        self.destruct()

        return retval

    def getAssembly(self):
        return self.target_machine.emit_assembly(self.llmod())

    def getLlvmIR(self):
        ret = self.module.getLlvmIR()

        for arg in self.module.entry_args:
            arg.destruct()  # may be a no-op if not necessary

        logging.debug("Returning...")
        self.destruct()

        return ret
