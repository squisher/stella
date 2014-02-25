#!/usr/bin/env python3
from llvm import *
from llvm.core import *
from llvm.ee import *

def power_jit(arg1_value, arg2_value):
    my_module = Module.new('my_module')

    ty_int = Type.int(64)
    ty_double = Type.double()
    ty_func = Type.function(ty_double, [ty_double, ty_int])
    ty_int32 = Type.int(32)

    f_power = my_module.add_function(ty_func, "power")
    f_power.args[0].name = "a"
    f_power.args[1].name = "b"

    bb = f_power.append_basic_block("entry")
    builder = Builder.new(bb)
    tmp = builder.trunc(f_power.args[1], ty_int32, '(i32)b')
    llvm_pow = Function.intrinsic(my_module, INTR_POWI, [ty_double])
    pow_result = builder.call(llvm_pow, [f_power.args[0], tmp])
    builder.ret(pow_result)

    print ('# IR ------------------------------------------------------------')
    print (my_module)

    #ee = ExecutionEngine.new(my_module)
    eb = EngineBuilder.new(my_module)
    target = eb.select_target()

    print('# Assembly', str(target),'----------------------------------------')
    print(target.emit_assembly(my_module))


    print ('# Running ------------------------------------------------------------')
    ee = eb.create()

    arg1 = GenericValue.real(ty_double, arg1_value)
    arg2 = GenericValue.int(ty_int, arg2_value)

    retval = ee.run_function(f_power, [arg1, arg2])

    return retval.as_real(ty_double)

if __name__ == '__main__':
    print (power_jit(42.0,1))
