Things to try
-------------

* pyflakes - https://pypi.python.org/pypi/pyflakes
    * https://github.com/fschulze/pytest-flakes
* codecheckers - https://bitbucket.org/RonnyPfannschmidt/pytest-codecheckers/
    * Dead? Has bugs with *-s*
* re-analysing the type of a function because a return type changed should be
  using a use-graph of function calls as to avoid unecessary re-types. For an
  example of this happening see langconstr.return_without_init().

Questions to ask
----------------

* Why doesn't LLVM use exceptions? abort() is harmful.
* Can py.test integrate faulthandler and tail stdout?

Current Work
------------

```
In [1]: dis(for1)
 41           0 LOAD_CONST               1 (0) 
              3 STORE_FAST               1 (r) 

 42           6 SETUP_LOOP              30 (to 39)      ForLoop
              9 LOAD_GLOBAL              0 (range)      $init
             12 LOAD_FAST                0 (x) 
             15 CALL_FUNCTION            1 
             18 GET_ITER             
        >>   19 FOR_ITER                16 (to 38)      
             22 STORE_FAST               2 (i) 

 43          25 LOAD_FAST                1 (r)          $body
             28 LOAD_FAST                2 (i) 
             31 INPLACE_ADD          
             32 STORE_FAST               1 (r) 
             35 JUMP_ABSOLUTE           19              $jump, move down one, place test and increment here
        >>   38 POP_BLOCK            

 44     >>   39 LOAD_FAST                1 (r) 
             42 RETURN_VALUE         
```

_Idea_: use disassemble to create the bytecodes for the init, test and increment part of the for loop
```for i in range(x):```
is not flexible enough
=>
```
i=0 # init

if i>=x then jump end :test

$body

i=i+1
jump test

:end
```

LLVMpy
------

Things to fix:
* abort() still sometimes called (llvm issue?)
* Exception not formatted correctly:
```
/home/squisher/usr/rsc/stella/stella/codegen.py in run(self, stats)
    101     def run(self, stats):
    102         logging.debug("Verifying... ")
--> 103         self.module.llvm.verify()
    104 
    105         logging.debug("Preparing execution...")

/home/squisher/usr/rsc/stella/llvmpy/build/lib.linux-x86_64-3.2/llvm/core.py in verify(self)
    591         broken = api.llvm.verifyModule(self._ptr, action, errio)
    592         if broken:
--> 593             raise llvm.LLVMException(errio.getvalue())
    594 
    595     def to_bitcode(self, fileobj=None):

LLVMException: b'Found return instr that returns non-void in Function of void return type!\n  ret i64 0\n voidBroken module found, compilation terminated.\nFound return instr that returns non-void in Function of void return type!\n  ret void <badref>\n voidBroken module found, compilation terminated.\nBroken module found, compilation terminated.\n'
```
  * This exception was thrown in current_work() in this state:
```
[master d9c8574] Add void type (WIP)
 3 files changed, 28 insertions(+), 1 deletion(-)
```
