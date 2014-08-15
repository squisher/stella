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


LLVMpy
------

Things to fix:
* abort() still sometimes called (llvm issue?) For example:
```
python3.2: Instructions.cpp:1488: llvm::InsertElementInst::InsertElementInst(llvm::Value *, llvm::Value *, llvm::Value *, const llvm::Twine &, llvm::Instruction *): Assertion `isValidOperands(Vec, Elt, Index) && "Invalid insertelement instruction operands!"' failed.
zsh: abort      ipython3 -i ipythoninit.py
``` @ `[master 992ea27] Store an array element (WIP)`
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

EuroPython 2014
---------------

Find a couple of example commits which crash llvm in current_work().

Papers
------

* The "cut" of python: what features the language has and how it fits the structure of certain programs
    * certain -> kmc simulations, what else? It needs to be broader, better defined.
    * "simple" programs?
    * it's not a DSL. But it's also not completely general. What's the right name for it?
        * it is "restricted", but I need to make sure it doesn't sound like "rpy"
* More is less - how less integration features better performing programs
    * easier to implement
    * is this big enough for a paper on its own or merge with next?
* OO features with zero run-time overhead for better structuring of high performance code
