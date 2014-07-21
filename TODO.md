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

Stella:
```
define i64 @array_allocation() {
entry:
  %0 = alloca [5 x i64]
  %a = alloca [5 x i64]*
  store [5 x i64]* %0, [5 x i64]** %a
  ret i64 0
}
```

clang O0
```
define i32 @main(i32 %argc, i8** %argv) #0 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %i = alloca i32, align 4
  %a = alloca [5 x i32], align 16
  store i32 0, i32* %retval
  store i32 %argc, i32* %argc.addr, align 4
  store i8** %argv, i8*** %argv.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32* %i, align 4
  %cmp = icmp slt i32 %0, 5
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32* %i, align 4
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds [5 x i32]* %a, i32 0, i64 %idxprom
  store i32 42, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %2 = load i32* %i, align 4
  %inc = add nsw i32 %2, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %3 = load i32* %i, align 4
  ret i32 %3
}
```

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
