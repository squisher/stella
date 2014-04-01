Things to try
-------------

* pyflakes - https://pypi.python.org/pypi/pyflakes
    * https://github.com/fschulze/pytest-flakes
* codecheckers - https://bitbucket.org/RonnyPfannschmidt/pytest-codecheckers/
    * Dead? Has bugs with *-s*

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

 42           6 SETUP_LOOP              30 (to 39) 
              9 LOAD_GLOBAL              0 (range) 
             12 LOAD_FAST                0 (x) 
             15 CALL_FUNCTION            1 
             18 GET_ITER             
        >>   19 FOR_ITER                16 (to 38) 
             22 STORE_FAST               2 (i) 

 43          25 LOAD_FAST                1 (r) 
             28 LOAD_FAST                2 (i) 
             31 INPLACE_ADD          
             32 STORE_FAST               1 (r) 
             35 JUMP_ABSOLUTE           19 
        >>   38 POP_BLOCK            

 44     >>   39 LOAD_FAST                1 (r) 
             42 RETURN_VALUE         
```

_Idea_: use disassemble to create the bytecodes for the init, test and increment part of the for loop
```for i in range(x):```
=>
```
i=0 # init

if i>=x then jump end :test

$body

i=i+1
jump test

:end
```
