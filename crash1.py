#!/usr/bin/env python

from stella import stella

# test3[f0-args0]

def power(a,b):          return a**b
delta = 1e-7

x = power(0,42)
print ("power(0,42) =", x)

stella(power, debug='print')(0,42)

y = stella(power)(0,42)
print ("stella(power)(0,42) =", y)
print ("met:", x-y < delta and type(x) == type(y))
