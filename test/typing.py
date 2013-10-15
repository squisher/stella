#!/usr/bin/env python

from stella import stella
from random import randint
from test import *

def return_bool(): return True
def return_arg(x): return x

make_eq_test(__name__, return_bool, [])
make_eq_test(__name__, return_arg, single_args([True, False, 0, 1, 42.0, -42.5]))

if __name__ == '__main__':
    print(stella(return_bool)())
