from math import log

cdef extern from "mtwist-1.1/mtwist.c":
    double mt_drand()
    void mt_seed32new(unsigned int)

def uniform():
    return mt_drand()

def exp(p):
    u = 1.0 - uniform()
    return -log(u)/p

def seed(s):
    mt_seed32new (s)
