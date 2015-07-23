#include <stdio.h>
#include <stdlib.h>

long long fib(long long x) {
    if (x == 0)
        return 1;
    if (x == 1)
        return 1;
    long long grandparent = 1;
    long long parent = 1;
    long long me = 0;
    int i;
    for (i=2; i<x; i++) {
        me = parent + grandparent;
        grandparent = parent;
        parent = me;
    }
    return me;
}

int main(int argc, char ** argv) {
    long long r = 0;
    const int {{x_init}};

    r += fib(x);

    printf ("%lld\n", r);
    exit (0);
}
