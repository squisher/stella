#include <stdio.h>
#include <stdlib.h>

int fib(int x) {
    if (x <= 2) {
        return 1;
    } else {
        return fib(x-1) + fib(x-2);
    }
}

int main(int argc, char ** argv) {
    int i;
    long long r = 0;

    for (i=0; i<4; i++) {
        r += fib(42);
    }

    printf ("%lld\n", r);
    exit (0);
}
