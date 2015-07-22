#include <stdio.h>
#include <stdlib.h>

long long fib(long long x) {
    if (x <= 2) {
        return 1;
    } else {
        return fib(x-1) + fib(x-2);
    }
}

int main(int argc, char ** argv) {
    long long r = 0;
    const int {{x_init}};

    r += fib(x);

    printf ("%lld\\n", r);
    exit (0);
}
