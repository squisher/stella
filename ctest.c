#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char ** argv) {
    printf ("floor(2.5) = %f\n", floor(2.5));
    printf ("5 / -2 = %d\n", 5 / -2);
    printf ("5.0 / -2 = %f\n", 5.0 / -2);
    printf ("5 / -2.0 = %f\n", 5 / -2.0);
    printf ("\n");

    /* test case chained() */
    int a = 341433;
    int b = 673069;
    printf ("a=%d, b=%dfor (a-b)/b*a;\n", a, b);
    float rf = ((a-b)/(float)b)*a;
    double rd = ((a-b)/(double)b)*a;
    printf ("Python =%10.10f\nStella =%10.10f\n", -168231.59941699886, -168231.609375);
    printf ("Cfloat =%10.10f\nCdouble=%10.10lf\n", rf, rd);

    printf ("-5 %% 2 = %d|%d\n", -5%2, (-5)%2);
    return 0;
}
