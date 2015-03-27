//#include <stdlib.h>
//#include <stdio.h>

typedef struct {
    int x;
    long long y;
} tCoord;

tCoord* second(tCoord* ts) {
    return &(ts[1]);
}

int main(int argc, char ** argv) {
    tCoord c[2];
    tCoord *t;
    c[1].x = 42;
    t = second(c);
    return t->x;
}
