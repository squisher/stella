//#include <stdlib.h>
//#include <stdio.h>

typedef struct {
    int x;
    int y;
} tCoord;

void setx(tCoord *t, int x) {
    t->x = x;
}

int main(int argc, char ** argv) {
    tCoord c[2];
    setx(&(c[1]), 1);
    return c[1].x;
}
